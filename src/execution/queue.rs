//! Liquidation intents waiting for an execution worker.
//!
//! The queue holds the newest evaluation of each account. The liquidator loop replaces intents as
//! it re-evaluates accounts, and the workers take the most profitable one nobody is working, so a
//! slow send occupies one worker and never holds up evaluation or the other intents.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Condvar, Mutex, MutexGuard,
    },
    time::{Duration, Instant},
};

use solana_pubkey::Pubkey;

use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;

/// How long a waiting worker sleeps before re-checking cooldowns and the stop flag.
const WORKER_WAIT: Duration = Duration::from_millis(250);

/// Lost races in a row, with the account unchanged between them, that an account retries at once;
/// from the next one on it cools down, so a bundle that fails unseen every time (one behind a push
/// isn't simulated) can't be resent every cycle. A race lost to another liquidation changes the
/// account and starts the count over.
const FREE_RETRIES: u32 = 2;

/// Lost races further apart than this aren't in a row.
const LOST_RACE_WINDOW: Duration = Duration::from_secs(30);

pub struct QueuedIntent {
    pub intent: PreparedLiquidatableAccount,
    pub queued_at: Instant,
}

pub struct ExecutionQueue {
    state: Mutex<State>,
    ready: Condvar,
    /// How long an account is left alone after an attempt on it.
    cooldown: Duration,
    /// The newest slot Geyser delivered each account at.
    seen: Mutex<HashMap<Pubkey, u64>>,
    /// How many accounts wait for their state after a landing, so an update takes the state lock
    /// only when one may.
    awaiting_count: AtomicUsize,
}

#[derive(Default)]
struct State {
    pending: HashMap<Pubkey, QueuedIntent>,
    in_flight: HashSet<Pubkey>,
    cooldowns: HashMap<Pubkey, Instant>,
    /// Accounts cooling down after a landing, until Geyser delivers them at the slot it landed at.
    awaiting: HashMap<Pubkey, u64>,
    /// Lost races in a row per account, when the last was, and the slot Geyser had delivered the
    /// account at then.
    lost_races: HashMap<Pubkey, (u32, Instant, Option<u64>)>,
    stopped: bool,
}

impl ExecutionQueue {
    pub fn new(cooldown: Duration) -> Self {
        Self {
            state: Mutex::new(State::default()),
            ready: Condvar::new(),
            cooldown,
            seen: Mutex::new(HashMap::new()),
            awaiting_count: AtomicUsize::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replaces the intents of the `checked` accounts with `intents`: an account that was checked
    /// and is no longer liquidatable leaves the queue.
    pub fn offer(&self, checked: &[Pubkey], intents: Vec<PreparedLiquidatableAccount>) {
        let mut state = self.lock();
        for address in checked {
            state.pending.remove(address);
        }
        let queued_at = Instant::now();
        for intent in intents {
            state.pending.insert(
                intent.liquidatee_account.address,
                QueuedIntent { intent, queued_at },
            );
        }
        drop(state);
        self.ready.notify_all();
    }

    /// Blocks until an intent is ready and hands it out; `None` once the queue is stopped.
    pub fn take(&self) -> Option<QueuedIntent> {
        let mut state = self.lock();
        loop {
            if state.stopped {
                return None;
            }
            let ready = state.take_ready(Instant::now());
            // Expired cooldowns no longer wait for their state.
            self.awaiting_count
                .store(state.awaiting.len(), Ordering::SeqCst);
            if let Some(queued) = ready {
                return Some(queued);
            }
            state = self
                .ready
                .wait_timeout(state, WORKER_WAIT)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Marks the attempt on `address` finished; it may be taken again after the cooldown, or at
    /// once without one (`cool_down` false).
    pub fn release(&self, address: Pubkey, cool_down: bool) {
        let mut state = self.lock();
        state.in_flight.remove(&address);
        if cool_down {
            state
                .cooldowns
                .insert(address, Instant::now() + self.cooldown);
        }
        drop(state);
        self.ready.notify_all();
    }

    /// After an attempt that landed at `slot`: the account may be taken again once Geyser has
    /// delivered it at that slot, which the next attempt is sized on, or after the cooldown at the
    /// latest.
    pub fn release_after_landing(&self, address: Pubkey, slot: u64) {
        let mut state = self.lock();
        state.in_flight.remove(&address);
        state.lost_races.remove(&address);
        // Waiting before `seen` is read: an update recorded after that read sees the count and
        // frees the account itself.
        state
            .cooldowns
            .insert(address, Instant::now() + self.cooldown);
        state.awaiting.insert(address, slot);
        self.awaiting_count
            .store(state.awaiting.len(), Ordering::SeqCst);
        let seen = self
            .seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&address)
            .copied();
        if seen.is_some_and(|seen| seen >= slot) {
            state.awaiting.remove(&address);
            state.cooldowns.remove(&address);
            self.awaiting_count
                .store(state.awaiting.len(), Ordering::SeqCst);
        }
        drop(state);
        self.ready.notify_all();
    }

    /// After an attempt that lost a race: the account may be taken again at once, unless it lost
    /// more than [`FREE_RETRIES`] in a row, when it cools down.
    pub fn release_after_lost_race(&self, address: Pubkey) {
        let now = Instant::now();
        let seen = self
            .seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&address)
            .copied();
        let cool_down = {
            let mut state = self.lock();
            state
                .lost_races
                .retain(|_, (_, last, _)| now.saturating_duration_since(*last) < LOST_RACE_WINDOW);
            let (count, last, at) = state.lost_races.entry(address).or_insert((0, now, seen));
            if *at != seen {
                *count = 0;
            }
            *count += 1;
            *last = now;
            *at = seen;
            *count > FREE_RETRIES
        };
        self.release(address, cool_down);
    }

    /// Geyser delivered `address` at `slot`: an account waiting for that state is free to go.
    pub fn account_updated(&self, address: Pubkey, slot: u64) {
        {
            let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
            let newest = seen.entry(address).or_insert(slot);
            *newest = (*newest).max(slot);
        }
        if self.awaiting_count.load(Ordering::SeqCst) == 0 {
            return;
        }
        let mut state = self.lock();
        if state
            .awaiting
            .get(&address)
            .is_some_and(|landed| slot >= *landed)
        {
            state.awaiting.remove(&address);
            state.cooldowns.remove(&address);
            self.awaiting_count
                .store(state.awaiting.len(), Ordering::SeqCst);
            drop(state);
            self.ready.notify_all();
        }
    }

    pub fn stop(&self) {
        self.lock().stopped = true;
        self.ready.notify_all();
    }
}

impl State {
    /// The most profitable intent not being worked or cooling down, now marked in flight.
    fn take_ready(&mut self, now: Instant) -> Option<QueuedIntent> {
        self.cooldowns.retain(|_, until| *until > now);
        let cooldowns = &self.cooldowns;
        self.awaiting
            .retain(|address, _| cooldowns.contains_key(address));
        let address = self
            .pending
            .iter()
            .filter(|(address, _)| {
                !self.in_flight.contains(*address) && !self.cooldowns.contains_key(*address)
            })
            .max_by_key(|(_, queued)| queued.intent.profit)
            .map(|(address, _)| *address)?;
        let queued = self.pending.remove(&address)?;
        self.in_flight.insert(address);
        Some(queued)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wrappers::marginfi_account::{MarginfiAccountWrapper, ObservationAccounts};
    use fixed::types::I80F48;

    fn intent(address: Pubkey, profit: i32) -> PreparedLiquidatableAccount {
        PreparedLiquidatableAccount {
            attempt_id: 0,
            liquidatee_account: MarginfiAccountWrapper::new(address, bytemuck::Zeroable::zeroed()),
            observation_accounts: ObservationAccounts {
                observation_accounts: vec![],
                bank_pks: vec![],
                kamino_reserves: Default::default(),
                juplend_states: Default::default(),
            },
            asset_bank: Pubkey::default(),
            liab_bank: Pubkey::default(),
            asset_amount: I80F48::ONE,
            liab_amount: I80F48::ONE,
            profit: I80F48::from_num(profit),
            estimated_repay_usd: profit as f64,
            binding_cap: "test",
            push: None,
            without_push: None,
            sandwich_pair: None,
        }
    }

    fn take_now(queue: &ExecutionQueue) -> Option<(Pubkey, I80F48)> {
        queue
            .lock()
            .take_ready(Instant::now())
            .map(|q| (q.intent.liquidatee_account.address, q.intent.profit))
    }

    #[test]
    fn hands_out_the_most_profitable_intent_first() {
        let queue = ExecutionQueue::new(Duration::ZERO);
        let (a, b, c) = (
            Pubkey::new_unique(),
            Pubkey::new_unique(),
            Pubkey::new_unique(),
        );
        queue.offer(&[a, b, c], vec![intent(a, 1), intent(b, 3), intent(c, 2)]);

        assert_eq!(take_now(&queue).unwrap().0, b);
        assert_eq!(take_now(&queue).unwrap().0, c);
        assert_eq!(take_now(&queue).unwrap().0, a);
        assert!(take_now(&queue).is_none());
    }

    #[test]
    fn a_re_evaluation_replaces_or_withdraws_the_intent() {
        let queue = ExecutionQueue::new(Duration::ZERO);
        let (a, b, untouched) = (
            Pubkey::new_unique(),
            Pubkey::new_unique(),
            Pubkey::new_unique(),
        );
        queue.offer(
            &[a, b, untouched],
            vec![intent(a, 1), intent(b, 1), intent(untouched, 1)],
        );

        // `a` re-evaluated with a new size, `b` re-evaluated and no longer liquidatable.
        queue.offer(&[a, b], vec![intent(a, 5)]);

        assert_eq!(take_now(&queue), Some((a, I80F48::from_num(5))));
        assert_eq!(take_now(&queue).unwrap().0, untouched);
        assert!(take_now(&queue).is_none());
    }

    #[test]
    fn an_account_in_flight_is_not_handed_out_twice() {
        let queue = ExecutionQueue::new(Duration::ZERO);
        let a = Pubkey::new_unique();
        queue.offer(&[a], vec![intent(a, 1)]);
        assert!(take_now(&queue).is_some());

        // Re-evaluated while the first attempt is still running.
        queue.offer(&[a], vec![intent(a, 2)]);
        assert!(take_now(&queue).is_none());

        queue.release(a, false);
        assert_eq!(take_now(&queue), Some((a, I80F48::from_num(2))));
    }

    #[test]
    fn an_account_cools_down_unless_its_attempt_lost_a_race() {
        let queue = ExecutionQueue::new(Duration::from_secs(3));
        let a = Pubkey::new_unique();
        queue.offer(&[a], vec![intent(a, 1)]);
        assert!(take_now(&queue).is_some());
        queue.release(a, true);
        queue.offer(&[a], vec![intent(a, 1)]);

        let now = Instant::now();
        assert!(queue.lock().take_ready(now).is_none());
        assert!(queue
            .lock()
            .take_ready(now + Duration::from_secs(4))
            .is_some());

        // A lost race changed nothing: it may go again at once.
        queue.release(a, false);
        queue.offer(&[a], vec![intent(a, 1)]);
        assert!(queue.lock().take_ready(Instant::now()).is_some());
    }

    #[test]
    fn after_a_landing_an_account_goes_once_geyser_delivers_its_state() {
        let queue = ExecutionQueue::new(Duration::from_secs(3));
        let (a, b) = (Pubkey::new_unique(), Pubkey::new_unique());
        for address in [a, b] {
            queue.offer(&[address], vec![intent(address, 1)]);
            assert!(take_now(&queue).is_some());
        }

        // Landed at slot 10; an older update changes nothing, the landing's own frees it.
        queue.release_after_landing(a, 10);
        queue.offer(&[a], vec![intent(a, 1)]);
        queue.account_updated(a, 9);
        assert!(queue.lock().take_ready(Instant::now()).is_none());
        queue.account_updated(a, 10);
        assert!(queue.lock().take_ready(Instant::now()).is_some());

        // Its state already delivered when the attempt ends: no wait at all.
        queue.account_updated(b, 20);
        queue.release_after_landing(b, 20);
        queue.offer(&[b], vec![intent(b, 1)]);
        assert!(queue.lock().take_ready(Instant::now()).is_some());
    }

    #[test]
    fn only_a_few_lost_races_in_a_row_retry_at_once() {
        let queue = ExecutionQueue::new(Duration::from_secs(3));
        let a = Pubkey::new_unique();
        let attempt = || {
            queue.offer(&[a], vec![intent(a, 1)]);
            take_now(&queue).is_some()
        };
        assert!(attempt());
        for _ in 0..FREE_RETRIES {
            queue.release_after_lost_race(a);
            assert!(attempt(), "retried at once");
        }
        queue.release_after_lost_race(a);
        assert!(!attempt(), "cooling down");

        // Lost to a liquidation that changed the account: retried at once.
        queue.lock().cooldowns.clear();
        assert!(attempt());
        queue.account_updated(a, 5);
        queue.release_after_lost_race(a);
        assert!(attempt());

        // A landing starts the count over (its state, at slot 5, is already delivered).
        queue.release_after_landing(a, 5);
        assert!(attempt());
        for _ in 0..FREE_RETRIES {
            queue.release_after_lost_race(a);
            assert!(attempt(), "retried at once");
        }
    }

    #[test]
    fn take_returns_none_once_stopped() {
        let queue = ExecutionQueue::new(Duration::ZERO);
        let a = Pubkey::new_unique();
        queue.offer(&[a], vec![intent(a, 1)]);
        queue.stop();

        assert!(queue.take().is_none());
    }
}
