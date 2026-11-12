//! Liquidation intents waiting for an execution worker.
//!
//! The queue holds the newest evaluation of each account. The liquidator loop replaces intents as
//! it re-evaluates accounts, and the workers take the most profitable one nobody is working, so a
//! slow send occupies one worker and never holds up evaluation or the other intents.

use std::{
    collections::{HashMap, HashSet},
    sync::{Condvar, Mutex, MutexGuard},
    time::{Duration, Instant},
};

use solana_pubkey::Pubkey;

use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;

/// How long a waiting worker sleeps before re-checking cooldowns and the stop flag.
const WORKER_WAIT: Duration = Duration::from_millis(250);

pub struct QueuedIntent {
    pub intent: PreparedLiquidatableAccount,
    pub queued_at: Instant,
}

pub struct ExecutionQueue {
    state: Mutex<State>,
    ready: Condvar,
    /// How long an account is left alone after an attempt on it.
    cooldown: Duration,
}

#[derive(Default)]
struct State {
    pending: HashMap<Pubkey, QueuedIntent>,
    in_flight: HashSet<Pubkey>,
    cooldowns: HashMap<Pubkey, Instant>,
    stopped: bool,
}

impl ExecutionQueue {
    pub fn new(cooldown: Duration) -> Self {
        Self {
            state: Mutex::new(State::default()),
            ready: Condvar::new(),
            cooldown,
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
            if let Some(queued) = state.take_ready(Instant::now()) {
                return Some(queued);
            }
            state = self
                .ready
                .wait_timeout(state, WORKER_WAIT)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Marks the attempt on `address` finished; it may be taken again after the cooldown.
    pub fn release(&self, address: Pubkey) {
        let mut state = self.lock();
        state.in_flight.remove(&address);
        state
            .cooldowns
            .insert(address, Instant::now() + self.cooldown);
        drop(state);
        self.ready.notify_all();
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
            push_oracles: Vec::new(),
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

        queue.release(a);
        assert_eq!(take_now(&queue), Some((a, I80F48::from_num(2))));
    }

    #[test]
    fn an_account_cools_down_after_an_attempt() {
        let queue = ExecutionQueue::new(Duration::from_secs(3));
        let a = Pubkey::new_unique();
        queue.offer(&[a], vec![intent(a, 1)]);
        assert!(take_now(&queue).is_some());
        queue.release(a);
        queue.offer(&[a], vec![intent(a, 1)]);

        let now = Instant::now();
        assert!(queue.lock().take_ready(now).is_none());
        assert!(queue
            .lock()
            .take_ready(now + Duration::from_secs(4))
            .is_some());
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
