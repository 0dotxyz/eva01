//! When eva last tried to liquidate, so rebalancing stays out of a burst's way.
//!
//! Rebalancing competes with liquidations for the swap venues' rate limits, the RPC and the
//! wallet's tokens, and nothing about it is urgent. It runs only once liquidations have been quiet
//! for a while, and a pass in progress stops as soon as one starts.
//!
//! An account that keeps failing is retried each time its backoff expires, for as long as it stays
//! liquidatable. That is no burst, so its retries don't count: after a failure, attempts on the
//! same account are ignored until one succeeds or it has rested for [`REPEAT_FAILURE_WINDOW`].

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use solana_pubkey::Pubkey;

/// Rebalancing waits this long after the last liquidation attempt.
pub const REBALANCE_QUIET_PERIOD: Duration = Duration::from_secs(5 * 60);

/// Rebalancing runs at least this often, quiet or not (between attempts, and still stopping for
/// one), so the seized collateral is sold and the inventory refilled even in a long burst.
pub const MAX_REBALANCE_INTERVAL: Duration = Duration::from_secs(30 * 60);

/// How long after a failed attempt the same account's next attempts are taken for retries.
pub const REPEAT_FAILURE_WINDOW: Duration = MAX_REBALANCE_INTERVAL;

#[derive(Default)]
pub struct LiquidationActivity {
    /// The counted attempts in progress.
    in_flight: AtomicUsize,
    /// When the last counted attempt started or ended.
    last_attempt: Mutex<Option<Instant>>,
    /// When each account's last attempt failed, while within [`REPEAT_FAILURE_WINDOW`].
    failures: Mutex<HashMap<Pubkey, Instant>>,
}

/// One liquidation attempt in progress, until dropped. It fails unless marked succeeded.
pub struct Attempt<'a> {
    activity: &'a LiquidationActivity,
    liquidatee: Pubkey,
    /// Whether it counts as liquidation activity, which a retry after a failure doesn't.
    counted: bool,
    succeeded: bool,
}

impl Attempt<'_> {
    pub fn succeeded(&mut self) {
        self.succeeded = true;
    }
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        let activity = self.activity;
        let mut failures = activity.failures.lock().unwrap_or_else(|e| e.into_inner());
        if self.succeeded {
            failures.remove(&self.liquidatee);
        } else {
            failures.insert(self.liquidatee, Instant::now());
        }
        drop(failures);
        if self.counted {
            activity.in_flight.fetch_sub(1, Ordering::SeqCst);
            activity.mark();
        }
    }
}

impl LiquidationActivity {
    /// Records an attempt on `liquidatee` that lasts until the returned guard is dropped.
    pub fn attempt(&self, liquidatee: Pubkey) -> Attempt<'_> {
        let now = Instant::now();
        let counted = {
            let mut failures = self.failures.lock().unwrap_or_else(|e| e.into_inner());
            failures.retain(|_, at| now.saturating_duration_since(*at) < REPEAT_FAILURE_WINDOW);
            !failures.contains_key(&liquidatee)
        };
        if counted {
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            self.mark();
        }
        Attempt {
            activity: self,
            liquidatee,
            counted,
            succeeded: false,
        }
    }

    fn mark(&self) {
        *self.last_attempt.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    fn last_attempt(&self) -> Option<Instant> {
        *self.last_attempt.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether an attempt is in progress, or one started or ended after `since`.
    pub fn attempted_since(&self, since: Instant) -> bool {
        self.in_flight.load(Ordering::SeqCst) > 0
            || self.last_attempt().is_some_and(|at| at > since)
    }

    /// Whether a rebalancing pass should run now, the last one having run at `last_pass`.
    pub fn rebalance_due(&self, last_pass: Instant, now: Instant) -> bool {
        rebalance_due(
            self.in_flight.load(Ordering::SeqCst) > 0,
            self.last_attempt(),
            last_pass,
            now,
        )
    }
}

/// Never during an attempt. Otherwise once liquidations attempted since the last pass have been
/// quiet for [`REBALANCE_QUIET_PERIOD`], and at least every [`MAX_REBALANCE_INTERVAL`].
fn rebalance_due(
    in_flight: bool,
    last_attempt: Option<Instant>,
    last_pass: Instant,
    now: Instant,
) -> bool {
    if in_flight {
        return false;
    }
    if now.saturating_duration_since(last_pass) >= MAX_REBALANCE_INTERVAL {
        return true;
    }
    last_attempt.is_some_and(|at| {
        at > last_pass && now.saturating_duration_since(at) >= REBALANCE_QUIET_PERIOD
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebalancing_waits_for_liquidations_to_go_quiet() {
        let pass = Instant::now();
        let minute = Duration::from_secs(60);
        let attempt = pass + minute;

        // Nothing attempted since the pass: only the interval brings the next one.
        assert!(!rebalance_due(false, None, pass, pass + 10 * minute));
        assert!(rebalance_due(
            false,
            None,
            pass,
            pass + MAX_REBALANCE_INTERVAL
        ));
        // An attempt since: due once it has been quiet long enough.
        assert!(!rebalance_due(
            false,
            Some(attempt),
            pass,
            attempt + 4 * minute
        ));
        assert!(rebalance_due(
            false,
            Some(attempt),
            pass,
            attempt + REBALANCE_QUIET_PERIOD
        ));
        // Never while an attempt is in progress, however long it has been.
        assert!(!rebalance_due(
            true,
            Some(attempt),
            pass,
            pass + 2 * MAX_REBALANCE_INTERVAL
        ));
    }

    #[test]
    fn a_pass_notices_an_attempt_that_starts_or_ends_during_it() {
        let activity = LiquidationActivity::default();
        let started = Instant::now();
        assert!(!activity.attempted_since(started));
        let mut attempt = activity.attempt(Pubkey::new_unique());
        assert!(activity.attempted_since(started));
        attempt.succeeded();
        drop(attempt);
        assert!(activity.attempted_since(started));
        assert!(!activity.attempted_since(Instant::now()));
    }

    #[test]
    fn retries_of_a_failing_account_are_no_activity() {
        let activity = LiquidationActivity::default();
        let (failing, other) = (Pubkey::new_unique(), Pubkey::new_unique());
        // Its first failure counts: it may be the start of a burst.
        drop(activity.attempt(failing));

        let after_failure = Instant::now();
        let retry = activity.attempt(failing);
        assert!(
            !activity.attempted_since(after_failure),
            "a retry in progress"
        );
        drop(retry);
        assert!(
            !activity.attempted_since(after_failure),
            "a retry that failed again"
        );

        // Another account still counts, and a success makes the failing one count again.
        drop(activity.attempt(other));
        assert!(activity.attempted_since(after_failure));
        let mut retry = activity.attempt(failing);
        retry.succeeded();
        drop(retry);
        let after_success = Instant::now();
        drop(activity.attempt(failing));
        assert!(activity.attempted_since(after_success));
    }
}
