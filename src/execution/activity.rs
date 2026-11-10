//! When eva last tried to liquidate, so rebalancing stays out of a burst's way.
//!
//! Rebalancing competes with liquidations for the swap venues' rate limits, the RPC and the
//! wallet's tokens, and nothing about it is urgent. It runs only once liquidations have been quiet
//! for a while, and a pass in progress stops as soon as one starts.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

/// Rebalancing waits this long after the last liquidation attempt.
pub const REBALANCE_QUIET_PERIOD: Duration = Duration::from_secs(5 * 60);

/// Rebalancing runs at least this often, quiet or not (between attempts, and still stopping for
/// one), so the seized collateral is sold and the inventory refilled even in a long burst.
pub const MAX_REBALANCE_INTERVAL: Duration = Duration::from_secs(30 * 60);

#[derive(Default)]
pub struct LiquidationActivity {
    in_flight: AtomicUsize,
    /// When the last attempt started or ended.
    last_attempt: Mutex<Option<Instant>>,
}

/// One liquidation attempt in progress, until dropped.
pub struct Attempt<'a>(&'a LiquidationActivity);

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.0.mark();
    }
}

impl LiquidationActivity {
    /// Records an attempt that lasts until the returned guard is dropped.
    pub fn attempt(&self) -> Attempt<'_> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.mark();
        Attempt(self)
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
        let attempt = activity.attempt();
        assert!(activity.attempted_since(started));
        drop(attempt);
        assert!(activity.attempted_since(started));
        assert!(!activity.attempted_since(Instant::now()));
    }
}
