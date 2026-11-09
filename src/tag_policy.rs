//! Decides which unhealthy accounts to tag.
//!
//! A tag lets the receivership premium an account pays grow over time, so a liquidation that is
//! not worth it today may be later. The tag costs a transaction fee whether or not it pays off,
//! and it only pays off if we are the ones liquidating the account once the premium has grown.
//! Accounts that other liquidators are working get chunked away long before that, so an account
//! is tagged only after it has stayed unhealthy, with nobody touching its position, for
//! [`TAG_MIN_UNHEALTHY_FOR`].
//!
//! The policy only remembers what it saw since startup: after a restart the wait starts over.
//! Accounts already tagged need no tracking, since `liquidation_tagged_at` is on the account and
//! every evaluation reads it.

use fixed::types::I80F48;
use solana_pubkey::Pubkey;
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use crate::utils::rate_limiter::RateLimiter;

/// How long an account must stay unhealthy with unchanged balances before it is tagged.
const TAG_MIN_UNHEALTHY_FOR: Duration = Duration::from_secs(30 * 60);

/// Minimum time between tag attempts on the same account, in case a tag does not land.
const TAG_RETRY_INTERVAL: Duration = Duration::from_secs(600);

/// A watched account no evaluation has reported for this long (it was closed, or lost its
/// liabilities and is no longer evaluated) is forgotten. Watched accounts are re-evaluated on
/// every update to their banks or oracles, i.e. every few seconds.
const WATCH_TTL: Duration = Duration::from_secs(3600);

/// Active balances as `(shares, bank)` for deposits and for liabilities. Shares only change when
/// someone deposits, withdraws, borrows, repays or liquidates, not as interest accrues.
pub type Balances = (Vec<(I80F48, Pubkey)>, Vec<(I80F48, Pubkey)>);

/// An unhealthy account the latest evaluation chose not to liquidate.
pub struct Candidate<'a> {
    pub address: Pubkey,
    /// Untagged, and the premium a tag can grow to would make the liquidation worth it.
    pub taggable: bool,
    pub balances: &'a Balances,
}

struct Watch {
    /// Since when the account has been unhealthy with these balances.
    since: Instant,
    last_seen: Instant,
    balances: Balances,
}

pub struct TagPolicy {
    watch: HashMap<Pubkey, Watch>,
    attempts: RateLimiter<Pubkey>,
}

impl TagPolicy {
    pub fn new() -> Self {
        Self {
            watch: HashMap::new(),
            attempts: RateLimiter::new(TAG_RETRY_INTERVAL),
        }
    }

    /// Records one evaluation of the `checked` accounts, of which `candidates` were unhealthy but
    /// not liquidated, and returns the accounts due a tag now with how long each has waited. A
    /// returned account counts as attempted.
    pub fn observe<'a>(
        &mut self,
        checked: &[Pubkey],
        candidates: impl IntoIterator<Item = Candidate<'a>>,
        now: Instant,
    ) -> Vec<(Pubkey, Duration)> {
        let candidates: Vec<Candidate> = candidates.into_iter().collect();

        // A checked account that is no longer a candidate healed, was liquidated or became worth
        // liquidating: if it turns up again, its wait starts over.
        let still_candidates: HashSet<Pubkey> = candidates.iter().map(|c| c.address).collect();
        for address in checked {
            if !still_candidates.contains(address) {
                self.watch.remove(address);
            }
        }
        self.watch
            .retain(|_, watch| now.saturating_duration_since(watch.last_seen) < WATCH_TTL);
        self.attempts.prune(now);

        let mut due = Vec::new();
        for candidate in candidates {
            if !candidate.taggable {
                self.watch.remove(&candidate.address);
                continue;
            }
            let watch = self
                .watch
                .entry(candidate.address)
                .or_insert_with(|| Watch {
                    since: now,
                    last_seen: now,
                    balances: candidate.balances.clone(),
                });
            watch.last_seen = now;
            if watch.balances != *candidate.balances {
                watch.since = now;
                watch.balances = candidate.balances.clone();
            }
            let waited = now.saturating_duration_since(watch.since);
            if waited >= TAG_MIN_UNHEALTHY_FOR && self.attempts.allow(candidate.address, now) {
                due.push((candidate.address, waited));
            }
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn balances(liability_shares: i32) -> Balances {
        let bank = Pubkey::new_from_array([1; 32]);
        (vec![], vec![(I80F48::from_num(liability_shares), bank)])
    }

    fn candidate(address: Pubkey, balances: &Balances) -> Candidate<'_> {
        Candidate {
            address,
            taggable: true,
            balances,
        }
    }

    fn minutes(n: u64) -> Duration {
        Duration::from_secs(n * 60)
    }

    #[test]
    fn tags_only_after_thirty_minutes_unhealthy() {
        let mut policy = TagPolicy::new();
        let account = Pubkey::new_unique();
        let held = balances(100);
        let start = Instant::now();

        for at in [0, 10, 29] {
            let due = policy.observe(&[account], [candidate(account, &held)], start + minutes(at));
            assert!(due.is_empty(), "tagged after {at} min");
        }
        let due = policy.observe(&[account], [candidate(account, &held)], start + minutes(30));
        assert_eq!(due, vec![(account, minutes(30))]);
    }

    #[test]
    fn a_balance_change_restarts_the_wait() {
        let mut policy = TagPolicy::new();
        let account = Pubkey::new_unique();
        let (before, after) = (balances(100), balances(90));
        let start = Instant::now();

        policy.observe(&[account], [candidate(account, &before)], start);
        policy.observe(
            &[account],
            [candidate(account, &after)],
            start + minutes(20),
        );

        let due = policy.observe(
            &[account],
            [candidate(account, &after)],
            start + minutes(40),
        );
        assert!(due.is_empty());
        let due = policy.observe(
            &[account],
            [candidate(account, &after)],
            start + minutes(50),
        );
        assert_eq!(due, vec![(account, minutes(30))]);
    }

    #[test]
    fn leaving_the_candidates_restarts_the_wait() {
        let mut policy = TagPolicy::new();
        let account = Pubkey::new_unique();
        let held = balances(100);
        let start = Instant::now();

        policy.observe(&[account], [candidate(account, &held)], start);
        // Checked and healthy (or liquidatable) for one cycle.
        policy.observe(&[account], [], start + minutes(20));

        let due = policy.observe(&[account], [candidate(account, &held)], start + minutes(31));
        assert!(due.is_empty());
    }

    #[test]
    fn an_unchecked_account_keeps_its_wait() {
        let mut policy = TagPolicy::new();
        let (watched, other) = (Pubkey::new_unique(), Pubkey::new_unique());
        let held = balances(100);
        let start = Instant::now();

        policy.observe(&[watched], [candidate(watched, &held)], start);
        policy.observe(&[other], [], start + minutes(20));

        let due = policy.observe(&[watched], [candidate(watched, &held)], start + minutes(30));
        assert_eq!(due.len(), 1);
    }

    #[test]
    fn untaggable_accounts_are_never_due() {
        let mut policy = TagPolicy::new();
        let account = Pubkey::new_unique();
        let held = balances(100);
        let start = Instant::now();
        let untaggable = || Candidate {
            address: account,
            taggable: false,
            balances: &held,
        };

        policy.observe(&[account], [untaggable()], start);
        let due = policy.observe(&[account], [untaggable()], start + minutes(60));
        assert!(due.is_empty());
    }

    #[test]
    fn retries_a_tag_only_after_the_retry_interval() {
        let mut policy = TagPolicy::new();
        let account = Pubkey::new_unique();
        let held = balances(100);
        let start = Instant::now();
        let mut observe =
            |at: u64| policy.observe(&[account], [candidate(account, &held)], start + minutes(at));

        observe(0);
        assert_eq!(observe(30).len(), 1);
        assert!(observe(35).is_empty());
        assert_eq!(observe(40).len(), 1);
    }

    #[test]
    fn forgets_accounts_no_longer_evaluated() {
        let mut policy = TagPolicy::new();
        let account = Pubkey::new_unique();
        let held = balances(100);
        let start = Instant::now();

        policy.observe(&[account], [candidate(account, &held)], start);
        policy.observe(&[], [], start + minutes(61));

        assert!(policy.watch.is_empty());
    }
}
