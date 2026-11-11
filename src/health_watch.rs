//! Which borrowers a price or bank update can't have made liquidatable, so they skip evaluation.
//!
//! An account's maintenance health is H = A − L: its weighted assets less its weighted
//! liabilities, each a sum over positions of shares × unit value × weight, where a unit value is
//! the bank's maintenance price times its share value. If no position's unit value has moved by
//! more than r since the account was evaluated, H has moved by at most (A + L)·r. So the account
//! stays healthy while every move stays below its margin H / (A + L). Only the positions of the
//! banks that moved need checking: the others were checked when they last moved.

use std::collections::HashMap;

use solana_pubkey::Pubkey;

/// The share of an account's margin a move may use before it is evaluated again. The bound itself
/// is exact and assumes every position moves against the account at once; the other half is room
/// for what unit values may not capture exactly (an integration bank's exchange rate, a stake
/// pool's rate).
const MARGIN_USED: f64 = 0.5;

/// Below this margin an account is evaluated on every update of its banks: those are the accounts
/// most likely to become liquidatable next.
const MIN_WATCHED_MARGIN: f64 = 0.01;

/// An account's margin: the relative move of its positions' unit values that would bring its
/// health to zero in the worst case, its weighted assets falling and its weighted liabilities
/// rising by that much at once.
pub fn margin(weighted_assets: f64, weighted_liabilities: f64) -> f64 {
    (weighted_assets - weighted_liabilities) / (weighted_assets + weighted_liabilities)
}

/// One position of a watched account: its bank, side, and unit value when evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct WatchedPosition {
    pub bank: Pubkey,
    pub liability: bool,
    pub unit_value: f64,
}

/// A healthy account's margin and its positions' unit values, as last evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct Watch {
    /// The relative move of any position's unit value it tolerates.
    tolerance: f64,
    positions: Vec<WatchedPosition>,
}

impl Watch {
    /// The watch for an account with `margin` (H / (A + L)), or `None` when it is too close to the
    /// threshold to skip any update.
    pub fn new(margin: f64, positions: Vec<WatchedPosition>) -> Option<Self> {
        (margin.is_finite() && margin >= MIN_WATCHED_MARGIN).then_some(Self {
            tolerance: margin * MARGIN_USED,
            positions,
        })
    }

    /// Whether a position whose bank and side `current` holds (its unit value now) has moved by
    /// at least the tolerance.
    fn moved(&self, current: &HashMap<(Pubkey, bool), f64>) -> bool {
        self.positions.iter().any(|position| {
            current
                .get(&(position.bank, position.liability))
                .is_some_and(|now| {
                    position.unit_value <= 0.0
                        || (now / position.unit_value - 1.0).abs() >= self.tolerance
                })
        })
    }
}

#[derive(Default)]
pub struct HealthWatch {
    watches: HashMap<Pubkey, Watch>,
}

impl HealthWatch {
    /// What `account`'s last evaluation found: a watch when it is healthy with room to spare,
    /// otherwise nothing, so every update of its banks evaluates it.
    pub fn set(&mut self, account: Pubkey, watch: Option<Watch>) {
        match watch {
            Some(watch) => self.watches.insert(account, watch),
            None => self.watches.remove(&account),
        };
    }

    /// Whether `account` must be evaluated, given the current unit values of the banks that moved.
    pub fn needs_evaluation(
        &self,
        account: &Pubkey,
        current: &HashMap<(Pubkey, bool), f64>,
    ) -> bool {
        self.watches
            .get(account)
            .is_none_or(|watch| watch.moved(current))
    }

    /// Whether `account` is watched: healthy with room to spare when last evaluated, and no
    /// update has needed it evaluated since.
    pub fn is_watched(&self, account: &Pubkey) -> bool {
        self.watches.contains_key(account)
    }

    pub fn len(&self) -> usize {
        self.watches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(bank: Pubkey, liability: bool, unit_value: f64) -> WatchedPosition {
        WatchedPosition {
            bank,
            liability,
            unit_value,
        }
    }

    #[test]
    fn an_account_is_evaluated_once_a_move_reaches_its_margin() {
        let (sol, usdc) = (Pubkey::new_unique(), Pubkey::new_unique());
        let account = Pubkey::new_unique();
        let mut watch = HealthWatch::default();
        // A 10% margin tolerates 5% moves.
        watch.set(
            account,
            Watch::new(
                0.10,
                vec![position(sol, false, 100.0), position(usdc, true, 1.0)],
            ),
        );

        let moved =
            |bank: Pubkey, liability: bool, now: f64| HashMap::from([((bank, liability), now)]);
        assert!(!watch.needs_evaluation(&account, &moved(sol, false, 95.1)));
        assert!(watch.needs_evaluation(&account, &moved(sol, false, 94.9)));
        assert!(!watch.needs_evaluation(&account, &moved(usdc, true, 1.049)));
        assert!(watch.needs_evaluation(&account, &moved(usdc, true, 1.051)));
        // The other side of the same bank, or another bank, isn't this account's position.
        assert!(!watch.needs_evaluation(&account, &moved(sol, true, 50.0)));
        assert!(!watch.needs_evaluation(&account, &moved(Pubkey::new_unique(), false, 0.0)));
    }

    #[test]
    fn a_move_within_the_tolerance_leaves_health_positive_in_the_worst_case() {
        // 10 SOL at $100, weight 0.8, against $700 of USDC debt, weight 1.0.
        let (assets, liabilities) = (10.0 * 100.0 * 0.8, 700.0);
        let margin = margin(assets, liabilities);
        assert!((margin - 100.0 / 1500.0).abs() < 1e-12, "6.67%");

        // The worst case: collateral falls and debt rises by the same share.
        let health_after = |r: f64| assets * (1.0 - r) - liabilities * (1.0 + r);
        assert!(
            health_after(margin).abs() < 1e-9,
            "the full margin reaches zero"
        );
        let tolerance = margin * MARGIN_USED;
        assert!((tolerance - 0.0333).abs() < 1e-3, "3.33%");
        assert!(
            (health_after(tolerance) - 50.0).abs() < 1e-9,
            "$50 left at the tolerance"
        );

        // The watch skips the account up to that move in SOL, and evaluates it past it.
        let (sol, usdc, account) = (
            Pubkey::new_unique(),
            Pubkey::new_unique(),
            Pubkey::new_unique(),
        );
        let mut watch = HealthWatch::default();
        watch.set(
            account,
            Watch::new(
                margin,
                vec![position(sol, false, 100.0), position(usdc, true, 1.0)],
            ),
        );
        let sol_at = |price: f64| HashMap::from([((sol, false), price)]);
        assert!(!watch.needs_evaluation(&account, &sol_at(96.8)), "-3.2%");
        assert!(watch.needs_evaluation(&account, &sol_at(96.6)), "-3.4%");
    }

    #[test]
    fn an_account_too_close_to_the_threshold_is_never_watched() {
        // $1,000 against $990: a 0.5% margin, under the 1% floor.
        let thin = margin(1000.0, 990.0);
        assert!(thin < MIN_WATCHED_MARGIN);
        assert!(Watch::new(thin, vec![]).is_none());
        // $1,000 against $950: 2.6%, watched with a 1.3% tolerance.
        assert!(Watch::new(margin(1000.0, 950.0), vec![]).is_some());
    }

    #[test]
    fn unwatched_and_thin_margin_accounts_are_always_evaluated() {
        let account = Pubkey::new_unique();
        let mut watch = HealthWatch::default();
        let any_move = HashMap::new();
        assert!(
            watch.needs_evaluation(&account, &any_move),
            "never evaluated"
        );

        watch.set(account, Watch::new(0.005, vec![]));
        assert!(
            watch.needs_evaluation(&account, &any_move),
            "too thin to watch"
        );

        watch.set(account, Watch::new(0.5, vec![]));
        assert!(!watch.needs_evaluation(&account, &any_move));
        watch.set(account, None);
        assert!(
            watch.needs_evaluation(&account, &any_move),
            "forgotten once unhealthy"
        );
    }
}
