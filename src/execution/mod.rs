//! Liquidation execution layer.
//!
//! The scan loop decides *which* accounts to liquidate and hands each one off as a
//! prepared liquidation. A [`LiquidationStrategy`] turns it into an [`ExecutionPlan`]
//! (an ordered set of transactions), and the executor lands it as a Jito bundle with a
//! sequential-RPC fallback. This separates the decision (scanning/ranking) from execution
//! (tx assembly + submission), and lets different funding methods (inventory now, flashloan
//! later) plug in behind the same trait.

pub mod executor;
pub mod inventory;

use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;
use anyhow::Result;
use solana_transaction::versioned::VersionedTransaction;

/// An ordered set of transactions that, executed in sequence, perform one liquidation.
///
/// For the inventory strategy this is `[liquidate]` (collateral sold inside the tx), sent directly
/// over RPC, or `[buy] [liquidate]`, submitted as a single atomic Jito bundle with a sequential RPC
/// fallback.
pub struct ExecutionPlan {
    /// Transactions in execution order. Earlier txs' state is visible to later ones.
    pub txs: Vec<VersionedTransaction>,
}

/// A way of funding and assembling a liquidation. Each implementation is a fully-configured
/// object (it owns the clients/cache it needs), so assembly only needs the intent.
///
/// Two kinds of plan, executed differently:
/// - a *sandwich* funds the repay from the seized collateral inside the liquidation tx, touches no
///   wallet inventory, and so can run in parallel with any other liquidation;
/// - an *inventory* plan spends the wallet's holdings (a JIT buy from swap_mint, or liability
///   already held), so only one may be assembled and landed at a time: sizing reads the wallet
///   balance, and two plans sized against the same balance would both believe they own it.
pub trait LiquidationStrategy {
    /// Stable name for logs/metrics (e.g. "inventory", "flashloan").
    fn name(&self) -> &'static str;

    /// Assemble the self-funded single-tx plan, or say why the inventory plan is needed instead.
    fn assemble_sandwich(&self, intent: &PreparedLiquidatableAccount) -> Result<Sandwich>;

    /// Assemble the plan that spends wallet inventory. Called with the inventory lock held, so the
    /// wallet balance it reads is stable until the plan has landed or failed. `collateral_sellable`
    /// is what the sandwich attempt learned: whether any venue quotes the seized collateral.
    fn assemble_with_inventory(
        &self,
        intent: &PreparedLiquidatableAccount,
        collateral_sellable: bool,
    ) -> Result<ExecutionPlan>;
}

/// The outcome of trying to fund a liquidation from its own collateral.
pub enum Sandwich {
    Plan(ExecutionPlan),
    /// No sandwich (no route, doesn't fit, simulation fails). `collateral_sellable` is false only
    /// when no venue quoted the collateral at all, which an inventory plan must respect for
    /// collateral the rebalancer could then never sell.
    Unavailable {
        collateral_sellable: bool,
    },
}
