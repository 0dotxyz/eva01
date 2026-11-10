//! Liquidation execution layer.
//!
//! The scan loop decides *which* accounts to liquidate and hands each one off as a
//! prepared liquidation. A [`LiquidationStrategy`] turns it into an [`ExecutionPlan`]
//! (an ordered set of transactions), and the executor lands it: a single transaction directly over
//! RPC, several as a Jito bundle with a sequential-RPC fallback. This separates the decision
//! (scanning/ranking) from execution (tx assembly + submission), and lets different funding
//! methods (inventory now, flashloan later) plug in behind the same trait.

pub mod executor;
pub mod inventory;
pub mod queue;
pub mod reservations;

use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;
use anyhow::Result;
use reservations::Claim;
use solana_transaction::versioned::VersionedTransaction;

/// An ordered set of transactions that, executed in sequence, perform one liquidation.
///
/// For the inventory strategy this is `[liquidate]` (collateral sold inside the tx, or the repay
/// paid from holdings), sent directly over RPC, or `[buy] [liquidate]`, submitted as a single
/// atomic Jito bundle with a sequential RPC fallback.
pub struct ExecutionPlan {
    /// Transactions in execution order. Earlier txs' state is visible to later ones.
    pub txs: Vec<VersionedTransaction>,
    /// The wallet tokens the plan spends or receives, held until it has landed or failed and the
    /// balances have been re-read. `None` for a plan that touches no wallet inventory.
    pub claim: Option<Claim>,
}

/// A way of funding and assembling a liquidation. Each implementation is a fully-configured
/// object (it owns the clients/cache it needs), so assembly only needs the intent.
///
/// Two kinds of plan, executed differently:
/// - a *sandwich* funds the repay from the seized collateral inside the liquidation tx, touches no
///   wallet inventory, and so can run in parallel with any other liquidation;
/// - an *inventory* plan spends the wallet's holdings (a JIT buy from swap_mint, or liability
///   already held). It claims the tokens it spends in the shared reservations, so any number of
///   them also run in parallel without two ever counting the same tokens.
pub trait LiquidationStrategy {
    /// Stable name for logs/metrics (e.g. "inventory", "flashloan").
    fn name(&self) -> &'static str;

    /// Assemble the self-funded single-tx plan, or say why the inventory plan is needed instead.
    fn assemble_sandwich(&self, intent: &PreparedLiquidatableAccount) -> Result<Sandwich>;

    /// Assemble the plan that spends wallet inventory, claiming the tokens it spends.
    fn assemble_with_inventory(
        &self,
        intent: &PreparedLiquidatableAccount,
    ) -> Result<ExecutionPlan>;
}

/// The outcome of trying to fund a liquidation from its own collateral.
pub enum Sandwich {
    Plan(ExecutionPlan),
    /// No sandwich (no route, doesn't fit, simulation fails): the inventory plan is next.
    Unavailable,
}
