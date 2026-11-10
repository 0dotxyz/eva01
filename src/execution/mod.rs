//! Liquidation execution layer.
//!
//! The scan loop decides *which* accounts to liquidate and hands each one off as a
//! prepared liquidation. A [`LiquidationStrategy`] turns it into an [`ExecutionPlan`], always
//! one transaction, and the executor sends it directly over RPC. This separates the decision
//! (scanning/ranking) from execution (tx assembly + submission), and lets different funding
//! methods plug in behind the same trait.

pub mod executor;
pub mod inventory;
pub mod queue;
pub mod reservations;

use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;
use anyhow::Result;
use reservations::Claim;
use solana_transaction::versioned::VersionedTransaction;

/// One liquidation in one transaction: the seized collateral sold inside it, or the repay paid
/// from the wallet's holdings.
pub struct ExecutionPlan {
    pub tx: VersionedTransaction,
    /// The wallet tokens the plan repays with, held until it has landed or failed and the
    /// balances have been re-read. `None` for a plan that touches no wallet inventory.
    pub claim: Option<Claim>,
}

/// How a liquidation will be landed, which decides how it is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    /// Sent alone over RPC: a v1 transaction, simulated while it is built.
    Rpc,
    /// Sent in a Jito bundle after a price push: a v0 transaction, since the block engine cannot
    /// decode v1, and not simulated alone, since it relies on prices the push has not landed
    /// yet. The executor simulates the whole bundle instead.
    Bundle,
}

/// A way of funding and assembling a liquidation. Each implementation is a fully-configured
/// object (it owns the clients/cache it needs), so assembly only needs the intent.
///
/// Two kinds of plan, executed differently:
/// - a *sandwich* funds the repay from the seized collateral inside the liquidation tx, touches no
///   wallet inventory, and so can run in parallel with any other liquidation;
/// - an *inventory* plan repays from the liability the wallet already holds. It claims the tokens
///   it repays with in the shared reservations, so any number of them also run in parallel
///   without two ever counting the same tokens.
pub trait LiquidationStrategy {
    /// Stable name for logs/metrics (e.g. "inventory", "flashloan").
    fn name(&self) -> &'static str;

    /// Assemble the self-funded single-tx plan, or say why the inventory plan is needed instead.
    fn assemble_sandwich(
        &self,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> Result<Sandwich>;

    /// Assemble the plan that spends wallet inventory, claiming the tokens it spends.
    fn assemble_with_inventory(
        &self,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> Result<ExecutionPlan>;
}

/// The outcome of trying to fund a liquidation from its own collateral. Short-lived, so the plan
/// stays inline.
#[allow(clippy::large_enum_variant)]
pub enum Sandwich {
    Plan(ExecutionPlan),
    /// No sandwich (no route, doesn't fit): the wallet-funded plan is next.
    Unavailable,
    /// The sandwich was built but its simulation failed. With newer prices than the chain holds,
    /// pushing them may make it land.
    Rejected,
}
