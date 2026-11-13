//! Liquidation execution layer.
//!
//! The scan loop decides *which* accounts to liquidate and hands each one off as a
//! prepared liquidation. A [`LiquidationStrategy`] turns it into an [`ExecutionPlan`], always
//! one transaction, and the executor sends it directly over RPC. This separates the decision
//! (scanning/ranking) from execution (tx assembly + submission), and lets different funding
//! methods plug in behind the same trait.

pub mod activity;
pub mod executor;
pub mod inventory;
pub mod queue;
pub mod reservations;

use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;
use anyhow::Result;
use fixed::types::I80F48;
use reservations::Claim;
use solana_transaction::versioned::VersionedTransaction;

/// One liquidation in one transaction: the seized collateral sold inside it, or the repay paid
/// from the wallet's holdings.
pub struct ExecutionPlan {
    pub tx: VersionedTransaction,
    /// The wallet tokens the plan repays with, held until it has landed or failed and the
    /// balances have been re-read. `None` for a plan that touches no wallet inventory.
    pub claim: Option<Claim>,
    /// What the plan is expected to earn (USD): the intent's profit, or the share of it the
    /// wallet can fund. Sizes the bundle tip and picks between plans.
    pub profit: I80F48,
}

/// How a liquidation will be landed, which decides how it is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    /// Sent alone over RPC: a v1 transaction, simulated while it is built.
    Rpc,
    /// Sent in a Jito bundle, behind a price push when it needs one: a v0 transaction, since the
    /// block engine cannot decode v1. Not simulated: a bundle that fails doesn't land and costs
    /// nothing, and one behind a push relies on prices the chain doesn't hold yet.
    Bundle,
    /// In a bundle whose earlier transaction refreshes the Kamino and JupLend accounts the
    /// liquidation reads (same slot, as both venues require), so the liquidation leaves those
    /// instructions out and its route has their room.
    BundleRefreshed,
}

impl Landing {
    /// Whether the liquidation refreshes the Kamino and JupLend accounts itself.
    pub fn refreshes_in_tx(self) -> bool {
        self != Landing::BundleRefreshed
    }

    pub fn is_bundle(self) -> bool {
        self != Landing::Rpc
    }
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
    /// Assemble the self-funded single-tx plan, or say why the inventory plan is needed instead.
    fn assemble_sandwich(
        &self,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> Result<Sandwich>;

    /// The SOL amount (lamports) worth `usd` at the oracle price, for sizing a tip.
    fn lamports_for_value(&self, usd: I80F48) -> Result<u64>;

    /// The Kamino and JupLend refreshes the liquidation of `intent` (or of its sandwich pair) relies
    /// on, in order; empty when it reads neither venue.
    fn refresh_ixs(
        &self,
        intent: &PreparedLiquidatableAccount,
    ) -> Result<Vec<solana_instruction::Instruction>>;

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
