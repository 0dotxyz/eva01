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
use solana_program::pubkey::Pubkey;
use solana_sdk::transaction::VersionedTransaction;

/// An ordered set of transactions that, executed in sequence, perform one liquidation.
///
/// For the inventory strategy this is `[crank?] [buy?] [liquidate]`. The executor submits
/// `txs` as a single atomic Jito bundle, falling back to sending them sequentially over RPC.
pub struct ExecutionPlan {
    /// Transactions in execution order. Earlier txs' state is visible to later ones.
    pub txs: Vec<VersionedTransaction>,
    /// Temporary LUTs created while assembling (e.g. to fit an oversized tx). The executor
    /// deactivates these after submission.
    pub temp_luts: Vec<Pubkey>,
}

/// A way of funding and assembling a liquidation. Each implementation is a fully-configured
/// object (it owns the clients/cache it needs), so assembly only needs the intent.
pub trait LiquidationStrategy {
    /// Stable name for logs/metrics (e.g. "inventory", "flashloan").
    fn name(&self) -> &'static str;

    /// Assemble the transactions for this intent, or `None` if this strategy cannot handle it
    /// (e.g. collateral whose extra refresh ixs won't fit) so the caller can skip or try another.
    fn assemble(&self, intent: &PreparedLiquidatableAccount) -> Result<Option<ExecutionPlan>>;
}
