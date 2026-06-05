//! Inventory-funded liquidation strategy.
//!
//! Funds each liquidation just-in-time: it buys only the *shortfall* of the liability token
//! (from the swap_mint reserve, e.g. USDC) via a separate Jupiter **ExactOut** swap tx, then
//! liquidates. The seized collateral is sold back to swap_mint by the rebalancer on a later
//! pass. No pre-held per-liability inventory is required beyond the swap_mint reserve.
//!
//! Produces an ordered plan `[buy?] [liquidate]`; the executor handles simulate-first cranking,
//! the Jito tip, and submission.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use jupiter_swap_api_client::{
    quote::{QuoteRequest, SwapMode},
    swap::SwapRequest,
    transaction_config::TransactionConfig,
    JupiterSwapApiClient,
};
use log::{debug, info};
use solana_program::pubkey::Pubkey;
use solana_sdk::{signer::Signer, transaction::VersionedTransaction};
use tokio::runtime::{Builder, Runtime};

use crate::wrappers::liquidator_account::{LiquidatorAccount, PROFIT_SHARE};

use super::{ExecutionPlan, LiquidationIntent, LiquidationStrategy};

pub struct InventoryStrategy {
    liquidator_account: Arc<LiquidatorAccount>,
    jupiter: JupiterSwapApiClient,
    tokio_rt: Runtime,
    swap_mint: Pubkey,
    slippage_bps: u16,
}

impl InventoryStrategy {
    pub fn new(
        liquidator_account: Arc<LiquidatorAccount>,
        swap_mint: Pubkey,
        jup_swap_api_url: String,
        jupiter_api_key: String,
        slippage_bps: u16,
    ) -> Result<Self> {
        let jupiter = JupiterSwapApiClient::new(jup_swap_api_url, jupiter_api_key)
            .map_err(|e| anyhow!("Failed to build Jupiter client: {e}"))?;
        let tokio_rt = Builder::new_current_thread()
            .thread_name("inventory-strategy")
            .enable_all()
            .build()?;

        Ok(Self {
            liquidator_account,
            jupiter,
            tokio_rt,
            swap_mint,
            slippage_bps,
        })
    }

    /// Build a signed `swap_mint -> output_mint` ExactOut buy tx for exactly `out_amount` tokens.
    fn build_buy_tx(&self, output_mint: Pubkey, out_amount: u64) -> Result<VersionedTransaction> {
        let quote = self
            .tokio_rt
            .block_on(self.jupiter.quote(&QuoteRequest {
                input_mint: self.swap_mint,
                output_mint,
                amount: out_amount,
                swap_mode: Some(SwapMode::ExactOut),
                slippage_bps: self.slippage_bps,
                ..QuoteRequest::default()
            }))
            .map_err(|e| anyhow!("Jupiter quote failed: {e}"))?;

        let swap_resp = self
            .tokio_rt
            .block_on(self.jupiter.swap(
                &SwapRequest {
                    user_public_key: self.liquidator_account.signer.pubkey(),
                    quote_response: quote,
                    config: TransactionConfig {
                        // Keep WSOL as a plain SPL token in the ATA; never auto-wrap/unwrap.
                        wrap_and_unwrap_sol: false,
                        ..TransactionConfig::default()
                    },
                },
                None,
            ))
            .map_err(|e| anyhow!("Jupiter swap build failed: {e}"))?;

        // Jupiter returns a ready VersionedTransaction built for our wallet; sign it.
        let unsigned: VersionedTransaction = bincode::deserialize(&swap_resp.swap_transaction)
            .map_err(|e| anyhow!("Failed to deserialize Jupiter swap tx: {e}"))?;
        let signed =
            VersionedTransaction::try_new(unsigned.message, &[&self.liquidator_account.signer])?;
        Ok(signed)
    }
}

/// Amount of the liability token to buy: the part of `repay_amount` not already in the wallet.
/// Uses the same `u64` truncation as the repay ix so the buy lands exactly what repay consumes.
fn buy_shortfall(repay_amount: I80F48, wallet_balance: u64) -> u64 {
    repay_amount.to_num::<u64>().saturating_sub(wallet_balance)
}

impl LiquidationStrategy for InventoryStrategy {
    fn name(&self) -> &'static str {
        "inventory"
    }

    fn assemble(&self, intent: &LiquidationIntent) -> Result<Option<ExecutionPlan>> {
        let liab_bank = self
            .liquidator_account
            .cache
            .banks
            .try_get_bank(&intent.liab_bank)?;
        let liab_mint = liab_bank.bank.mint;

        // JIT-buy covers the liability, so we always liquidate the full amount (no proportional
        // reduction). The repay ix consumes `repay_amount` of the liab token.
        let asset_amount = intent.asset_amount;
        let repay_amount = intent
            .liab_amount
            .checked_mul(I80F48::from_num(1.0 - PROFIT_SHARE))
            .ok_or_else(|| anyhow!("repay amount overflow"))?;

        let wallet_liab = self
            .liquidator_account
            .get_token_balance_for_mint(&liab_mint)
            .unwrap_or(0);
        let shortfall = buy_shortfall(repay_amount, wallet_liab);

        let mut txs = Vec::new();
        let mut temp_luts = Vec::new();
        if shortfall > 0 {
            info!(
                "InventoryStrategy: buying shortfall of {} {} (have {}, need {})",
                shortfall,
                liab_mint,
                wallet_liab,
                repay_amount.to_num::<u64>()
            );
            txs.push(self.build_buy_tx(liab_mint, shortfall)?);
        } else {
            debug!(
                "InventoryStrategy: wallet already covers {} {}; no buy needed",
                repay_amount.to_num::<u64>(),
                liab_mint
            );
        }

        let (liquidate_tx, _ixs, temp_lut) =
            self.liquidator_account
                .build_liquidate_tx(intent, asset_amount, repay_amount)?;
        txs.push(liquidate_tx);
        if let Some(lut_key) = temp_lut {
            temp_luts.push(lut_key);
        }

        Ok(Some(ExecutionPlan {
            txs,
            est_profit: intent.profit,
            // Full lamport cost (crank + tip + fees) is owned by the executor / a follow-up;
            // the executor currently gates on est_profit.
            est_cost_lamports: 0,
            temp_luts,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixed_macro::types::I80F48;

    #[test]
    fn test_buy_shortfall_partial() {
        assert_eq!(buy_shortfall(I80F48!(1000), 600), 400);
    }

    #[test]
    fn test_buy_shortfall_covered() {
        assert_eq!(buy_shortfall(I80F48!(1000), 1000), 0);
        assert_eq!(buy_shortfall(I80F48!(1000), 1500), 0);
    }

    #[test]
    fn test_buy_shortfall_empty_wallet() {
        assert_eq!(buy_shortfall(I80F48!(1000), 0), 1000);
    }

    #[test]
    fn test_buy_shortfall_truncates_like_repay() {
        // repay ix uses to_num::<u64>() (truncation); the shortfall must match it.
        assert_eq!(buy_shortfall(I80F48!(1000.9), 0), 1000);
    }
}
