//! Inventory-funded liquidation strategy.
//!
//! Funds each liquidation just-in-time: it buys the *shortfall* of the liability token (from the
//! swap_mint reserve, e.g. USDC) via a separate DEX **ExactIn** swap tx, then liquidates. The
//! seized collateral — and any liability bought in excess of what the repay consumes — is sold
//! back to swap_mint by the rebalancer on a later pass. No pre-held per-liability inventory is
//! required beyond the swap_mint reserve.
//!
//! ExactIn (rather than ExactOut) because many routes don't support ExactOut: we size the
//! swap_mint input from the liability's oracle price (+ a buffer), then let the DEX aggregator
//! verify the quote actually yields at least the shortfall, bumping the input once if it falls
//! short.
//!
//! Produces an ordered plan `[buy?] [liquidate]`; the executor handles simulate-first cranking,
//! the Jito tip, and submission.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::{debug, info};
use solana_dex_superagg::{buy_shortfall, DexSuperAggClient};
use solana_program::pubkey::Pubkey;
use solana_sdk::transaction::VersionedTransaction;
use tokio::runtime::{Builder, Runtime};

use crate::clock_manager;
use crate::wrappers::{
    liquidator_account::{LiquidatorAccount, PreparedLiquidatableAccount, PROFIT_SHARE},
    oracle::OracleWrapper,
    token_account::TokenAccountWrapper,
};

use super::{ExecutionPlan, LiquidationStrategy};

/// Extra margin over the oracle-derived input estimate, to absorb the oracle/market price gap and
/// rounding so the first ExactIn quote usually already covers the shortfall (the quote is still
/// verified, and the input bumped, if it doesn't).
const INPUT_BUFFER: f64 = 1.02;

pub struct InventoryStrategy {
    liquidator_account: Arc<LiquidatorAccount>,
    dex_client: Arc<DexSuperAggClient>,
    tokio_rt: Runtime,
    swap_mint: Pubkey,
    slippage_bps: u16,
}

impl InventoryStrategy {
    pub fn new(
        liquidator_account: Arc<LiquidatorAccount>,
        swap_mint: Pubkey,
        dex_client: Arc<DexSuperAggClient>,
        slippage_bps: u16,
    ) -> Result<Self> {
        let tokio_rt = Builder::new_current_thread()
            .thread_name("inventory-dex")
            .enable_all()
            .build()?;

        Ok(Self {
            liquidator_account,
            dex_client,
            tokio_rt,
            swap_mint,
            slippage_bps,
        })
    }

    /// A price-only token wrapper (balance 0) for `mint`, used solely for the oracle-price <-> token
    /// amount conversions when sizing the swap input. Avoids depending on a wallet token account for
    /// a liability we may hold none of.
    fn value_wrapper(&self, mint: &Pubkey) -> Result<TokenAccountWrapper<OracleWrapper>> {
        let cache = &self.liquidator_account.cache;
        let bank_address = cache.banks.try_get_account_for_mint(mint)?;
        let bank_wrapper = cache.banks.try_get_bank(&bank_address)?;
        let clock = clock_manager::get_clock(&cache.clock)?;
        let oracle_wrapper = OracleWrapper::build_lenient(cache, &clock, &bank_address)?;
        Ok(TokenAccountWrapper {
            balance: 0,
            bank_wrapper,
            oracle_wrapper,
        })
    }

    /// Build a signed `swap_mint -> output_mint` **ExactIn** buy tx that yields at least `min_out`
    /// of `output_mint`. Sizes the input from oracle prices, then delegates DEX quote validation,
    /// one retry bump, transaction request, and signing to `solana-dex-superagg`.
    fn build_buy_tx(&self, output_mint: Pubkey, min_out: u64) -> Result<VersionedTransaction> {
        // Size the swap_mint input from oracle prices: value(min_out of liab) -> swap_mint amount.
        let needed_value = self
            .value_wrapper(&output_mint)?
            .get_value_for_amount(I80F48::from_num(min_out))?;
        let swap_wrapper = self.value_wrapper(&self.swap_mint)?;
        let buffered_value = needed_value
            .checked_mul(I80F48::from_num(INPUT_BUFFER))
            .ok_or_else(|| anyhow!("input sizing overflow"))?;
        let input_amount = swap_wrapper
            .get_amount_from_value(buffered_value)?
            .to_num::<u64>();
        if input_amount == 0 {
            return Err(anyhow!(
                "computed zero swap_mint input for buying {} {}",
                min_out,
                output_mint
            ));
        }

        let mut route_config = self.dex_client.config().default_route_config();
        route_config.slippage_bps = Some(self.slippage_bps);
        route_config.wrap_and_unwrap_sol = false;

        let prepared = self.tokio_rt.block_on(
            self.dex_client
                .build_swap_transaction_for_min_out_with_route_config(
                    &self.swap_mint.to_string(),
                    &output_mint.to_string(),
                    input_amount,
                    min_out,
                    route_config,
                ),
        )?;
        debug!(
            "InventoryStrategy: ExactIn spends {} {} for {} {} (need {})",
            prepared.in_amount, self.swap_mint, prepared.out_amount, output_mint, min_out
        );

        Ok(prepared.transaction)
    }
}

impl LiquidationStrategy for InventoryStrategy {
    fn name(&self) -> &'static str {
        "inventory"
    }

    fn assemble(&self, intent: &PreparedLiquidatableAccount) -> Result<Option<ExecutionPlan>> {
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

        let (liquidate_tx, temp_lut) =
            self.liquidator_account
                .build_liquidate_tx(intent, asset_amount, repay_amount)?;
        txs.push(liquidate_tx);
        if let Some(lut_key) = temp_lut {
            temp_luts.push(lut_key);
        }

        Ok(Some(ExecutionPlan { txs, temp_luts }))
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
