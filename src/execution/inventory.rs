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
//! The buy never spends more than the shortfall's oracle value plus the liquidation's margin above
//! `min_profit`: anything the DEX charges above the oracle comes out of that margin.
//!
//! Produces an ordered plan `[buy?] [liquidate]`; the executor handles simulate-first cranking,
//! the Jito tip, and submission.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::{debug, info};
use marginfi_type_crate::constants::ASSET_TAG_STAKED;
use solana_dex_superagg::DexSuperAggClient;
use solana_program::pubkey::Pubkey;
use solana_sdk::transaction::VersionedTransaction;
use tokio::{
    runtime::{Builder, Runtime},
    sync::Semaphore,
};

use crate::clock_manager;
use crate::rebalancer::MIN_SELL_PRICE_VS_ORACLE;
use crate::wrappers::{
    liquidator_account::{LiquidatorAccount, PreparedLiquidatableAccount},
    oracle::OracleWrapper,
    token_account::TokenAccountWrapper,
};

use super::{ExecutionPlan, LiquidationStrategy};

/// Extra margin over the oracle-derived input estimate, to absorb the oracle/market price gap and
/// rounding so the first ExactIn quote usually already covers the shortfall (the quote is still
/// verified, and the input bumped, if it doesn't).
const INPUT_BUFFER: f64 = 1.02;

/// Liability still to buy: `repay_amount` truncated like the repay ix, minus what the wallet holds.
fn buy_shortfall(repay_amount: I80F48, wallet_balance: u64) -> u64 {
    repay_amount.to_num::<u64>().saturating_sub(wallet_balance)
}

pub struct InventoryStrategy {
    liquidator_account: Arc<LiquidatorAccount>,
    dex_client: Arc<DexSuperAggClient>,
    dex_gate: Arc<Semaphore>,
    tokio_rt: Runtime,
    swap_mint: Pubkey,
    min_profit: f64,
}

impl InventoryStrategy {
    pub fn new(
        liquidator_account: Arc<LiquidatorAccount>,
        swap_mint: Pubkey,
        dex_client: Arc<DexSuperAggClient>,
        dex_gate: Arc<Semaphore>,
        min_profit: f64,
    ) -> Result<Self> {
        // Multi-threaded so concurrent liquidations can `block_on` DEX quotes at the same time —
        // a current-thread runtime only supports one `block_on` at a time. A couple of worker
        // threads is plenty for the HTTP-bound DEX calls; parallelism comes from the callers.
        let tokio_rt = Builder::new_multi_thread()
            .thread_name("inventory-dex")
            .worker_threads(2)
            .enable_all()
            .build()?;

        Ok(Self {
            liquidator_account,
            dex_gate,
            dex_client,
            tokio_rt,
            swap_mint,
            min_profit,
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

    /// Refuse collateral the rebalancer couldn't sell afterwards (at [`MIN_SELL_PRICE_VS_ORACLE`] of
    /// its oracle value). For staked collateral: single-validator LSTs often have no DEX market.
    fn check_exit_liquidity(&self, asset_mint: Pubkey, asset_amount: u64) -> Result<()> {
        let asset_value = self
            .value_wrapper(&asset_mint)?
            .get_value_for_amount(I80F48::from_num(asset_amount))?;

        let quotes = self
            .tokio_rt
            .block_on(async {
                let _permit = self.dex_gate.acquire().await?;
                self.dex_client
                    .quote(asset_mint, self.swap_mint, asset_amount)
                    .await
            })
            .map_err(|e| anyhow!("no exit route for {} {}: {}", asset_amount, asset_mint, e))?;

        let best_out = quotes
            .first()
            .map(|(_, out_amount)| *out_amount)
            .unwrap_or(0);
        let exit_value = self
            .value_wrapper(&self.swap_mint)?
            .get_value_for_amount(I80F48::from_num(best_out))?;
        if exit_value < asset_value * MIN_SELL_PRICE_VS_ORACLE {
            return Err(anyhow!(
                "seized {} {} would sell for ${:.6} vs ${:.6} at the oracle; not seizing collateral \
                 the rebalancer can't exit",
                asset_amount,
                asset_mint,
                exit_value.to_num::<f64>(),
                asset_value.to_num::<f64>()
            ));
        }
        Ok(())
    }

    /// Build a signed `swap_mint -> output_mint` **ExactIn** buy tx that yields at least `min_out`
    /// of `output_mint`. Sizes the input from oracle prices, then delegates quoting, the capped
    /// input bump, transaction request, and signing to `solana-dex-superagg`.
    ///
    /// `profit_budget` is how much over-oracle cost (USD) the liquidation can absorb and still clear
    /// `min_profit`; the buy never spends more than the shortfall's oracle value plus that.
    fn build_buy_tx(
        &self,
        output_mint: Pubkey,
        min_out: u64,
        profit_budget: I80F48,
    ) -> Result<VersionedTransaction> {
        // Size the swap_mint input from oracle prices: value(min_out of liab) -> swap_mint amount.
        let needed_value = self
            .value_wrapper(&output_mint)?
            .get_value_for_amount(I80F48::from_num(min_out))?;
        let swap_wrapper = self.value_wrapper(&self.swap_mint)?;
        let to_swap_amount = |value: I80F48| -> Result<u64> {
            Ok(swap_wrapper.get_amount_from_value(value)?.to_num::<u64>())
        };

        let max_input = to_swap_amount(needed_value + profit_budget.max(I80F48::ZERO))?;
        let buffered_value = needed_value
            .checked_mul(I80F48::from_num(INPUT_BUFFER))
            .ok_or_else(|| anyhow!("input sizing overflow"))?;
        let input_amount = to_swap_amount(buffered_value)?.min(max_input);
        if input_amount == 0 {
            return Err(anyhow!(
                "computed zero swap_mint input for buying {} {}",
                min_out,
                output_mint
            ));
        }

        // One permit for the whole sequence: the min-out flow quotes, maybe re-quotes, then builds.
        let prepared = self
            .tokio_rt
            .block_on(async {
                let _permit = self.dex_gate.acquire().await?;
                self.dex_client
                    .build_swap_transaction(
                        self.swap_mint,
                        output_mint,
                        input_amount,
                        min_out,
                        Some(max_input),
                    )
                    .await
            })
            .map_err(|e| anyhow!("JIT buy of {} {}: {}", min_out, output_mint, e))?;
        debug!(
            "InventoryStrategy: ExactIn spends {} {} for {} {} via {:?} (need {}, cap {})",
            prepared.in_amount,
            self.swap_mint,
            prepared.out_amount,
            output_mint,
            prepared.aggregator,
            min_out,
            max_input
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

        let asset_bank = self
            .liquidator_account
            .cache
            .banks
            .try_get_bank(&intent.asset_bank)?;
        if asset_bank.bank.config.asset_tag == ASSET_TAG_STAKED {
            self.check_exit_liquidity(asset_bank.bank.mint, intent.asset_amount.to_num())?;
        }

        // JIT-buy covers the liability, so we always liquidate the full amount (no proportional
        // reduction). The repay ix consumes `repay_amount` of the liab token; the premium is
        // already the gap between the two amounts (see `compute_max_liquidatable_amounts_with_banks`).
        let asset_amount = intent.asset_amount;
        let repay_amount = intent.liab_amount;

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
            let profit_budget = intent.profit - I80F48::from_num(self.min_profit);
            txs.push(self.build_buy_tx(liab_mint, shortfall, profit_budget)?);
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
