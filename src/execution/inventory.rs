//! Liquidation strategy: every liquidation is one transaction.
//!
//! Preferred plan: start the receivership, withdraw the seized collateral, sell it for the
//! liability on a DEX, repay, end. The swap must guarantee enough liability to cover the whole
//! repay plus `min_profit` from the seized collateral alone, and the transaction must fit and
//! simulate. It touches no wallet inventory. The liability left over is the profit.
//!
//! Fallback, when no swap fits or quotes: repay from the liability the wallet already holds. The
//! rebalancer keeps a working balance of the common debt tokens for this. When the wallet holds
//! less than the repay, the liquidation shrinks to what it holds. The plan claims the tokens it
//! repays with, so concurrent plans never count the same tokens. The seized collateral stays in
//! the wallet for the rebalancer.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::{debug, info};
use solana_dex_superagg::{DexSuperAggClient, RouteRequest};
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;
use tokio::runtime::{Builder, Runtime};

use crate::clock_manager;
use crate::rebalancer::DexGate;
use crate::wrappers::{
    liquidator_account::{
        locked_accounts, LiquidatorAccount, PreparedLiquidatableAccount, MAX_TX_ACCOUNT_LOCKS,
    },
    oracle::OracleWrapper,
    token_account::TokenAccountWrapper,
};

use super::{
    reservations::TokenReservations, ExecutionPlan, Landing, LiquidationStrategy, Sandwich,
};

/// The share of a `repay_units` repay that `held` free wallet tokens cover, at most 1, or `None`
/// when they cover nothing.
fn wallet_share(repay_units: u64, held: u64) -> Option<I80F48> {
    if held == 0 || repay_units == 0 {
        return None;
    }
    Some(I80F48::from_num(held.min(repay_units)) / I80F48::from_num(repay_units))
}

/// Programs marginfi lets a swap call between the start and end of a receivership: the venues'
/// routers. The ATA program is whitelisted there too, but marginfi's instruction check rejects
/// any instruction with fewer than 8 bytes of data (`StartNotFirst`), which an ATA creation is,
/// so a route that creates an intermediate token account can never land; leaving the ATA program
/// out makes the aggregator pick a route without one instead.
const RECEIVERSHIP_SWAP_PROGRAMS: [Pubkey; 3] = [
    Pubkey::from_str_const("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"),
    Pubkey::from_str_const("T1TANpTeScyeqVzzgNViGDNrkQ6qHz9KrSBS4aNXvGT"),
    Pubkey::from_str_const("DF1ow4tspfHX9JwWJsAb9epbkA8hmpSEAtxXy1V27QBH"),
];

pub struct InventoryStrategy {
    liquidator_account: Arc<LiquidatorAccount>,
    dex_client: Arc<DexSuperAggClient>,
    dex_gate: Arc<DexGate>,
    tokio_rt: Runtime,
    slippage_bps: u16,
    min_profit: f64,
    reservations: Arc<TokenReservations>,
}

impl InventoryStrategy {
    pub fn new(
        liquidator_account: Arc<LiquidatorAccount>,
        dex_client: Arc<DexSuperAggClient>,
        dex_gate: Arc<DexGate>,
        slippage_bps: u16,
        min_profit: f64,
        reservations: Arc<TokenReservations>,
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
            slippage_bps,
            min_profit,
            reservations,
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

    /// Builds and simulates the one-transaction plan: the liquidation with the seized collateral
    /// sold for the liability inside it. The swap must guarantee the repay plus `min_profit` for
    /// at most the seized amount, and may only add as many accounts as the liquidation leaves
    /// under the account-lock limit, so the route is rejected by the aggregator rather than by
    /// the chain.
    fn build_sandwich_tx(
        &self,
        intent: &PreparedLiquidatableAccount,
        asset_mint: Pubkey,
        liab_mint: Pubkey,
        landing: Landing,
    ) -> Result<VersionedTransaction> {
        // What the withdraw hands the wallet, which is what the swap can sell.
        let asset_bank = self
            .liquidator_account
            .cache
            .banks
            .try_get_bank(&intent.asset_bank)?;
        let asset_amount: u64 = self
            .liquidator_account
            .seized_underlying_amount(&asset_bank, intent.asset_amount)?;
        let repay_amount: u64 = intent.liab_amount.to_num();
        let min_profit: u64 = self
            .value_wrapper(&liab_mint)?
            .get_amount_from_value(I80F48::from_num(self.min_profit))?
            .to_num();

        let (liquidation_ixs, _, _) = self.liquidator_account.liquidation_ixs(
            intent,
            intent.asset_amount,
            intent.liab_amount,
            None,
        )?;
        let transaction_accounts: Vec<Pubkey> =
            locked_accounts(&self.liquidator_account.signer.pubkey(), &liquidation_ixs)
                .into_iter()
                .collect();
        // A v0 transaction also loads the compute-budget program; a v1 header carries the budget.
        let budget_program = usize::from(landing == Landing::Bundle);
        let max_accounts =
            MAX_TX_ACCOUNT_LOCKS.saturating_sub(transaction_accounts.len() + budget_program);

        let request = RouteRequest {
            allowed_programs: Some(RECEIVERSHIP_SWAP_PROGRAMS.to_vec()),
            max_accounts,
            transaction_accounts,
            ..RouteRequest::new(
                self.liquidator_account.signer.pubkey(),
                asset_mint,
                liab_mint,
                asset_amount,
                repay_amount.saturating_add(min_profit),
                self.slippage_bps,
            )
        };
        let swap = self
            .tokio_rt
            .block_on(async {
                let _permit = self.dex_gate.acquire().await?;
                self.dex_client.build_swap_instructions(&request).await
            })?
            .swap_remove(0);
        info!(
            "InventoryStrategy: selling {} {} for at least {} {} in-tx via {:?} (repaying {})",
            asset_amount, asset_mint, swap.min_out_amount, liab_mint, swap.aggregator, repay_amount
        );
        let tx = self.liquidator_account.build_liquidate_tx(
            intent,
            intent.asset_amount,
            intent.liab_amount,
            Some(&swap),
            landing,
        )?;
        // A failure caught here falls back to the wallet-funded plan; the RPC preflight checks it
        // again. A bundled liquidation relies on prices its push has not landed yet, so the
        // executor simulates it inside the bundle instead.
        if landing == Landing::Rpc {
            self.liquidator_account.simulate_tx(&tx)?;
        }
        Ok(tx)
    }

    /// Builds the fallback plan: the liquidation repaid from the liability the wallet holds and no
    /// other plan has claimed. When that is less than the repay, the liquidation shrinks to it:
    /// the repay, and with it the seized collateral, since `end_liquidation` caps the latter at the
    /// repay plus the premium.
    ///
    /// The seized collateral stays in the wallet for the rebalancer to sell, except staked
    /// collateral, which is kept. Accounts holding staked collateral can only borrow SOL, which the
    /// wallet keeps a working balance of.
    fn build_wallet_plan(
        &self,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
        landing: Landing,
    ) -> Result<ExecutionPlan> {
        let mut claim = self.reservations.new_claim();
        let wallet_balance = self
            .liquidator_account
            .get_token_balance_for_mint(&liab_mint)
            .unwrap_or(0);
        let repay_units: u64 = intent.liab_amount.to_num();
        let held =
            self.reservations
                .claim_available(&mut claim, liab_mint, repay_units, wallet_balance);
        let Some(share) = wallet_share(repay_units, held) else {
            return Err(anyhow!(
                "the wallet holds no free {} to repay with ({} held, all claimed)",
                liab_mint,
                wallet_balance
            ));
        };

        let (asset_amount, repay_amount) = if share < I80F48::ONE {
            let profit = intent.profit * share;
            if profit < I80F48::from_num(self.min_profit) {
                return Err(anyhow!(
                    "the wallet's free {} {} covers {:.4} of the repay, for ${:.6} profit",
                    held,
                    liab_mint,
                    share.to_num::<f64>(),
                    profit.to_num::<f64>()
                ));
            }
            info!(
                "InventoryStrategy: repaying {} of {} {} from the wallet; liquidating {:.4} of the plan",
                held,
                repay_units,
                liab_mint,
                share.to_num::<f64>()
            );
            (intent.asset_amount * share, I80F48::from_num(held))
        } else {
            debug!(
                "InventoryStrategy: the wallet covers the repay of {} {}",
                repay_units, liab_mint
            );
            (intent.asset_amount, intent.liab_amount)
        };

        let tx = self.liquidator_account.build_liquidate_tx(
            intent,
            asset_amount,
            repay_amount,
            None,
            landing,
        )?;
        Ok(ExecutionPlan {
            tx,
            claim: Some(claim),
        })
    }
}

impl LiquidationStrategy for InventoryStrategy {
    fn name(&self) -> &'static str {
        "inventory"
    }

    fn assemble_sandwich(
        &self,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> Result<Sandwich> {
        let cache = &self.liquidator_account.cache;
        let liab_mint = cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint;
        let asset_mint = cache.banks.try_get_bank(&intent.asset_bank)?.bank.mint;

        match self.build_sandwich_tx(intent, asset_mint, liab_mint, landing) {
            Ok(tx) => Ok(Sandwich::Plan(ExecutionPlan { tx, claim: None })),
            Err(e) => {
                info!(
                    "InventoryStrategy: not selling the collateral in-tx for {} ({}); repaying from the wallet instead",
                    intent.liquidatee_account.address, e
                );
                Ok(Sandwich::Unavailable)
            }
        }
    }

    fn assemble_with_inventory(
        &self,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> Result<ExecutionPlan> {
        let cache = &self.liquidator_account.cache;
        let liab_mint = cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint;
        self.build_wallet_plan(intent, liab_mint, landing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixed_macro::types::I80F48;

    #[test]
    fn the_wallet_covers_a_share_of_the_repay() {
        assert_eq!(wallet_share(1000, 1000), Some(I80F48!(1)));
        assert_eq!(wallet_share(1000, 5000), Some(I80F48!(1)));
        assert_eq!(wallet_share(1000, 250), Some(I80F48!(0.25)));
    }

    #[test]
    fn an_empty_wallet_covers_nothing() {
        assert_eq!(wallet_share(1000, 0), None);
        assert_eq!(wallet_share(0, 1000), None);
    }
}
