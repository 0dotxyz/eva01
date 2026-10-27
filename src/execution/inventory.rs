//! Inventory-funded liquidation strategy.
//!
//! Preferred plan, one transaction: start the receivership, withdraw the seized collateral, sell
//! it for the liability on a DEX, repay, end. The swap must guarantee enough liability to cover
//! the whole repay plus `min_profit` from the seized collateral alone, and the transaction must
//! fit and simulate. The liability left over is the profit; the rebalancer sells it to swap_mint
//! later.
//!
//! Fallback, two transactions: buy the liability shortfall from the swap_mint reserve (USDC) in a
//! separate DEX swap, capped at what the wallet holds, then liquidate. The buy is ExactIn, since
//! many routes don't support ExactOut: the input is sized from the liability's oracle price plus a
//! buffer, and the aggregator raises it if the quote falls short. It never spends more than the
//! shortfall's oracle value plus the liquidation's margin above `min_profit`. The rebalancer sells
//! the seized collateral and any excess liability afterwards.
//!
//! A sandwich touches no wallet inventory, so the executor runs sandwiches in parallel and sends
//! each directly over RPC. The fallback spends the wallet, so the executor assembles and lands one
//! at a time, as a Jito bundle with a sequential RPC fallback.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::{debug, info};
use marginfi_type_crate::constants::ASSET_TAG_STAKED;
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

use super::{ExecutionPlan, LiquidationStrategy, Sandwich};

/// Liability still to buy: `repay_amount` truncated like the repay ix, minus what the wallet holds.
fn buy_shortfall(repay_amount: I80F48, wallet_balance: u64) -> u64 {
    repay_amount.to_num::<u64>().saturating_sub(wallet_balance)
}

/// Whether a failed sandwich means no venue would quote the collateral at all, as opposed to a
/// route that exists but pays too little, doesn't fit, or fails to build.
fn no_venue_quoted(error: &anyhow::Error) -> bool {
    let text = error.to_string();
    text.contains("No aggregator returned a quote") || text.contains("NO_ROUTES_FOUND")
}

/// swap_mint to spend on a buy: what the oracle says the shortfall costs, or everything the wallet
/// holds when that is less. A smaller buy still liquidates, just a smaller share; refusing would
/// liquidate nothing until the rebalancer refills the wallet.
fn affordable_input(wanted: u64, wallet_swap_mint: u64) -> u64 {
    wanted.min(wallet_swap_mint)
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
    swap_mint: Pubkey,
    slippage_bps: u16,
    min_profit: f64,
}

impl InventoryStrategy {
    pub fn new(
        liquidator_account: Arc<LiquidatorAccount>,
        swap_mint: Pubkey,
        dex_client: Arc<DexSuperAggClient>,
        dex_gate: Arc<DexGate>,
        slippage_bps: u16,
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
            slippage_bps,
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

    /// Builds a signed ExactIn buy of `output_mint`, spending the oracle value of `amount` in
    /// swap_mint, or all of the `wallet_swap_mint` the wallet holds if that is less. Returns the
    /// transaction and how much it guarantees to deliver, which may be less than `amount` when the
    /// market is worse than the oracle or the wallet can't afford the whole buy; the caller
    /// shrinks the liquidation to what it will hold.
    fn build_buy_tx(
        &self,
        output_mint: Pubkey,
        amount: u64,
        wallet_swap_mint: u64,
    ) -> Result<(VersionedTransaction, u64)> {
        let value = self
            .value_wrapper(&output_mint)?
            .get_value_for_amount(I80F48::from_num(amount))?;
        let wanted_input: u64 = self
            .value_wrapper(&self.swap_mint)?
            .get_amount_from_value(value)?
            .to_num();
        let input_amount = affordable_input(wanted_input, wallet_swap_mint);
        if input_amount == 0 {
            return Err(anyhow!(
                "no {} to buy {} {} with (wallet holds {}, wanted {})",
                self.swap_mint,
                amount,
                output_mint,
                wallet_swap_mint,
                wanted_input
            ));
        }
        if input_amount < wanted_input {
            info!(
                "InventoryStrategy: the wallet holds {} {}, short of the {} the buy wants; buying what it can",
                wallet_swap_mint, self.swap_mint, wanted_input
            );
        }

        let prepared = self
            .tokio_rt
            .block_on(async {
                let _permit = self.dex_gate.acquire().await?;
                self.dex_client
                    .build_swap_transaction(self.swap_mint, output_mint, input_amount, 0)
                    .await
            })
            .map_err(|e| anyhow!("JIT buy of {} {}: {}", amount, output_mint, e))?;
        debug!(
            "InventoryStrategy: spending {} {} for at least {} {} via {:?} (wanted {})",
            prepared.in_amount,
            self.swap_mint,
            prepared.min_out_amount,
            output_mint,
            prepared.aggregator,
            amount
        );
        Ok((prepared.transaction, prepared.min_out_amount))
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
        let max_accounts = MAX_TX_ACCOUNT_LOCKS.saturating_sub(transaction_accounts.len());

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
        )?;
        // A failure caught here falls back to the buy plan; the RPC preflight checks it again.
        self.liquidator_account.simulate_tx(&tx)?;
        Ok(tx)
    }

    /// Builds the fallback plan: buy the liability shortfall from swap_mint, then liquidate. When
    /// the buy guarantees less than the shortfall, the liquidation shrinks to what the wallet will
    /// hold: the repay, and with it the seized collateral, since `end_liquidation` caps the latter
    /// at the repay plus the premium.
    fn build_buy_plan(
        &self,
        intent: &PreparedLiquidatableAccount,
        asset_mint: Pubkey,
        asset_tag: u8,
        liab_mint: Pubkey,
        collateral_sellable: bool,
    ) -> Result<ExecutionPlan> {
        let mut asset_amount = intent.asset_amount;
        let mut repay_amount = intent.liab_amount;
        // Staked collateral is often a single-validator LST with no DEX market; seizing it would
        // leave the rebalancer holding what it can't sell. The sandwich attempt just asked every
        // venue for a quote, so its answer is reused rather than quoted again.
        if asset_tag == ASSET_TAG_STAKED && !collateral_sellable {
            return Err(anyhow!(
                "no venue quotes the staked collateral {}; not seizing what can't be sold",
                asset_mint
            ));
        }

        let wallet_liab = self
            .liquidator_account
            .get_token_balance_for_mint(&liab_mint)
            .unwrap_or(0);
        let wallet_swap_mint = self
            .liquidator_account
            .get_token_balance_for_mint(&self.swap_mint)
            .unwrap_or(0);
        let shortfall = buy_shortfall(repay_amount, wallet_liab);

        let mut txs = Vec::new();
        if shortfall > 0 {
            info!(
                "InventoryStrategy: buying shortfall of {} {} (have {}, need {})",
                shortfall,
                liab_mint,
                wallet_liab,
                repay_amount.to_num::<u64>()
            );
            let (buy_tx, bought) = self.build_buy_tx(liab_mint, shortfall, wallet_swap_mint)?;
            txs.push(buy_tx);

            if bought < shortfall {
                let covered = I80F48::from_num(wallet_liab + bought);
                let ratio = covered / repay_amount;
                // The liability not bought was still paid for, at the oracle price.
                let missing_value = self
                    .value_wrapper(&liab_mint)?
                    .get_value_for_amount(I80F48::from_num(shortfall - bought))?;
                let profit = intent.profit * ratio - missing_value;
                if profit < I80F48::from_num(self.min_profit) {
                    return Err(anyhow!(
                        "buying {} {} only guarantees {}; the liquidation it covers would make ${:.6}",
                        shortfall,
                        liab_mint,
                        bought,
                        profit.to_num::<f64>()
                    ));
                }
                info!(
                    "InventoryStrategy: the buy guarantees {} of {} {}; liquidating {:.4} of the plan",
                    bought,
                    shortfall,
                    liab_mint,
                    ratio.to_num::<f64>()
                );
                asset_amount *= ratio;
                repay_amount = covered;
            }
        } else {
            debug!(
                "InventoryStrategy: wallet already covers {} {}; no buy needed",
                repay_amount.to_num::<u64>(),
                liab_mint
            );
        }

        let liquidate_tx =
            self.liquidator_account
                .build_liquidate_tx(intent, asset_amount, repay_amount, None)?;
        txs.push(liquidate_tx);

        Ok(ExecutionPlan { txs })
    }
}

impl LiquidationStrategy for InventoryStrategy {
    fn name(&self) -> &'static str {
        "inventory"
    }

    fn assemble_sandwich(&self, intent: &PreparedLiquidatableAccount) -> Result<Sandwich> {
        let cache = &self.liquidator_account.cache;
        let liab_mint = cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint;
        let asset_mint = cache.banks.try_get_bank(&intent.asset_bank)?.bank.mint;

        match self.build_sandwich_tx(intent, asset_mint, liab_mint) {
            Ok(tx) => Ok(Sandwich::Plan(ExecutionPlan { txs: vec![tx] })),
            Err(e) => {
                info!(
                    "InventoryStrategy: not selling the collateral in-tx for {} ({}); buying the liability instead",
                    intent.liquidatee_account.address, e
                );
                Ok(Sandwich::Unavailable {
                    collateral_sellable: !no_venue_quoted(&e),
                })
            }
        }
    }

    fn assemble_with_inventory(
        &self,
        intent: &PreparedLiquidatableAccount,
        collateral_sellable: bool,
    ) -> Result<ExecutionPlan> {
        let cache = &self.liquidator_account.cache;
        let liab_mint = cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint;
        let asset_bank = cache.banks.try_get_bank(&intent.asset_bank)?;
        self.build_buy_plan(
            intent,
            asset_bank.bank.mint,
            asset_bank.bank.config.asset_tag,
            liab_mint,
            collateral_sellable,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixed_macro::types::I80F48;

    #[test]
    fn only_a_missing_quote_marks_collateral_unsellable() {
        assert!(no_venue_quoted(&anyhow!(
            "No aggregator returned a quote (Jupiter: 429; Titan: timeout)"
        )));
        assert!(no_venue_quoted(&anyhow!(
            "Jupiter quote failed: NO_ROUTES_FOUND"
        )));
        assert!(!no_venue_quoted(&anyhow!(
            "No route pays 541005 X for at most 535916 Y"
        )));
        assert!(!no_venue_quoted(&anyhow!(
            "transaction loads 75 accounts (max 64)"
        )));
        assert!(!no_venue_quoted(&anyhow!(
            "Could not build swap instructions on any venue (Jupiter: adds 27 accounts)"
        )));
    }

    #[test]
    fn affordable_input_is_capped_by_the_wallet() {
        assert_eq!(affordable_input(1_000, 1_500), 1_000);
        assert_eq!(affordable_input(1_000, 400), 400);
        assert_eq!(affordable_input(1_000, 0), 0);
    }

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
