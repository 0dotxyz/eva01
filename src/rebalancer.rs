use crate::{
    cache::Cache,
    config::Eva01Config,
    execution::reservations::{Claim, TokenReservations},
    utils::{accessor, rate_limiter::RateLimiter},
    wrappers::{oracle::OracleWrapper, token_account::TokenAccountWrapper},
};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{error, info, warn};
use marginfi_type_crate::constants::ASSET_TAG_STAKED;
use solana_commitment_config::CommitmentLevel;
use solana_dex_superagg::{
    ClientConfig, DexSuperAggClient, DflowConfig, JupiterConfig, TitanConfig,
};
use solana_keypair::Keypair;
use solana_pubkey::Pubkey;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    runtime::{Builder, Runtime},
    sync::{Semaphore, SemaphorePermit},
};

/// Concurrent DEX operations allowed across the liquidation workers and the rebalancer. Each one
/// costs several aggregator round-trips, so an unbounded burst trips the Jupiter rate limit (429)
/// and the failing targets then sit in the executor's backoff.
const MAX_CONCURRENT_DEX_CALLS: usize = 2;
/// Minimum spacing between DEX operations starting. Two concurrent operations that each make
/// two or three Jupiter requests still burst past its per-second gateway limit at startup; this
/// keeps the request rate steady instead.
const DEX_CALL_MIN_INTERVAL: Duration = Duration::from_millis(500);

/// Admission to the DEX aggregators: at most [`MAX_CONCURRENT_DEX_CALLS`] operations in flight,
/// starting no closer together than [`DEX_CALL_MIN_INTERVAL`].
pub struct DexGate {
    semaphore: Semaphore,
    last_grant: tokio::sync::Mutex<Instant>,
}

impl DexGate {
    fn new() -> Self {
        Self {
            semaphore: Semaphore::new(MAX_CONCURRENT_DEX_CALLS),
            last_grant: tokio::sync::Mutex::new(Instant::now() - DEX_CALL_MIN_INTERVAL),
        }
    }

    /// Waits for a slot and for the spacing, then holds the slot until the permit is dropped.
    pub async fn acquire(&self) -> anyhow::Result<SemaphorePermit<'_>> {
        let permit = self.semaphore.acquire().await?;
        let mut last_grant = self.last_grant.lock().await;
        let since = last_grant.elapsed();
        if since < DEX_CALL_MIN_INTERVAL {
            tokio::time::sleep(DEX_CALL_MIN_INTERVAL - since).await;
        }
        *last_grant = Instant::now();
        Ok(permit)
    }
}

/// Don't bother selling a position worth less than this (USD); the swap fee/dust isn't worth it.
const MIN_REBALANCE_VALUE: I80F48 = I80F48!(1.0);

/// Never sell for less than this fraction of the oracle value: further off, the market (or a pushed
/// pool) is wrong and holding until the next pass beats dumping. Within the ~8.5% liquidation margin.
const MIN_SELL_PRICE_VS_ORACLE: I80F48 = I80F48!(0.95);

/// A token the sweep cannot price (e.g. a bank whose oracle is disabled) is reported this often,
/// not on every pass.
const SKIP_WARN_INTERVAL: Duration = Duration::from_secs(1200);

/// One rebalancing pass: how many swaps it attempted, each of which may have changed the wallet
/// even if it failed, and the tokens it claimed to spend. The claim is released once the wallet
/// balances have been re-read, so swapped tokens never look free to a liquidation.
pub struct RebalancePass {
    pub trades: usize,
    pub claim: Claim,
}

/// What a pass does with one token, by the value of its unclaimed balance.
#[derive(Debug, PartialEq)]
enum Trade {
    Hold,
    /// Sell this much (USD) for the swap mint.
    Sell(I80F48),
    /// Buy this much (USD) with the swap mint.
    Buy(I80F48),
}

/// An inventory token is kept between `target` and twice `target`. Outside that band it is
/// brought back to the middle, one and a half times `target`, so every trade is at least half the
/// target and the balance lands well inside the band: small price moves or a liquidation's
/// leftovers never trigger another trade, and the pass never trades cents back and forth.
fn inventory_trade(free_value: I80F48, target: I80F48) -> Trade {
    let middle = target * I80F48!(1.5);
    if free_value < target {
        Trade::Buy(middle - free_value)
    } else if free_value > target * I80F48!(2) {
        Trade::Sell(free_value - middle)
    } else {
        Trade::Hold
    }
}

/// The rebalancer keeps the liquidator holding the swap mint (USDC) as its reserve and a working
/// balance of each inventory token, the common debt tokens a liquidation can repay from the
/// wallet when its swap does not fit. Every other token it ends up with (a liquidation's leftover
/// liability or seized collateral) is sold back to USDC. Tokens a liquidation in flight has
/// claimed are left alone.
pub struct Rebalancer {
    swap_mint: Pubkey,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    dex_client: Arc<DexSuperAggClient>,
    dex_gate: Arc<DexGate>,
    empty_stake_banks: HashSet<Pubkey>,
    unsweepable_mints: HashMap<Pubkey, u64>,
    /// Staked collateral held, by mint, as last reported.
    held_stake: HashMap<Pubkey, u64>,
    /// Reports each unpriceable token at most once per [`SKIP_WARN_INTERVAL`].
    skip_warnings: RateLimiter<Pubkey>,
    reservations: Arc<TokenReservations>,
    inventory_mints: HashSet<Pubkey>,
    /// The working balance per inventory token (USD).
    inventory_target: I80F48,
}

impl Rebalancer {
    pub fn new(
        config: Eva01Config,
        cache: Arc<Cache>,
        reservations: Arc<TokenReservations>,
    ) -> anyhow::Result<Self> {
        let swap_mint = config.swap_mint;
        let inventory_mints = config
            .inventory_mints
            .iter()
            .copied()
            .filter(|mint| *mint != swap_mint)
            .collect();
        let inventory_target = I80F48::from_num(config.inventory_target_usd);

        let tokio_rt = Builder::new_multi_thread()
            .thread_name("rebalancer")
            .worker_threads(2)
            .enable_all()
            .build()?;

        let client_config = ClientConfig {
            rpc_url: config.rpc_url.clone(),
            slippage_bps: config.slippage_bps,
            compute_unit_price_micro_lamports: config.compute_unit_price_micro_lamports,
            jupiter: Some(JupiterConfig {
                jup_swap_api_url: config.jup_swap_api_url.clone(),
                api_key: config.jupiter_api_key.clone(),
            }),
            titan: Some(TitanConfig {
                titan_ws_endpoint: config.titan_ws_endpoint.clone(),
                titan_api_key: config.titan_api_key.clone(),
            }),
            dflow: config.dflow_api_key.clone().map(|api_key| DflowConfig {
                api_url: "https://quote-api.dflow.net".to_string(),
                api_key,
            }),
            ..Default::default()
        };
        let signer = Keypair::try_from(config.wallet_keypair.as_slice())?;
        let dex_client = Arc::new(DexSuperAggClient::new(client_config, signer)?);

        Ok(Self {
            swap_mint,
            tokio_rt,
            cache,
            dex_client,
            dex_gate: Arc::new(DexGate::new()),
            empty_stake_banks: HashSet::new(),
            unsweepable_mints: HashMap::new(),
            held_stake: HashMap::new(),
            skip_warnings: RateLimiter::new(SKIP_WARN_INTERVAL),
            reservations,
            inventory_mints,
            inventory_target,
        })
    }

    pub fn dex_client(&self) -> Arc<DexSuperAggClient> {
        Arc::clone(&self.dex_client)
    }

    pub fn dex_gate(&self) -> Arc<DexGate> {
        Arc::clone(&self.dex_gate)
    }

    /// Bring every inventory token back into its band, and sell every other unclaimed token the
    /// wallet holds (above the dust floor) back to the swap mint.
    pub fn run(&mut self) -> anyhow::Result<RebalancePass> {
        let mut trades = 0;
        let mut claim = self.reservations.new_claim();
        info!("Running the Rebalancing process...");

        for mint in self.cache.mints.get_mints() {
            if mint == self.swap_mint || self.empty_stake_banks.contains(&mint) {
                continue;
            }

            let token = match self.cache.tokens.try_get_token_for_mint(&mint) {
                Ok(token) => token,
                Err(_) => continue,
            };

            // Mints backed only by integration banks (Kamino/JupLend) have no bank that
            // prices the raw SPL token, so they can never be swept.
            // Expected to only ever happen on staging.
            if self.cache.banks.try_get_account_for_mint(&mint).is_err() {
                let balance = self.token_balance(&token);
                // The condition is permanent, so report a mint only when its held amount changes.
                if self.unsweepable_mints.insert(mint, balance) != Some(balance) {
                    if balance > 0 {
                        warn!(
                            "Holding {} of {} that rebalancing cannot sweep: no non-integration bank prices this mint.",
                            balance, mint
                        );
                    } else {
                        info!(
                            "Excluding the mint {} from rebalancing: integration banks only.",
                            mint
                        );
                    }
                }
                continue;
            }

            // Seized stake is kept, not sold: a single-validator LST rarely has a DEX market, and
            // liquidations seizing it are rare enough to just hold what they leave.
            if self.is_staked_collateral(&mint) {
                let balance = self.token_balance(&token);
                if balance > 0 && self.held_stake.insert(mint, balance) != Some(balance) {
                    info!(
                        "Keeping {} of the staked collateral {}: seized stake is held, not sold.",
                        balance, mint
                    );
                }
                continue;
            }

            let wrapper = match self.cache.try_get_token_wrapper_lenient(&mint, &token) {
                Ok(wrapper) => wrapper,
                Err(e) => {
                    // Ignore empty stake banks.
                    if e.to_string().contains("Stake pool supply is zero") {
                        self.empty_stake_banks.insert(mint);
                    } else if self.skip_warnings.allow(mint, Instant::now()) {
                        warn!("Skipping the token {} in rebalancing: {}", mint, e);
                    }
                    continue;
                }
            };

            // A liquidation in flight may be about to repay with some of it.
            let free = self.reservations.unclaimed(&mint, wrapper.balance);
            let free_value = match wrapper.get_value_for_amount(I80F48::from_num(free)) {
                Ok(value) => value,
                Err(e) => {
                    if self.skip_warnings.allow(mint, Instant::now()) {
                        warn!(
                            "Skipping the token {} in rebalancing (price unavailable): {}",
                            mint, e
                        );
                    }
                    continue;
                }
            };

            let trade = if self.inventory_mints.contains(&mint) {
                inventory_trade(free_value, self.inventory_target)
            } else if free_value >= MIN_REBALANCE_VALUE {
                Trade::Sell(free_value)
            } else {
                Trade::Hold
            };
            // One token failing to trade must not stop the pass for the others.
            let traded = match trade {
                Trade::Hold => Ok(false),
                Trade::Sell(value) => {
                    let units = if value >= free_value {
                        Ok(free)
                    } else {
                        wrapper
                            .get_amount_from_value(value)
                            .map(|units| units.to_num::<u64>().min(free))
                    };
                    units.and_then(|units| self.sell(&mut claim, mint, &wrapper, units))
                }
                Trade::Buy(value) => self.buy(&mut claim, mint, value),
            };
            match traded {
                Ok(traded) => trades += usize::from(traded),
                Err(e) => {
                    if self.skip_warnings.allow(mint, Instant::now()) {
                        warn!("Rebalancing {} failed: {}", mint, e);
                    }
                }
            }
        }

        info!("The Rebalancing process is complete.");

        Ok(RebalancePass { trades, claim })
    }

    /// Sells `units` of `mint` for the swap mint, or whatever of them is still unclaimed. Returns
    /// whether a swap was attempted.
    fn sell(
        &self,
        claim: &mut Claim,
        mint: Pubkey,
        wrapper: &TokenAccountWrapper<OracleWrapper>,
        units: u64,
    ) -> anyhow::Result<bool> {
        // Claimed so no liquidation counts the tokens being sold. A liquidation may have claimed
        // some since the free balance was read; then this sells only what is still free.
        let units = self
            .reservations
            .claim_available(claim, mint, units, wrapper.balance);
        if units == 0 {
            return Ok(false);
        }
        let value = wrapper.get_value_for_amount(I80F48::from_num(units))?;
        info!(
            "Selling {} of {} (~${}) back to the swap mint.",
            units,
            mint,
            value.to_num::<f64>()
        );
        if let Err(e) = self.swap(units, mint, self.swap_mint, value) {
            error!("Rebalance swap failed for {}: {}", mint, e);
        }
        Ok(true)
    }

    /// Buys `value` (USD) of `mint` with the unclaimed swap mint. Returns whether a swap was
    /// attempted.
    fn buy(&mut self, claim: &mut Claim, mint: Pubkey, value: I80F48) -> anyhow::Result<bool> {
        let reserve_token = self.cache.tokens.try_get_token_for_mint(&self.swap_mint)?;
        let reserve = self
            .cache
            .try_get_token_wrapper_lenient(&self.swap_mint, &reserve_token)?;
        let wanted = reserve.get_amount_from_value(value)?.to_num::<u64>();
        let units =
            self.reservations
                .claim_available(claim, self.swap_mint, wanted, reserve.balance);
        // Less than half the buy means the reserve is nearly empty; topping up a few cents at a
        // time would only churn.
        if units == 0 || units < wanted / 2 {
            if self.skip_warnings.allow(self.swap_mint, Instant::now()) {
                warn!(
                    "Not buying {} of inventory: only {} of the {} {} it needs is free",
                    mint, units, wanted, self.swap_mint
                );
            }
            return Ok(false);
        }
        let spent = reserve.get_value_for_amount(I80F48::from_num(units))?;
        info!(
            "Buying ~${} of the inventory token {} with {} of the swap mint.",
            spent.to_num::<f64>(),
            mint,
            units
        );
        if let Err(e) = self.swap(units, self.swap_mint, mint, spent) {
            error!("Inventory buy failed for {}: {}", mint, e);
        }
        Ok(true)
    }

    /// Whether `mint` is the LST of a staked collateral bank.
    fn is_staked_collateral(&self, mint: &Pubkey) -> bool {
        self.cache
            .banks
            .try_get_account_for_mint(mint)
            .and_then(|bank| self.cache.banks.try_get_bank(&bank))
            .is_ok_and(|bank| bank.bank.config.asset_tag == ASSET_TAG_STAKED)
    }

    fn token_balance(&self, token: &Pubkey) -> u64 {
        self.cache
            .tokens
            .try_get_account(token)
            .ok()
            .and_then(|account| accessor::amount(&account.data).ok())
            .unwrap_or(0)
    }

    /// Swap `amount` of `input_mint` (worth `value` USD at the oracle) for `output_mint`, refusing
    /// any route that pays less than [`MIN_SELL_PRICE_VS_ORACLE`] of that value.
    fn swap(
        &self,
        amount: u64,
        input_mint: Pubkey,
        output_mint: Pubkey,
        value: I80F48,
    ) -> anyhow::Result<u64> {
        if input_mint == output_mint {
            return Err(anyhow::anyhow!(
                "Input and output mints cannot be the same: {:?}",
                input_mint
            ));
        }
        if amount == 0 {
            return Err(anyhow::anyhow!("Amount cannot be zero: {:?}", input_mint));
        }

        let output_token = self.cache.tokens.try_get_token_for_mint(&output_mint)?;
        let min_out = self
            .cache
            .try_get_token_wrapper_lenient(&output_mint, &output_token)?
            .get_amount_from_value(value * MIN_SELL_PRICE_VS_ORACLE)?
            .to_num::<u64>();

        info!(
            "Swapping {} tokens of mint {} to mint {} (at least {}) ...",
            amount, input_mint, output_mint, min_out
        );

        let (prepared, signature) = self.tokio_rt.block_on(async {
            // The gate admits aggregator calls only; waiting for the confirmation must not hold a
            // slot the liquidation workers need for their quotes.
            let prepared = {
                let _permit = self.dex_gate.acquire().await?;
                self.dex_client
                    .build_swap_transaction(input_mint, output_mint, amount, min_out)
                    .await?
            };
            let signature = self
                .dex_client
                .send_and_confirm(&prepared.transaction, CommitmentLevel::Confirmed)
                .await?;
            anyhow::Ok((prepared, signature))
        })?;

        info!(
            "Swap successful via {:?}! Transaction: {}, at least {} tokens of mint {}",
            prepared.aggregator, signature, prepared.min_out_amount, output_mint
        );

        Ok(prepared.min_out_amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: I80F48 = I80F48!(50);

    #[test]
    fn an_inventory_token_inside_its_band_is_held() {
        assert_eq!(inventory_trade(I80F48!(50), TARGET), Trade::Hold);
        assert_eq!(inventory_trade(I80F48!(75), TARGET), Trade::Hold);
        assert_eq!(inventory_trade(I80F48!(100), TARGET), Trade::Hold);
    }

    #[test]
    fn a_short_inventory_token_is_topped_up_to_the_middle() {
        assert_eq!(
            inventory_trade(I80F48!(49.99), TARGET),
            Trade::Buy(I80F48!(25.01))
        );
        assert_eq!(inventory_trade(I80F48!(0), TARGET), Trade::Buy(I80F48!(75)));
    }

    #[test]
    fn an_overfull_inventory_token_is_sold_down_to_the_middle() {
        assert_eq!(
            inventory_trade(I80F48!(100.01), TARGET),
            Trade::Sell(I80F48!(25.01))
        );
        assert_eq!(
            inventory_trade(I80F48!(500), TARGET),
            Trade::Sell(I80F48!(425))
        );
    }

    #[test]
    fn every_trade_is_at_least_half_the_target() {
        let mut value = I80F48!(0);
        while value < I80F48!(300) {
            match inventory_trade(value, TARGET) {
                Trade::Buy(v) | Trade::Sell(v) => assert!(v >= TARGET / I80F48!(2)),
                Trade::Hold => {}
            }
            value += I80F48!(0.25);
        }
    }
}
