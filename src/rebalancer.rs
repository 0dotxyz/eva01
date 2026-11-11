use crate::{
    cache::Cache,
    config::Eva01Config,
    execution::{
        activity::LiquidationActivity,
        reservations::{Claim, TokenReservations},
    },
    utils::{accessor, rate_limiter::RateLimiter},
    wrappers::{oracle::OracleWrapper, token_account::TokenAccountWrapper},
};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{error, info, warn};
use marginfi_type_crate::constants::ASSET_TAG_STAKED;
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::{CommitmentConfig, CommitmentLevel};
use solana_dex_superagg::{
    ClientConfig, DexSuperAggClient, DflowConfig, JupiterConfig, TitanConfig,
};
use solana_keypair::Keypair;
use solana_pubkey::Pubkey;
use solana_signer::Signer;
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

/// How long a swap venue gets to quote, or to build its route, before it is left out.
const SWAP_VENUE_TIMEOUT: Duration = Duration::from_secs(1);

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

/// A token whose swap failed is left alone this long: a pass runs after every liquidation
/// attempt, and retrying a failing route on each one spends the DEX quota liquidations need.
const TRADE_RETRY_INTERVAL: Duration = Duration::from_secs(300);

/// When each token's last swap failed, so it is retried only after [`TRADE_RETRY_INTERVAL`].
#[derive(Default)]
struct TradeCooldown {
    failed_at: HashMap<Pubkey, Instant>,
}

impl TradeCooldown {
    fn failed(&mut self, mint: Pubkey, now: Instant) {
        self.failed_at.insert(mint, now);
    }

    fn ready(&mut self, mint: &Pubkey, now: Instant) -> bool {
        match self.failed_at.get(mint) {
            Some(at) if now.duration_since(*at) < TRADE_RETRY_INTERVAL => false,
            Some(_) => {
                self.failed_at.remove(mint);
                true
            }
            None => true,
        }
    }
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
    /// The rebalancer's own swaps, under the venues' default deadlines: nothing races them.
    dex_client: Arc<DexSuperAggClient>,
    /// The liquidations' swaps, under [`SWAP_VENUE_TIMEOUT`].
    liquidation_dex_client: Arc<DexSuperAggClient>,
    dex_gate: Arc<DexGate>,
    empty_stake_banks: HashSet<Pubkey>,
    unsweepable_mints: HashMap<Pubkey, u64>,
    /// Staked collateral held, by mint, as last reported.
    held_stake: HashMap<Pubkey, u64>,
    /// Reports each unpriceable token at most once per [`SKIP_WARN_INTERVAL`].
    skip_warnings: RateLimiter<Pubkey>,
    reservations: Arc<TokenReservations>,
    /// A pass stops as soon as a liquidation is attempted.
    activity: Arc<LiquidationActivity>,
    inventory_mints: HashSet<Pubkey>,
    /// The working balance per inventory token (USD).
    inventory_target: I80F48,
    cooldown: TradeCooldown,
    /// Where the last pass stopped early, which the next one starts from.
    resume_at: Option<Pubkey>,
    /// Recreates the wallet's wSOL account when a route has closed it.
    rpc_client: RpcClient,
    signer: Keypair,
}

const WSOL: Pubkey = spl_token::native_mint::ID;

/// The size of an SPL token account, whose rent a recreated wSOL account costs.
const TOKEN_ACCOUNT_SIZE: usize = 165;

impl Rebalancer {
    pub fn new(
        config: Eva01Config,
        cache: Arc<Cache>,
        reservations: Arc<TokenReservations>,
        activity: Arc<LiquidationActivity>,
    ) -> anyhow::Result<Self> {
        let swap_mint = config.swap_mint;
        let inventory_mints: HashSet<Pubkey> = config
            .inventory_mints
            .iter()
            .copied()
            .filter(|mint| *mint != swap_mint)
            .collect();
        let inventory_target = I80F48::from_num(config.inventory_target_usd);
        for mint in &inventory_mints {
            if cache.tokens.try_get_token_for_mint(mint).is_err() {
                warn!(
                    "The inventory token {} has no wallet token account (no bank uses it?); it will never be bought",
                    mint
                );
            }
        }

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
        // A liquidation's venue slower than this loses the race to the others, or the
        // liquidation to a competitor, anyway.
        let liquidation_config = ClientConfig {
            quote_timeout: SWAP_VENUE_TIMEOUT,
            build_timeout: SWAP_VENUE_TIMEOUT,
            ..client_config.clone()
        };
        let signer = || Keypair::try_from(config.wallet_keypair.as_slice());
        let dex_client = Arc::new(DexSuperAggClient::new(client_config, signer()?)?);
        let liquidation_dex_client =
            Arc::new(DexSuperAggClient::new(liquidation_config, signer()?)?);

        Ok(Self {
            swap_mint,
            tokio_rt,
            cache,
            dex_client,
            liquidation_dex_client,
            dex_gate: Arc::new(DexGate::new()),
            empty_stake_banks: HashSet::new(),
            unsweepable_mints: HashMap::new(),
            held_stake: HashMap::new(),
            skip_warnings: RateLimiter::new(SKIP_WARN_INTERVAL),
            reservations,
            activity,
            inventory_mints,
            inventory_target,
            cooldown: TradeCooldown::default(),
            resume_at: None,
            rpc_client: RpcClient::new_with_commitment(
                config.rpc_url.clone(),
                CommitmentConfig::confirmed(),
            ),
            signer: signer()?,
        })
    }

    /// The swap client liquidations quote and build on.
    pub fn liquidation_dex_client(&self) -> Arc<DexSuperAggClient> {
        Arc::clone(&self.liquidation_dex_client)
    }

    pub fn dex_gate(&self) -> Arc<DexGate> {
        Arc::clone(&self.dex_gate)
    }

    /// Bring every inventory token back into its band, and sell every other unclaimed token the
    /// wallet holds (above the dust floor) back to the swap mint. Stops early, between tokens, once
    /// a liquidation is attempted.
    pub fn run(&mut self) -> anyhow::Result<RebalancePass> {
        let mut trades = 0;
        let mut claim = self.reservations.new_claim();
        let started = Instant::now();
        info!("Running the Rebalancing process...");
        // Everything that repays or seizes SOL needs the account; a closed one makes those fail
        // until it exists again.
        if self.wsol_account_closed() {
            if let Err(e) = self.recreate_wsol_account(0) {
                error!("Failed to recreate the wSOL account: {:?}", e);
            }
        }

        for mint in pass_order(self.cache.mints.get_mints(), self.resume_at) {
            if self.activity.attempted_since(started) {
                info!("Stopping the Rebalancing process: a liquidation was attempted.");
                // The next pass starts here, or in a long burst the same first few tokens would
                // be the only ones ever traded.
                self.resume_at = Some(mint);
                return Ok(RebalancePass { trades, claim });
            }
            if mint == self.swap_mint
                || self.empty_stake_banks.contains(&mint)
                || !self.cooldown.ready(&mint, Instant::now())
            {
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

        self.resume_at = None;
        info!("The Rebalancing process is complete.");

        Ok(RebalancePass { trades, claim })
    }

    /// Sells `units` of `mint` for the swap mint, or whatever of them is still unclaimed. Returns
    /// whether a swap was attempted.
    fn sell(
        &mut self,
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
            self.cooldown.failed(mint, Instant::now());
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
            self.cooldown.failed(mint, Instant::now());
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
        // A route that closes the wSOL account pays the SOL out natively; measured to wrap it back.
        let native_before = if output_mint == WSOL {
            Some(self.rpc_client.get_balance(&self.signer.pubkey())?)
        } else {
            None
        };

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

        if let Some(before) = native_before {
            self.wrap_unwrapped_sol(before, signature);
        }

        Ok(prepared.min_out_amount)
    }

    /// Whether the cache shows the wallet's wSOL account closed (Geyser reports a closed account
    /// with no data).
    fn wsol_account_closed(&self) -> bool {
        self.cache
            .tokens
            .try_get_token_for_mint(&WSOL)
            .and_then(|token| self.cache.tokens.try_get_account(&token))
            .is_ok_and(|account| account.data.is_empty())
    }

    /// After a buy of SOL: if the route closed the wSOL account, recreates it and wraps the SOL
    /// the swap paid out natively (`before` is the native balance just before it).
    fn wrap_unwrapped_sol(&self, before: u64, signature: solana_signature::Signature) {
        let wsol_account = associated_wsol_account(&self.signer.pubkey());
        let closed = match self
            .rpc_client
            .get_account_with_commitment(&wsol_account, CommitmentConfig::confirmed())
        {
            Ok(response) => response.value.is_none(),
            Err(e) => {
                warn!(
                    "Could not check the wSOL account after {}: {}",
                    signature, e
                );
                return;
            }
        };
        if !closed {
            return;
        }
        let returned = match self.rpc_client.get_balance(&self.signer.pubkey()) {
            Ok(after) => after.saturating_sub(before),
            Err(e) => {
                warn!(
                    "Could not read the native balance after {}: {}",
                    signature, e
                );
                0
            }
        };
        warn!(
            "The swap {} closed the wSOL account and paid {} lamports out natively; recreating it",
            signature, returned
        );
        if let Err(e) = self.recreate_wsol_account(returned) {
            error!("Failed to recreate the wSOL account: {:?}", e);
        }
    }

    /// Creates the wallet's wSOL account and wraps up to `lamports` of native SOL into it, less
    /// the account's rent (which the close refunded).
    fn recreate_wsol_account(&self, lamports: u64) -> anyhow::Result<()> {
        let owner = self.signer.pubkey();
        let wsol_account = associated_wsol_account(&owner);
        let mut instructions = vec![
            spl_associated_token_account::instruction::create_associated_token_account_idempotent(
                &owner,
                &owner,
                &WSOL,
                &spl_token::ID,
            ),
        ];
        let rent = self
            .rpc_client
            .get_minimum_balance_for_rent_exemption(TOKEN_ACCOUNT_SIZE)?;
        let wrapped = lamports.saturating_sub(rent);
        if wrapped > 0 {
            instructions.push(solana_system_interface::instruction::transfer(
                &owner,
                &wsol_account,
                wrapped,
            ));
            instructions.push(spl_token::instruction::sync_native(
                &spl_token::ID,
                &wsol_account,
            )?);
        }
        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        let tx = solana_transaction::Transaction::new_signed_with_payer(
            &instructions,
            Some(&owner),
            &[&self.signer],
            blockhash,
        );
        let signature = self.rpc_client.send_and_confirm_transaction(&tx)?;
        info!(
            "Recreated the wSOL account {} with {} lamports wrapped: {}",
            wsol_account, wrapped, signature
        );
        Ok(())
    }
}

/// The mints in a fixed order, starting from `resume_at` when it is still one of them.
fn pass_order(mut mints: Vec<Pubkey>, resume_at: Option<Pubkey>) -> Vec<Pubkey> {
    mints.sort();
    if let Some(start) = resume_at.and_then(|mint| mints.iter().position(|m| *m == mint)) {
        mints.rotate_left(start);
    }
    mints
}

fn associated_wsol_account(owner: &Pubkey) -> Pubkey {
    spl_associated_token_account::get_associated_token_address_with_program_id(
        owner,
        &WSOL,
        &spl_token::ID,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: I80F48 = I80F48!(50);

    #[test]
    fn a_pass_resumes_where_the_last_one_stopped() {
        let mut mints: Vec<Pubkey> = (0..4).map(|_| Pubkey::new_unique()).collect();
        mints.sort();
        let shuffled = vec![mints[2], mints[0], mints[3], mints[1]];
        assert_eq!(pass_order(shuffled.clone(), None), mints);
        assert_eq!(
            pass_order(shuffled.clone(), Some(mints[2])),
            vec![mints[2], mints[3], mints[0], mints[1]]
        );
        assert_eq!(pass_order(shuffled, Some(Pubkey::new_unique())), mints);
    }

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
    fn a_failed_token_waits_before_trading_again() {
        let mut cooldown = TradeCooldown::default();
        let mint = Pubkey::new_unique();
        let start = Instant::now();
        assert!(cooldown.ready(&mint, start));
        cooldown.failed(mint, start);
        assert!(!cooldown.ready(&mint, start + TRADE_RETRY_INTERVAL / 2));
        assert!(cooldown.ready(&mint, start + TRADE_RETRY_INTERVAL));
        assert!(cooldown.ready(&Pubkey::new_unique(), start));
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
