use crate::{
    cache::Cache,
    config::{Eva01Config, TokenThresholds},
};
use fixed::types::I80F48;
use log::{debug, error, info, warn};
use solana_dex_superagg::{
    client::DexSuperAggClient,
    config::{ClientConfig, JupiterConfig, RoutingStrategy, SharedConfig, TitanConfig},
};
use solana_program::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentLevel;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::runtime::{Builder, Runtime};

/// Base cooldown after the first no-route swap failure; doubles with each consecutive failure.
const ROUTE_BACKOFF_BASE: Duration = Duration::from_secs(60);
/// Cap on the per-token no-route backoff cooldown.
const ROUTE_BACKOFF_MAX: Duration = Duration::from_secs(3600);

/// Per-token backoff state for seized tokens we can't currently find a swap route for.
struct RouteBackoff {
    /// Consecutive no-route failures (drives the exponential cooldown).
    failures: u32,
    /// Don't re-attempt the swap before this time.
    retry_after: Instant,
}

/// Exponential backoff: `BASE * 2^(failures-1)`, capped at `ROUTE_BACKOFF_MAX`.
fn backoff_delay(failures: u32) -> Duration {
    let shift = failures.saturating_sub(1).min(16);
    let secs = ROUTE_BACKOFF_BASE
        .as_secs()
        .saturating_mul(1u64 << shift)
        .min(ROUTE_BACKOFF_MAX.as_secs());
    Duration::from_secs(secs)
}

/// The rebalancer sells excess/seized tokens back into the swap_mint reserve, keeping each
/// token at or below its configured max threshold. Liquidations JIT-buy their own liabilities,
/// so the rebalancer is sell-only.
pub struct Rebalancer {
    swap_mint: Pubkey,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    default_token_max_threshold: I80F48,
    token_thresholds: HashMap<Pubkey, TokenThresholds>,
    dex_client: Arc<DexSuperAggClient>,
    empty_stake_banks: HashSet<Pubkey>,
    /// Tokens with no current swap route, backed off so we don't re-attempt them every cycle.
    /// Routes can appear later, so this is a time-based backoff rather than a permanent ban.
    route_backoff: HashMap<Pubkey, RouteBackoff>,
}

impl Rebalancer {
    pub fn new(config: Eva01Config, cache: Arc<Cache>) -> anyhow::Result<Self> {
        let swap_mint = config.swap_mint;

        let tokio_rt = Builder::new_multi_thread()
            .thread_name("rebalancer")
            .worker_threads(2)
            .enable_all()
            .build()?;

        let default_token_max_threshold = config.default_token_max_threshold;
        let token_thresholds = config.token_thresholds;

        // Convert wallet keypair to JSON string format expected by solana-dex-superagg
        let wallet_keypair_str = serde_json::to_string(&config.wallet_keypair)?;

        // Create ClientConfig for DexSuperAggClient
        let shared_config = SharedConfig {
            rpc_url: config.rpc_url.clone(),
            slippage_bps: config.slippage_bps,
            wallet_keypair: Some(wallet_keypair_str),
            compute_unit_price_micro_lamports: config.compute_unit_price_micro_lamports,
            routing_strategy: Some(RoutingStrategy::BestPrice),
            retry_tx_landing: 3,
            commitment_level: CommitmentLevel::Confirmed,
        };

        let jupiter_config = JupiterConfig {
            jup_swap_api_url: config.jup_swap_api_url.clone(),
            api_key: Some(config.jupiter_api_key.clone()),
        };

        let titan_config = Some(TitanConfig {
            titan_ws_endpoint: config.titan_ws_endpoint.clone(),
            titan_api_key: Some(config.titan_api_key.clone()),
        });

        let client_config = ClientConfig {
            shared: shared_config,
            jupiter: jupiter_config,
            titan: titan_config,
            dflow: None,
        };

        let dex_client = Arc::new(DexSuperAggClient::new(client_config)?);

        Ok(Self {
            swap_mint,
            tokio_rt,
            cache,
            default_token_max_threshold,
            token_thresholds,
            dex_client,
            empty_stake_banks: HashSet::new(),
            route_backoff: HashMap::new(),
        })
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        info!("Running the Rebalancing process...");

        if let Err(e) = self.sell_excessive_tokens() {
            error!("Failed to rebalance the Liquidator's tokens: {}", e);
        }

        info!("The Rebalancing process is complete.");

        Ok(())
    }

    /// Sell any token whose USD value exceeds its max threshold back into the swap_mint reserve.
    /// Liquidations fund themselves just-in-time (JIT-buy), so the rebalancer no longer buys —
    /// it only offloads seized collateral / excess inventory into swap_mint.
    fn sell_excessive_tokens(&mut self) -> anyhow::Result<()> {
        for mint in self.cache.mints.get_mints() {
            debug!("Processing token {}...", mint);
            if mint == self.swap_mint || self.empty_stake_banks.contains(&mint) {
                continue;
            }

            // Skip tokens we recently failed to find a swap route for (illiquid seized assets),
            // so we don't re-attempt + re-log them every cycle while the backoff is active.
            if let Some(backoff) = self.route_backoff.get(&mint) {
                if Instant::now() < backoff.retry_after {
                    debug!(
                        "Skipping {} in rebalancing: no swap route, backing off ({} consecutive failures)",
                        mint, backoff.failures
                    );
                    continue;
                }
            }

            let token = self.cache.tokens.try_get_token_for_mint(&mint)?;
            let wrapper = match self.cache.try_get_token_wrapper_lenient(&mint, &token) {
                Ok(wrapper) => wrapper,
                Err(e) => {
                    // Ignore empty stake banks.
                    if e.to_string().contains("Stake pool supply is zero") {
                        self.empty_stake_banks.insert(mint);
                    } else {
                        // SwitchboardStalePrice here at startup is harmless — SwbPriceFetcher
                        // populates synthetic oracle accounts on its first 30-second cycle.
                        warn!("Skipping the token {} in rebalancing: {}", mint, e);
                    }
                    continue;
                }
            };

            let value = wrapper.get_value()?;
            let max_value = self
                .token_thresholds
                .get(&mint)
                .map(|t| t.max_value)
                .unwrap_or(self.default_token_max_threshold);

            if value > max_value {
                info!("The value of {} tokens is higher than set threshold: {} > {}. Selling ${} worth of tokens.", mint, value.to_num::<f64>(), max_value.to_num::<f64>(), (value - max_value / 2).to_num::<f64>());
                let amount_to_swap = wrapper.get_amount_from_value(value - max_value / 2)?;
                match self.swap(amount_to_swap.to_num(), mint, self.swap_mint) {
                    Ok(swapped_amount) => {
                        // Route is healthy again; clear any prior backoff.
                        self.route_backoff.remove(&mint);
                        info!("Got {} back from the swap.", swapped_amount);
                    }
                    Err(e) if e.to_string().contains("No aggregators available") => {
                        let entry = self.route_backoff.entry(mint).or_insert(RouteBackoff {
                            failures: 0,
                            retry_after: Instant::now(),
                        });
                        entry.failures = entry.failures.saturating_add(1);
                        let delay = backoff_delay(entry.failures);
                        entry.retry_after = Instant::now() + delay;
                        warn!(
                            "No swap route for {}; backing off {}s (consecutive failures: {})",
                            mint,
                            delay.as_secs(),
                            entry.failures
                        );
                    }
                    Err(e) => {
                        error!("Swap failed: {}", e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute a swap using the unified DEX aggregator client with best price strategy
    fn swap(&self, amount: u64, input_mint: Pubkey, output_mint: Pubkey) -> anyhow::Result<u64> {
        if input_mint == output_mint {
            return Err(anyhow::anyhow!(
                "Input and output mints cannot be the same: {:?}",
                input_mint
            ));
        }
        if amount == 0 {
            return Err(anyhow::anyhow!("Amount cannot be zero: {:?}", input_mint));
        }

        info!(
            "Swapping {} tokens of mint {} to mint {} ...",
            amount, input_mint, output_mint
        );

        // Keep WSOL as a plain SPL token in the ATA: never auto-wrap/unwrap. Otherwise a
        // USDT -> WSOL buy would unwrap the output to native SOL, leaving the WSOL ATA
        // empty and the rebalancer buying WSOL again every cycle.
        let result = self.tokio_rt.block_on(self.dex_client.swap(
            &input_mint.to_string(),
            &output_mint.to_string(),
            amount,
            false,
        ))?;

        info!(
            "Swap successful! Transaction: {}, Output: {} tokens of mint {}",
            result.swap_result.signature, result.swap_result.out_amount, output_mint
        );

        if let Some(agg) = result.swap_result.aggregator_used {
            info!("Aggregator used: {:?}", agg);
        }

        Ok(result.swap_result.out_amount)
    }
}
