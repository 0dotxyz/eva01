use crate::{cache::Cache, config::Eva01Config, utils::accessor};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{error, info, warn};
use solana_commitment_config::CommitmentLevel;
use solana_dex_superagg::{
    ClientConfig, DexSuperAggClient, DflowConfig, JupiterConfig, TitanConfig,
};
use solana_program::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::{
    runtime::{Builder, Runtime},
    sync::Semaphore,
};

/// Concurrent DEX operations allowed across the liquidation workers and the rebalancer. Each one
/// costs several aggregator round-trips, so an unbounded burst trips the Jupiter rate limit (429)
/// and the failing targets then sit in the executor's backoff.
const MAX_CONCURRENT_DEX_CALLS: usize = 2;

/// Don't bother selling a position worth less than this (USD); the swap fee/dust isn't worth it.
const MIN_REBALANCE_VALUE: I80F48 = I80F48!(1.0);

/// Never sell for less than this fraction of the oracle value: further off, the market (or a pushed
/// pool) is wrong and holding until the next pass beats dumping. Within the ~8.5% liquidation margin.
pub const MIN_SELL_PRICE_VS_ORACLE: I80F48 = I80F48!(0.95);

/// The rebalancer keeps the liquidator holding only the swap mint (USDC): every other token it ends
/// up with — seized collateral from a liquidation, or a JIT-buy overshoot — is sold back to USDC on
/// the next pass. There is no inventory to maintain and nothing is bought here (liabilities are
/// funded just-in-time during liquidation), so it is a pure sell-to-USDC sweep.
pub struct Rebalancer {
    swap_mint: Pubkey,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    dex_client: Arc<DexSuperAggClient>,
    dex_gate: Arc<Semaphore>,
    empty_stake_banks: HashSet<Pubkey>,
    unsweepable_mints: HashMap<Pubkey, u64>,
}

impl Rebalancer {
    pub fn new(config: Eva01Config, cache: Arc<Cache>) -> anyhow::Result<Self> {
        let swap_mint = config.swap_mint;

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
            dex_gate: Arc::new(Semaphore::new(MAX_CONCURRENT_DEX_CALLS)),
            empty_stake_banks: HashSet::new(),
            unsweepable_mints: HashMap::new(),
        })
    }

    pub fn dex_client(&self) -> Arc<DexSuperAggClient> {
        Arc::clone(&self.dex_client)
    }

    pub fn dex_gate(&self) -> Arc<Semaphore> {
        Arc::clone(&self.dex_gate)
    }

    /// Sell every non-swap-mint token the wallet holds (above the dust floor) back to the swap mint.
    pub fn run(&mut self) -> anyhow::Result<()> {
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
                let balance = self
                    .cache
                    .tokens
                    .try_get_account(&token)
                    .ok()
                    .and_then(|account| accessor::amount(&account.data).ok())
                    .unwrap_or(0);
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

            let wrapper = match self.cache.try_get_token_wrapper_lenient(&mint, &token) {
                Ok(wrapper) => wrapper,
                Err(e) => {
                    // Ignore empty stake banks.
                    if e.to_string().contains("Stake pool supply is zero") {
                        self.empty_stake_banks.insert(mint);
                    } else {
                        warn!("Skipping the token {} in rebalancing: {}", mint, e);
                    }
                    continue;
                }
            };

            if wrapper.balance == 0 {
                continue;
            }

            let value = match wrapper.get_value() {
                Ok(value) => value,
                Err(e) => {
                    warn!(
                        "Skipping the token {} in rebalancing (price unavailable): {}",
                        mint, e
                    );
                    continue;
                }
            };

            if value < MIN_REBALANCE_VALUE {
                continue;
            }

            info!(
                "Selling {} of {} (~${}) back to the swap mint.",
                wrapper.balance,
                mint,
                value.to_num::<f64>()
            );
            if let Err(e) = self.swap(wrapper.balance, mint, self.swap_mint, value) {
                error!("Rebalance swap failed for {}: {}", mint, e);
            }
        }

        info!("The Rebalancing process is complete.");

        Ok(())
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
            let _permit = self.dex_gate.acquire().await?;
            // `max_in = amount`: sell exactly the balance, never bump the input.
            let prepared = self
                .dex_client
                .build_swap_transaction(input_mint, output_mint, amount, min_out, Some(amount))
                .await?;
            let signature = self
                .dex_client
                .send_and_confirm(&prepared.transaction, CommitmentLevel::Confirmed)
                .await?;
            anyhow::Ok((prepared, signature))
        })?;

        info!(
            "Swap successful via {:?}! Transaction: {}, Output: {} tokens of mint {}",
            prepared.aggregator, signature, prepared.out_amount, output_mint
        );

        Ok(prepared.out_amount)
    }
}
