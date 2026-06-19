use crate::{cache::Cache, config::Eva01Config};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{error, info, warn};
use solana_commitment_config::CommitmentLevel;
use solana_dex_superagg::{
    client::DexSuperAggClient,
    config::{ClientConfig, JupiterConfig, RoutingStrategy, SharedConfig, TitanConfig},
};
use solana_program::pubkey::Pubkey;
use std::{collections::HashSet, sync::Arc};
use tokio::runtime::{Builder, Runtime};

/// Don't bother selling a position worth less than this (USD); the swap fee/dust isn't worth it.
const MIN_REBALANCE_VALUE: I80F48 = I80F48!(0.5);

/// The rebalancer keeps the liquidator holding only the swap mint (USDC): every other token it ends
/// up with — seized collateral from a liquidation, or a JIT-buy overshoot — is sold back to USDC on
/// the next pass. There is no inventory to maintain and nothing is bought here (liabilities are
/// funded just-in-time during liquidation), so it is a pure sell-to-USDC sweep.
pub struct Rebalancer {
    swap_mint: Pubkey,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    dex_client: Arc<DexSuperAggClient>,
    empty_stake_banks: HashSet<Pubkey>,
}

impl Rebalancer {
    pub fn new(config: Eva01Config, cache: Arc<Cache>) -> anyhow::Result<Self> {
        let swap_mint = config.swap_mint;

        let tokio_rt = Builder::new_multi_thread()
            .thread_name("rebalancer")
            .worker_threads(2)
            .enable_all()
            .build()?;

        // Convert wallet keypair to JSON string format expected by solana-dex-superagg
        let wallet_keypair_str = serde_json::to_string(&config.wallet_keypair)?;

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
            dex_client,
            empty_stake_banks: HashSet::new(),
        })
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

            let wrapper = match self.cache.try_get_token_wrapper_lenient(&mint, &token) {
                Ok(wrapper) => wrapper,
                Err(e) => {
                    // Ignore empty stake banks; SwitchboardStalePrice at startup is harmless
                    // (SwbPriceFetcher populates synthetic oracle accounts on its first cycle).
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
            if let Err(e) = self.swap(wrapper.balance, mint, self.swap_mint) {
                error!("Rebalance swap failed for {}: {}", mint, e);
            }
        }

        info!("The Rebalancing process is complete.");

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

        const WSOL: Pubkey = Pubkey::from_str_const("So11111111111111111111111111111111111111112");
        let wrap_and_unwrap_sol = input_mint == WSOL || output_mint == WSOL;

        let result = self.tokio_rt.block_on(self.dex_client.swap(
            &input_mint.to_string(),
            &output_mint.to_string(),
            amount,
            wrap_and_unwrap_sol,
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
