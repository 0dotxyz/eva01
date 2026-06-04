use crate::{
    cache::Cache,
    config::{Eva01Config, TokenThresholds},
    wrappers::{oracle::OracleWrapper, token_account::TokenAccountWrapper},
};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
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
};
use tokio::runtime::{Builder, Runtime};

const SLIPPAGE_MULTIPLIER: I80F48 = I80F48!(1.05);

/// The rebalancer is responsible to maintain the appropriate amounts of tokens on token accounts.
/// Guided primarily by token_thresholds and specific requests from the liquidator.
pub struct Rebalancer {
    swap_mint: Pubkey,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    default_token_max_threshold: I80F48,
    token_thresholds: HashMap<Pubkey, TokenThresholds>,
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
        })
    }

    pub fn run(&mut self, missing_tokens: HashMap<Pubkey, I80F48>) -> anyhow::Result<()> {
        info!("Running the Rebalancing process...");

        let swap_token_address = self.cache.tokens.try_get_token_for_mint(&self.swap_mint)?;
        let swap_wrapper = self
            .cache
            .try_get_token_wrapper_lenient(&self.swap_mint, &swap_token_address)?;

        if let Err(e) = self.handle_token_accounts(missing_tokens, &swap_wrapper) {
            error!("Failed to handle the Liquidator's tokens: {}", e);
        }

        info!("The Rebalancing process is complete.");

        Ok(())
    }

    fn handle_token_accounts(
        &mut self,
        missing_tokens: HashMap<Pubkey, I80F48>,
        swap_wrapper: &TokenAccountWrapper<OracleWrapper>,
    ) -> anyhow::Result<()> {
        let missing_mint_to_value =
            self.sell_excessive_tokens_and_collect_missing(missing_tokens)?;

        self.buy_missing_tokens(swap_wrapper, missing_mint_to_value)
    }

    fn sell_excessive_tokens_and_collect_missing(
        &mut self,
        bank_to_amount: HashMap<Pubkey, I80F48>,
    ) -> anyhow::Result<HashMap<Pubkey, I80F48>> {
        let mut mint_to_value: HashMap<Pubkey, I80F48> = HashMap::new();
        for mint in self.cache.mints.get_mints() {
            debug!("Processing token {}...", mint);
            if mint == self.swap_mint || self.empty_stake_banks.contains(&mint) {
                continue;
            }

            let token = self.cache.tokens.try_get_token_for_mint(&mint)?;
            let wrapper = self.cache.try_get_token_wrapper_lenient(&mint, &token);
            if let Err(e) = wrapper {
                // Ignore empty stake banks
                if e.to_string().contains("Stake pool supply is zero") {
                    self.empty_stake_banks.insert(mint);
                } else {
                    // SwitchboardStalePrice here at startup is harmless — SwbPriceFetcher
                    // populates synthetic oracle accounts on its first 30-second cycle.
                    warn!("Skipping the token {} in rebalancing: {}", mint, e);
                }
                continue;
            }

            let wrapper = wrapper.unwrap();

            if let Some(&amount) = bank_to_amount.get(&wrapper.bank_wrapper.address) {
                let value_to_swap = wrapper.get_value_for_amount(amount)?;
                let missing_value = if value_to_swap < I80F48::from_num(1) {
                    I80F48::from_num(1)
                } else {
                    value_to_swap
                        .checked_mul(SLIPPAGE_MULTIPLIER)
                        .ok_or_else(|| anyhow::anyhow!("Failed to calculate missing token value"))?
                };
                mint_to_value.insert(mint, missing_value);
                continue;
            }

            let value = wrapper.get_value()?;
            let min_value = self
                .token_thresholds
                .get(&mint)
                .map(|t| t.min_value)
                .unwrap_or(I80F48::ZERO);
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
                        info!("Got {} back from the swap.", swapped_amount);
                    }
                    Err(e) => {
                        error!("Swap failed: {}", e);
                    }
                }
            } else if value < min_value {
                // Buy only the shortfall to reach min, not the full min_value.
                let needed_value = min_value - value;
                info!("The value of {} tokens is lower than set threshold: {} < {}. Will buy ${} worth of tokens.", mint, value.to_num::<f64>(), min_value.to_num::<f64>(), needed_value.to_num::<f64>());
                mint_to_value.insert(mint, needed_value);
            }
        }
        Ok(mint_to_value)
    }

    fn buy_missing_tokens(
        &mut self,
        swap_token_wrapper: &TokenAccountWrapper<OracleWrapper>,
        mint_to_value: HashMap<Pubkey, I80F48>,
    ) -> anyhow::Result<()> {
        // We only spend swap_mint that is actually in the wallet (no MarginFi withdraw).
        // Cap each buy to the remaining wallet balance and stop once it is spent, instead of
        // sending swaps that revert on-chain with "insufficient funds".
        let mut available = I80F48::from_num(swap_token_wrapper.balance);
        for mint in self.cache.mints.get_mints() {
            if mint == self.swap_mint {
                continue;
            }

            if let Some(&value_to_swap) = mint_to_value.get(&mint) {
                if available <= I80F48::ZERO {
                    warn!(
                        "No {} left in the wallet to fund buys; skipping remaining tokens.",
                        self.swap_mint
                    );
                    break;
                }
                let desired = swap_token_wrapper.get_amount_from_value(value_to_swap)?;
                let amount_to_swap = desired.min(available);
                match self.swap(amount_to_swap.to_num(), self.swap_mint, mint) {
                    Ok(_) => {
                        available -= amount_to_swap;
                    }
                    Err(e) => error!("Swap failed: {}", e),
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
