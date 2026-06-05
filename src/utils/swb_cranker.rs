use anyhow::Result;
use log::warn;
use solana_client::{
    client_error::{ClientError, ClientErrorKind},
    nonblocking::rpc_client::RpcClient as NonBlockingRpcClient,
    rpc_client::RpcClient,
    rpc_config::RpcSendTransactionConfig,
    rpc_request::RpcError,
};
use solana_sdk::{
    commitment_config::{CommitmentConfig, CommitmentLevel},
    genesis_config::ClusterType,
    message::{v0, VersionedMessage},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    transaction::VersionedTransaction,
};
use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};
use switchboard_on_demand_client::{
    CrossbarClient, FetchUpdateManyParams, Gateway, PullFeed, QueueAccountData, SbContext,
};
use tokio::runtime::{Builder, Runtime};

use crate::config::Eva01Config;

pub const SWB_STALE_PRICE_ERROR_CODE: &str = "17a1";
pub const SWB_STALE_PRICE_ERROR_CODE_NUMBER: u32 = 6049;
pub const SWB_STALE_HANDLED_ERROR: &str = "STALE HANDLED";

const CHUNK_SIZE: usize = 6;
const ORACLE_QUARANTINE_DURATION: Duration = Duration::from_secs(10 * 60);

pub struct SwbCranker {
    tokio_rt: Runtime,
    rpc_client: RpcClient,
    non_blocking_rpc_client: NonBlockingRpcClient,
    swb_gateway: Gateway,
    crossbar: Option<CrossbarClient>,
    payer: Keypair,
    oracle_quarantine: Mutex<HashMap<Pubkey, Instant>>,
}

impl SwbCranker {
    pub fn new(config: &Eva01Config, cache: &crate::cache::Cache) -> Result<Self> {
        let payer = Keypair::from_bytes(&config.wallet_keypair)?;

        let tokio_rt = Builder::new_multi_thread()
            .thread_name("SwbCranker")
            .worker_threads(4)
            .enable_all()
            .build()?;

        let rpc_client =
            RpcClient::new_with_commitment(config.rpc_url.clone(), CommitmentConfig::confirmed());
        let non_blocking_rpc_client = NonBlockingRpcClient::new_with_commitment(
            config.rpc_url.clone(),
            CommitmentConfig::confirmed(),
        );
        let queue = tokio_rt.block_on(QueueAccountData::load(
            &non_blocking_rpc_client,
            &config.swb_program_id,
        ))?;

        // Prefer private gateway from env; fall back to first on-chain gateway
        let (swb_gateway, crossbar) = if let Some(url) = config.crossbar_api_url.as_ref() {
            let crossbar = CrossbarClient::new(url.as_str(), true);
            (
                tokio_rt.block_on(queue.fetch_gateway_from_crossbar(&crossbar))?,
                Some(crossbar),
            )
        } else {
            (
                tokio_rt.block_on(queue.fetch_gateways(&non_blocking_rpc_client))?[0].clone(),
                None,
            )
        };

        let _ = cache; // cache parameter reserved for future use (e.g. pre-filtering oracles)

        Ok(Self {
            tokio_rt,
            rpc_client,
            non_blocking_rpc_client,
            swb_gateway,
            crossbar,
            payer,
            oracle_quarantine: Mutex::new(HashMap::new()),
        })
    }

    pub fn crank_oracles(&self, swb_oracles: Vec<Pubkey>) -> Result<()> {
        let swb_oracles = self.filter_quarantined_oracles(&swb_oracles, "crank");
        if swb_oracles.is_empty() {
            return Ok(());
        }

        // Run simulations to get more details on potential failures, if crossbar is available.
        if let Some(crossbar) = self.crossbar.as_ref() {
            let result = self
                .tokio_rt
                .block_on(crossbar.simulate_solana_feeds(ClusterType::MainnetBeta, &swb_oracles));
            if let Err(result) = result {
                warn!("SWB Simulation failed: {:?}", result);
            }
        }

        for (chunk_index, chunk) in swb_oracles.chunks(CHUNK_SIZE).enumerate() {
            let chunk_oracles = chunk.to_vec();
            if let Err(err) = self.crank_oracles_internal(chunk_oracles.clone()) {
                warn!(
                    "SWB crank failed for chunk {} ({} feeds): {}. Retrying feeds individually.",
                    chunk_index,
                    chunk_oracles.len(),
                    err
                );

                let mut recovered_count = 0usize;
                let mut failed_individual: Vec<(Pubkey, anyhow::Error)> = Vec::new();
                for oracle in chunk_oracles {
                    match self.crank_oracles_internal(vec![oracle]) {
                        Ok(()) => recovered_count += 1,
                        Err(single_err) => failed_individual.push((oracle, single_err)),
                    }
                }

                if failed_individual.is_empty() {
                    continue;
                }

                if recovered_count > 0 {
                    let failed_oracles: Vec<Pubkey> = failed_individual
                        .iter()
                        .map(|(oracle, _)| *oracle)
                        .collect();
                    self.quarantine_oracles(
                        &failed_oracles,
                        "crank",
                        "individual crank failures after partial recovery",
                    );
                } else {
                    warn!(
                        "SWB crank chunk {} failed for all feeds even individually ({} feeds). Skipping this chunk without quarantine.",
                        chunk_index,
                        failed_individual.len()
                    );
                }
            }
        }
        Ok(())
    }

    fn crank_oracles_internal(&self, swb_oracles: Vec<Pubkey>) -> Result<()> {
        let tx = self.build_crank_transaction(swb_oracles)?;

        self.rpc_client
            .send_and_confirm_transaction_with_spinner_and_config(
                &tx,
                CommitmentConfig::confirmed(),
                RpcSendTransactionConfig {
                    skip_preflight: false,
                    preflight_commitment: Some(CommitmentLevel::Processed),
                    ..Default::default()
                },
            )?;

        Ok(())
    }

    fn build_crank_transaction(&self, swb_oracles: Vec<Pubkey>) -> Result<VersionedTransaction> {
        self.tokio_rt
            .block_on(self.build_crank_transaction_async(swb_oracles))
    }

    /// Build a signed, ready-to-send crank transaction for the given feeds. Used by the executor
    /// to prepend a crank to a liquidation bundle so the feeds are fresh in the same block.
    pub fn build_crank_tx(&self, swb_oracles: Vec<Pubkey>) -> Result<VersionedTransaction> {
        self.build_crank_transaction(swb_oracles)
    }

    async fn build_crank_transaction_async(
        &self,
        swb_oracles: Vec<Pubkey>,
    ) -> Result<VersionedTransaction> {
        let (crank_ix, crank_lut) = PullFeed::fetch_update_consensus_ix(
            SbContext::new(),
            &self.non_blocking_rpc_client,
            FetchUpdateManyParams {
                feeds: swb_oracles,
                payer: self.payer.pubkey(),
                gateway: self.swb_gateway.clone(),
                crossbar: self.crossbar.clone(),
                num_signatures: Some(1),
                ..Default::default()
            },
        )
        .await?;

        let blockhash = self
            .non_blocking_rpc_client
            .get_latest_blockhash_with_commitment(CommitmentConfig::confirmed())
            .await?
            .0;

        let tx = VersionedTransaction::try_new(
            VersionedMessage::V0(v0::Message::try_compile(
                &self.payer.pubkey(),
                &crank_ix,
                &crank_lut,
                blockhash,
            )?),
            &[&self.payer],
        )?;

        Ok(tx)
    }

    fn filter_quarantined_oracles(&self, oracles: &[Pubkey], context: &str) -> Vec<Pubkey> {
        let now = Instant::now();
        let mut quarantine_guard = match self.oracle_quarantine.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    "SWB oracle quarantine lock poisoned while filtering for {}. Continuing with recovered state.",
                    context
                );
                poisoned.into_inner()
            }
        };

        quarantine_guard.retain(|_, until| *until > now);

        let mut active_oracles: Vec<Pubkey> = Vec::with_capacity(oracles.len());
        let mut skipped_count = 0usize;
        for oracle in oracles {
            if quarantine_guard.contains_key(oracle) {
                skipped_count += 1;
            } else {
                active_oracles.push(*oracle);
            }
        }

        if skipped_count > 0 {
            warn!(
                "Skipping {} quarantined SWB feeds for {} (cooldown {}s).",
                skipped_count,
                context,
                ORACLE_QUARANTINE_DURATION.as_secs()
            );
        }

        active_oracles
    }

    fn quarantine_oracles(&self, oracles: &[Pubkey], context: &str, reason: &str) {
        if oracles.is_empty() {
            return;
        }
        let until = Instant::now() + ORACLE_QUARANTINE_DURATION;

        let mut quarantine_guard = match self.oracle_quarantine.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    "SWB oracle quarantine lock poisoned while quarantining for {}. Continuing with recovered state.",
                    context
                );
                poisoned.into_inner()
            }
        };

        for oracle in oracles {
            quarantine_guard.insert(*oracle, until);
        }

        warn!(
            "Quarantined {} SWB feeds for {} ({}s cooldown): {:?}",
            oracles.len(),
            context,
            ORACLE_QUARANTINE_DURATION.as_secs(),
            oracles
        );
        warn!("SWB quarantine reason for {}: {}", context, reason);
    }
}

pub fn is_stale_swb_price_error(err: &ClientError) -> bool {
    if let ClientErrorKind::RpcError(RpcError::RpcResponseError { message, .. }) = err.kind() {
        message.contains(SWB_STALE_PRICE_ERROR_CODE)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_client::client_error::ClientError;
    use solana_client::client_error::ClientErrorKind;
    use solana_client::rpc_request::RpcResponseErrorData;

    #[test]
    fn test_is_stale_swb_price_true_transaction_error() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::RpcError(RpcError::RpcResponseError {
                code: -32000,
                message: SWB_STALE_PRICE_ERROR_CODE.to_string(),
                data: RpcResponseErrorData::Empty,
            }),
        };
        assert!(is_stale_swb_price_error(&err));
    }

    #[test]
    fn test_is_stale_swb_price_false_wrong_custom_code() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::RpcError(RpcError::RpcResponseError {
                code: -32000,
                message: "12a4".to_string(),
                data: RpcResponseErrorData::Empty,
            }),
        };
        assert!(!is_stale_swb_price_error(&err));
    }

    #[test]
    fn test_is_stale_swb_price_false_other_instruction_error() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::RpcError(RpcError::ParseError("Test error".to_string())),
        };
        assert!(!is_stale_swb_price_error(&err));
    }

    #[test]
    fn test_is_stale_swb_price_false_wrong_code() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::Custom("Some other error".to_string()),
        };
        assert!(!is_stale_swb_price_error(&err));
    }

    #[test]
    fn test_is_stale_swb_price_false_other_kind() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::Io(std::io::Error::new(std::io::ErrorKind::Other, "io error")),
        };
        assert!(!is_stale_swb_price_error(&err));
    }
}
