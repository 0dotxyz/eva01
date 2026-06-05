use anyhow::Result;
use solana_client::nonblocking::rpc_client::RpcClient as NonBlockingRpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    message::{v0, VersionedMessage},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    transaction::VersionedTransaction,
};
use switchboard_on_demand_client::{
    CrossbarClient, FetchUpdateManyParams, Gateway, PullFeed, QueueAccountData, SbContext,
};
use tokio::runtime::{Builder, Runtime};

use crate::config::Eva01Config;

pub const SWB_STALE_PRICE_ERROR_CODE: &str = "17a1";
pub const SWB_STALE_PRICE_ERROR_CODE_NUMBER: u32 = 6049;
pub const SWB_STALE_HANDLED_ERROR: &str = "STALE HANDLED";

/// Builds Switchboard on-demand crank transactions. Cranking is done by the executor as part of
/// a liquidation bundle (simulate-first), so this only needs to *build* a signed crank tx.
pub struct SwbCranker {
    tokio_rt: Runtime,
    non_blocking_rpc_client: NonBlockingRpcClient,
    swb_gateway: Gateway,
    crossbar: Option<CrossbarClient>,
    payer: Keypair,
}

impl SwbCranker {
    pub fn new(config: &Eva01Config, cache: &crate::cache::Cache) -> Result<Self> {
        let payer = Keypair::from_bytes(&config.wallet_keypair)?;

        let tokio_rt = Builder::new_multi_thread()
            .thread_name("SwbCranker")
            .worker_threads(4)
            .enable_all()
            .build()?;

        let non_blocking_rpc_client = NonBlockingRpcClient::new_with_commitment(
            config.rpc_url.clone(),
            CommitmentConfig::confirmed(),
        );
        let queue = tokio_rt.block_on(QueueAccountData::load(
            &non_blocking_rpc_client,
            &config.swb_program_id,
        ))?;

        // Prefer private gateway from env; fall back to first on-chain gateway.
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

        let _ = cache; // reserved for future use (e.g. pre-filtering oracles)

        Ok(Self {
            tokio_rt,
            non_blocking_rpc_client,
            swb_gateway,
            crossbar,
            payer,
        })
    }

    /// Build a signed, ready-to-send crank transaction for the given feeds. Used by the executor
    /// to prepend a crank to a liquidation bundle so the feeds are fresh in the same block.
    pub fn build_crank_tx(&self, swb_oracles: Vec<Pubkey>) -> Result<VersionedTransaction> {
        self.tokio_rt
            .block_on(self.build_crank_transaction_async(swb_oracles))
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
}
