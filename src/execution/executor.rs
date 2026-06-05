//! Submission + orchestration engine for the execution layer.
//!
//! `execute()` turns a [`LiquidationIntent`] into a landed liquidation:
//! 1. ask the strategy to assemble the txs (`[buy?] [liquidate]`),
//! 2. profit-gate,
//! 3. simulate-first: only prepend a crank tx when the program reports a stale oracle,
//! 4. submit as an atomic Jito bundle (with a tip), falling back to sequential RPC sends.
//!
//! The tip is added only on the bundle path; the sequential fallback sends the core txs as-is.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use solana_client::{rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig};
use solana_sdk::{
    commitment_config::{CommitmentConfig, CommitmentLevel},
    message::{v0, VersionedMessage},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    system_instruction,
    transaction::VersionedTransaction,
};

use crate::utils::{jito::JitoClient, swb_cranker::SwbCranker};

use super::{LiquidationIntent, LiquidationStrategy};

/// Number of status polls before a bundle is considered not landed.
const BUNDLE_CONFIRM_ATTEMPTS: usize = 10;

/// Don't re-crank a feed cranked more recently than this (dedup across intents in a drain).
const CRANK_DEDUP_COOLDOWN: Duration = Duration::from_secs(30);

pub struct Executor {
    jito: JitoClient,
    rpc_client: RpcClient,
    swb_cranker: Arc<SwbCranker>,
    signer: Keypair,
    /// RPC URL that supports `simulateBundle` (used for simulate-first crank detection).
    rpc_url: String,
    sim_api_key: Option<String>,
    tip_lamports: u64,
    tip_accounts: Vec<Pubkey>,
    /// Minimum estimated profit (USD) to bother executing.
    min_profit_usd: u64,
    /// Feeds cranked recently, to avoid double-cranking across intents in the same drain.
    recently_cranked: Mutex<HashMap<Pubkey, Instant>>,
}

impl Executor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        jito: JitoClient,
        rpc_client: RpcClient,
        swb_cranker: Arc<SwbCranker>,
        signer: Keypair,
        rpc_url: String,
        sim_api_key: Option<String>,
        tip_lamports: u64,
        min_profit_usd: u64,
    ) -> Self {
        // Fetch tip accounts once; if unavailable, bundles can't be tipped and we fall back to
        // sequential sends, so degrade gracefully rather than failing startup.
        let tip_accounts = match jito.get_tip_accounts() {
            Ok(accounts) if !accounts.is_empty() => accounts,
            Ok(_) => {
                warn!("Jito returned no tip accounts; bundles will fall back to sequential sends");
                Vec::new()
            }
            Err(e) => {
                warn!("Failed to fetch Jito tip accounts ({e}); bundles will fall back to sequential sends");
                Vec::new()
            }
        };

        Self {
            jito,
            rpc_client,
            swb_cranker,
            signer,
            rpc_url,
            sim_api_key,
            tip_lamports,
            tip_accounts,
            min_profit_usd,
            recently_cranked: Mutex::new(HashMap::new()),
        }
    }

    /// Assemble, gate, simulate-first, and land a single liquidation.
    pub fn execute(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &LiquidationIntent,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;

        let Some(plan) = strategy.assemble(intent)? else {
            debug!(
                "Strategy '{}' cannot handle {}; skipping",
                strategy.name(),
                liquidatee
            );
            return Ok(());
        };

        // Profit gate (USD). Full lamport-cost-aware gating (convert est_cost via SOL price) is a
        // follow-up; for now we gate on profit and log the estimated cost.
        if plan.est_profit < self.min_profit_usd {
            info!(
                "Skipping {}: est profit ${} < min ${} (est cost {} lamports)",
                liquidatee, plan.est_profit, self.min_profit_usd, plan.est_cost_lamports
            );
            return Ok(());
        }

        // Simulate-first: only pay for a crank if the program actually reports a stale oracle.
        let mut txs = plan.txs;
        let sim = self
            .jito
            .simulate_bundle(&self.rpc_url, self.sim_api_key.as_deref(), &txs, &[])?;
        if !sim.succeeded {
            if sim.is_stale_price_failure() {
                if let Some(crank_tx) = self.build_crank_if_needed(intent)? {
                    info!("Prepending SWB crank to bundle for {}", liquidatee);
                    txs.insert(0, crank_tx);
                }
            } else {
                warn!(
                    "Skipping {}: simulation failed: {}",
                    liquidatee,
                    sim.error_message.unwrap_or_default()
                );
                return Ok(());
            }
        }

        self.submit(&txs)
    }

    /// Build a crank tx for the intent's stale feeds, unless they were all cranked very recently.
    fn build_crank_if_needed(
        &self,
        intent: &LiquidationIntent,
    ) -> Result<Option<VersionedTransaction>> {
        let oracles = &intent.observation_accounts.swb_oracles;
        if oracles.is_empty() {
            return Ok(None);
        }

        let now = Instant::now();
        let mut guard = self
            .recently_cranked
            .lock()
            .map_err(|_| anyhow!("recently_cranked mutex poisoned"))?;
        guard.retain(|_, t| now.duration_since(*t) < CRANK_DEDUP_COOLDOWN);

        if oracles.iter().all(|o| guard.contains_key(o)) {
            debug!(
                "All SWB feeds for {} cranked within cooldown; skipping crank",
                intent.liquidatee_account.address
            );
            return Ok(None);
        }

        let crank_tx = self.swb_cranker.build_crank_tx(oracles.clone())?;
        for o in oracles {
            guard.insert(*o, now);
        }
        Ok(Some(crank_tx))
    }

    /// Land the ordered transactions: try as one atomic Jito bundle (with a tip), and if that
    /// fails to land, fall back to sending them sequentially over RPC (no tip, non-atomic).
    pub fn submit(&self, txs: &[VersionedTransaction]) -> Result<()> {
        if txs.is_empty() {
            return Err(anyhow!("Executor::submit called with no transactions"));
        }
        match self.try_bundle(txs) {
            Ok(bundle_id) => {
                info!("Bundle landed: {} ({} txs)", bundle_id, txs.len());
                Ok(())
            }
            Err(e) => {
                warn!(
                    "Bundle submission failed ({}); falling back to sequential send",
                    e
                );
                self.submit_sequential(txs)
            }
        }
    }

    /// Append a tip tx and submit the bundle to the block engine.
    fn try_bundle(&self, txs: &[VersionedTransaction]) -> Result<String> {
        let bundle_txs = self.with_tip(txs)?;
        self.jito
            .send_bundle_and_confirm(&bundle_txs, BUNDLE_CONFIRM_ATTEMPTS)
    }

    /// Clone the core txs and append a tip transfer tx (required for Jito bundles).
    fn with_tip(&self, txs: &[VersionedTransaction]) -> Result<Vec<VersionedTransaction>> {
        let tip_account = self
            .tip_accounts
            .first()
            .ok_or_else(|| anyhow!("No Jito tip account available"))?;
        let ix =
            system_instruction::transfer(&self.signer.pubkey(), tip_account, self.tip_lamports);
        let blockhash = self.rpc_client.get_latest_blockhash()?;
        let msg = v0::Message::try_compile(&self.signer.pubkey(), &[ix], &[], blockhash)?;
        let tip_tx = VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])?;

        let mut out = txs.to_vec();
        out.push(tip_tx);
        Ok(out)
    }

    /// Send each transaction in order, confirming before moving on. Used when the bundle path is
    /// unavailable; loses atomicity, so a later failure can leave earlier txs applied.
    fn submit_sequential(&self, txs: &[VersionedTransaction]) -> Result<()> {
        for (i, tx) in txs.iter().enumerate() {
            let sig = self
                .rpc_client
                .send_and_confirm_transaction_with_spinner_and_config(
                    tx,
                    CommitmentConfig::confirmed(),
                    RpcSendTransactionConfig {
                        skip_preflight: false,
                        preflight_commitment: Some(CommitmentLevel::Processed),
                        ..Default::default()
                    },
                )
                .map_err(|e| {
                    anyhow!("Sequential send failed at tx {}/{}: {}", i + 1, txs.len(), e)
                })?;
            info!("Sequential tx {}/{} confirmed: {}", i + 1, txs.len(), sig);
        }
        Ok(())
    }
}
