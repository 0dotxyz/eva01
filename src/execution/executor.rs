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
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
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

use crate::cache::Cache;
use crate::metrics::{
    FAILED_LIQUIDATIONS, LIQUIDATION_ATTEMPTS, LIQUIDATION_LATENCY_SECONDS, LIQUIDATION_SUCCESSES,
};
use crate::utils::{
    jito::{default_tip_accounts, BundleOutcome, JitoClient, TipEstimator},
    swb_cranker::SwbCranker,
};

use super::{LiquidationIntent, LiquidationStrategy};

/// Number of status polls before a bundle is considered not landed.
const BUNDLE_CONFIRM_ATTEMPTS: usize = 10;

/// Don't re-crank a feed cranked more recently than this (dedup across intents in a drain).
const CRANK_DEDUP_COOLDOWN: Duration = Duration::from_secs(30);

pub struct Executor {
    jito: JitoClient,
    rpc_client: RpcClient,
    cache: Arc<Cache>,
    swb_cranker: Arc<SwbCranker>,
    signer: Keypair,
    /// RPC URL that supports `simulateBundle` (used for simulate-first crank detection).
    rpc_url: String,
    /// Single key for both `sendBundle` (uuid) and `simulateBundle` (Bearer).
    bundle_api_key: Option<String>,
    tip_estimator: TipEstimator,
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
        cache: Arc<Cache>,
        swb_cranker: Arc<SwbCranker>,
        signer: Keypair,
        rpc_url: String,
        bundle_api_key: Option<String>,
        tip_max_lamports: u64,
        min_profit_usd: u64,
    ) -> Self {
        // The Jito tip accounts are static and well-known; use the hardcoded set rather than a
        // network fetch (the REST tip_accounts endpoint is unreliable).
        let tip_accounts = default_tip_accounts();

        Self {
            jito,
            rpc_client,
            cache,
            swb_cranker,
            signer,
            rpc_url,
            bundle_api_key,
            tip_estimator: TipEstimator::new(tip_max_lamports),
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

        // Gate on profit *before* assembling, so we don't run swap quotes or create LUTs for
        // sub-profit targets. (Full lamport-cost-aware gating via SOL price is a follow-up.)
        if intent.profit < self.min_profit_usd {
            info!(
                "Skipping {}: est profit ${} < min ${}",
                liquidatee, intent.profit, self.min_profit_usd
            );
            return Ok(());
        }

        LIQUIDATION_ATTEMPTS.inc();
        let _latency = LIQUIDATION_LATENCY_SECONDS.start_timer();

        let Some(plan) = strategy.assemble(intent)? else {
            debug!(
                "Strategy '{}' cannot handle {}; skipping",
                strategy.name(),
                liquidatee
            );
            return Ok(());
        };

        // Simulate-first crank detection. The outcome dispatches four ways:
        //  - ran & ok                 -> bundle send, no crank
        //  - ran & stale (0x17a1)     -> prepend crank, bundle send
        //  - ran & other prog error   -> skip (doomed on-chain; don't burn tip+fees)
        //  - couldn't run (infra err) -> crank all feeds + sequential RPC send (no bundle)
        // Temp LUTs created during assembly are always cleaned up afterwards, whatever the path.
        let temp_luts = plan.temp_luts;
        let mut txs = plan.txs;
        let result = match self.jito.simulate_bundle(
            &self.rpc_url,
            self.bundle_api_key.as_deref(),
            &txs,
            &[],
        ) {
            Ok(sim) if sim.succeeded => self.submit(&txs),
            Ok(sim) if sim.is_stale_price_failure() => {
                if let Some(crank_tx) = self.build_crank_if_needed(intent)? {
                    info!("Prepending SWB crank to bundle for {}", liquidatee);
                    txs.insert(0, crank_tx);
                }
                self.submit(&txs)
            }
            Ok(sim) => {
                FAILED_LIQUIDATIONS.inc();
                warn!(
                    "Skipping {}: simulation reports the liquidation would fail (tx index {:?}): {}",
                    liquidatee,
                    sim.failed_tx_index,
                    sim.error_message.unwrap_or_default()
                );
                Ok(())
            }
            Err(e) => {
                warn!(
                    "simulateBundle unavailable for {} ({}); cranking all feeds and sending sequentially",
                    liquidatee, e
                );
                if let Some(crank_tx) = self.build_crank_if_needed(intent)? {
                    txs.insert(0, crank_tx);
                }
                self.submit_sequential(&txs)
            }
        };

        self.deactivate_temp_luts(&temp_luts);
        result
    }

    /// Deactivate any temporary LUTs created during assembly (best-effort; logs on failure), and
    /// close earlier deactivated LUTs whose cooldown has elapsed to reclaim their rent.
    fn deactivate_temp_luts(&self, temp_luts: &[Pubkey]) {
        if temp_luts.is_empty() {
            return;
        }
        self.cache
            .try_close_deactivated_luts(&self.rpc_client, &self.signer);
        for lut_key in temp_luts {
            if let Err(e) =
                self.cache
                    .deactivate_targeted_lut(&self.rpc_client, &self.signer, *lut_key)
            {
                warn!("Failed to deactivate temporary LUT {lut_key}: {e}");
            }
        }
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

    /// Land the ordered transactions as an atomic Jito bundle (with a tip). Only fall back to
    /// sequential RPC when the bundle was *never accepted* (infra error / rejection). If it was
    /// accepted but didn't confirm in time, leave it in-flight (no resend) to avoid double
    /// execution — the next cycle re-evaluates and retries if it didn't land.
    pub fn submit(&self, txs: &[VersionedTransaction]) -> Result<()> {
        if txs.is_empty() {
            return Err(anyhow!("Executor::submit called with no transactions"));
        }
        match self.try_bundle(txs) {
            Ok(BundleOutcome::Confirmed(bundle_id)) => {
                LIQUIDATION_SUCCESSES.inc();
                info!("Bundle landed: {} ({} txs)", bundle_id, txs.len());
                Ok(())
            }
            Ok(BundleOutcome::Unconfirmed(bundle_id)) => {
                warn!(
                    "Bundle {} accepted but unconfirmed; leaving in-flight (will retry next cycle if it didn't land)",
                    bundle_id
                );
                Ok(())
            }
            Err(e) => {
                warn!(
                    "Bundle was not accepted ({}); falling back to sequential send",
                    e
                );
                self.submit_sequential(txs)
            }
        }
    }

    /// Append a tip tx and submit the bundle to the block engine.
    fn try_bundle(&self, txs: &[VersionedTransaction]) -> Result<BundleOutcome> {
        let bundle_txs = self.with_tip(txs)?;
        self.jito
            .send_bundle_and_confirm(&bundle_txs, BUNDLE_CONFIRM_ATTEMPTS)
    }

    /// Clone the core txs and append a tip transfer tx (required for Jito bundles).
    fn with_tip(&self, txs: &[VersionedTransaction]) -> Result<Vec<VersionedTransaction>> {
        if self.tip_accounts.is_empty() {
            return Err(anyhow!("No Jito tip account available"));
        }
        // Pick a tip account pseudo-randomly to spread tip load across them (Jito's guidance).
        let idx = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as usize)
            .unwrap_or(0)
            % self.tip_accounts.len();
        let tip_account = &self.tip_accounts[idx];
        let tip_lamports = self.tip_estimator.current_tip();
        let ix = system_instruction::transfer(&self.signer.pubkey(), tip_account, tip_lamports);
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
                    anyhow!(
                        "Sequential send failed at tx {}/{}: {}",
                        i + 1,
                        txs.len(),
                        e
                    )
                })?;
            info!("Sequential tx {}/{} confirmed: {}", i + 1, txs.len(), sig);
        }
        LIQUIDATION_SUCCESSES.inc();
        Ok(())
    }
}
