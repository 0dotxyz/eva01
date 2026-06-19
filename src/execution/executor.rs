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
use solana_commitment_config::{CommitmentConfig, CommitmentLevel};
use solana_sdk::{
    message::{v0, VersionedMessage},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    transaction::VersionedTransaction,
};
use solana_system_interface::instruction::transfer;

use crate::cache::Cache;
use crate::utils::{
    jito::{default_tip_accounts, BundleOutcome, JitoClient, TipEstimator},
    swb_cranker::SwbCranker,
};

use super::{LiquidationIntent, LiquidationStrategy};

/// Number of status polls before a bundle is considered not landed.
const BUNDLE_CONFIRM_ATTEMPTS: usize = 10;

/// Don't re-crank a feed cranked more recently than this (dedup across intents in a drain).
const CRANK_DEDUP_COOLDOWN: Duration = Duration::from_secs(30);

/// Re-check a liability mint that has no swap route only after this long (routes can appear).
const NO_ROUTE_QUARANTINE_TTL: Duration = Duration::from_secs(3600);

/// Base / cap for the per-target exponential backoff after a transient assemble failure
/// (e.g. a Jupiter `429`): skip the target for `BASE * 2^(failures-1)`, capped at `MAX`.
const TARGET_BACKOFF_BASE: Duration = Duration::from_secs(30);
const TARGET_BACKOFF_MAX: Duration = Duration::from_secs(900);

/// Per-target backoff state after a transient (non-route) assemble failure.
struct TargetBackoff {
    /// Consecutive assemble failures (drives the exponential cooldown).
    failures: u32,
    /// Don't re-attempt the target before this time.
    retry_after: Instant,
}

/// Exponential backoff: `BASE * 2^(failures-1)`, capped at `TARGET_BACKOFF_MAX`.
fn target_backoff_delay(failures: u32) -> Duration {
    let shift = failures.saturating_sub(1).min(16);
    let secs = TARGET_BACKOFF_BASE
        .as_secs()
        .saturating_mul(1u64 << shift)
        .min(TARGET_BACKOFF_MAX.as_secs());
    Duration::from_secs(secs)
}

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
    /// Liability mints with no swap route (NO_ROUTES_FOUND); quarantined to stop re-quoting them.
    quarantined_mints: Mutex<HashMap<Pubkey, Instant>>,
    /// Per-liquidatee backoff after a transient assemble failure (rate limits, network, etc.).
    failed_targets: Mutex<HashMap<Pubkey, TargetBackoff>>,
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
            quarantined_mints: Mutex::new(HashMap::new()),
            failed_targets: Mutex::new(HashMap::new()),
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

        // Skip targets we recently failed to assemble, *before* incurring an attempt or a Jupiter
        // quote: a liab mint with no swap route is quarantined; a target that hit a transient
        // error (rate limit, etc.) is backed off. Both expire so recovering targets retry.
        let liab_mint = self.intent_liab_mint(intent);
        if let Some(mint) = liab_mint {
            if self.is_mint_quarantined(&mint) {
                debug!(
                    "Skipping {}: liab mint {} has no swap route (quarantined)",
                    liquidatee, mint
                );
                return Ok(());
            }
        }
        if self.is_target_backed_off(&liquidatee) {
            debug!(
                "Skipping {}: backing off after a recent execution failure",
                liquidatee
            );
            return Ok(());
        }

        let plan = match strategy.assemble(intent) {
            Ok(Some(plan)) => {
                // Assembly succeeded (quote went through or no buy was needed): clear any backoff.
                self.clear_target_backoff(&liquidatee);
                plan
            }
            Ok(None) => {
                debug!(
                    "Strategy '{}' cannot handle {}; skipping",
                    strategy.name(),
                    liquidatee
                );
                return Ok(());
            }
            Err(e) => {
                // Record a quarantine/backoff so we stop hammering this target, then propagate so
                // the caller logs + counts the failure once (subsequent cycles skip silently).
                self.note_assemble_failure(&liquidatee, liab_mint, &e);
                return Err(e);
            }
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

    /// The liability mint for an intent, looked up via the bank cache (`None` if unavailable).
    fn intent_liab_mint(&self, intent: &LiquidationIntent) -> Option<Pubkey> {
        self.cache
            .banks
            .try_get_bank(&intent.liab_bank)
            .ok()
            .map(|b| b.bank.mint)
    }

    /// Whether a liability mint is currently quarantined (no swap route). Prunes expired entries.
    fn is_mint_quarantined(&self, mint: &Pubkey) -> bool {
        let now = Instant::now();
        let Ok(mut guard) = self.quarantined_mints.lock() else {
            return false;
        };
        guard.retain(|_, until| now < *until);
        guard.contains_key(mint)
    }

    /// Whether a target is currently backed off after a transient failure. Prunes expired entries.
    fn is_target_backed_off(&self, target: &Pubkey) -> bool {
        let now = Instant::now();
        let Ok(mut guard) = self.failed_targets.lock() else {
            return false;
        };
        guard.retain(|_, b| now < b.retry_after);
        guard.contains_key(target)
    }

    /// Clear a target's backoff once it assembles successfully again.
    fn clear_target_backoff(&self, target: &Pubkey) {
        if let Ok(mut guard) = self.failed_targets.lock() {
            guard.remove(target);
        }
    }

    /// Record an assemble failure: quarantine the liab mint when it has no swap route
    /// (`NO_ROUTES_FOUND`), otherwise apply an exponential per-target backoff.
    fn note_assemble_failure(
        &self,
        target: &Pubkey,
        liab_mint: Option<Pubkey>,
        e: &anyhow::Error,
    ) {
        if e.to_string().contains("NO_ROUTES_FOUND") {
            if let Some(mint) = liab_mint {
                if let Ok(mut guard) = self.quarantined_mints.lock() {
                    guard.insert(mint, Instant::now() + NO_ROUTE_QUARANTINE_TTL);
                }
                warn!(
                    "Quarantining liab mint {} for {}s: no swap route",
                    mint,
                    NO_ROUTE_QUARANTINE_TTL.as_secs()
                );
                return;
            }
        }

        if let Ok(mut guard) = self.failed_targets.lock() {
            let entry = guard.entry(*target).or_insert(TargetBackoff {
                failures: 0,
                retry_after: Instant::now(),
            });
            entry.failures = entry.failures.saturating_add(1);
            let delay = target_backoff_delay(entry.failures);
            entry.retry_after = Instant::now() + delay;
            debug!(
                "Backing off {} for {}s (consecutive assemble failures: {})",
                target,
                delay.as_secs(),
                entry.failures
            );
        }
    }

    /// Deactivate any temporary LUTs created during assembly (best-effort; logs on failure), and
    /// close earlier deactivated LUTs whose cooldown has elapsed to reclaim their rent.
    fn deactivate_temp_luts(&self, temp_luts: &[Pubkey]) {
        if temp_luts.is_empty() {
            return;
        }
        self.cache
            .try_close_deactivated_luts(&self.rpc_client, &self.signer);
        // The cache tracks a single targeted LUT (the one `build_liquidate_tx` just created), so
        // deactivate it once rather than per key; the signature reads the key from cache state.
        if let Err(e) = self
            .cache
            .deactivate_targeted_lut(&self.rpc_client, &self.signer)
        {
            warn!("Failed to deactivate temporary LUT(s) {temp_luts:?}: {e}");
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
        let ix = transfer(&self.signer.pubkey(), tip_account, tip_lamports);
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
        Ok(())
    }
}
