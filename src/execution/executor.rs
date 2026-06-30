//! Submission + orchestration engine for the execution layer.
//!
//! `try_execute()` turns a prepared liquidation into a landed liquidation:
//! 1. ask the strategy to assemble the txs (`[buy?] [liquidate]`),
//! 2. simulate-first: only prepend a crank tx when the program reports a stale oracle,
//! 3. submit as an atomic Jito bundle (with a tip), falling back to sequential RPC sends.
//!
//! The tip is added only on the bundle path; the sequential fallback sends the core txs as-is.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use solana_client::{rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig};
use solana_commitment_config::{CommitmentConfig, CommitmentLevel};
use solana_sdk::{pubkey::Pubkey, signature::Keypair, transaction::VersionedTransaction};

use crate::cache::Cache;
use crate::utils::{
    jito::{BundleOutcome, JitoClient, TipEstimator},
    swb_cranker::SwbCranker,
};
use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;

use super::{ExecutionPlan, LiquidationStrategy};

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

enum BackoffState {
    /// Liability mint has no swap route until this instant.
    Mint(Instant),
    /// Liquidatee target hit transient assemble failures.
    Target(TargetBackoff),
}

impl BackoffState {
    fn retry_after(&self) -> Instant {
        match self {
            Self::Mint(retry_after) => *retry_after,
            Self::Target(backoff) => backoff.retry_after,
        }
    }
}

/// Exponential backoff: `BASE * 2^(failures-1)`, capped at `TARGET_BACKOFF_MAX`.
fn target_backoff_delay(failures: u32) -> Duration {
    let shift = failures.saturating_sub(1).min(8);
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
    /// Feeds cranked recently, to avoid double-cranking across intents in the same drain.
    recently_cranked: Mutex<HashMap<Pubkey, Instant>>,
    /// Liability-mint no-route quarantines and per-liquidatee transient backoffs.
    backoffs: Mutex<HashMap<Pubkey, BackoffState>>,
    /// Count sequential fallback uses so logs reveal how often the non-bundle path is exercised.
    sequential_fallbacks: AtomicU64,
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
    ) -> Self {
        Self {
            jito,
            rpc_client,
            cache,
            swb_cranker,
            signer,
            rpc_url,
            bundle_api_key,
            tip_estimator: TipEstimator::new(tip_max_lamports),
            recently_cranked: Mutex::new(HashMap::new()),
            backoffs: Mutex::new(HashMap::new()),
            sequential_fallbacks: AtomicU64::new(0),
        }
    }

    /// Assemble, gate, simulate-first, and land a single liquidation.
    pub fn try_execute(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;

        // Skip targets we recently failed to assemble, *before* incurring an attempt or a Jupiter
        // quote: a liab mint with no swap route is quarantined; a target that hit a transient
        // error (rate limit, etc.) is backed off. Both expire so recovering targets retry.
        let liab_mint = self.intent_liab_mint(intent)?;
        if self.is_backed_off(&liab_mint)? {
            debug!(
                "Skipping {}: liab mint {} has no swap route (quarantined)",
                liquidatee, liab_mint
            );
            return Ok(());
        }
        if self.is_backed_off(&liquidatee)? {
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
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                return Err(e);
            }
        };

        // Simulate-first crank detection. The outcome dispatches four ways:
        //  - ran & ok                 -> bundle send, no crank
        //  - ran & stale (0x17a1)     -> prepend crank, bundle send
        //  - ran & other prog error   -> skip (doomed on-chain; don't burn tip+fees)
        //  - couldn't run (infra err) -> crank all feeds + sequential RPC send (no bundle)
        // Temp LUTs created during assembly are always cleaned up afterwards, whatever the path.
        let ExecutionPlan { mut txs, temp_luts } = plan;
        let result = match self.jito.simulate_bundle(
            &self.rpc_url,
            self.bundle_api_key.as_deref(),
            &txs,
            &[],
        ) {
            Ok(sim) if sim.succeeded => self.submit(&txs),
            Ok(sim) if sim.is_stale_price_failure() => {
                if let Some(crank_tx) = self.build_crank_if_needed(intent) {
                    info!("Prepending SWB crank to bundle for {}", liquidatee);
                    txs.insert(0, crank_tx);
                }
                // Re-simulate with the crank applied: a stale oracle made the first sim fail, so
                // only submit if the cranked bundle now actually succeeds. Otherwise we'd pay to
                // land a bundle that still reverts once the real price is posted (e.g. the account
                // turns out healthy, 0x17b4) — which is exactly what Jito drops as "Failed".
                match self.jito.simulate_bundle(
                    &self.rpc_url,
                    self.bundle_api_key.as_deref(),
                    &txs,
                    &[],
                ) {
                    Ok(sim2) if sim2.succeeded => self.submit(&txs),
                    Ok(sim2) => {
                        warn!(
                            "Skipping {}: bundle still fails after crank (tx index {:?}): {}",
                            liquidatee,
                            sim2.failed_tx_index,
                            sim2.error_message.unwrap_or_default()
                        );
                        Ok(())
                    }
                    Err(e) => {
                        warn!(
                            "Skipping {}: re-simulation after crank failed: {}",
                            liquidatee, e
                        );
                        Ok(())
                    }
                }
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
                if let Some(crank_tx) = self.build_crank_if_needed(intent) {
                    txs.insert(0, crank_tx);
                }
                self.submit_sequential(&txs)
            }
        };

        self.deactivate_temp_luts(temp_luts);
        result
    }

    /// The liability mint for an intent, looked up via the bank cache.
    fn intent_liab_mint(&self, intent: &PreparedLiquidatableAccount) -> Result<Pubkey> {
        Ok(self.cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint)
    }

    /// Whether a mint or target is currently backed off. Prunes expired entries.
    fn is_backed_off(&self, key: &Pubkey) -> Result<bool> {
        let now = Instant::now();
        let mut guard = self
            .backoffs
            .lock()
            .map_err(|_| anyhow!("execution backoffs mutex poisoned"))?;
        guard.retain(|_, state| now < state.retry_after());
        Ok(guard.contains_key(key))
    }

    /// Clear a target's backoff once it assembles successfully again.
    fn clear_target_backoff(&self, target: &Pubkey) {
        if let Ok(mut guard) = self.backoffs.lock() {
            if matches!(guard.get(target), Some(BackoffState::Target(_))) {
                guard.remove(target);
            }
        }
    }

    /// Record an assemble failure: quarantine the liab mint when it has no swap route
    /// (`NO_ROUTES_FOUND`), otherwise apply an exponential per-target backoff.
    fn note_assemble_failure(
        &self,
        target: &Pubkey,
        liab_mint: Pubkey,
        e: &anyhow::Error,
    ) -> Result<()> {
        let mut guard = self
            .backoffs
            .lock()
            .map_err(|_| anyhow!("execution backoffs mutex poisoned"))?;

        if e.to_string().contains("NO_ROUTES_FOUND") {
            guard.insert(
                liab_mint,
                BackoffState::Mint(Instant::now() + NO_ROUTE_QUARANTINE_TTL),
            );
            warn!(
                "Quarantining liab mint {} for {}s: no swap route",
                liab_mint,
                NO_ROUTE_QUARANTINE_TTL.as_secs()
            );
            return Ok(());
        }

        let entry = guard.entry(*target).or_insert_with(|| {
            BackoffState::Target(TargetBackoff {
                failures: 0,
                retry_after: Instant::now(),
            })
        });
        let entry = match entry {
            BackoffState::Target(entry) => entry,
            BackoffState::Mint(_) => {
                *entry = BackoffState::Target(TargetBackoff {
                    failures: 0,
                    retry_after: Instant::now(),
                });
                match entry {
                    BackoffState::Target(entry) => entry,
                    BackoffState::Mint(_) => unreachable!("target backoff entry was just inserted"),
                }
            }
        };
        entry.failures = entry.failures.saturating_add(1);
        let delay = target_backoff_delay(entry.failures);
        entry.retry_after = Instant::now() + delay;
        debug!(
            "Backing off {} for {}s (consecutive assemble failures: {})",
            target,
            delay.as_secs(),
            entry.failures
        );
        Ok(())
    }

    /// Deactivate any temporary LUTs created during assembly (best-effort; logs on failure), and
    /// close earlier deactivated LUTs whose cooldown has elapsed to reclaim their rent.
    fn deactivate_temp_luts(&self, temp_luts: Vec<Pubkey>) {
        if temp_luts.is_empty() {
            return;
        }
        let cache = self.cache.clone();
        let rpc_url = self.rpc_url.clone();
        let signer_bytes = self.signer.to_bytes();
        thread::spawn(move || {
            let signer = match Keypair::try_from(signer_bytes.as_slice()) {
                Ok(signer) => signer,
                Err(e) => {
                    warn!("Failed to rebuild signer for temporary LUT cleanup: {e}");
                    return;
                }
            };
            let rpc_client = RpcClient::new_with_commitment(rpc_url, CommitmentConfig::confirmed());
            cache.try_close_deactivated_luts(&rpc_client, &signer);
            // The cache tracks a single targeted LUT (the one just created), so deactivate it once
            // rather than per key; the signature reads the key from cache state.
            if let Err(e) = cache.deactivate_targeted_lut(&rpc_client, &signer) {
                warn!("Failed to deactivate temporary LUT(s) {temp_luts:?}: {e}");
            }
        });
    }

    /// Build a crank tx for the intent's stale feeds, unless they were all cranked very recently.
    fn build_crank_if_needed(
        &self,
        intent: &PreparedLiquidatableAccount,
    ) -> Option<VersionedTransaction> {
        let oracles = &intent.observation_accounts.swb_oracles;
        if oracles.is_empty() {
            return None;
        }

        let now = Instant::now();
        let mut guard = match self.recently_cranked.lock() {
            Ok(guard) => guard,
            Err(_) => {
                warn!("recently_cranked mutex poisoned");
                return None;
            }
        };
        guard.retain(|_, t| now.duration_since(*t) < CRANK_DEDUP_COOLDOWN);

        if oracles.iter().all(|o| guard.contains_key(o)) {
            debug!(
                "All SWB feeds for {} cranked within cooldown; skipping crank",
                intent.liquidatee_account.address
            );
            return None;
        }

        let crank_tx = match self.swb_cranker.build_crank_transaction(oracles.clone()) {
            Ok(crank_tx) => crank_tx,
            Err(e) => {
                warn!(
                    "Failed to build SWB crank tx for {}: {}",
                    intent.liquidatee_account.address, e
                );
                return None;
            }
        };
        for o in oracles {
            guard.insert(*o, now);
        }
        Some(crank_tx)
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
        let mut bundle_txs = txs.to_vec();
        bundle_txs.push(
            self.tip_estimator
                .build_tip_transaction(&self.signer, &self.rpc_client)?,
        );
        self.jito
            .send_bundle_and_confirm(&bundle_txs, BUNDLE_CONFIRM_ATTEMPTS)
    }

    /// Send each transaction in order, confirming before moving on. Used when the bundle path is
    /// unavailable; loses atomicity, so a later failure can leave earlier txs applied.
    fn submit_sequential(&self, txs: &[VersionedTransaction]) -> Result<()> {
        let fallback_count = self.sequential_fallbacks.fetch_add(1, Ordering::Relaxed) + 1;
        warn!(
            "Sequential fallback #{}: sending {} transaction(s)",
            fallback_count,
            txs.len()
        );
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
