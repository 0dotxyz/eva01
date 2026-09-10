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
use solana_sdk::{
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    transaction::VersionedTransaction,
};

use crate::cache::Cache;
use crate::utils::{
    jito::{BundleOutcome, JitoClient, TipEstimator},
    swb_cranker::SwbCranker,
};
use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;

use super::{ExecutionPlan, LiquidationStrategy};

/// Number of Jito status polls before checking the chain directly. Kept small: the public status
/// endpoint is frequently rate-limited, so the RPC signature check below is the real confirmation
/// and we want to reach the sequential fallback quickly (while the txs' blockhash is still fresh).
const BUNDLE_CONFIRM_ATTEMPTS: usize = 3;

/// After Jito reports a bundle accepted-but-unconfirmed (or its status endpoint is rate-limited),
/// poll the RPC for the liquidation tx itself this many times, this interval apart, before giving
/// up. The on-chain signature is the authoritative "did it land" — independent of Jito's throttled
/// `getBundleStatuses`.
const SIGNATURE_CONFIRM_ATTEMPTS: usize = 10;
const SIGNATURE_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Re-check a liability mint that has no swap route only after this long (routes can appear).
const NO_ROUTE_QUARANTINE_TTL: Duration = Duration::from_secs(3600);

/// After a crank actually LANDS, treat its feeds as fresh for this long so the sim-unavailable
/// path doesn't re-crank them unconditionally every attempt. Kept comfortably below the Switchboard
/// pull-feed staleness window, so a feed that genuinely goes stale is still re-cranked in time.
const CRANK_LANDED_COOLDOWN: Duration = Duration::from_secs(30);

/// Base delays for the per-target exponential backoff: skip the target for
/// `BASE * 2^(failures-1)`, capped at `TARGET_BACKOFF_MAX`.
///
/// Assemble failures are typically rate limits (a Jupiter `429`), which need a real pause;
/// an execution failure can just be a lost race, so it starts short and escalates.
const ASSEMBLE_BACKOFF_BASE: Duration = Duration::from_secs(30);
const EXECUTION_BACKOFF_BASE: Duration = Duration::from_secs(5);
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

/// Exponential backoff: `base * 2^(failures-1)`, capped at `TARGET_BACKOFF_MAX`.
fn target_backoff_delay(base: Duration, failures: u32) -> Duration {
    let shift = failures.saturating_sub(1).min(16);
    let secs = base
        .as_secs()
        .saturating_mul(1u64 << shift)
        .min(TARGET_BACKOFF_MAX.as_secs());
    Duration::from_secs(secs)
}

/// Bump `target`'s consecutive-failure count and push out its retry time.
fn bump_target_backoff(
    backoffs: &mut HashMap<Pubkey, BackoffState>,
    target: &Pubkey,
    base: Duration,
    kind: &str,
) {
    let failures = match backoffs.get(target) {
        Some(BackoffState::Target(backoff)) => backoff.failures,
        _ => 0,
    }
    .saturating_add(1);
    let delay = target_backoff_delay(base, failures);
    backoffs.insert(
        *target,
        BackoffState::Target(TargetBackoff {
            failures,
            retry_after: Instant::now() + delay,
        }),
    );
    debug!(
        "Backing off {} for {}s (consecutive {} failures: {})",
        target,
        delay.as_secs(),
        kind,
        failures
    );
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
    /// SWB feeds whose crank actually LANDED, with when. Used only to avoid re-cranking still-fresh
    /// feeds on the sim-unavailable path (where staleness can't be detected); the sim-stale path
    /// always cranks regardless, since the sim proves the feed is stale on-chain.
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
            Ok(Some(plan)) => plan,
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
        let mut cranked = false;
        let mut attempt_failed = false;
        let result = match self.jito.simulate_bundle(
            &self.rpc_url,
            self.bundle_api_key.as_deref(),
            &txs,
            &[],
        ) {
            Ok(sim) if sim.succeeded => self.submit(&txs),
            Ok(sim) if sim.is_stale_price_failure() => {
                // Sim proves the feed is stale on-chain: always crank (force).
                if let Some(crank_tx) = self.build_crank_if_needed(intent, true) {
                    info!("Prepending SWB crank to bundle for {}", liquidatee);
                    txs.insert(0, crank_tx);
                    cranked = true;
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
                        attempt_failed = true;
                        Ok(())
                    }
                    Err(e) => {
                        warn!(
                            "Skipping {}: re-simulation after crank failed: {}",
                            liquidatee, e
                        );
                        attempt_failed = true;
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
                attempt_failed = true;
                Ok(())
            }
            Err(e) => {
                warn!(
                    "simulateBundle unavailable for {} ({}); cranking all feeds and sending sequentially",
                    liquidatee, e
                );
                // Staleness can't be detected here: crank defensively, but skip feeds we already
                // landed a crank for recently (still fresh) instead of cranking unconditionally.
                if let Some(crank_tx) = self.build_crank_if_needed(intent, false) {
                    txs.insert(0, crank_tx);
                    cranked = true;
                }
                self.submit_sequential(&txs)
            }
        };

        // Record the crank as landed only when the submission actually succeeded, so a stale feed
        // is never wrongly treated as fresh by the sim-unavailable path above.
        if cranked && result.is_ok() {
            self.record_landed_cranks(&intent.observation_accounts.swb_oracles);
        }

        // Back off a target whose execution failed just like a failed assembly: every retry
        // re-quotes the DEX and creates + deactivates a temporary LUT without landing anything.
        if attempt_failed || result.is_err() {
            self.note_execution_failure(&liquidatee);
        } else {
            self.clear_target_backoff(&liquidatee);
        }

        self.deactivate_temp_luts(temp_luts);
        result
    }

    /// The liability mint for an intent, looked up via the bank cache.
    fn intent_liab_mint(&self, intent: &PreparedLiquidatableAccount) -> Result<Pubkey> {
        Ok(self.cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint)
    }

    /// Whether a mint or target is currently backed off. Expired target entries are kept for one
    /// more `TARGET_BACKOFF_MAX` window so their failure counter — and thus the escalating delay —
    /// survives a retry that fails again.
    fn is_backed_off(&self, key: &Pubkey) -> Result<bool> {
        let now = Instant::now();
        let mut guard = self
            .backoffs
            .lock()
            .map_err(|_| anyhow!("execution backoffs mutex poisoned"))?;
        guard.retain(|_, state| match state {
            BackoffState::Mint(retry_after) => now < *retry_after,
            BackoffState::Target(backoff) => now < backoff.retry_after + TARGET_BACKOFF_MAX,
        });
        Ok(guard.get(key).is_some_and(|state| now < state.retry_after()))
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

        bump_target_backoff(&mut guard, target, ASSEMBLE_BACKOFF_BASE, "assemble");
        Ok(())
    }

    /// Record an execution failure: the plan assembled, but the liquidation could not be landed
    /// (simulation says it would revert, or the send failed). Same exponential backoff as an
    /// assemble failure.
    fn note_execution_failure(&self, target: &Pubkey) {
        if let Ok(mut guard) = self.backoffs.lock() {
            bump_target_backoff(&mut guard, target, EXECUTION_BACKOFF_BASE, "execution");
        }
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
            // Deactivate exactly the LUT(s) this liquidation created — by key, no shared slot — so
            // concurrent liquidations never step on each other's LUTs.
            for lut_key in temp_luts {
                if let Err(e) = cache.deactivate_lut(&rpc_client, &signer, lut_key) {
                    warn!("Failed to deactivate temporary LUT {lut_key}: {e}");
                }
            }
        });
    }

    /// Build a crank tx for the intent's SWB feeds.
    ///
    /// `force` (sim reported the feed stale on-chain): always crank — the sim is authoritative, and
    /// a prior *submitted-but-unlanded* crank must not suppress it or the account stays
    /// un-liquidatable while its feed is still stale. Self-limiting: once the crank lands, the sim
    /// stops reporting stale.
    ///
    /// `!force` (sim was unavailable, so staleness can't be detected — we'd otherwise crank
    /// defensively every attempt): skip if we LANDED a crank for all these feeds recently, since
    /// they're still fresh. Only landed cranks are recorded (see `record_landed_cranks`).
    fn build_crank_if_needed(
        &self,
        intent: &PreparedLiquidatableAccount,
        force: bool,
    ) -> Option<VersionedTransaction> {
        let oracles = &intent.observation_accounts.swb_oracles;
        if oracles.is_empty() {
            return None;
        }

        if !force {
            if let Ok(mut guard) = self.recently_cranked.lock() {
                let now = Instant::now();
                guard.retain(|_, t| now.duration_since(*t) < CRANK_LANDED_COOLDOWN);
                if oracles.iter().all(|o| guard.contains_key(o)) {
                    debug!(
                        "Skipping crank for {}: all SWB feeds landed-cranked within cooldown",
                        intent.liquidatee_account.address
                    );
                    return None;
                }
            }
        }

        match self.swb_cranker.build_crank_transaction(oracles.clone()) {
            Ok(crank_tx) => Some(crank_tx),
            Err(e) => {
                warn!(
                    "Failed to build SWB crank tx for {}: {}",
                    intent.liquidatee_account.address, e
                );
                None
            }
        }
    }

    /// Mark these SWB feeds as freshly cranked, once their crank has actually landed. Only landed
    /// cranks count, so an accepted-but-unconfirmed bundle never suppresses a later needed crank.
    fn record_landed_cranks(&self, oracles: &[Pubkey]) {
        if oracles.is_empty() {
            return;
        }
        if let Ok(mut guard) = self.recently_cranked.lock() {
            let now = Instant::now();
            for oracle in oracles {
                guard.insert(*oracle, now);
            }
        }
    }

    /// Land the ordered transactions as an atomic Jito bundle (with a tip). Only fall back to
    /// sequential RPC when the bundle was *never accepted* (infra error / rejection). If it was
    /// accepted but didn't confirm in time, leave it in-flight (no resend) to avoid double
    /// execution — the next cycle re-evaluates and retries if it didn't land.
    pub fn submit(&self, txs: &[VersionedTransaction]) -> Result<()> {
        if txs.is_empty() {
            return Err(anyhow!("Executor::submit called with no transactions"));
        }
        // The liquidation tx is the last core tx (any crank/buy precede it); its signature is our
        // authoritative "did it land" check when Jito's bundle status is unavailable/lagging.
        let liquidation_sig = txs.last().and_then(|tx| tx.signatures.first()).copied();
        match self.try_bundle(txs) {
            Ok(BundleOutcome::Confirmed(bundle_id)) => {
                info!("Bundle landed: {} ({} txs)", bundle_id, txs.len());
                Ok(())
            }
            Ok(BundleOutcome::Unconfirmed(bundle_id)) => {
                // Jito's `getBundleStatuses` can lag or be rate-limited (`-32097`); confirm against
                // the chain directly before deciding it didn't land.
                match liquidation_sig {
                    Some(sig) => match self.confirm_on_chain(&sig) {
                        Ok(true) => {
                            info!(
                                "Bundle {} landed ({} txs), confirmed on-chain via {}",
                                bundle_id,
                                txs.len(),
                                sig
                            );
                            Ok(())
                        }
                        Ok(false) => {
                            // The bundle was accepted but never landed (common on the rate-limited
                            // public block engine). Don't spin re-submitting bundles that won't
                            // land — fall back to a direct RPC send, which is what actually lands.
                            // Idempotent: the txs keep their signatures, so a late-landing bundle
                            // just makes the resend a no-op ("already processed").
                            warn!(
                                "Bundle {} accepted but liquidation tx {} not on-chain; falling back to sequential send",
                                bundle_id, sig
                            );
                            self.submit_sequential(txs)
                        }
                        Err(e) => {
                            // Bundles are atomic: a reverted liquidation tx means nothing applied,
                            // so the next cycle can safely retry.
                            warn!(
                                "Bundle {} did not land (liquidation tx {}: {}); next cycle retries",
                                bundle_id, sig, e
                            );
                            Ok(())
                        }
                    },
                    None => {
                        warn!(
                            "Bundle {} accepted but unconfirmed and no signature to verify; leaving in-flight",
                            bundle_id
                        );
                        Ok(())
                    }
                }
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

    /// Authoritative confirmation: poll the RPC for the liquidation tx landing on-chain,
    /// independent of Jito's throttled bundle status. Returns `Ok(true)` once it reaches
    /// `confirmed`, `Ok(false)` if it never appears within the window (still possibly in-flight),
    /// and `Err` if it landed with an on-chain error (atomic bundle reverted).
    fn confirm_on_chain(&self, sig: &Signature) -> Result<bool> {
        for _ in 0..SIGNATURE_CONFIRM_ATTEMPTS {
            match self.rpc_client.get_signature_statuses(&[*sig]) {
                Ok(resp) => {
                    if let Some(Some(status)) = resp.value.into_iter().next() {
                        if let Some(err) = &status.err {
                            return Err(anyhow!("reverted on-chain: {:?}", err));
                        }
                        if status.satisfies_commitment(CommitmentConfig::confirmed()) {
                            return Ok(true);
                        }
                    }
                }
                Err(e) => debug!("signature status poll for {} failed: {}", sig, e),
            }
            thread::sleep(SIGNATURE_POLL_INTERVAL);
        }
        Ok(false)
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
