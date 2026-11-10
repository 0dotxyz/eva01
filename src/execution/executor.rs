//! Submission + orchestration engine for the execution layer.
//!
//! `try_execute()` turns a prepared liquidation into a landed liquidation:
//! 1. ask the strategy for a sandwich (`[liquidate]`, self-funded): a single tx that touches no
//!    wallet inventory, sent directly over RPC;
//! 2. otherwise ask for the inventory plan, which claims the wallet tokens it spends:
//!    `[liquidate]` paid from holdings is sent directly over RPC like a sandwich; `[buy]
//!    [liquidate]` is simulated first, then sent as an atomic Jito bundle (with a tip), falling
//!    back to sequential RPC sends. The claim is released once the plan has landed or failed and
//!    the wallet balances have been re-read.
//!
//! Nothing waits for another plan: any number of liquidations run at once, bounded by the workers.
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
use solana_keypair::Keypair;
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;

use crate::cache::Cache;
use crate::utils::jito::{BundleOutcome, JitoClient, TipEstimator};
use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;

use super::{ExecutionPlan, LiquidationStrategy, Sandwich};

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

/// A transaction sent over RPC is re-sent this often until it confirms or its blockhash expires:
/// a single send can be dropped on the way to the leader, which is indistinguishable from a slow
/// confirmation, and the signature makes every re-send idempotent.
const RESEND_INTERVAL: Duration = Duration::from_secs(2);
/// How often the status of an in-flight RPC send is polled.
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Hard ceiling on waiting for an RPC send, in case the blockhash validity check misbehaves.
const SEND_MAX_WAIT: Duration = Duration::from_secs(90);

/// Reads of the wallet balances made while settling a plan, this far apart, before giving up on
/// a lagging RPC node and releasing the plan's claim on the last read.
const SETTLE_ATTEMPTS: usize = 10;
const SETTLE_RETRY_INTERVAL: Duration = Duration::from_millis(200);

/// Re-check a liability mint that has no swap route only after this long (routes can appear).
const NO_ROUTE_QUARANTINE_TTL: Duration = Duration::from_secs(3600);

/// Base delays for the per-target exponential backoff: skip the target for
/// `BASE * 2^(failures-1)`, capped at `TARGET_BACKOFF_MAX`.
///
/// Assemble failures are typically rate limits (a Jupiter `429`), which need a real pause;
/// an execution failure can just be a lost race, so it starts short and escalates.
const ASSEMBLE_BACKOFF_BASE: Duration = Duration::from_secs(30);
const EXECUTION_BACKOFF_BASE: Duration = Duration::from_secs(5);
const TARGET_BACKOFF_MAX: Duration = Duration::from_secs(900);

/// A target rejected twice in a row by `end_liquidation`'s premium cap (6090) is left alone this
/// long. Once is a price move; twice is usually a dust repay whose rounding the cap can never
/// accept, and every retry spends a DEX quote.
const PREMIUM_CAP_BACKOFF: Duration = Duration::from_secs(1200);

/// Whether a failure is `end_liquidation` rejecting the seizure as above the premium cap.
fn is_premium_cap_failure(message: &str) -> bool {
    message.contains("0x17ca") || message.contains("LiquidationPremiumTooHigh")
}

/// Per-target backoff state after a transient (non-route) assemble failure.
struct TargetBackoff {
    /// Consecutive assemble failures (drives the exponential cooldown).
    failures: u32,
    /// The last failure was the premium cap.
    premium_cap: bool,
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

/// Bump `target`'s consecutive-failure count and push out its retry time. A second premium-cap
/// failure in a row backs the target off for [`PREMIUM_CAP_BACKOFF`].
fn bump_target_backoff(
    backoffs: &mut HashMap<Pubkey, BackoffState>,
    target: &Pubkey,
    base: Duration,
    kind: &str,
    premium_cap: bool,
) {
    let (failures, repeated_premium_cap) = match backoffs.get(target) {
        Some(BackoffState::Target(backoff)) => {
            (backoff.failures, premium_cap && backoff.premium_cap)
        }
        _ => (0, false),
    };
    let failures = failures.saturating_add(1);
    let delay = if repeated_premium_cap {
        PREMIUM_CAP_BACKOFF
    } else {
        target_backoff_delay(base, failures)
    };
    backoffs.insert(
        *target,
        BackoffState::Target(TargetBackoff {
            failures,
            premium_cap,
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
    signer: Keypair,
    /// RPC URL that supports `simulateBundle` (used for the simulate-first check).
    rpc_url: String,
    /// Single key for both `sendBundle` (uuid) and `simulateBundle` (Bearer).
    bundle_api_key: Option<String>,
    tip_estimator: TipEstimator,
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
        signer: Keypair,
        rpc_url: String,
        bundle_api_key: Option<String>,
        tip_max_lamports: u64,
    ) -> Self {
        Self {
            jito,
            rpc_client,
            cache,
            signer,
            rpc_url,
            bundle_api_key,
            tip_estimator: TipEstimator::new(tip_max_lamports),
            backoffs: Mutex::new(HashMap::new()),
            sequential_fallbacks: AtomicU64::new(0),
        }
    }

    /// Whether `try_execute` would skip `intent` for now: its liability mint has no swap route, or
    /// the target failed recently.
    pub fn is_on_hold(&self, intent: &PreparedLiquidatableAccount) -> Result<bool> {
        Ok(self.is_backed_off(&self.intent_liab_mint(intent)?)?
            || self.is_backed_off(&intent.liquidatee_account.address)?)
    }

    /// Assemble, gate, and land a single liquidation, sized on the current state by the caller.
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

        // A sandwich funds itself from the seized collateral: no wallet inventory, no bundle, no
        // serialization. Simulated during assembly; the RPC preflight is the second check.
        match strategy.assemble_sandwich(intent) {
            Ok(Sandwich::Plan(ExecutionPlan { txs, .. })) => {
                let result = self.send_over_rpc(&txs, "Sandwich");
                self.note_result(&liquidatee, &result);
                self.refresh_wallet_balances();
                return result;
            }
            Ok(Sandwich::Unavailable) => {}
            Err(e) => {
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                return Err(e);
            }
        };

        debug!(
            "{}: no sandwich for {}; funding it from the wallet",
            strategy.name(),
            liquidatee
        );
        self.execute_with_inventory(strategy, intent, liab_mint)
    }

    /// Brings the cached wallet balances up to the moment of the call, after a liquidation or sell
    /// that may have changed them; the Geyser updates reporting that are only applied when the
    /// liquidator loop drains them.
    ///
    /// The balances are re-read until the read is at or past the slot the RPC was at when this was
    /// called, which is at or past the slot of any transaction the caller saw confirm. The token
    /// cache never lets an older read replace a newer one, so once this returns, the tokens the
    /// caller spent can no longer look free, and its claim on them can be released.
    pub fn refresh_wallet_balances(&self) {
        let settled_at = match self
            .rpc_client
            .get_slot_with_commitment(CommitmentConfig::processed())
        {
            Ok(slot) => slot,
            Err(e) => {
                warn!("Failed to read the slot before refreshing the wallet balances: {e}");
                0
            }
        };
        for _ in 0..SETTLE_ATTEMPTS {
            match self.cache.refresh_wallet_tokens(&self.rpc_client) {
                Ok(read_at) if read_at >= settled_at => return,
                Ok(read_at) => debug!(
                    "Wallet balances read at slot {read_at}, short of {settled_at}; reading again"
                ),
                Err(e) => warn!("Failed to refresh the wallet balances: {e}"),
            }
            thread::sleep(SETTLE_RETRY_INTERVAL);
        }
        warn!(
            "The wallet balances could not be read at or past slot {settled_at}; releasing the claim on the last read"
        );
    }

    /// Assemble and land a plan that spends wallet inventory, then re-read the wallet balances
    /// before releasing the tokens it claimed, so the spent tokens never look free.
    fn execute_with_inventory(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let plan = match strategy.assemble_with_inventory(intent) {
            Ok(plan) => plan,
            Err(e) => {
                // Record a quarantine/backoff so we stop hammering this target, then propagate so
                // the caller logs + counts the failure once (subsequent cycles skip silently).
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                return Err(e);
            }
        };

        let ExecutionPlan { txs, claim } = plan;
        let result = self.land_inventory_plan(&liquidatee, &txs);
        self.refresh_wallet_balances();
        drop(claim);
        result
    }

    /// One transaction goes directly over RPC, whose preflight is the simulation; a bundle would
    /// add nothing but the block engine's latency. A buy plus liquidation is simulated first, then
    /// sent as an atomic bundle.
    fn land_inventory_plan(&self, liquidatee: &Pubkey, txs: &[VersionedTransaction]) -> Result<()> {
        if txs.len() == 1 {
            let result = self.send_over_rpc(txs, "Inventory");
            self.note_result(liquidatee, &result);
            return result;
        }

        // Simulate-first. The outcome dispatches three ways:
        //  - ran & ok                 -> bundle send
        //  - ran & program error      -> skip (doomed on-chain; don't burn tip+fees)
        //  - couldn't run (infra err) -> sequential RPC send (no bundle)
        let mut failure: Option<String> = None;
        let result = match self.jito.simulate_bundle(
            &self.rpc_url,
            self.bundle_api_key.as_deref(),
            txs,
            &[],
        ) {
            Ok(sim) if sim.succeeded => self.submit(txs),
            Ok(sim) => {
                let message = sim.error_message.unwrap_or_default();
                warn!(
                    "Skipping {}: simulation reports the liquidation would fail (tx index {:?}): {}",
                    liquidatee, sim.failed_tx_index, message
                );
                failure = Some(message);
                Ok(())
            }
            Err(e) => {
                warn!(
                    "simulateBundle unavailable for {} ({}); sending sequentially",
                    liquidatee, e
                );
                self.submit_sequential(txs)
            }
        };

        match (&failure, &result) {
            (Some(message), _) => self.note_execution_failure(liquidatee, message),
            (None, result) => self.note_result(liquidatee, result),
        }
        result
    }

    /// Back off a target whose execution failed just like a failed assembly: every retry
    /// re-quotes the DEX without landing anything.
    fn note_result(&self, liquidatee: &Pubkey, result: &Result<()>) {
        match result {
            Ok(()) => self.clear_target_backoff(liquidatee),
            Err(e) => self.note_execution_failure(liquidatee, &e.to_string()),
        }
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
        Ok(guard
            .get(key)
            .is_some_and(|state| now < state.retry_after()))
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

        bump_target_backoff(&mut guard, target, ASSEMBLE_BACKOFF_BASE, "assemble", false);
        Ok(())
    }

    /// Record an execution failure: the plan assembled, but the liquidation could not be landed
    /// (simulation says it would revert, or the send failed). Same exponential backoff as an
    /// assemble failure, and a long one for a repeated premium-cap rejection.
    fn note_execution_failure(&self, target: &Pubkey, message: &str) {
        if let Ok(mut guard) = self.backoffs.lock() {
            bump_target_backoff(
                &mut guard,
                target,
                EXECUTION_BACKOFF_BASE,
                "execution",
                is_premium_cap_failure(message),
            );
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
        // The liquidation tx is the last core tx (any buy precedes it); its signature is our
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
        self.send_over_rpc(txs, "Sequential")
    }

    /// Send each transaction over RPC in order, with preflight, confirming before moving on.
    fn send_over_rpc(&self, txs: &[VersionedTransaction], label: &str) -> Result<()> {
        for (i, tx) in txs.iter().enumerate() {
            match self.send_with_resend(tx) {
                Ok(sig) => info!("{} tx {}/{} confirmed: {}", label, i + 1, txs.len(), sig),
                Err(e) => {
                    return Err(anyhow!(
                        "{} send failed at tx {}/{}: {}",
                        label,
                        i + 1,
                        txs.len(),
                        e
                    ));
                }
            }
        }
        Ok(())
    }

    /// Sends `tx` with preflight, then re-sends it every [`RESEND_INTERVAL`] (without preflight)
    /// while polling its status, until it is confirmed, reverts, or its blockhash expires.
    fn send_with_resend(&self, tx: &VersionedTransaction) -> Result<Signature> {
        let blockhash = *tx.message.recent_blockhash();
        let sig = self.rpc_client.send_transaction_with_config(
            tx,
            RpcSendTransactionConfig {
                skip_preflight: false,
                preflight_commitment: Some(CommitmentLevel::Processed),
                ..Default::default()
            },
        )?;

        let started = Instant::now();
        let mut last_sent = Instant::now();
        let mut resends = 0u32;
        loop {
            thread::sleep(STATUS_POLL_INTERVAL);

            match self.rpc_client.get_signature_statuses(&[sig]) {
                Ok(resp) => {
                    if let Some(Some(status)) = resp.value.into_iter().next() {
                        if let Some(err) = status.err {
                            return Err(anyhow!("{sig} reverted on-chain: {err:?}"));
                        }
                        if status.satisfies_commitment(CommitmentConfig::confirmed()) {
                            if resends > 0 {
                                debug!("{sig} confirmed after {resends} re-send(s)");
                            }
                            return Ok(sig);
                        }
                    }
                }
                Err(e) => debug!("status poll for {sig} failed: {e}"),
            }

            let blockhash_valid = self
                .rpc_client
                .is_blockhash_valid(&blockhash, CommitmentConfig::processed())
                .unwrap_or(true);
            if !blockhash_valid || started.elapsed() > SEND_MAX_WAIT {
                return Err(anyhow!(
                    "{sig} not confirmed after {}s and {resends} re-send(s) (blockhash valid: {blockhash_valid})",
                    started.elapsed().as_secs()
                ));
            }

            if last_sent.elapsed() >= RESEND_INTERVAL {
                // Already-processed and similar rejections mean it landed or is landing; the
                // status poll above settles it either way.
                if let Err(e) = self.rpc_client.send_transaction_with_config(
                    tx,
                    RpcSendTransactionConfig {
                        skip_preflight: true,
                        ..Default::default()
                    },
                ) {
                    debug!("re-send of {sig} rejected: {e}");
                }
                last_sent = Instant::now();
                resends += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retry_in(backoffs: &HashMap<Pubkey, BackoffState>, target: &Pubkey) -> Duration {
        backoffs[target]
            .retry_after()
            .saturating_duration_since(Instant::now())
    }

    #[test]
    fn a_second_premium_cap_failure_in_a_row_backs_off_long() {
        let mut backoffs = HashMap::new();
        let target = Pubkey::new_unique();
        bump_target_backoff(
            &mut backoffs,
            &target,
            EXECUTION_BACKOFF_BASE,
            "execution",
            true,
        );
        assert!(retry_in(&backoffs, &target) <= EXECUTION_BACKOFF_BASE);
        bump_target_backoff(
            &mut backoffs,
            &target,
            EXECUTION_BACKOFF_BASE,
            "execution",
            true,
        );
        assert!(retry_in(&backoffs, &target) > TARGET_BACKOFF_MAX);
    }

    #[test]
    fn other_failures_keep_the_exponential_backoff() {
        let mut backoffs = HashMap::new();
        let target = Pubkey::new_unique();
        bump_target_backoff(
            &mut backoffs,
            &target,
            EXECUTION_BACKOFF_BASE,
            "execution",
            true,
        );
        bump_target_backoff(
            &mut backoffs,
            &target,
            EXECUTION_BACKOFF_BASE,
            "execution",
            false,
        );
        assert!(retry_in(&backoffs, &target) <= EXECUTION_BACKOFF_BASE * 2);
    }

    #[test]
    fn recognizes_the_premium_cap_error() {
        assert!(is_premium_cap_failure(
            "Error processing Instruction 4: custom program error: 0x17ca"
        ));
        assert!(is_premium_cap_failure(
            "Error Code: LiquidationPremiumTooHigh. Error Number: 6090."
        ));
        assert!(!is_premium_cap_failure("custom program error: 0x17b4"));
    }
}
