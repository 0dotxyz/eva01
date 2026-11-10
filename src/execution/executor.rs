//! Submission + orchestration engine for the execution layer.
//!
//! `try_execute()` turns a prepared liquidation into a landed liquidation, always one liquidation
//! transaction:
//! 1. the sandwich (`[liquidate]` with the seized collateral sold in-tx), which touches no wallet
//!    inventory;
//! 2. otherwise the liquidation repaid from the wallet's holdings, which claims the tokens it
//!    repays with until it has landed or failed and the wallet balances have been re-read.
//!
//! It goes alone over RPC, unless it was sized on Pyth prices the chain does not hold yet (from
//! Hermes): then the prices are pushed ahead of it in one Jito bundle, `[push..] [liquidate]
//! [tip]`, simulated first. A bundle that doesn't land falls back to sending the push and then the
//! liquidation over RPC; a push alone is harmless.
//!
//! Nothing waits for another plan: any number of liquidations run at once, bounded by the workers.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use solana_client::{rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig};
use solana_commitment_config::{CommitmentConfig, CommitmentLevel};
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;

use crate::cache::{pyth_prices::price_update, Cache};
use crate::utils::{
    jito::{BundleOutcome, JitoClient, TipEstimator},
    pyth_cranker::price_feed_address,
    pyth_push::{PythPusher, MAX_PUSH_TXS},
};
use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;
use solana_keypair::Keypair;

use super::{ExecutionPlan, Landing, LiquidationStrategy, Sandwich};

/// Bundle status polls before checking the chain for the liquidation itself.
const BUNDLE_CONFIRM_ATTEMPTS: usize = 3;
/// Polls of the liquidation's signature after an unconfirmed bundle, this far apart.
const SIGNATURE_CONFIRM_ATTEMPTS: usize = 6;
const SIGNATURE_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// What landing a liquidation behind a price push needs.
pub struct PricePushLanding {
    pub pusher: PythPusher,
    pub jito: JitoClient,
    pub tip_estimator: TipEstimator,
    /// Pays the tip.
    pub signer: Keypair,
    /// An RPC endpoint that serves `simulateBundle`.
    pub rpc_url: String,
    pub bundle_api_key: Option<String>,
}

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
    rpc_client: RpcClient,
    cache: Arc<Cache>,
    /// Liability-mint no-route quarantines and per-liquidatee transient backoffs.
    backoffs: Mutex<HashMap<Pubkey, BackoffState>>,
    /// Lands liquidations sized on prices the chain does not hold yet; `None` when price pushing
    /// is off.
    price_push: Option<PricePushLanding>,
}

impl Executor {
    pub fn new(
        rpc_client: RpcClient,
        cache: Arc<Cache>,
        price_push: Option<PricePushLanding>,
    ) -> Self {
        Self {
            rpc_client,
            cache,
            backoffs: Mutex::new(HashMap::new()),
            price_push,
        }
    }

    /// The Pyth feeds `intent` was priced on that are newer in the cache than on chain, which a
    /// liquidation sized on them must push first. Every feed the account's health reads counts,
    /// since the program re-checks health on all of them.
    fn pending_pushes(&self, intent: &PreparedLiquidatableAccount) -> Vec<[u8; 32]> {
        let mut feeds = Vec::new();
        for oracle in &intent.observation_accounts.observation_accounts {
            let Some(update) = self
                .cache
                .oracles
                .try_get_account(oracle)
                .ok()
                .as_ref()
                .and_then(price_update)
            else {
                continue;
            };
            let feed_id = update.price_message.feed_id;
            let behind = self
                .cache
                .pyth_prices
                .onchain(oracle)
                .is_some_and(|onchain| update.price_message.publish_time > onchain);
            if behind && price_feed_address(&feed_id) == *oracle && !feeds.contains(&feed_id) {
                feeds.push(feed_id);
            }
        }
        feeds
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

        if self.price_push.is_some() {
            let feeds = self.pending_pushes(intent);
            if !feeds.is_empty() {
                return self.execute_with_price_push(strategy, intent, liab_mint, &feeds);
            }
        }

        // A sandwich funds itself from the seized collateral and touches no wallet inventory.
        // Simulated during assembly; the RPC preflight is the second check.
        match strategy.assemble_sandwich(intent, Landing::Rpc) {
            Ok(Sandwich::Plan(ExecutionPlan { tx, .. })) => {
                let result = self.send_over_rpc(&tx, "Sandwich");
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
        let plan = match strategy.assemble_with_inventory(intent, Landing::Rpc) {
            Ok(plan) => plan,
            Err(e) => {
                // Record a quarantine/backoff so we stop hammering this target, then propagate so
                // the caller logs + counts the failure once (subsequent cycles skip silently).
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                return Err(e);
            }
        };

        let ExecutionPlan { tx, claim } = plan;
        let result = self.send_over_rpc(&tx, "Wallet-funded");
        self.note_result(&liquidatee, &result);
        self.refresh_wallet_balances();
        drop(claim);
        result
    }

    /// Builds the liquidation for a bundle, the sandwich first and the wallet-funded plan
    /// otherwise, and lands it behind a push of `feeds`.
    fn execute_with_price_push(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
        feeds: &[[u8; 32]],
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let plan = match strategy.assemble_sandwich(intent, Landing::Bundle) {
            Ok(Sandwich::Plan(plan)) => Ok(plan),
            Ok(Sandwich::Unavailable) => strategy.assemble_with_inventory(intent, Landing::Bundle),
            Err(e) => Err(e),
        };
        let ExecutionPlan { tx, claim } = match plan {
            Ok(plan) => plan,
            Err(e) => {
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                return Err(e);
            }
        };
        let result = self.land_behind_price_push(&liquidatee, feeds, &tx);
        self.note_result(&liquidatee, &result);
        self.refresh_wallet_balances();
        drop(claim);
        result
    }

    /// Simulates `[push..] [liquidation]`, then sends it as a Jito bundle with a tip. A bundle that
    /// doesn't land is followed by the push and then the liquidation over RPC.
    fn land_behind_price_push(
        &self,
        liquidatee: &Pubkey,
        feeds: &[[u8; 32]],
        liquidation: &VersionedTransaction,
    ) -> Result<()> {
        let landing = self
            .price_push
            .as_ref()
            .ok_or_else(|| anyhow!("price pushing is off"))?;
        let push = landing.pusher.push_transactions(feeds)?;
        let mut bundle = push.txs.clone();
        bundle.push(liquidation.clone());
        debug_assert!(push.txs.len() <= MAX_PUSH_TXS);

        match landing.jito.simulate_bundle(
            &landing.rpc_url,
            landing.bundle_api_key.as_deref(),
            &bundle,
            &[],
        ) {
            Ok(sim) if sim.succeeded => {}
            Ok(sim) => {
                return Err(anyhow!(
                    "the price push and liquidation would fail (tx index {:?}): {}",
                    sim.failed_tx_index,
                    sim.error_message.unwrap_or_default()
                ));
            }
            Err(e) => warn!(
                "simulateBundle unavailable for {} ({}); sending without it",
                liquidatee, e
            ),
        }

        info!(
            "Liquidating {} behind a push of {} Pyth feed(s)",
            liquidatee,
            feeds.len()
        );
        if self.send_bundle(landing, &bundle)? {
            return Ok(());
        }

        // The bundle didn't land: push over RPC, then liquidate on the pushed prices.
        for (i, tx) in push.txs.iter().enumerate() {
            if let Err(e) = self.send_with_resend(tx) {
                warn!(
                    "The price push for {} failed at tx {}: {}",
                    liquidatee,
                    i + 1,
                    e
                );
                if i > 0 {
                    // The first transaction created the VAA account; reclaim its rent.
                    match landing.pusher.reclaim_transaction(&push.encoded_vaa) {
                        Ok(reclaim) => {
                            if let Err(e) = self.send_with_resend(&reclaim) {
                                warn!(
                                    "Failed to reclaim the VAA account {}: {}",
                                    push.encoded_vaa, e
                                );
                            }
                        }
                        Err(e) => warn!("Failed to build the VAA reclaim: {e}"),
                    }
                }
                // Somebody else may have pushed the same prices; the liquidation's preflight
                // decides.
                break;
            }
        }
        self.send_over_rpc(liquidation, "Pushed")
    }

    /// Sends `bundle` with a tip appended. Returns whether its liquidation, the last transaction,
    /// landed.
    fn send_bundle(
        &self,
        landing: &PricePushLanding,
        bundle: &[VersionedTransaction],
    ) -> Result<bool> {
        let liquidation_sig = bundle
            .last()
            .and_then(|tx| tx.signatures.first())
            .copied()
            .ok_or_else(|| anyhow!("empty bundle"))?;
        let mut txs = bundle.to_vec();
        txs.push(
            landing
                .tip_estimator
                .build_tip_transaction(&landing.signer, &self.rpc_client)?,
        );
        match landing
            .jito
            .send_bundle_and_confirm(&txs, BUNDLE_CONFIRM_ATTEMPTS)
        {
            Ok(BundleOutcome::Confirmed(id)) => {
                info!("Bundle landed: {} ({} txs)", id, txs.len());
                Ok(true)
            }
            Ok(BundleOutcome::Unconfirmed(id)) => {
                for _ in 0..SIGNATURE_CONFIRM_ATTEMPTS {
                    if let Ok(resp) = self.rpc_client.get_signature_statuses(&[liquidation_sig]) {
                        if let Some(Some(status)) = resp.value.into_iter().next() {
                            if let Some(err) = status.err {
                                return Err(anyhow!("bundle {id} reverted: {err:?}"));
                            }
                            if status.satisfies_commitment(CommitmentConfig::confirmed()) {
                                info!("Bundle {} landed, confirmed via {}", id, liquidation_sig);
                                return Ok(true);
                            }
                        }
                    }
                    thread::sleep(SIGNATURE_POLL_INTERVAL);
                }
                warn!("Bundle {id} accepted but not on chain; pushing over RPC instead");
                Ok(false)
            }
            Err(e) => {
                warn!("The bundle was not accepted ({e}); pushing over RPC instead");
                Ok(false)
            }
        }
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

    /// Sends the liquidation over RPC, with preflight, and waits for it to confirm or fail.
    fn send_over_rpc(&self, tx: &VersionedTransaction, label: &str) -> Result<()> {
        let sig = self
            .send_with_resend(tx)
            .map_err(|e| anyhow!("{label} send failed: {e}"))?;
        info!("{label} tx confirmed: {sig}");
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
