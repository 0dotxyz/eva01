//! Submission + orchestration engine for the execution layer.
//!
//! `try_execute()` turns a prepared liquidation into a landed liquidation, always one liquidation
//! transaction, fastest first:
//! 1. repaid from the wallet's holdings, when that earns enough of the profit: no swap quote. It
//!    claims the tokens it repays with until it has landed or failed and the wallet balances have
//!    been re-read;
//! 2. otherwise the sandwich (`[liquidate]` with the seized collateral sold in-tx), which touches
//!    no wallet inventory.
//!
//! It goes as a Jito bundle, `[push..] [liquidate] [tip]`, unsimulated, whenever the block engine
//! has a request slot free right now, and as a single simulated transaction over RPC otherwise.
//! The push, when the chain's Pyth prices aren't enough (decided at evaluation), is never sent
//! outside its bundle: on its own it would hand the prices to whoever liquidates next. A bundle
//! that doesn't land is retried on a later cycle.
//!
//! Nothing waits for another plan: any number of liquidations run at once, bounded by the workers.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::{debug, info, warn};
use solana_client::{rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig};
use solana_commitment_config::{CommitmentConfig, CommitmentLevel};
use solana_message::Hash;
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;

use crate::cache::{pyth_prices::price_update, Cache};
use crate::utils::{
    jito::{BundleSend, InflightStatus, JitoClient, JitoSlot, TipEstimator},
    pyth_cranker::price_feed_address,
    pyth_push::{PricePush, PythPusher},
};
use crate::wrappers::liquidator_account::PreparedLiquidatableAccount;
use solana_keypair::Keypair;

use super::{ExecutionPlan, Landing, LiquidationStrategy, Sandwich};

/// How often a watched bundle's status is asked of the block engine.
const INFLIGHT_CHECK_INTERVAL: Duration = Duration::from_secs(2);
/// Polls of the liquidation's signature, after an unconfirmed bundle, this far apart.
const SIGNATURE_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Whether the RPC simulates a transaction before forwarding it. Every transaction is checked
/// exactly once: one eva already simulated skips it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Preflight {
    Run,
    Skip,
}

/// A price push that couldn't be built, so the liquidation behind it never went out.
#[derive(Debug)]
struct PushFailure(String);

impl std::fmt::Display for PushFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PushFailure {}

/// What landing a liquidation as a Jito bundle needs, behind a price push or not.
pub struct BundleLanding {
    pub pusher: Arc<PythPusher>,
    pub jito: JitoClient,
    pub tip_estimator: TipEstimator,
    /// Pays the tip.
    pub signer: Keypair,
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

/// The feed of `oracle` when eva prices it ahead of the chain and can push it: only sponsored
/// feeds, since nobody else can push to the others and eva never prices them from Hermes.
pub fn pushable_feed(cache: &Cache, oracle: &Pubkey) -> Option<[u8; 32]> {
    let update = price_update(&cache.oracles.try_get_account(oracle).ok()?)?;
    let feed_id = update.price_message.feed_id;
    let behind = cache
        .pyth_prices
        .onchain(oracle)
        .is_some_and(|onchain| update.price_message.publish_time > onchain);
    (behind && sponsored_feed_address(&feed_id) == *oracle).then_some(feed_id)
}

/// `price_feed_address`, derived once per feed: evaluation asks for it on every account near the
/// threshold.
fn sponsored_feed_address(feed_id: &[u8; 32]) -> Pubkey {
    static ADDRESSES: std::sync::OnceLock<std::sync::RwLock<HashMap<[u8; 32], Pubkey>>> =
        std::sync::OnceLock::new();
    let addresses = ADDRESSES.get_or_init(Default::default);
    if let Some(address) = addresses
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(feed_id)
    {
        return *address;
    }
    let address = price_feed_address(feed_id);
    addresses
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(*feed_id, address);
    address
}

/// The feeds of `oracles` still ahead of the chain, which a liquidation sized on their prices
/// must push first.
pub fn pending_feeds(cache: &Cache, oracles: &[Pubkey]) -> Vec<[u8; 32]> {
    let mut feeds = Vec::new();
    for feed_id in oracles
        .iter()
        .filter_map(|oracle| pushable_feed(cache, oracle))
    {
        if !feeds.contains(&feed_id) {
            feeds.push(feed_id);
        }
    }
    feeds
}

/// Whether `profit` is at least `share` of `full`.
fn earns_enough(profit: I80F48, full: I80F48, share: I80F48) -> bool {
    profit >= full * share
}

pub struct Executor {
    rpc_client: RpcClient,
    cache: Arc<Cache>,
    /// Liability-mint no-route quarantines and per-liquidatee transient backoffs.
    backoffs: Mutex<HashMap<Pubkey, BackoffState>>,
    /// Lands liquidations as Jito bundles, behind a price push when they need one; `None` when
    /// bundles (and so price pushes) are off.
    bundles: Option<BundleLanding>,
    /// The share of the full liquidation's profit a wallet-funded plan must earn to go before
    /// the plans that need a swap quote.
    wallet_min_profit_share: I80F48,
}

impl Executor {
    pub fn new(
        rpc_client: RpcClient,
        cache: Arc<Cache>,
        bundles: Option<BundleLanding>,
        wallet_min_profit_share: I80F48,
    ) -> Self {
        Self {
            rpc_client,
            cache,
            backoffs: Mutex::new(HashMap::new()),
            bundles,
            wallet_min_profit_share,
        }
    }

    fn pending_pushes(&self, intent: &PreparedLiquidatableAccount) -> Vec<[u8; 32]> {
        pending_feeds(&self.cache, &intent.push_oracles)
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

        // The evaluation sized the liquidation on the prices the chain holds after pushing
        // `push_oracles`; those still ahead of the chain go out first.
        let pending = if self.bundles.is_some() {
            self.pending_pushes(intent)
        } else {
            Vec::new()
        };
        let must_push = !pending.is_empty();

        if let Some(bundles) = &self.bundles {
            // A push never goes out alone, so a liquidation behind one waits for the block
            // engine. Any other takes a bundle only when a request slot is free right now, and
            // otherwise goes over RPC at once instead of queueing behind the other workers.
            if must_push {
                // A push that can't be built gives way to the smaller liquidation the chain's
                // prices already allow, when there is one.
                let unpushed = intent.without_push.as_deref();
                if let Some(unpushed) = unpushed {
                    if bundles.pusher.check(&pending).is_err() {
                        return self.execute_without_push(strategy, bundles, unpushed);
                    }
                }
                let slot = match bundles.jito.try_reserve() {
                    Some(slot) => slot,
                    None => {
                        if let Some(landed) = self.wallet_plan_while_waiting(strategy, intent) {
                            return landed;
                        }
                        bundles.jito.reserve()
                    }
                };
                let result = self.execute_as_bundle(strategy, intent, liab_mint, &pending, slot);
                // The stream may have moved on since the check.
                if let (Err(e), Some(unpushed)) = (&result, unpushed) {
                    if e.downcast_ref::<PushFailure>().is_some() {
                        self.clear_target_backoff(&liquidatee);
                        return self.execute_without_push(strategy, bundles, unpushed);
                    }
                }
                return result;
            }
            if let Some(slot) = bundles.jito.try_reserve() {
                return self.execute_as_bundle(strategy, intent, liab_mint, &[], slot);
            }
            debug!("No bundle slot free for {liquidatee}; liquidating over RPC");
        }
        self.execute_over_rpc(strategy, intent, liab_mint, &pending)
    }

    /// Whether a wallet-funded plan earns enough of the full liquidation's profit to go before
    /// any plan that needs a swap quote.
    fn wallet_plan_suffices(
        &self,
        plan: &ExecutionPlan,
        intent: &PreparedLiquidatableAccount,
    ) -> bool {
        earns_enough(plan.profit, intent.profit, self.wallet_min_profit_share)
    }

    /// The wallet-funded plan, when it earns enough to go before the sandwich.
    fn sufficient_wallet_plan(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> std::result::Result<ExecutionPlan, String> {
        let liquidatee = intent.liquidatee_account.address;
        let short = match strategy.assemble_with_inventory(intent, landing) {
            Ok(plan) if self.wallet_plan_suffices(&plan, intent) => return Ok(plan),
            // Dropped, releasing its claim while the sandwich is quoted.
            Ok(plan) => format!(
                "the wallet funds only ${:.4} of the ${:.4} profit",
                plan.profit.to_num::<f64>(),
                intent.profit.to_num::<f64>()
            ),
            Err(e) => e.to_string(),
        };
        debug!("No wallet-funded plan for {liquidatee}: {short}");
        Err(short)
    }

    /// While a liquidation behind a price push waits for a bundle slot: the wallet-funded plan over
    /// RPC, if the chain's prices already let it pass (another push may have landed). Its
    /// simulation fits in the wait; a sandwich isn't tried, as its quote would be spent twice.
    fn wallet_plan_while_waiting(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
    ) -> Option<Result<()>> {
        let plan = self
            .sufficient_wallet_plan(strategy, intent, Landing::Rpc)
            .ok()?;
        if self.simulate(&plan.tx).is_err() {
            return None;
        }
        let liquidatee = intent.liquidatee_account.address;
        info!("{liquidatee} passes on the chain's prices already; liquidating it over RPC");
        Some(self.land_wallet_plan(&liquidatee, plan, Preflight::Skip))
    }

    /// Single transactions over RPC, each simulated: the wallet-funded plan when it earns enough,
    /// then the sandwich. With a feed behind, a plan the chain's prices reject falls back to a
    /// price push in a bundle.
    fn execute_over_rpc(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
        pending: &[[u8; 32]],
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let wallet_short = match self.sufficient_wallet_plan(strategy, intent, Landing::Rpc) {
            Ok(plan) => {
                return self.land_plain_wallet_plan(strategy, intent, liab_mint, plan, pending)
            }
            Err(short) => short,
        };

        // A sandwich funds itself from the seized collateral and touches no wallet inventory.
        match strategy.assemble_sandwich(&intent.for_sandwich(), Landing::Rpc) {
            Ok(Sandwich::Plan(ExecutionPlan { tx, .. })) => {
                // Simulated while it was built: no second check on the way out.
                // It claims no wallet tokens, so nothing waits for the balances to be re-read.
                let result = self.send_over_rpc(&tx, "Sandwich", Preflight::Skip);
                self.note_result(&liquidatee, &result);
                result
            }
            // It failed only on the chain's prices: push them.
            Ok(Sandwich::Rejected) if !pending.is_empty() => {
                self.execute_behind_push(strategy, intent, liab_mint, pending)
            }
            Ok(_) => self.no_plan(&liquidatee, liab_mint, &wallet_short),
            Err(e) => {
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                Err(e)
            }
        }
    }

    /// `unpushed`, a liquidation the chain's prices allow: in a bundle when a request slot is free,
    /// otherwise over RPC. Skipped while its liability mint is quarantined.
    fn execute_without_push(
        &self,
        strategy: &dyn LiquidationStrategy,
        bundles: &BundleLanding,
        unpushed: &PreparedLiquidatableAccount,
    ) -> Result<()> {
        let liquidatee = unpushed.liquidatee_account.address;
        let liab_mint = self.intent_liab_mint(unpushed)?;
        if self.is_backed_off(&liab_mint)? {
            debug!("Skipping {liquidatee} without a push: liab mint {liab_mint} is quarantined");
            return Ok(());
        }
        info!("Liquidating {liquidatee} without a push, on the chain's prices: the push can't be built");
        match bundles.jito.try_reserve() {
            Some(slot) => self.execute_as_bundle(strategy, unpushed, liab_mint, &[], slot),
            None => self.execute_over_rpc(strategy, unpushed, liab_mint, &[]),
        }
    }

    /// Neither plan: the wallet funds too little and no sandwich fits. Backed off like a failed
    /// assembly.
    fn no_plan(&self, liquidatee: &Pubkey, liab_mint: Pubkey, wallet_short: &str) -> Result<()> {
        let e = anyhow!("no plan for {liquidatee}: {wallet_short}, and no sandwich fits");
        self.note_assemble_failure(liquidatee, liab_mint, &e)?;
        Err(e)
    }

    /// Lands a wallet-funded plan over RPC. With a feed behind, it is simulated here to choose
    /// between it and a price push, and that simulation is its only check; otherwise the send's
    /// preflight is.
    fn land_plain_wallet_plan(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
        plan: ExecutionPlan,
        pending: &[[u8; 32]],
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        if pending.is_empty() {
            return self.land_wallet_plan(&liquidatee, plan, Preflight::Run);
        }
        if let Err(e) = self.simulate(&plan.tx) {
            debug!(
                "The wallet-funded liquidation of {} fails on the chain's prices ({e}); pushing them",
                liquidatee
            );
            // Releases its claim; the pushed plan claims afresh.
            drop(plan);
            return self.execute_behind_push(strategy, intent, liab_mint, pending);
        }
        self.land_wallet_plan(&liquidatee, plan, Preflight::Skip)
    }

    /// A bundle behind a push of `feeds`, once the block engine admits it.
    fn execute_behind_push(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
        feeds: &[[u8; 32]],
    ) -> Result<()> {
        let bundles = self
            .bundles
            .as_ref()
            .ok_or_else(|| anyhow!("bundles are off"))?;
        let slot = bundles.jito.reserve();
        self.execute_as_bundle(strategy, intent, liab_mint, feeds, slot)
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

    /// Lands a plan that spends wallet inventory, then re-reads the wallet balances before
    /// releasing the tokens it claimed, so the spent tokens never look free.
    fn land_wallet_plan(
        &self,
        liquidatee: &Pubkey,
        plan: ExecutionPlan,
        preflight: Preflight,
    ) -> Result<()> {
        let ExecutionPlan { tx, claim, .. } = plan;
        let result = self.send_over_rpc(&tx, "Wallet-funded", preflight);
        self.note_result(liquidatee, &result);
        self.refresh_wallet_balances();
        drop(claim);
        result
    }

    /// Simulates `tx` against the chain's current state; a failure carries the program's last logs.
    fn simulate(&self, tx: &VersionedTransaction) -> Result<()> {
        let simulation = self.rpc_client.simulate_transaction(tx)?.value;
        match simulation.err {
            None => Ok(()),
            Some(err) => {
                let logs = simulation.logs.unwrap_or_default();
                let tail = &logs[logs.len().saturating_sub(3)..];
                Err(anyhow!("simulation failed: {err:?}; logs: {tail:?}"))
            }
        }
    }

    /// Why a bundle that didn't land didn't, as far as simulating its liquidation alone now tells
    /// (after the fact, off the landing path).
    fn why_not_landed(&self, bundle: &[VersionedTransaction], behind_push: bool) -> String {
        if behind_push {
            return "lost or dropped (behind a push, so not simulated alone)".to_string();
        }
        match bundle.last().map(|tx| self.simulate(tx)) {
            Some(Ok(())) => "its liquidation passes simulation now, so it lost or expired".into(),
            Some(Err(e)) => format!("its liquidation fails: {e}"),
            None => "empty bundle".into(),
        }
    }

    /// Lands the liquidation as an unsimulated Jito bundle in `slot`, behind a push of `feeds` when
    /// there are any: the wallet-funded plan when it earns enough, otherwise the sandwich.
    fn execute_as_bundle(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
        feeds: &[[u8; 32]],
        slot: JitoSlot<'_>,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let bundles = self
            .bundles
            .as_ref()
            .ok_or_else(|| anyhow!("bundles are off"))?;
        // The push first: without it there is nothing to bundle, and no quote is spent.
        let push = if feeds.is_empty() {
            None
        } else {
            match bundles.pusher.push_transactions(feeds) {
                Ok(push) => Some(push),
                Err(e) => {
                    let e = PushFailure(format!(
                        "can't push the {} Pyth feed(s) it needs: {e}",
                        feeds.len()
                    ));
                    self.note_execution_failure(&liquidatee, &e.0);
                    return Err(e.into());
                }
            }
        };

        let wallet_short = match self.sufficient_wallet_plan(strategy, intent, Landing::Bundle) {
            Ok(plan) => {
                return self.land_bundle(
                    strategy,
                    bundles,
                    intent,
                    push.as_ref(),
                    plan,
                    "wallet-funded",
                    slot,
                )
            }
            Err(short) => short,
        };
        match strategy.assemble_sandwich(&intent.for_sandwich(), Landing::Bundle) {
            Ok(Sandwich::Plan(plan)) => self.land_bundle(
                strategy,
                bundles,
                intent,
                push.as_ref(),
                plan,
                "sandwich",
                slot,
            ),
            Ok(_) => self.no_plan(&liquidatee, liab_mint, &wallet_short),
            Err(e) => {
                self.note_assemble_failure(&liquidatee, liab_mint, &e)?;
                Err(e)
            }
        }
    }

    /// Sends `[push..] [liquidation] [tip]`, the tip a share of the plan's profit, and settles it.
    fn land_bundle(
        &self,
        strategy: &dyn LiquidationStrategy,
        bundles: &BundleLanding,
        intent: &PreparedLiquidatableAccount,
        push: Option<&PricePush>,
        plan: ExecutionPlan,
        kind: &str,
        slot: JitoSlot<'_>,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let ExecutionPlan { tx, claim, profit } = plan;
        let profit_lamports = strategy.lamports_for_value(profit).unwrap_or_else(|e| {
            warn!("Can't price the profit of {liquidatee} in SOL ({e}); tipping the floor");
            0
        });
        let tip = bundles.tip_estimator.tip_for(profit_lamports);
        let mut bundle = push.map(|push| push.txs.clone()).unwrap_or_default();
        bundle.push(tx);
        info!(
            "Liquidating {} in a bundle: {} plan for ${:.4}, tip {} lamports{}",
            liquidatee,
            kind,
            profit.to_num::<f64>(),
            tip,
            push.map_or(String::new(), |push| format!(
                ", behind a {}-tx price push",
                push.txs.len()
            ))
        );

        let result = match self.send_bundle(bundles, &bundle, tip, slot) {
            Ok(true) => Ok(()),
            Ok(false) => Err(anyhow!(
                "the bundle for {liquidatee} didn't land: {}",
                self.why_not_landed(&bundle, push.is_some())
            )),
            Err(e) => Err(e),
        };
        self.note_result(&liquidatee, &result);
        // A sandwich claims no wallet tokens, so nothing waits for the balances to be re-read.
        if claim.is_some() {
            self.refresh_wallet_balances();
        }
        drop(claim);
        result
    }

    /// Sends `bundle` with a tip transaction appended. Returns whether its liquidation, the last
    /// transaction, landed.
    fn send_bundle(
        &self,
        bundles: &BundleLanding,
        bundle: &[VersionedTransaction],
        tip: u64,
        slot: JitoSlot<'_>,
    ) -> Result<bool> {
        let liquidation = bundle.last().ok_or_else(|| anyhow!("empty bundle"))?;
        let liquidation_sig = *liquidation
            .signatures
            .first()
            .ok_or_else(|| anyhow!("unsigned liquidation"))?;
        // The tip goes out with the liquidation's blockhash, so it expires with it.
        let blockhash = *liquidation.message.recent_blockhash();
        let mut txs = bundle.to_vec();
        txs.push(
            bundles
                .tip_estimator
                .build_tip_transaction(&bundles.signer, blockhash, tip)?,
        );
        match slot.send_bundle(&txs) {
            Ok(BundleSend::Accepted(id)) => {
                debug!("Bundle {id} sent; watching its liquidation {liquidation_sig}");
                self.landed_before_expiry(
                    &liquidation_sig,
                    &blockhash,
                    Some((&bundles.jito, id.as_str())),
                )
            }
            Ok(BundleSend::Rejected(reason)) => {
                warn!("The block engine rejected the bundle: {reason}");
                Ok(false)
            }
            // The request may still have reached the block engine, which gave no id to ask about:
            // only the blockhash's expiry settles it.
            Err(e) => {
                warn!(
                    "Sending the bundle failed ({e}); watching its liquidation {liquidation_sig}"
                );
                self.landed_before_expiry(&liquidation_sig, &blockhash, None)
            }
        }
    }

    /// Whether `sig` lands while its `blockhash` is valid. Until settled, the caller keeps its claim
    /// and reports no failure, since the bundle may still land. A bundle the block engine reports
    /// failed or unknown settles it early, as most lost auctions do.
    fn landed_before_expiry(
        &self,
        sig: &Signature,
        blockhash: &Hash,
        bundle: Option<(&JitoClient, &str)>,
    ) -> Result<bool> {
        let started = Instant::now();
        let mut last_inflight_check = Instant::now();
        loop {
            match self.landed_status(sig) {
                Some(Ok(())) => {
                    info!("Bundle landed late, confirmed via {sig}");
                    return Ok(true);
                }
                Some(Err(reverted)) => return Err(reverted),
                None => {}
            }
            let dead = match bundle {
                Some((jito, id)) if last_inflight_check.elapsed() >= INFLIGHT_CHECK_INTERVAL => {
                    last_inflight_check = Instant::now();
                    // Only in a slot free right now: sends need them more.
                    jito.try_reserve().is_some_and(|slot| {
                        slot.inflight_bundle_status(id)
                            .ok()
                            .flatten()
                            .is_some_and(InflightStatus::is_dead)
                    })
                }
                _ => false,
            };
            let valid = self
                .rpc_client
                .is_blockhash_valid(blockhash, CommitmentConfig::processed())
                .unwrap_or(true);
            if dead || !valid || started.elapsed() > SEND_MAX_WAIT {
                // It may have been processed in the blockhash's last slots.
                return match self.landed_status(sig) {
                    Some(Ok(())) => Ok(true),
                    Some(Err(reverted)) => Err(reverted),
                    None => {
                        if dead {
                            warn!("The block engine gave up on the bundle with {sig}");
                        } else {
                            warn!("The bundle with {sig} expired without landing");
                        }
                        Ok(false)
                    }
                };
            }
            thread::sleep(SIGNATURE_POLL_INTERVAL);
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

    /// Sends the liquidation over RPC and waits for it to confirm or fail.
    fn send_over_rpc(
        &self,
        tx: &VersionedTransaction,
        label: &str,
        preflight: Preflight,
    ) -> Result<()> {
        let sig = self
            .send_with_resend(tx, preflight)
            .map_err(|e| anyhow!("{label} send failed: {e}"))?;
        info!("{label} tx confirmed: {sig}");
        Ok(())
    }

    /// Whether `sig` is on chain: `Some(Ok)` once confirmed, `Some(Err)` if it reverted, `None`
    /// if it isn't known or the status can't be read.
    fn landed_status(&self, sig: &Signature) -> Option<Result<()>> {
        let status = self
            .rpc_client
            .get_signature_statuses(&[*sig])
            .ok()?
            .value
            .into_iter()
            .next()??;
        if let Some(err) = status.err {
            return Some(Err(anyhow!("{sig} reverted on-chain: {err:?}")));
        }
        status
            .satisfies_commitment(CommitmentConfig::confirmed())
            .then_some(Ok(()))
    }

    /// Sends `tx`, with the RPC's preflight simulation unless eva already simulated it, then
    /// re-sends it every [`RESEND_INTERVAL`] (without preflight) while polling its status, until it
    /// is confirmed, reverts, or its blockhash expires. Only this one transaction is waited on:
    /// every liquidation runs on its own worker.
    fn send_with_resend(
        &self,
        tx: &VersionedTransaction,
        preflight: Preflight,
    ) -> Result<Signature> {
        let blockhash = *tx.message.recent_blockhash();
        let sig = match self.rpc_client.send_transaction_with_config(
            tx,
            RpcSendTransactionConfig {
                skip_preflight: preflight == Preflight::Skip,
                preflight_commitment: Some(CommitmentLevel::Processed),
                ..Default::default()
            },
        ) {
            Ok(sig) => sig,
            Err(e) => {
                // A transaction that already landed, from a bundle that landed late for instance,
                // is rejected by preflight; the chain says whether that is a success.
                let sig = tx
                    .signatures
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("unsigned transaction"))?;
                return match self.landed_status(&sig) {
                    Some(Ok(())) => Ok(sig),
                    Some(Err(reverted)) => Err(reverted),
                    None => Err(e.into()),
                };
            }
        };

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
                // Processed in the blockhash's last slots, it confirms after the hash expired.
                if let Some(landed) = self.landed_status(&sig) {
                    return landed.map(|()| sig);
                }
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
    use crate::cache::pyth_prices::test_price_update_account;
    use solana_clock::Clock;
    use std::collections::HashSet;

    #[test]
    fn a_wallet_plan_goes_first_from_its_share_of_the_profit() {
        let share = I80F48::from_num(0.3);
        let full = I80F48::from_num(10);
        assert!(earns_enough(full, full, share));
        assert!(earns_enough(I80F48::from_num(3.01), full, share));
        assert!(!earns_enough(I80F48::from_num(2.99), full, share));
    }

    #[test]
    fn only_sponsored_feeds_ahead_of_the_chain_are_pushed() {
        let mut cache = Cache::new(
            Pubkey::new_unique(),
            Pubkey::new_unique(),
            Arc::new(Mutex::new(Clock::default())),
            HashSet::new(),
        );
        let (ahead, synced, unknown) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let mut add = |feed: [u8; 32], address: Pubkey, cached: i64, onchain: Option<i64>| {
            cache
                .oracles
                .try_insert(address, test_price_update_account(feed, cached, 1))
                .unwrap();
            if let Some(onchain) = onchain {
                cache
                    .pyth_prices
                    .record_onchain(address, &test_price_update_account(feed, onchain, 1));
            }
            address
        };
        let ahead_feed = add(ahead, price_feed_address(&ahead), 110, Some(100));
        let synced_feed = add(synced, price_feed_address(&synced), 100, Some(100));
        // Not the sponsored PDA: nobody can push it, and Hermes never overwrites it.
        let unsponsored = add(ahead, Pubkey::new_unique(), 110, Some(100));
        // The chain's publish time was never seen: nothing to compare against.
        let never_seen = add(unknown, price_feed_address(&unknown), 110, None);

        let oracles = [ahead_feed, synced_feed, unsponsored, never_seen, ahead_feed];
        assert_eq!(pending_feeds(&cache, &oracles), vec![ahead]);
    }

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
