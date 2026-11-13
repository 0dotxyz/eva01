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
//! With bundles on it always goes as a Jito bundle, `[push..] [refresh] [liquidate] [tip]`,
//! unsimulated, waiting for a request slot when all are held; with bundles off, as a single
//! transaction over RPC.
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
use crate::transaction_limits::MAX_BUNDLE_TXS;
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
    /// Whether the Kamino and JupLend refreshes may go ahead of the liquidation.
    pub refresh_ahead: bool,
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

/// Counts bundles built, to make each one's push, refresh and tip transactions unique.
static NEXT_SEND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How long an account is left alone after an attempt found no plan: no wallet repay and no
/// sandwich (no route, no room, or a swap venue's error). Short and fixed, since a route or the
/// wallet can change at any moment and a retry costs one quote.
const NO_PLAN_HOLD: Duration = Duration::from_secs(20);

/// How long an account is left alone after `end_liquidation` rejected the seizure as above the
/// premium cap (6090): the prices moved or the sizing doesn't match the program's.
const PREMIUM_CAP_HOLD: Duration = Duration::from_secs(60);

/// Bundles in a row for one account that don't land before it is held for [`NO_PLAN_HOLD`].
const MISSES_BEFORE_HOLD: u32 = 3;

/// Misses further apart than this aren't "in a row".
const MISS_WINDOW: Duration = Duration::from_secs(60);

/// How long an account is left alone after its transaction over RPC (bundles off) failed.
const RPC_FAILURE_HOLD: Duration = Duration::from_secs(5);

/// Whether a failure is `end_liquidation` rejecting the seizure as above the premium cap.
fn is_premium_cap_failure(message: &str) -> bool {
    message.contains("0x17ca") || message.contains("LiquidationPremiumTooHigh")
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
    (behind && price_feed_address(&feed_id) == *oracle).then_some(feed_id)
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

/// What became of a bundle sent to the block engine.
enum BundleOutcome {
    Landed,
    /// Accepted for the auction (or possibly so) and didn't land: lost, or failed unseen.
    NotLanded,
    /// Refused or never delivered: it took no part in an auction.
    Refused,
}

/// Whether `profit` is at least `share` of `full`.
fn earns_enough(profit: I80F48, full: I80F48, share: I80F48) -> bool {
    profit >= full * share
}

pub struct Executor {
    rpc_client: RpcClient,
    cache: Arc<Cache>,
    /// Liability-mint no-route quarantines and per-liquidatee transient backoffs.
    /// Accounts left alone until the instant given, after an attempt that found no plan or failed.
    backoffs: Mutex<HashMap<Pubkey, Instant>>,
    /// Bundles in a row that didn't land, per account, and when the first of them was sent.
    misses: Mutex<HashMap<Pubkey, (u32, Instant)>>,
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
            misses: Mutex::new(HashMap::new()),
            bundles,
            wallet_min_profit_share,
        }
    }

    fn pending_pushes(&self, intent: &PreparedLiquidatableAccount) -> Vec<[u8; 32]> {
        pending_feeds(&self.cache, &intent.push_oracles)
    }

    /// Whether `try_execute` would skip `intent` for now: the target failed recently.
    pub fn is_on_hold(&self, intent: &PreparedLiquidatableAccount) -> Result<bool> {
        self.is_backed_off(&intent.liquidatee_account.address)
    }

    /// Assemble, gate, and land a single liquidation, sized on the current state by the caller.
    pub fn try_execute(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
    ) -> Result<()> {
        // A target that failed recently (rate limit, no plan, a lost bundle) is backed off before
        // any quote is spent; the backoff expires so a recovering target retries.
        if self.is_backed_off(&intent.liquidatee_account.address)? {
            info!(
                "LIQUIDATION_ATTEMPT_END attempt_id={} account={} outcome=skipped reason=backed_off",
                intent.attempt_id, intent.liquidatee_account.address
            );
            return Ok(());
        }
        if !intent.push_oracles.is_empty() {
            let oracle_state: Vec<String> = intent
                .push_oracles
                .iter()
                .map(|oracle| {
                    match self
                        .cache
                        .oracles
                        .try_get_account(oracle)
                        .ok()
                        .and_then(|account| price_update(&account))
                    {
                        Some(update) => {
                            let message = update.price_message;
                            format!(
                                "{}:cached_time={},chain_time={:?},price={}e{}",
                                oracle,
                                message.publish_time,
                                self.cache.pyth_prices.onchain(oracle),
                                message.price,
                                message.exponent
                            )
                        }
                        None => format!(
                            "{}:cached=unavailable,chain_time={:?}",
                            oracle,
                            self.cache.pyth_prices.onchain(oracle)
                        ),
                    }
                })
                .collect();
            info!(
                "LIQUIDATION_ORACLE_STATE attempt_id={} account={} feeds={:?}",
                intent.attempt_id, intent.liquidatee_account.address, oracle_state
            );
        }
        let started = Instant::now();
        let result = self.try_execute_inner(strategy, intent);
        match &result {
            Ok(()) => info!(
                "LIQUIDATION_ATTEMPT_END attempt_id={} account={} outcome=success elapsed_ms={}",
                intent.attempt_id,
                intent.liquidatee_account.address,
                started.elapsed().as_millis()
            ),
            Err(_) => warn!(
                "LIQUIDATION_ATTEMPT_END attempt_id={} account={} outcome=failure elapsed_ms={}",
                intent.attempt_id,
                intent.liquidatee_account.address,
                started.elapsed().as_millis()
            ),
        }
        result
    }

    fn try_execute_inner(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
    ) -> Result<()> {
        let liab_mint = self.intent_liab_mint(intent)?;

        // The evaluation sized the liquidation on the prices the chain holds after pushing
        // `push_oracles`; those still ahead of the chain go out first.
        let pending = if self.bundles.is_some() {
            self.pending_pushes(intent)
        } else {
            Vec::new()
        };
        let must_push = !pending.is_empty();

        let Some(bundles) = &self.bundles else {
            return self.execute_over_rpc(strategy, intent, liab_mint);
        };
        // While the block engine is down, what needs no push goes over RPC. A push never goes out
        // without its bundle, so a liquidation that needs one waits.
        if bundles.jito.is_down() {
            if !must_push {
                return self.execute_over_rpc(strategy, intent, liab_mint);
            }
            if let Some(unpushed) = intent.without_push.as_deref() {
                let liab_mint = self.intent_liab_mint(unpushed)?;
                return self.execute_over_rpc(strategy, unpushed, liab_mint);
            }
            return Err(anyhow!(
                "the block engine is down and the liquidation needs a price push"
            ));
        }
        // Every liquidation goes as a bundle. A slot is all but always free at the key's rate,
        // and a wait for one is a few milliseconds.
        if must_push {
            // A push that can't be built gives way to the smaller liquidation the chain's prices
            // already allow, when there is one.
            let unpushed = intent.without_push.as_deref();
            if let Some(unpushed) = unpushed {
                if bundles.pusher.check(&pending).is_err() {
                    return self.execute_without_push(strategy, unpushed);
                }
            }
            let result = self.execute_as_bundle(strategy, intent, liab_mint, &pending);
            // The stream may have moved on since the check.
            if let (Err(e), Some(unpushed)) = (&result, unpushed) {
                if e.downcast_ref::<PushFailure>().is_some() {
                    return self.execute_without_push(strategy, unpushed);
                }
            }
            return result;
        }
        self.execute_as_bundle(strategy, intent, liab_mint, &[])
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

    /// Single transactions over RPC, with bundles off: the wallet-funded plan when it earns
    /// enough (checked by the send's preflight), then the sandwich (simulated while built).
    fn execute_over_rpc(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        liab_mint: Pubkey,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let wallet_short = match self.sufficient_wallet_plan(strategy, intent, Landing::Rpc) {
            Ok(plan) => {
                return self.land_wallet_plan(&liquidatee, intent.attempt_id, plan, Preflight::Run)
            }
            Err(short) => short,
        };

        // A sandwich funds itself from the seized collateral and touches no wallet inventory.
        match strategy.assemble_sandwich(&intent.for_sandwich(), Landing::Rpc) {
            Ok(Sandwich::Plan(ExecutionPlan { tx, .. })) => {
                // Simulated while it was built: no second check on the way out.
                // It claims no wallet tokens, so nothing waits for the balances to be re-read.
                info!(
                    "LIQUIDATION_RPC_SUBMIT attempt_id={} account={} signature={:?}",
                    intent.attempt_id,
                    liquidatee,
                    tx.signatures.first().map(ToString::to_string)
                );
                let result = self.send_over_rpc(&tx, "Sandwich", Preflight::Skip);
                self.note_result(&liquidatee, &result);
                result
            }
            no_sandwich => {
                // What the wallet funds is better than nothing when no sandwich can go.
                if let Some(plan) = self.partial_wallet_plan(strategy, intent, Landing::Rpc) {
                    return self.land_wallet_plan(
                        &liquidatee,
                        intent.attempt_id,
                        plan,
                        Preflight::Run,
                    );
                }
                self.no_plan(&liquidatee, liab_mint, &wallet_short, no_sandwich.err())
            }
        }
    }

    /// `unpushed`, a liquidation the chain's prices allow, in a bundle without a push.
    fn execute_without_push(
        &self,
        strategy: &dyn LiquidationStrategy,
        unpushed: &PreparedLiquidatableAccount,
    ) -> Result<()> {
        let liquidatee = unpushed.liquidatee_account.address;
        let liab_mint = self.intent_liab_mint(unpushed)?;
        info!("Liquidating {liquidatee} without a push, on the chain's prices: the push can't be built");
        self.execute_as_bundle(strategy, unpushed, liab_mint, &[])
    }

    /// A wallet-funded plan for whatever share of `intent` the wallet funds, even below
    /// [`LIQUIDATION_WALLET_MIN_PROFIT_SHARE`]: when no sandwich can go, it is the only plan. `None`
    /// when the wallet funds less than the minimum profit.
    fn partial_wallet_plan(
        &self,
        strategy: &dyn LiquidationStrategy,
        intent: &PreparedLiquidatableAccount,
        landing: Landing,
    ) -> Option<ExecutionPlan> {
        let plan = strategy.assemble_with_inventory(intent, landing).ok()?;
        info!(
            "{}: no sandwich; repaying the share the wallet funds, for ${:.4}",
            intent.liquidatee_account.address,
            plan.profit.to_num::<f64>()
        );
        Some(plan)
    }

    /// Neither plan: the wallet funds too little and no sandwich fits (`sandwich_error`, when it
    /// failed outright). The account is held briefly.
    fn no_plan(
        &self,
        liquidatee: &Pubkey,
        liab_mint: Pubkey,
        wallet_short: &str,
        sandwich_error: Option<anyhow::Error>,
    ) -> Result<()> {
        let e = match sandwich_error {
            Some(e) => anyhow!("no plan for {liquidatee} (debt {liab_mint}): {wallet_short}, and the sandwich failed: {e}"),
            None => anyhow!("no plan for {liquidatee} (debt {liab_mint}): {wallet_short}, and no sandwich fits"),
        };
        self.note_assemble_failure(liquidatee, &e);
        Err(e)
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
        attempt_id: u64,
        plan: ExecutionPlan,
        preflight: Preflight,
    ) -> Result<()> {
        let ExecutionPlan { tx, claim, .. } = plan;
        info!(
            "LIQUIDATION_RPC_SUBMIT attempt_id={} account={} signature={:?}",
            attempt_id,
            liquidatee,
            tx.signatures.first().map(ToString::to_string)
        );
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
                Err(anyhow!("simulation failed: {err:?}; full_logs: {logs:?}"))
            }
        }
    }

    /// Why a bundle that didn't land didn't, as far as simulating its liquidation alone now tells
    /// (after the fact, off the landing path): only for a liquidation that went alone, since one
    /// behind a push or a refresh would fail the solo simulation for their absence.
    fn why_not_landed(&self, bundle: &[VersionedTransaction], behind_others: bool) -> String {
        if behind_others {
            return "lost or dropped (behind a push or a refresh, so not simulated alone)"
                .to_string();
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
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let bundles = self
            .bundles
            .as_ref()
            .ok_or_else(|| anyhow!("bundles are off"))?;
        // Makes this bundle's push, refresh and tip transactions its own.
        let send = NEXT_SEND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The Kamino and JupLend refreshes go ahead of the liquidation when the bundle has room:
        // with the push, or in a transaction of their own. The liquidation's route gets their
        // accounts. Otherwise the liquidation keeps them.
        let refresh = if bundles.refresh_ahead {
            strategy.refresh_ixs(intent).unwrap_or_else(|e| {
                debug!("{liquidatee}: refreshes kept in the liquidation ({e})");
                Vec::new()
            })
        } else {
            Vec::new()
        };
        let mut landing = Landing::Bundle;
        // The push first: without it there is nothing to bundle, and no quote is spent.
        let push = if feeds.is_empty() {
            None
        } else {
            let carrying = if refresh.is_empty() {
                None
            } else {
                bundles.pusher.push_transactions(feeds, &refresh, send).ok()
            };
            if carrying.is_some() {
                landing = Landing::BundleRefreshed;
            }
            match carrying.map_or_else(|| bundles.pusher.push_transactions(feeds, &[], send), Ok) {
                Ok(push) => Some(push),
                Err(e) => {
                    // Not held: the feed may be pushable, or no longer needed, any moment.
                    let e = PushFailure(format!(
                        "can't push the {} Pyth feed(s) it needs: {e}",
                        feeds.len()
                    ));
                    return Err(e.into());
                }
            }
        };
        if !feeds.is_empty() {
            info!(
                "LIQUIDATION_PRICE_PUSH attempt_id={} account={} feed_ids={:?} push_tx_count={}",
                intent.attempt_id,
                liquidatee,
                feeds
                    .iter()
                    .map(|feed| bs58::encode(feed).into_string())
                    .collect::<Vec<_>>(),
                push.as_ref().map_or(0, |push| push.txs.len())
            );
        }
        // Push, refresh, liquidation and tip.
        let push_txs = push.as_ref().map_or(0, |push| push.txs.len());
        let mut refresh_tx = None;
        if !refresh.is_empty() && landing == Landing::Bundle && push_txs + 3 <= MAX_BUNDLE_TXS {
            match bundles.pusher.transaction(&refresh, send) {
                Ok(Some(tx)) => {
                    refresh_tx = Some(tx);
                    landing = Landing::BundleRefreshed;
                }
                Ok(None) => debug!("{liquidatee}: its refreshes don't fit a transaction"),
                Err(e) => debug!("{liquidatee}: refreshes kept in the liquidation ({e})"),
            }
        }

        let wallet_short = match self.sufficient_wallet_plan(strategy, intent, landing) {
            Ok(plan) => {
                return self.land_bundle(
                    strategy,
                    bundles,
                    intent,
                    push.as_ref(),
                    refresh_tx,
                    plan,
                    "wallet-funded",
                    send,
                )
            }
            Err(short) => short,
        };
        match strategy.assemble_sandwich(&intent.for_sandwich(), landing) {
            Ok(Sandwich::Plan(plan)) => self.land_bundle(
                strategy,
                bundles,
                intent,
                push.as_ref(),
                refresh_tx,
                plan,
                "sandwich",
                send,
            ),
            no_sandwich => {
                // What the wallet funds is better than nothing when no sandwich can go.
                if let Some(plan) = self.partial_wallet_plan(strategy, intent, landing) {
                    return self.land_bundle(
                        strategy,
                        bundles,
                        intent,
                        push.as_ref(),
                        refresh_tx,
                        plan,
                        "wallet-funded (partial)",
                        send,
                    );
                }
                self.no_plan(&liquidatee, liab_mint, &wallet_short, no_sandwich.err())
            }
        }
    }

    /// Sends `[push..] [refresh] [liquidation] [tip]`, the tip a share of the plan's profit, and
    /// settles it.
    #[allow(clippy::too_many_arguments)]
    fn land_bundle(
        &self,
        strategy: &dyn LiquidationStrategy,
        bundles: &BundleLanding,
        intent: &PreparedLiquidatableAccount,
        push: Option<&PricePush>,
        refresh_tx: Option<VersionedTransaction>,
        plan: ExecutionPlan,
        kind: &str,
        send: u64,
    ) -> Result<()> {
        let liquidatee = intent.liquidatee_account.address;
        let ExecutionPlan { tx, claim, profit } = plan;
        let profit_lamports = strategy.lamports_for_value(profit).unwrap_or_else(|e| {
            warn!("Can't price the profit of {liquidatee} in SOL ({e}); tipping the floor");
            0
        });
        // A few lamports off or on per send, so two bundles' tip transactions are never the same,
        // within the cap.
        let tip = bundles.tip_estimator.tip_for(profit_lamports, send % 1_000);
        let mut bundle = push.map(|push| push.txs.clone()).unwrap_or_default();
        let push_signatures: Vec<String> = bundle
            .iter()
            .filter_map(|tx| tx.signatures.first().map(ToString::to_string))
            .collect();
        let behind_refresh = refresh_tx.is_some();
        bundle.extend(refresh_tx);
        bundle.push(tx);
        let alone = bundle.len() == 1;
        info!(
            "LIQUIDATION_BUNDLE attempt_id={} account={} plan={} estimated_profit_usd={:.6} tip_lamports={} push_tx_count={} push_signatures={:?}",
            intent.attempt_id,
            liquidatee,
            kind,
            profit.to_num::<f64>(),
            tip,
            push.map_or(0, |push| push.txs.len()),
            push_signatures
        );

        // The request slot is taken only now, everything built, so none is held while quoting.
        let slot = bundles.jito.reserve();
        let mut accepted = false;
        let result = match self.send_bundle(intent.attempt_id, bundles, &bundle, tip, slot) {
            Ok(BundleOutcome::Landed) => Ok(()),
            Ok(BundleOutcome::NotLanded) => {
                accepted = true;
                Err(anyhow!(
                    "the bundle for {liquidatee} didn't land: {}",
                    self.why_not_landed(&bundle, push.is_some() || behind_refresh)
                ))
            }
            Ok(BundleOutcome::Refused) => Err(anyhow!("the bundle for {liquidatee} wasn't sent")),
            Err(e) => Err(e),
        };
        // A bundle that didn't land costs nothing and holds nothing back: the account is evaluated
        // afresh. Only the premium cap holds it, as retrying the same sizing fails the same way.
        match &result {
            Ok(()) => {
                self.clear_target_backoff(&liquidatee);
                self.misses_reset(&liquidatee);
            }
            // Only when the liquidation went alone does its simulation tell.
            Err(e) if alone && is_premium_cap_failure(&e.to_string()) => {
                self.hold(&liquidatee, PREMIUM_CAP_HOLD, "premium cap")
            }
            // A lost auction is routine; a bundle that keeps not landing may fail for a reason
            // nothing reports (it isn't simulated behind a push or refresh), so it rests briefly.
            Err(_) if accepted => {
                if self.miss(&liquidatee) >= MISSES_BEFORE_HOLD {
                    self.misses_reset(&liquidatee);
                    self.hold(
                        &liquidatee,
                        NO_PLAN_HOLD,
                        &format!("{MISSES_BEFORE_HOLD} bundles in a row didn't land"),
                    );
                }
            }
            Err(_) => {}
        }
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
        attempt_id: u64,
        bundles: &BundleLanding,
        bundle: &[VersionedTransaction],
        tip: u64,
        slot: JitoSlot<'_>,
    ) -> Result<BundleOutcome> {
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
                info!("LIQUIDATION_BUNDLE_ACCEPTED attempt_id={} bundle_id={} liquidation_signature={} txs_before_liquidation={}",
                    attempt_id, id, liquidation_sig, bundle.len().saturating_sub(1));
                Ok(
                    // "0x0": already processed, so there's no bundle to ask about; the liquidation's
                    // signature settles it.
                    match self.landed_before_expiry(
                        &liquidation_sig,
                        &blockhash,
                        (id != "0x0").then_some((&bundles.jito, id.as_str())),
                    )? {
                        true => BundleOutcome::Landed,
                        false => BundleOutcome::NotLanded,
                    },
                )
            }
            Ok(BundleSend::Rejected(reason)) => {
                warn!(
                    "LIQUIDATION_BUNDLE_REJECTED attempt_id={} liquidation_signature={} reason={}",
                    attempt_id, liquidation_sig, reason
                );
                Ok(BundleOutcome::Refused)
            }
            // Taken as not sent: nothing waits for it.
            Ok(BundleSend::Unavailable(reason)) => {
                warn!(
                    "LIQUIDATION_BUNDLE_UNAVAILABLE attempt_id={} liquidation_signature={} reason={}",
                    attempt_id, liquidation_sig, reason
                );
                Ok(BundleOutcome::Refused)
            }
            // Never sent: the bundle couldn't be encoded.
            Err(e) => {
                warn!(
                    "LIQUIDATION_BUNDLE_SEND_ERROR attempt_id={} liquidation_signature={} error={}",
                    attempt_id, liquidation_sig, e
                );
                Ok(BundleOutcome::Refused)
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

    /// After a transaction over RPC (bundles off): clears the account's hold when it landed,
    /// otherwise holds it briefly, or longer for the premium cap.
    fn note_result(&self, liquidatee: &Pubkey, result: &Result<()>) {
        match result {
            Ok(()) => {
                self.clear_target_backoff(liquidatee);
                self.misses_reset(liquidatee);
            }
            Err(_) => self.hold(liquidatee, RPC_FAILURE_HOLD, "RPC failure"),
        }
    }

    /// The liability mint for an intent, looked up via the bank cache.
    fn intent_liab_mint(&self, intent: &PreparedLiquidatableAccount) -> Result<Pubkey> {
        Ok(self.cache.banks.try_get_bank(&intent.liab_bank)?.bank.mint)
    }

    /// Whether the account `key` is held after a recent attempt.
    fn is_backed_off(&self, key: &Pubkey) -> Result<bool> {
        let now = Instant::now();
        let mut guard = self
            .backoffs
            .lock()
            .map_err(|_| anyhow!("execution backoffs mutex poisoned"))?;
        guard.retain(|_, until| now < *until);
        Ok(guard.contains_key(key))
    }

    /// Counts a bundle for `target` that didn't land; the count in a row.
    fn miss(&self, target: &Pubkey) -> u32 {
        let now = Instant::now();
        let mut misses = self.misses.lock().unwrap_or_else(|e| e.into_inner());
        misses.retain(|_, (_, first)| now.saturating_duration_since(*first) < MISS_WINDOW);
        let (count, _) = misses.entry(*target).or_insert((0, now));
        *count += 1;
        *count
    }

    fn misses_reset(&self, target: &Pubkey) {
        self.misses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(target);
    }

    /// Leaves `target` alone for `hold`.
    fn hold(&self, target: &Pubkey, hold: Duration, why: &str) {
        let mut guard = self.backoffs.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(*target, Instant::now() + hold);
        debug!("Holding {target} for {}s ({why})", hold.as_secs());
    }

    /// Clears a target's hold once it lands.
    fn clear_target_backoff(&self, target: &Pubkey) {
        self.backoffs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(target);
    }

    /// No plan for `target`: no wallet repay and no sandwich.
    fn note_assemble_failure(&self, target: &Pubkey, e: &anyhow::Error) {
        self.hold(target, NO_PLAN_HOLD, &format!("no plan: {e}"));
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

    fn executor() -> Executor {
        Executor::new(
            RpcClient::new("http://127.0.0.1:1".to_string()),
            Arc::new(Cache::new(
                Pubkey::new_unique(),
                Pubkey::new_unique(),
                Arc::new(Mutex::new(Clock::default())),
                HashSet::new(),
            )),
            None,
            I80F48::ZERO,
        )
    }

    #[test]
    fn holds_are_short_fixed_and_only_for_failures_that_would_repeat() {
        let executor = executor();
        let account = Pubkey::new_unique();

        // No plan: held briefly, never escalating.
        for _ in 0..5 {
            executor.note_assemble_failure(&account, &anyhow!("NO_ROUTES_FOUND"));
        }
        assert!(executor.is_backed_off(&account).unwrap());
        let until = executor.backoffs.lock().unwrap()[&account];
        assert!(until.saturating_duration_since(Instant::now()) <= NO_PLAN_HOLD);

        // Landing clears it.
        executor.note_result(&account, &Ok(()));
        assert!(!executor.is_backed_off(&account).unwrap());

        // Over RPC every failure holds briefly, the premium cap included.
        executor.note_result(&account, &Err(anyhow!("custom program error: 0x17ca")));
        let until = executor.backoffs.lock().unwrap()[&account];
        assert!(until.saturating_duration_since(Instant::now()) <= RPC_FAILURE_HOLD);
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
