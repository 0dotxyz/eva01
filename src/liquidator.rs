use crate::{
    cache::Cache,
    clock_manager,
    config::Eva01Config,
    execution::{executor::Executor, inventory::InventoryStrategy},
    geyser::{AccountType, GeyserUpdate},
    rebalancer::Rebalancer,
    utils::{self, jito::JitoClient, log_genuine_error},
    wrappers::{
        bank::BankWrapper,
        liquidator_account::{
            is_seizable_asset_tag, LiquidatorAccount, PreparedLiquidatableAccount,
        },
        marginfi_account::{MarginfiAccountWrapper, ObservationAccounts},
        oracle::{oracle_publish_time, OracleWrapper, OracleWrapperTrait},
    },
};
use anchor_lang::AccountDeserialize;
use anyhow::{anyhow, Result};
use crossbeam::channel::{Receiver, RecvTimeoutError};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{debug, error, info};
use marginfi::{
    constants::{
        LIQUIDATION_BONUS_FEE_MINIMUM, LIQUIDATION_CLOSEOUT_DOLLAR_THRESHOLD,
        LIQUIDATION_TAG_MAX_PREMIUM,
    },
    state::{
        bank::BankImpl, liquidation_record::tag_adjusted_premium,
        marginfi_account::get_health_components,
    },
};
use marginfi_type_crate::{
    constants::BANKRUPT_THRESHOLD,
    types::{
        reconcile_emode_configs, BalanceSide, Bank, BankOperationalState, EmodeConfig,
        HealthPriceMode, MarginfiAccount, MarginfiGroup, OracleSetup, ReconciledEmodeConfig,
        RequirementType,
    },
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use solana_account::Account;
use solana_account_info::IntoAccountInfo;
use solana_client::rpc_client::RpcClient;
use solana_clock::Clock;
use solana_commitment_config::CommitmentConfig;
use solana_keypair::Keypair;
use solana_pubkey::Pubkey;
use std::sync::atomic::Ordering;
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc, Condvar, Mutex, MutexGuard},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Max liquidations executed concurrently. Bounds in-flight RPC/sim/bundle load while still
/// letting a burst of liquidatable accounts (e.g. a price crash) be worked in parallel rather
/// than one-at-a-time.
const MAX_CONCURRENT_LIQUIDATIONS: usize = 8;

/// How often the rebalancing sweep runs when no liquidation has been attempted.
const IDLE_REBALANCE_INTERVAL: Duration = Duration::from_secs(60);

/// Taken off the premium `end_liquidation` allows, so price moves between evaluation and landing
/// cannot push the seizure over the cap.
const PREMIUM_SAFETY_MARGIN: I80F48 = I80F48!(0.005);

/// Liquidations are sized to this share of the most the program allows, leaving room for price
/// moves between evaluation and landing; profit is judged on the same share.
const LIQUIDATION_SIZE_HAIRCUT: I80F48 = I80F48!(0.90);

/// Minimum time between tag attempts on the same account, so a failing tag is not resent on every
/// update.
const TAG_RETRY_INTERVAL: Duration = Duration::from_secs(600);

/// An account is tagged only after it has been unhealthy, with its position untouched, for this
/// long. A tag costs a fee and pays off only if we liquidate the account ourselves once the premium
/// has grown; an account other liquidators are willing to work gets chunked away long before that.
const TAG_MIN_UNHEALTHY_FOR: Duration = Duration::from_secs(30 * 60);

/// An account under tag observation that has not been evaluated for this long is forgotten.
const TAG_WATCH_TTL: Duration = Duration::from_secs(3600);

/// How often each unhealthy account we are not liquidating is reported.
const SKIP_REPORT_INTERVAL: Duration = Duration::from_secs(600);

/// After a liquidation attempt the same account is left alone this long, so the next attempt sees
/// the state the attempt produced (which arrives over Geyser) rather than the state it started from.
const ATTEMPT_COOLDOWN: Duration = Duration::from_secs(3);

/// How long a worker waits for new intents before re-checking cooldowns and the stop flag.
const WORKER_WAIT: Duration = Duration::from_millis(250);

pub struct Liquidator {
    rebalancer: Rebalancer,
    executor: Arc<Executor>,
    strategy: Arc<InventoryStrategy>,
    evaluator: Arc<Evaluator>,
    queue: Arc<ExecutionQueue>,
    workers: Vec<JoinHandle<()>>,
    liquidator_account: Arc<LiquidatorAccount>,
    /// When each account was last tagged (or a tag last failed).
    tag_attempts: HashMap<Pubkey, Instant>,
    /// Unhealthy accounts waiting out [`TAG_MIN_UNHEALTHY_FOR`].
    tag_watch: HashMap<Pubkey, TagWatch>,
    /// When each unhealthy account we are not liquidating was last reported.
    skip_reports: HashMap<Pubkey, Instant>,
    geyser_rx: Receiver<GeyserUpdate>,
    stop_liquidator: Arc<AtomicBool>,
    cache: Arc<Cache>,
}

/// Finds the liquidatable accounts in the cache and sizes their liquidations. Shared by the
/// liquidator loop (which evaluates every updated account) and the execution workers (which
/// re-evaluate an account on the current state right before liquidating it).
struct Evaluator {
    cache: Arc<Cache>,
    min_profit: f64,
}

/// Liquidation intents waiting for a worker: the newest evaluation of each account. The liquidator
/// loop replaces intents as it re-evaluates, and the workers take the most profitable one that is
/// not already being worked, so a slow send never holds up evaluation or the other intents.
struct ExecutionQueue {
    state: Mutex<QueueState>,
    ready: Condvar,
    /// Set by a worker after an attempt, since the wallet may now hold tokens to sweep.
    rebalance_requested: AtomicBool,
}

#[derive(Default)]
struct QueueState {
    pending: HashMap<Pubkey, QueuedIntent>,
    in_flight: HashSet<Pubkey>,
    cooldowns: HashMap<Pubkey, Instant>,
    stopped: bool,
}

struct QueuedIntent {
    intent: PreparedLiquidatableAccount,
    queued_at: Instant,
}

impl ExecutionQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(QueueState::default()),
            ready: Condvar::new(),
            rebalance_requested: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replaces the intents of the `checked` accounts with `intents`: an account that was checked
    /// and is no longer liquidatable leaves the queue.
    fn offer(&self, checked: &[Pubkey], intents: Vec<PreparedLiquidatableAccount>) {
        let mut state = self.lock();
        for address in checked {
            state.pending.remove(address);
        }
        let queued_at = Instant::now();
        for intent in intents {
            state.pending.insert(
                intent.liquidatee_account.address,
                QueuedIntent { intent, queued_at },
            );
        }
        drop(state);
        self.ready.notify_all();
    }

    /// The most profitable intent not being worked or cooling down; `None` once stopped.
    fn take(&self) -> Option<QueuedIntent> {
        let mut state = self.lock();
        loop {
            if state.stopped {
                return None;
            }
            let now = Instant::now();
            state.cooldowns.retain(|_, until| *until > now);
            let best = state
                .pending
                .iter()
                .filter(|(address, _)| {
                    !state.in_flight.contains(*address) && !state.cooldowns.contains_key(*address)
                })
                .max_by_key(|(_, queued)| queued.intent.profit)
                .map(|(address, _)| *address);
            if let Some(queued) = best.and_then(|address| state.pending.remove(&address)) {
                state
                    .in_flight
                    .insert(queued.intent.liquidatee_account.address);
                return Some(queued);
            }
            state = self
                .ready
                .wait_timeout(state, WORKER_WAIT)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn release(&self, address: Pubkey) {
        let mut state = self.lock();
        state.in_flight.remove(&address);
        state
            .cooldowns
            .insert(address, Instant::now() + ATTEMPT_COOLDOWN);
        drop(state);
        self.ready.notify_all();
    }

    fn request_rebalance(&self) {
        self.rebalance_requested.store(true, Ordering::Relaxed);
    }

    fn take_rebalance_request(&self) -> bool {
        self.rebalance_requested.swap(false, Ordering::Relaxed)
    }

    fn stop(&self) {
        self.lock().stopped = true;
        self.ready.notify_all();
    }
}

#[derive(Debug, Clone, Copy)]
struct LiquidationAmounts {
    max_liquidatable_asset_amount: I80F48,
    max_liquidatable_liab_amount: I80F48,
    /// The most `end_liquidation` lets this pair repay (USD), whether or not it is worth it.
    max_repay: I80F48,
    /// The premium the liquidation is sized at.
    premium: I80F48,
    liquidator_profit: I80F48,
    /// Unhealthy, untagged, solvent and not e-mode boosted: tagging would let the premium grow.
    taggable: bool,
}

impl LiquidationAmounts {
    fn none() -> Self {
        Self {
            max_liquidatable_asset_amount: I80F48::ZERO,
            max_liquidatable_liab_amount: I80F48::ZERO,
            max_repay: I80F48::ZERO,
            premium: I80F48::ZERO,
            liquidator_profit: I80F48::ZERO,
            taggable: false,
        }
    }

    fn is_liquidatable(&self) -> bool {
        !self.max_liquidatable_asset_amount.is_zero()
    }
}

/// The best bank pair for an unhealthy account and what it allows.
struct Sizing {
    asset_bank: Pubkey,
    liab_bank: Pubkey,
    amounts: LiquidationAmounts,
    maintenance_health: I80F48,
}

/// A bank's maintenance price over its equity price on one side.
fn maintenance_to_equity_price(
    bank: &BankWrapper,
    oracle: &OracleWrapper,
    side: impl Fn() -> BalanceSide,
) -> Result<I80F48> {
    let equity = bank.price(oracle, side(), RequirementType::Equity)?;
    bank.price(oracle, side(), RequirementType::Maintenance)?
        .checked_div(equity)
        .ok_or_else(|| anyhow!("no equity price for the bank {}", bank.address))
}

/// Whether the bank's deposits cover its borrows. The program's `check_utilization_ratio` answers
/// the same question but logs every failing call, which for a bank in bad debt is once per holder
/// per evaluation cycle.
fn bank_covers_its_borrows(bank: &Bank) -> Result<bool> {
    let total_assets = bank
        .get_asset_amount(bank.total_asset_shares.into())
        .map_err(|e| anyhow!("Couldn't calculate the bank's total assets: {}", e))?;
    let total_liabilities = bank
        .get_liability_amount(bank.total_liability_shares.into())
        .map_err(|e| anyhow!("Couldn't calculate the bank's total liabilities: {}", e))?;
    Ok(total_assets >= total_liabilities)
}

/// What `end_liquidation` bounds a receivership by: USD values at equity prices, health at
/// maintenance weights.
struct RepayBounds {
    liab_value: I80F48,
    asset_value: I80F48,
    /// Equity assets minus equity liabilities; negative in bad debt.
    equity: I80F48,
    maintenance_health: I80F48,
    asset_weight_maint: I80F48,
    liab_weight_maint: I80F48,
    /// Maintenance price over equity price, per side: turns equity dollars into health dollars.
    asset_price_ratio: I80F48,
    liab_price_ratio: I80F48,
    above_closeout: bool,
}

impl RepayBounds {
    /// Largest repayment (USD) `end_liquidation` accepts when seizing `repaid * (1 + premium)`:
    /// * the seizure fits the collateral and the repayment fits the debt;
    /// * an account that started solvent must stay solvent, i.e. the premium taken cannot exceed
    ///   its equity;
    /// * above the closeout threshold the account must stay unhealthy. Each equity dollar repaid
    ///   raises maintenance health by `l * rl - w * (1 + premium) * ra`; once that is no longer
    ///   positive the liquidation cannot heal the account and this bound no longer applies.
    fn max_repay(&self, premium: I80F48) -> I80F48 {
        let one_plus_premium = I80F48::ONE + premium;
        let mut max_repay = min(self.liab_value, self.asset_value / one_plus_premium);
        if self.equity >= I80F48::ZERO && premium > I80F48::ZERO {
            max_repay = min(max_repay, self.equity / premium);
        }
        let health_gain_per_repaid = self.liab_weight_maint * self.liab_price_ratio
            - self.asset_weight_maint * one_plus_premium * self.asset_price_ratio;
        if self.above_closeout && health_gain_per_repaid > I80F48::ZERO {
            max_repay = min(max_repay, -self.maintenance_health / health_gain_per_repaid);
        }
        I80F48::max(I80F48::ZERO, max_repay)
    }
}

#[allow(clippy::large_enum_variant)]
enum Evaluation {
    Liquidate(PreparedLiquidatableAccount),
    /// Unhealthy but not worth liquidating at the current premium.
    Unhealthy(UnhealthyAccount),
}

/// Active balances as `(shares, bank)` per side; unchanged while nobody touches the position.
type Balances = (Vec<(I80F48, Pubkey)>, Vec<(I80F48, Pubkey)>);

struct UnhealthyAccount {
    address: Pubkey,
    observation_accounts: ObservationAccounts,
    /// Tagging would let the premium grow enough to make the liquidation worth it.
    taggable: bool,
    tagged_at: i64,
    maintenance_health: I80F48,
    asset_bank: Pubkey,
    liab_bank: Pubkey,
    max_repay: I80F48,
    profit: I80F48,
    premium: I80F48,
    balances: Balances,
}

struct TagWatch {
    /// Since when the account has been unhealthy with these balances.
    since: Instant,
    last_seen: Instant,
    balances: Balances,
}

impl Liquidator {
    pub fn new(
        config: Eva01Config,
        liquidator_account: Arc<LiquidatorAccount>,
        geyser_rx: Receiver<GeyserUpdate>,
        stop_liquidator: Arc<AtomicBool>,
        cache: Arc<Cache>,
    ) -> Result<Self> {
        let rebalancer = Rebalancer::new(config.clone(), cache.clone())?;
        let dex_client = rebalancer.dex_client();
        let dex_gate = rebalancer.dex_gate();

        let jito = JitoClient::new(
            config.jito_block_engine_url.clone(),
            config.bundle_api_key.clone(),
        );
        let rpc_client =
            RpcClient::new_with_commitment(config.rpc_url.clone(), CommitmentConfig::confirmed());
        let signer = Keypair::try_from(config.wallet_keypair.as_slice())?;
        let executor = Executor::new(
            jito,
            rpc_client,
            cache.clone(),
            signer,
            config.rpc_url.clone(),
            config.bundle_api_key.clone(),
            config.jito_tip_max_lamports,
        );

        let strategy = InventoryStrategy::new(
            liquidator_account.clone(),
            config.swap_mint,
            dex_client,
            dex_gate,
            config.slippage_bps,
            config.min_profit,
        )?;

        Ok(Liquidator {
            rebalancer,
            executor: Arc::new(executor),
            strategy: Arc::new(strategy),
            evaluator: Arc::new(Evaluator {
                cache: cache.clone(),
                min_profit: config.min_profit,
            }),
            queue: Arc::new(ExecutionQueue::new()),
            workers: Vec::new(),
            liquidator_account,
            tag_attempts: HashMap::new(),
            tag_watch: HashMap::new(),
            skip_reports: HashMap::new(),
            geyser_rx,
            stop_liquidator,
            cache,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.rebalancer.run()?;
        self.spawn_workers()?;

        // Geyser only delivers changes, so an account that is already underwater at startup would
        // wait for an unrelated update to touch one of its banks. Sweep them all once up front.
        match self
            .cache
            .marginfi_accounts
            .try_get_accounts_with_liabilities()
        {
            Ok(accounts) => {
                info!("Running the initial Liquidation scan...");
                self.run_liquidation_cycle(accounts);
            }
            Err(error) => error!(
                "Failed to collect accounts for the initial scan: {:?}",
                error
            ),
        }

        info!("Staring the Liquidator loop.");
        let mut last_rebalance = Instant::now();
        while !self.stop_liquidator.load(Ordering::Relaxed) {
            debug!("Waiting for any data change...");
            let accounts_to_check = self.receive_geyser_updates()?;
            if !accounts_to_check.is_empty() {
                self.run_liquidation_cycle(accounts_to_check);
            }

            // Sell-only sweep: convert any seized collateral / JIT-buy overshoot back to USDC. Only
            // a liquidation attempt can leave such tokens behind, so the sweep runs after one, and
            // otherwise on a heartbeat.
            if self.queue.take_rebalance_request()
                || last_rebalance.elapsed() >= IDLE_REBALANCE_INTERVAL
            {
                last_rebalance = Instant::now();
                if let Err(error) = self.rebalancer.run() {
                    error!("Rebalancing failed: {:?}", error);
                }
            }
        }

        info!("Stopping the liquidation workers...");
        self.queue.stop();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                error!("A liquidation worker panicked.");
            }
        }
        info!("The Liquidator loop is stopped.");

        Ok(())
    }

    /// Starts the workers that take intents off the queue and land them. They outlive the
    /// evaluation cycles: a send waiting for confirmation occupies one worker, not the loop.
    fn spawn_workers(&mut self) -> Result<()> {
        for index in 0..MAX_CONCURRENT_LIQUIDATIONS {
            let queue = Arc::clone(&self.queue);
            let evaluator = Arc::clone(&self.evaluator);
            let executor = Arc::clone(&self.executor);
            let strategy = Arc::clone(&self.strategy);
            let worker = thread::Builder::new()
                .name(format!("liquidation-worker-{index}"))
                .spawn(move || Self::run_worker(&queue, &evaluator, &executor, &strategy))?;
            self.workers.push(worker);
        }
        Ok(())
    }

    fn run_worker(
        queue: &ExecutionQueue,
        evaluator: &Evaluator,
        executor: &Executor,
        strategy: &InventoryStrategy,
    ) {
        while let Some(QueuedIntent { intent, queued_at }) = queue.take() {
            let liquidatee = intent.liquidatee_account.address;
            // The intent was sized on the state at evaluation time. By now another liquidator may
            // have taken a chunk of the account or prices may have moved, so it is re-sized on the
            // current state and dropped if it is no longer worth it, instead of sending a
            // transaction preflight or the program would reject.
            match evaluator.re_evaluate(&liquidatee) {
                Ok(Some(fresh)) => {
                    info!(
                        "Liquidating {}: seizing {} for {} (est. profit ${:.4}, ${:.4} when queued {} ms ago)",
                        liquidatee,
                        fresh.asset_bank,
                        fresh.liab_bank,
                        fresh.profit.to_num::<f64>(),
                        intent.profit.to_num::<f64>(),
                        queued_at.elapsed().as_millis()
                    );
                    if let Err(e) = executor.try_execute(strategy, &fresh) {
                        error!(
                            "Failed to execute liquidation for {:?}: {:?}",
                            liquidatee, e
                        );
                    }
                    queue.request_rebalance();
                }
                Ok(None) => info!(
                    "Dropping the liquidation of {}: no longer worth it on the current state ({} ms after it was queued)",
                    liquidatee,
                    queued_at.elapsed().as_millis()
                ),
                Err(e) => error!(
                    "Failed to re-evaluate {} before liquidating it: {:?}",
                    liquidatee, e
                ),
            }
            queue.release(liquidatee);
        }
    }

    fn run_liquidation_cycle(&mut self, accounts_to_check: Vec<Pubkey>) {
        let started = Instant::now();
        match self.evaluator.check_accounts(&accounts_to_check) {
            Ok((liquidations, unhealthy)) => {
                let liquidatable = liquidations.len();
                let skipped = unhealthy.len();
                self.queue.offer(&accounts_to_check, liquidations);
                self.handle_unhealthy(&accounts_to_check, unhealthy);
                info!(
                    "Evaluated {} account(s) in {} ms: {} queued for liquidation, {} unhealthy but not worth liquidating.",
                    accounts_to_check.len(),
                    started.elapsed().as_millis(),
                    liquidatable,
                    skipped
                );
            }
            Err(err) => {
                error!("Failed to evaluate accounts in liquidation: {}", err);
            }
        }
    }

    fn receive_geyser_updates(&self) -> Result<Vec<Pubkey>> {
        let first_update = match self.geyser_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(update) => update,
            Err(RecvTimeoutError::Timeout) => return Ok(vec![]),
            Err(RecvTimeoutError::Disconnected) if self.stop_liquidator.load(Ordering::Relaxed) => {
                return Ok(vec![])
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("Geyser update channel disconnected"))
            }
        };

        let mut accounts_to_check = HashSet::new();
        self.process_geyser_update(first_update, &mut accounts_to_check);
        while let Ok(update) = self.geyser_rx.try_recv() {
            self.process_geyser_update(update, &mut accounts_to_check);
        }

        Ok(accounts_to_check.into_iter().collect())
    }

    fn process_geyser_update(&self, update: GeyserUpdate, account_addresses: &mut HashSet<Pubkey>) {
        if let Err(error) = self.handle_geyser_update(update, account_addresses) {
            log_genuine_error("Failed to process Geyser update", error);
        }
    }

    /// Our clock lags the chain, so a fresh Scope price can look "published in the future" and get
    /// rejected. Moving the clock up to each new publish time prevents that.
    fn advance_clock_to_oracle(&self, address: &Pubkey, account: &Account, banks: &[Pubkey]) {
        let scope_entries: Vec<u16> = banks
            .iter()
            .filter_map(|bank| self.cache.banks.try_get_bank(bank).ok())
            .filter(|wrapper| {
                matches!(
                    wrapper.bank.config.oracle_setup,
                    OracleSetup::Scope | OracleSetup::ScopeKamino | OracleSetup::ScopeJuplend
                ) && wrapper.bank.config.oracle_keys[0] == *address
            })
            .map(|wrapper| wrapper.bank.config.scope_entry_index)
            .collect();

        if let Some(publish_time) = oracle_publish_time(account, &scope_entries) {
            if let Err(e) = clock_manager::advance_to(&self.cache.clock, publish_time) {
                error!("Failed to advance the clock to the {address} publish time: {e}");
            }
        }
    }

    fn handle_geyser_update(
        &self,
        update: GeyserUpdate,
        account_addresses: &mut HashSet<Pubkey>,
    ) -> Result<()> {
        let GeyserUpdate {
            account_type,
            address,
            account,
        } = update;

        debug!("Processing the {:?} {:?} update.", account_type, address);

        match account_type {
            AccountType::Oracle => {
                let banks = self.cache.banks.get_banks_for_oracle(&address)?;
                self.advance_clock_to_oracle(&address, &account, &banks);
                self.cache.oracles.try_update(&address, account)?;

                for bank in banks {
                    self.add_accounts_for_bank(&bank, account_addresses)?;
                }
            }
            AccountType::Marginfi => {
                let mut data = account.data.as_slice();
                let marginfi_account = MarginfiAccount::try_deserialize(&mut data)?;
                let has_liabilities = self
                    .cache
                    .marginfi_accounts
                    .try_insert(MarginfiAccountWrapper::new(address, marginfi_account))?;

                // Only queue accounts that carry debt — a debt-free account can't be liquidated, so
                // evaluating it is wasted work. It rejoins the queue the moment it borrows (that
                // update will report liabilities).
                if has_liabilities {
                    account_addresses.insert(address);
                }
            }
            AccountType::Bank => {
                let bank = utils::decode_bank(&account.data)?;
                let is_new = self.cache.banks.try_get_bank(&address).is_err();
                self.cache.banks.try_insert(address, bank, account)?;
                if is_new {
                    self.liquidator_account.cover_bank_in_background(&address);
                }

                self.add_accounts_for_bank(&address, account_addresses)?;
            }
            AccountType::Token => {
                self.cache.tokens.try_update_account(address, account)?;
            }
        }
        Ok(())
    }

    fn add_accounts_for_bank(
        &self,
        bank: &Pubkey,
        account_addresses: &mut HashSet<Pubkey>,
    ) -> Result<()> {
        account_addresses.extend(
            self.cache
                .marginfi_accounts
                .try_get_accounts_for_bank_with_liabilities(bank)?,
        );
        Ok(())
    }

    /// Reports the unhealthy accounts we are not liquidating and tags the ones that have stayed
    /// unhealthy and untouched for [`TAG_MIN_UNHEALTHY_FOR`].
    fn handle_unhealthy(&mut self, checked: &[Pubkey], unhealthy: Vec<UnhealthyAccount>) {
        let now = Instant::now();

        // An account that was checked and is no longer merely unhealthy (it healed, someone
        // liquidated it, or it became worth liquidating) leaves the watch; the wait starts over if
        // it turns up again.
        let still_unhealthy: HashSet<Pubkey> = unhealthy.iter().map(|u| u.address).collect();
        for address in checked {
            if !still_unhealthy.contains(address) {
                self.tag_watch.remove(address);
            }
        }
        self.tag_watch
            .retain(|_, watch| now.duration_since(watch.last_seen) < TAG_WATCH_TTL);
        self.tag_attempts
            .retain(|_, at| now.duration_since(*at) < TAG_RETRY_INTERVAL);
        self.skip_reports
            .retain(|_, at| now.duration_since(*at) < SKIP_REPORT_INTERVAL);

        for account in unhealthy {
            self.report_skipped(&account, now);
            self.consider_tagging(account, now);
        }
    }

    fn report_skipped(&mut self, account: &UnhealthyAccount, now: Instant) {
        if self.skip_reports.contains_key(&account.address) {
            return;
        }
        self.skip_reports.insert(account.address, now);
        info!(
            "Not liquidating the unhealthy account {} (maintenance health ${:.4}): seizing {} for {} \
             may repay at most ${:.4} for ${:.4} profit at a {:.2}% premium, below the ${} minimum \
             (tagged_at {}, taggable {})",
            account.address,
            account.maintenance_health.to_num::<f64>(),
            account.asset_bank,
            account.liab_bank,
            account.max_repay.to_num::<f64>(),
            account.profit.to_num::<f64>(),
            account.premium.to_num::<f64>() * 100.0,
            self.evaluator.min_profit,
            account.tagged_at,
            account.taggable
        );
    }

    /// Tags `account` once it has been unhealthy with nobody touching it for
    /// [`TAG_MIN_UNHEALTHY_FOR`]. The tag fee is spent either way, so an account other liquidators
    /// are working (its balances keep changing) is left to them.
    fn consider_tagging(&mut self, account: UnhealthyAccount, now: Instant) {
        let UnhealthyAccount {
            address,
            observation_accounts,
            taggable,
            balances,
            ..
        } = account;
        if !taggable {
            self.tag_watch.remove(&address);
            return;
        }

        let watch = self.tag_watch.entry(address).or_insert_with(|| TagWatch {
            since: now,
            last_seen: now,
            balances: balances.clone(),
        });
        watch.last_seen = now;
        if watch.balances != balances {
            watch.since = now;
            watch.balances = balances;
        }
        let unhealthy_for = now.duration_since(watch.since);
        if unhealthy_for < TAG_MIN_UNHEALTHY_FOR || self.tag_attempts.contains_key(&address) {
            return;
        }

        // Tagged by someone else since the evaluation? The program would reject ours, fee paid.
        match self.cache.marginfi_accounts.try_get_account(&address) {
            Ok(current) if current.account.liquidation_tagged_at != 0 => {
                debug!("Not tagging {}: already tagged.", address);
                self.tag_watch.remove(&address);
                return;
            }
            Ok(_) => {}
            Err(e) => {
                debug!("Not tagging {}: {}", address, e);
                return;
            }
        }

        self.tag_attempts.insert(address, now);
        match self
            .liquidator_account
            .send_tag_tx(address, &observation_accounts)
        {
            Ok(signature) => info!(
                "Sent a tag for {} after {} min unhealthy with nobody liquidating it, so its premium grows: {}",
                address,
                unhealthy_for.as_secs() / 60,
                signature
            ),
            Err(err) => error!("Failed to tag {}: {:?}", address, err),
        }
    }
}

impl Evaluator {
    fn check_accounts(
        &self,
        account_addresses: &[Pubkey],
    ) -> Result<(Vec<PreparedLiquidatableAccount>, Vec<UnhealthyAccount>)> {
        let clock = clock_manager::get_clock(&self.cache.clock)?;

        // Evaluation is CPU-bound (oracle parsing + health math) and independent per account, so a
        // full scan of every borrower parallelises cleanly across cores.
        let evaluations: Vec<Evaluation> = account_addresses
            .par_iter()
            .filter_map(|account_address| {
                let account = self
                    .cache
                    .marginfi_accounts
                    .try_get_account(account_address)
                    .ok()?;

                match self.process_account(&account, clock.clone()) {
                    Ok(acc_opt) => acc_opt,
                    Err(err) => {
                        debug!("Failed to process account {:?}: {:?}", account.address, err);
                        None
                    }
                }
            })
            .collect();

        let mut liquidations = Vec::new();
        let mut unhealthy = Vec::new();
        for evaluation in evaluations {
            match evaluation {
                Evaluation::Liquidate(acc) => liquidations.push(acc),
                Evaluation::Unhealthy(acc) => unhealthy.push(acc),
            }
        }
        Ok((liquidations, unhealthy))
    }

    /// Sizes the liquidation of `address` on the current cache state; `None` when it is no longer
    /// worth liquidating.
    fn re_evaluate(&self, address: &Pubkey) -> Result<Option<PreparedLiquidatableAccount>> {
        let account = self.cache.marginfi_accounts.try_get_account(address)?;
        let clock = clock_manager::get_clock(&self.cache.clock)?;
        Ok(match self.process_account(&account, clock)? {
            Some(Evaluation::Liquidate(intent)) => Some(intent),
            _ => None,
        })
    }

    fn process_account(
        &self,
        account: &MarginfiAccountWrapper,
        clock: Clock,
    ) -> Result<Option<Evaluation>> {
        let (deposit_shares, liab_shares) = account.get_deposits_and_liabilities_shares();
        if deposit_shares.is_empty() || liab_shares.is_empty() {
            return Ok(None);
        }

        let deposit_values = self.get_value_of_shares(
            &clock,
            deposit_shares,
            &BalanceSide::Assets,
            RequirementType::Maintenance,
        )?;

        let liab_values = self.get_value_of_shares(
            &clock,
            liab_shares,
            &BalanceSide::Liabilities,
            RequirementType::Maintenance,
        )?;

        let (asset_banks, liab_banks) =
            match self.find_liquidation_bank_candidates(deposit_values, liab_values)? {
                Some(banks) => banks,
                None => return Ok(None),
            };

        let banks_to_include: Vec<Pubkey> = vec![];
        let banks_to_exclude: Vec<Pubkey> = vec![];
        let observation_accounts = MarginfiAccountWrapper::get_observation_accounts::<OracleWrapper>(
            &account.account.lending_account,
            &banks_to_include,
            &banks_to_exclude,
            &self.cache,
            &clock,
        )?;

        // Calculated max liquidatable amount is the defining factor for liquidation.
        let Some(Sizing {
            asset_bank: asset_bank_pk,
            liab_bank: liab_bank_pk,
            amounts: liquidation_amounts,
            maintenance_health,
        }) = self.compute_max_liquidatable_amounts_with_banks(
            &self.cache,
            clock,
            account,
            &self.cache.marginfi_group,
            observation_accounts.observation_accounts.as_slice(),
            &asset_banks,
            &liab_banks,
        )?
        else {
            return Ok(None);
        };

        if !liquidation_amounts.is_liquidatable() {
            return Ok(Some(Evaluation::Unhealthy(UnhealthyAccount {
                address: account.address,
                observation_accounts,
                taggable: liquidation_amounts.taggable,
                tagged_at: account.account.liquidation_tagged_at,
                maintenance_health,
                asset_bank: asset_bank_pk,
                liab_bank: liab_bank_pk,
                max_repay: liquidation_amounts.max_repay,
                profit: liquidation_amounts.liquidator_profit * LIQUIDATION_SIZE_HAIRCUT,
                premium: liquidation_amounts.premium,
                balances: account.get_deposits_and_liabilities_shares(),
            })));
        }

        let slippage_adjusted_asset_amount =
            liquidation_amounts.max_liquidatable_asset_amount * LIQUIDATION_SIZE_HAIRCUT;
        let slippage_adjusted_liab_amount =
            liquidation_amounts.max_liquidatable_liab_amount * LIQUIDATION_SIZE_HAIRCUT;

        debug!(
                "asset_amount_to_liquidate: {:?}, slippage_adjusted_asset_amount: {:?}, slippage_adjusted_liab_amount: {:?}",
                liquidation_amounts.max_liquidatable_asset_amount, slippage_adjusted_asset_amount, slippage_adjusted_liab_amount
            );

        Ok(Some(Evaluation::Liquidate(PreparedLiquidatableAccount {
            liquidatee_account: account.clone(),
            observation_accounts,
            asset_bank: asset_bank_pk,
            liab_bank: liab_bank_pk,
            asset_amount: slippage_adjusted_asset_amount,
            liab_amount: slippage_adjusted_liab_amount,
            profit: liquidation_amounts.liquidator_profit * LIQUIDATION_SIZE_HAIRCUT,
        })))
    }

    fn find_liquidation_bank_candidates(
        &self,
        deposit_values: Vec<(I80F48, Pubkey)>,
        liab_values: Vec<(I80F48, Pubkey)>,
    ) -> Result<Option<(Vec<Pubkey>, Vec<Pubkey>)>> {
        if deposit_values.is_empty() || liab_values.is_empty() {
            debug!(
                "No bank candidates: {} priced deposit(s), {} priced liability(-ies)",
                deposit_values.len(),
                liab_values.len()
            );
            return Ok(None);
        }

        let total_deposit_value = deposit_values
            .iter()
            .map(|(v, _)| v.to_num::<f64>())
            .sum::<f64>();
        if total_deposit_value < BANKRUPT_THRESHOLD {
            debug!(
                "No bank candidates: deposits worth ${total_deposit_value:.6} are below the \
                 bankruptcy threshold ${BANKRUPT_THRESHOLD}"
            );
            return Ok(None);
        }

        // Every deposit we can seize and every liability we can buy, so an unsupported or excluded
        // position doesn't block an otherwise liquidatable account.
        let usable = |bank_pk: &Pubkey, collateral: bool| {
            self.cache.banks.try_get_bank(bank_pk).is_ok_and(|bank| {
                !self.cache.mints.is_excluded(&bank.bank.mint)
                    && (!collateral || is_seizable_asset_tag(bank.bank.config.asset_tag))
            })
        };
        let asset_banks: Vec<Pubkey> = deposit_values
            .iter()
            .filter(|(_, bank_pk)| usable(bank_pk, true))
            .map(|(_, bank_pk)| *bank_pk)
            .collect();
        let liab_banks: Vec<Pubkey> = liab_values
            .iter()
            .filter(|(_, bank_pk)| usable(bank_pk, false))
            .map(|(_, bank_pk)| *bank_pk)
            .collect();

        match (asset_banks.is_empty(), liab_banks.is_empty()) {
            (false, false) => Ok(Some((asset_banks, liab_banks))),
            _ => {
                debug!(
                    "No bank candidates: no seizable deposit or no buyable liability \
                     (unsupported asset tag or excluded mint)"
                );
                Ok(None)
            }
        }
    }

    fn compute_max_liquidatable_amounts_with_banks(
        &self,
        cache: &Cache,
        clock: Clock,
        account: &MarginfiAccountWrapper,
        group: &MarginfiGroup,
        banks_and_oracles: &[Pubkey],
        asset_banks: &[Pubkey],
        liab_banks: &[Pubkey],
    ) -> Result<Option<Sizing>> {
        let mut accounts: Vec<_> = vec![];
        let mut bank_to_emode_config = HashMap::<Pubkey, EmodeConfig>::new();
        for pk in banks_and_oracles.iter() {
            let bank_account = cache.banks.try_get_bank(pk);
            let account = match bank_account {
                Ok(wrapper) => {
                    bank_to_emode_config.insert(*pk, wrapper.bank.emode.emode_config);
                    wrapper.account
                }
                Err(_) => cache.oracles.try_get_account(pk)?,
            };
            accounts.push(account);
        }

        let remaining_ais: Vec<_> = accounts
            .iter_mut()
            .zip(banks_and_oracles.iter())
            .map(|(account, pk)| (pk, account).into_account_info())
            .collect();

        let (total_weighted_assets, total_weighted_liabilities) = get_health_components(
            &account.account,
            group,
            &remaining_ais,
            RequirementType::Maintenance,
            &mut None,
            HealthPriceMode::Client(clock.clone()),
            &mut None,
        )?;

        let emode_configs = account
            .account
            .lending_account
            .get_active_balances_iter()
            .filter(|b| !b.is_empty(BalanceSide::Liabilities))
            .map(|b| {
                bank_to_emode_config
                    .remove(&b.bank_pk)
                    .expect("every liability bank is in the observation accounts")
            });
        let reconciled_emode_config =
            reconcile_emode_configs(emode_configs, RequirementType::Maintenance);

        let maintenance_health = total_weighted_assets - total_weighted_liabilities;
        debug!(
            "Account {} maintenance_health = {:?} (assets {:?}, liabilities {:?})",
            account.address, maintenance_health, total_weighted_assets, total_weighted_liabilities
        );
        if maintenance_health >= I80F48::ZERO {
            debug!(
                "Account {} is healthy: maintenance health ${:.6}",
                account.address,
                maintenance_health.to_num::<f64>()
            );
            return Ok(None);
        }

        let (total_assets_equity, total_liabs_equity) = get_health_components(
            &account.account,
            group,
            &remaining_ais,
            RequirementType::Equity,
            &mut None,
            HealthPriceMode::Client(clock.clone()),
            &mut None,
        )?;

        // Which collateral is seized and which debt is repaid decides how much the program lets a
        // liquidation take, so every pair is sized and the most profitable one wins.
        let mut best: Option<(Pubkey, Pubkey, LiquidationAmounts)> = None;
        let mut any_taggable = false;
        for asset_bank in asset_banks {
            for liab_bank in liab_banks {
                let amounts = cache
                    .banks
                    .try_get_bank(asset_bank)
                    .and_then(|asset_bank_wrapper| {
                        let liab_bank_wrapper = cache.banks.try_get_bank(liab_bank)?;
                        self.size_pair(
                            cache,
                            clock.clone(),
                            account,
                            &asset_bank_wrapper,
                            &liab_bank_wrapper,
                            &reconciled_emode_config,
                            maintenance_health,
                            total_assets_equity,
                            total_liabs_equity,
                        )
                    });
                let amounts = match amounts {
                    Ok(amounts) => amounts,
                    Err(e) => {
                        debug!(
                            "Account {}: can't size seizing {} for {}: {}",
                            account.address, asset_bank, liab_bank, e
                        );
                        continue;
                    }
                };
                any_taggable |= amounts.taggable;
                if best
                    .as_ref()
                    .is_none_or(|(_, _, b)| amounts.liquidator_profit > b.liquidator_profit)
                {
                    best = Some((*asset_bank, *liab_bank, amounts));
                }
            }
        }
        // Tagging helps every pair, so it is worth it when any pair would profit from it.
        Ok(best.map(|(asset_bank, liab_bank, amounts)| Sizing {
            asset_bank,
            liab_bank,
            amounts: LiquidationAmounts {
                taggable: any_taggable,
                ..amounts
            },
            maintenance_health,
        }))
    }

    /// How much seizing `asset_bank_wrapper` for `liab_bank_wrapper` may repay and earn, as
    /// `end_liquidation` measures it.
    #[allow(clippy::too_many_arguments)]
    fn size_pair(
        &self,
        cache: &Cache,
        clock: Clock,
        account: &MarginfiAccountWrapper,
        asset_bank_wrapper: &BankWrapper,
        liab_bank_wrapper: &BankWrapper,
        reconciled_emode_config: &ReconciledEmodeConfig,
        maintenance_health: I80F48,
        total_assets_equity: I80F48,
        total_liabs_equity: I80F48,
    ) -> Result<LiquidationAmounts> {
        let in_bad_debt = total_assets_equity < total_liabs_equity;
        let asset_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &asset_bank_wrapper.address)?;

        let liab_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &liab_bank_wrapper.address)?;

        let asset_weight_maint: I80F48 = asset_bank_wrapper
            .bank
            .get_asset_weight(RequirementType::Maintenance, reconciled_emode_config);
        let liab_weight_maint: I80F48 = liab_bank_wrapper.bank.config.liability_weight_maint.into();

        // `end_liquidation` holds the premium at its base for e-mode-boosted accounts and for
        // accounts already in bad debt; otherwise a tag grows it over time.
        let own_asset_weight_maint: I80F48 =
            asset_bank_wrapper.bank.config.asset_weight_maint.into();
        let emode_boosted = account.account.health_cache.is_emode_boosted()
            || asset_weight_maint > own_asset_weight_maint;
        let tagged_at = account.account.liquidation_tagged_at;
        let base_premium = I80F48::max(cache.liquidation_max_fee, LIQUIDATION_BONUS_FEE_MINIMUM);
        let premium = if emode_boosted || in_bad_debt {
            base_premium
        } else {
            tag_adjusted_premium(base_premium, tagged_at, clock.unix_timestamp)
        };
        let premium_taken = I80F48::max(I80F48::ZERO, premium - PREMIUM_SAFETY_MARGIN);

        let (asset_amount, _) = account.get_balance_for_bank(asset_bank_wrapper, &clock)?;
        let (_, liab_amount) = account.get_balance_for_bank(liab_bank_wrapper, &clock)?;

        let asset_value = asset_bank_wrapper.calc_value(
            &asset_oracle_wrapper,
            asset_amount,
            BalanceSide::Assets,
            RequirementType::Equity,
        )?;

        let liab_value = liab_bank_wrapper.calc_value(
            &liab_oracle_wrapper,
            liab_amount,
            BalanceSide::Liabilities,
            RequirementType::Equity,
        )?;

        let bounds = RepayBounds {
            liab_value,
            asset_value,
            equity: total_assets_equity - total_liabs_equity,
            maintenance_health,
            asset_weight_maint,
            liab_weight_maint,
            asset_price_ratio: maintenance_to_equity_price(
                asset_bank_wrapper,
                &asset_oracle_wrapper,
                || BalanceSide::Assets,
            )?,
            liab_price_ratio: maintenance_to_equity_price(
                liab_bank_wrapper,
                &liab_oracle_wrapper,
                || BalanceSide::Liabilities,
            )?,
            above_closeout: total_assets_equity >= LIQUIDATION_CLOSEOUT_DOLLAR_THRESHOLD,
        };
        let max_repay = bounds.max_repay(premium_taken);
        let one_plus_premium = I80F48::ONE + premium_taken;
        let liquidator_profit = max_repay * premium_taken;
        // Tag only when the premium a tag can grow to would make the liquidation worth it.
        let taggable = tagged_at == 0 && !in_bad_debt && !emode_boosted && {
            let max_premium = LIQUIDATION_TAG_MAX_PREMIUM - PREMIUM_SAFETY_MARGIN;
            bounds.max_repay(max_premium) * max_premium * LIQUIDATION_SIZE_HAIRCUT > self.min_profit
        };

        if liquidator_profit * LIQUIDATION_SIZE_HAIRCUT <= self.min_profit {
            debug!(
                "Account {} profit ${:.6} <= min profit ${} at premium {:.4} (tagged_at {}, taggable {})",
                account.address,
                liquidator_profit.to_num::<f64>(),
                self.min_profit,
                premium_taken.to_num::<f64>(),
                tagged_at,
                taggable
            );
            return Ok(LiquidationAmounts {
                max_repay,
                premium: premium_taken,
                liquidator_profit,
                taggable,
                ..LiquidationAmounts::none()
            });
        }

        let max_liquidatable_asset_amount = asset_bank_wrapper.calc_amount(
            &asset_oracle_wrapper,
            max_repay * one_plus_premium,
            BalanceSide::Assets,
            RequirementType::Equity,
        )?;

        let max_liquidatable_liab_amount = liab_bank_wrapper.calc_amount(
            &liab_oracle_wrapper,
            max_repay,
            BalanceSide::Liabilities,
            RequirementType::Equity,
        )?;

        debug!("Account {:?} liquidability evaluation:\nMaintenance health {:?}\n\
            Equity assets {:?}\nEquity liabilities {:?}\nPremium {:?} (base {:?}, tagged_at {}, emode boosted {})\n\
            Asset Bank {:?}\nAsset maint weight: {:?}\nAsset Amount {:?}\nAsset Value (USD) {:?}\n\
            Liab Bank {:?}\nLiab maint weight: {:?}\nLiab Amount {:?}\nLiab Value (USD) {:?}\n\
            Max Repay (USD) {:?}\nMax Liquidatable Asset Amount {:?}\nMax Liquidatable Liab Amount {:?}\nLiquidator profit (USD) {:?}",
            account.address, maintenance_health,
            total_assets_equity, total_liabs_equity, premium_taken, base_premium, tagged_at, emode_boosted,
            asset_bank_wrapper.address, asset_weight_maint, asset_amount, asset_value,
            liab_bank_wrapper.address, liab_weight_maint, liab_amount, liab_value,
            max_repay, max_liquidatable_asset_amount, max_liquidatable_liab_amount, liquidator_profit);

        Ok(LiquidationAmounts {
            max_liquidatable_asset_amount,
            max_liquidatable_liab_amount,
            max_repay,
            premium: premium_taken,
            liquidator_profit,
            taggable,
        })
    }

    fn get_value_of_shares(
        &self,
        clock: &Clock,
        shares: Vec<(I80F48, Pubkey)>,
        balance_side: &BalanceSide,
        requirement_type: RequirementType,
    ) -> Result<Vec<(I80F48, Pubkey)>> {
        let mut values: Vec<(I80F48, Pubkey)> = Vec::new();

        for (shares_amount, bank_pk) in shares {
            let bank_wrapper = self.cache.banks.try_get_bank(&bank_pk)?;
            let oracle_wrapper = OracleWrapper::build(&self.cache, clock, &bank_pk)?;

            if !matches!(
                bank_wrapper.bank.config.operational_state,
                BankOperationalState::Operational | BankOperationalState::ReduceOnly
            ) {
                continue;
            }

            if !bank_covers_its_borrows(&bank_wrapper.bank)? {
                debug!("Skipping bankrupt bank from evaluation: {}", bank_pk);
                continue;
            }

            let value = match balance_side {
                BalanceSide::Liabilities => {
                    let liabilities = bank_wrapper
                        .bank
                        .get_liability_amount(shares_amount)
                        .map_err(|e| anyhow!("Couldn't calculate liability amount for: {}", e))?;
                    bank_wrapper.calc_value(
                        &oracle_wrapper,
                        liabilities,
                        BalanceSide::Liabilities,
                        requirement_type,
                    )?
                }
                BalanceSide::Assets => {
                    let assets = bank_wrapper
                        .bank
                        .get_asset_amount(shares_amount)
                        .map_err(|e| anyhow!("Couldn't calculate asset amount for: {}", e))?;
                    bank_wrapper.calc_value(
                        &oracle_wrapper,
                        assets,
                        BalanceSide::Assets,
                        requirement_type,
                    )?
                }
            };

            values.push((value, bank_pk));
        }

        Ok(values)
    }
}

#[cfg(test)]
mod repay_bounds_tests {
    use super::*;

    /// The 0.1.12 patch notes' example: $100 SOL at maintenance weight 0.8, $85 USDC debt.
    fn example() -> RepayBounds {
        RepayBounds {
            liab_value: I80F48!(85),
            asset_value: I80F48!(100),
            equity: I80F48!(15),
            maintenance_health: I80F48!(-5),
            asset_weight_maint: I80F48!(0.8),
            liab_weight_maint: I80F48!(1),
            asset_price_ratio: I80F48::ONE,
            liab_price_ratio: I80F48::ONE,
            above_closeout: true,
        }
    }

    #[test]
    fn collateral_twap_above_spot_shrinks_the_stay_unhealthy_bound() {
        // SOL falling: its time-weighted price sits 1% above spot, so each equity dollar seized
        // removes only 0.99 health dollars: 1 - 0.8 * 1.05 * 0.99 = 0.1684 health per dollar.
        let bounds = RepayBounds {
            asset_price_ratio: I80F48!(0.99),
            ..example()
        };
        let repay = bounds.max_repay(I80F48!(0.05));
        assert!((repay - I80F48!(29.6912)).abs() < I80F48!(0.001));
    }

    #[test]
    fn base_premium_is_held_by_the_stay_unhealthy_bound() {
        // Each dollar repaid at 5% gains 1 - 0.8 * 1.05 = 0.16 health: 5 / 0.16.
        let repay = example().max_repay(I80F48!(0.05));
        assert!((repay - I80F48!(31.25)).abs() < I80F48!(0.0001));
    }

    #[test]
    fn max_premium_is_held_by_solvency() {
        // At 100% repaying lowers health, so only "stay solvent" binds: $15 equity / 1.0.
        assert_eq!(example().max_repay(I80F48!(1)), I80F48!(15));
    }

    #[test]
    fn bad_debt_has_no_solvency_bound() {
        let bounds = RepayBounds {
            equity: I80F48!(-5),
            ..example()
        };
        assert_eq!(bounds.max_repay(I80F48!(1)), I80F48!(50));
    }
}
