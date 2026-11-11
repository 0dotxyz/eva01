use crate::{
    cache::Cache,
    clock_manager,
    config::Eva01Config,
    execution::{
        activity::LiquidationActivity,
        executor::{BundleLanding, Executor},
        inventory::InventoryStrategy,
        queue::{ExecutionQueue, QueuedIntent},
        reservations::TokenReservations,
    },
    geyser::{AccountType, GeyserUpdate},
    rebalancer::Rebalancer,
    tag_policy::{Balances, Candidate, TagPolicy},
    utils::{
        self,
        jito::{JitoClient, TipEstimator},
        log_genuine_error,
        pyth_push::PythPusher,
        rate_limiter::RateLimiter,
    },
    wrappers::{
        bank::BankWrapper,
        liquidator_account::{
            is_seizable_asset_tag, LiquidationPair, LiquidatorAccount, PreparedLiquidatableAccount,
        },
        marginfi_account::{MarginfiAccountWrapper, ObservationAccounts},
        oracle::{oracle_publish_time, OracleWrapper, OracleWrapperTrait},
    },
};
use anchor_lang::AccountDeserialize;
use anyhow::{anyhow, Result};
use crossbeam::channel::{bounded, Receiver, RecvTimeoutError};
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
    constants::{ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO, BANKRUPT_THRESHOLD},
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
    panic::{self, AssertUnwindSafe},
    sync::{atomic::AtomicBool, Arc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Max liquidations executed concurrently. Bounds in-flight RPC/simulation load while still
/// letting a burst of liquidatable accounts (e.g. a price crash) be worked in parallel rather
/// than one-at-a-time.
const MAX_CONCURRENT_LIQUIDATIONS: usize = 8;

/// How often the rebalancer checks whether a pass is due (see `LiquidationActivity`).
const REBALANCE_CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Taken off the premium `end_liquidation` allows, so price moves between evaluation and landing
/// cannot push the seizure over the cap.
const PREMIUM_SAFETY_MARGIN: I80F48 = I80F48!(0.005);

/// Liquidations are sized to this share of the most the program allows, leaving room for price
/// moves between evaluation and landing; profit is judged on the same share.
const LIQUIDATION_SIZE_HAIRCUT: I80F48 = I80F48!(0.90);

/// A pair that is cheaper to execute wins unless the heavier one earns more than this share more:
/// landing first matters far more than the last few cents of premium.
const PAIR_PROFIT_TIE_MARGIN: I80F48 = I80F48!(0.30);

/// How much heavier seizing from a bank makes the liquidation transaction. An integration bank
/// (Kamino, JupLend) withdraws through its venue, adding the accounts an in-tx swap would need to
/// fit under the account-lock limit.
fn seize_weight(asset_tag: u8) -> u8 {
    u8::from(matches!(asset_tag, ASSET_TAG_KAMINO | ASSET_TAG_JUPLEND))
}

/// Whether a pair earning `profit` and seizing at `weight` should replace the best so far: a
/// lighter pair wins unless the heavier one earns more than [`PAIR_PROFIT_TIE_MARGIN`] more.
fn is_better_pair(profit: I80F48, weight: u8, best_profit: I80F48, best_weight: u8) -> bool {
    let margin = I80F48::ONE + PAIR_PROFIT_TIE_MARGIN;
    match weight.cmp(&best_weight) {
        std::cmp::Ordering::Equal => profit > best_profit,
        std::cmp::Ordering::Less => profit * margin >= best_profit,
        std::cmp::Ordering::Greater => profit > best_profit * margin,
    }
}

/// Of pairs earning `profit` and seizing at `weight`, in order: the most profitable (the lighter on
/// a tie), and, when it is heavy, the pair an in-tx swap should take instead if that differs (a
/// lighter one unless the heavier earns more than [`PAIR_PROFIT_TIE_MARGIN`] more). Indexes into
/// the pairs; `None` when there are none.
fn choose_pairs(pairs: impl Iterator<Item = (I80F48, u8)>) -> Option<(usize, Option<usize>)> {
    let mut main: Option<(usize, I80F48, u8)> = None;
    let mut sandwich: Option<(usize, I80F48, u8)> = None;
    for (index, (profit, weight)) in pairs.enumerate() {
        if main.is_none_or(|(_, best_profit, best_weight)| {
            profit > best_profit || (profit == best_profit && weight < best_weight)
        }) {
            main = Some((index, profit, weight));
        }
        if sandwich.is_none_or(|(_, best_profit, best_weight)| {
            is_better_pair(profit, weight, best_profit, best_weight)
        }) {
            sandwich = Some((index, profit, weight));
        }
    }
    let (main, _, main_weight) = main?;
    let sandwich = sandwich
        .map(|(index, _, _)| index)
        .filter(|index| main_weight > 0 && *index != main);
    Some((main, sandwich))
}

/// How often each unhealthy account we are not liquidating is reported.
const SKIP_REPORT_INTERVAL: Duration = Duration::from_secs(1200);

/// After an attempt the same account is left alone this long. An attempt only ends once its
/// transaction is confirmed or can no longer land, and Geyser streams the account at processed
/// commitment, so the update is already on its way; this covers its delivery and the loop applying
/// it, so the next attempt is sized on the state the last one produced.
const ATTEMPT_COOLDOWN: Duration = Duration::from_secs(3);

pub struct Liquidator {
    /// Moved onto its own thread by [`Liquidator::start`].
    rebalancer: Option<Rebalancer>,
    executor: Arc<Executor>,
    strategy: Arc<InventoryStrategy>,
    evaluator: Arc<Evaluator>,
    queue: Arc<ExecutionQueue>,
    /// The liquidation attempts the rebalancer stays out of the way of.
    activity: Arc<LiquidationActivity>,
    threads: Vec<JoinHandle<()>>,
    liquidator_account: Arc<LiquidatorAccount>,
    tag_policy: TagPolicy,
    skip_reports: RateLimiter<Pubkey>,
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
    /// Liquidations earning more than this are left to another instance (a dust instance's cap).
    max_profit: Option<f64>,
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
    /// The most profitable pair.
    asset_bank: Pubkey,
    liab_bank: Pubkey,
    amounts: LiquidationAmounts,
    /// The pair an in-tx swap should take instead, when the most profitable one seizes from a
    /// Kamino or JupLend bank and a lighter pair earns nearly as much (`is_better_pair`).
    sandwich: Option<(Pubkey, Pubkey, LiquidationAmounts)>,
    maintenance_health: I80F48,
    /// The account is healthy on the Pyth prices the chain holds; only newer ones make it
    /// liquidatable.
    needs_price_push: bool,
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

impl Liquidator {
    pub fn new(
        config: Eva01Config,
        liquidator_account: Arc<LiquidatorAccount>,
        geyser_rx: Receiver<GeyserUpdate>,
        stop_liquidator: Arc<AtomicBool>,
        cache: Arc<Cache>,
    ) -> Result<Self> {
        // Shared by the liquidation plans and the rebalancer, so neither spends tokens the other
        // has claimed.
        let reservations = TokenReservations::new();
        let activity = Arc::new(LiquidationActivity::default());
        let rebalancer = Rebalancer::new(
            config.clone(),
            cache.clone(),
            reservations.clone(),
            activity.clone(),
        )?;
        let dex_client = rebalancer.liquidation_dex_client();
        let dex_gate = rebalancer.dex_gate();

        let rpc_client =
            RpcClient::new_with_commitment(config.rpc_url.clone(), CommitmentConfig::confirmed());
        let bundles = if config.price_push_enabled {
            Some(BundleLanding {
                pusher: PythPusher::new(&config, cache.clone())?,
                jito: JitoClient::new(
                    config.jito_block_engine_url.clone(),
                    config.bundle_api_key.clone(),
                ),
                tip_estimator: TipEstimator::new(
                    config.jito_tip_max_lamports,
                    config.bundle_tip_profit_share,
                ),
                signer: Keypair::try_from(config.wallet_keypair.as_slice())?,
            })
        } else {
            None
        };
        let executor = Executor::new(
            rpc_client,
            cache.clone(),
            bundles,
            I80F48::from_num(config.wallet_min_profit_share),
        );

        let strategy = InventoryStrategy::new(
            liquidator_account.clone(),
            dex_client,
            dex_gate,
            config.slippage_bps,
            config.min_profit,
            reservations,
        )?;

        Ok(Liquidator {
            rebalancer: Some(rebalancer),
            executor: Arc::new(executor),
            strategy: Arc::new(strategy),
            evaluator: Arc::new(Evaluator {
                cache: cache.clone(),
                min_profit: config.min_profit,
                max_profit: config.max_profit,
            }),
            queue: Arc::new(ExecutionQueue::new(ATTEMPT_COOLDOWN)),
            activity,
            threads: Vec::new(),
            liquidator_account,
            tag_policy: TagPolicy::new(),
            skip_reports: RateLimiter::new(SKIP_REPORT_INTERVAL),
            geyser_rx,
            stop_liquidator,
            cache,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        let mut rebalancer = self
            .rebalancer
            .take()
            .ok_or_else(|| anyhow!("the Liquidator was already started"))?;
        if rebalancer.run()?.trades > 0 {
            self.executor.refresh_wallet_balances();
        }
        self.spawn_threads(rebalancer)?;

        // Updates only reach the cache through this loop, so everything Geyser and Hermes sent
        // since the cache was loaded (the startup rebalance alone takes seconds) is still queued.
        // Apply it first, or the initial scan prices every account on the load-time snapshot.
        let mut backlog = HashSet::new();
        while let Ok(update) = self.geyser_rx.try_recv() {
            self.process_geyser_update(update, &mut backlog);
        }

        // Geyser only delivers changes, so an account that is already underwater at startup would
        // wait for an unrelated update to touch one of its banks. Sweep them all once up front.
        match self
            .cache
            .marginfi_accounts
            .try_get_accounts_with_liabilities()
        {
            Ok(accounts) => {
                info!(
                    "Running the initial Liquidation scan (updates queued during startup applied; {} account(s) touched)...",
                    backlog.len()
                );
                self.run_liquidation_cycle(accounts);
            }
            Err(error) => error!(
                "Failed to collect accounts for the initial scan: {:?}",
                error
            ),
        }

        info!("Staring the Liquidator loop.");
        while !self.stop_liquidator.load(Ordering::Relaxed) {
            debug!("Waiting for any data change...");
            let accounts_to_check = self.receive_geyser_updates()?;
            if !accounts_to_check.is_empty() {
                self.run_liquidation_cycle(accounts_to_check);
            }
        }

        // The workers exit once the queue stops; the rebalancer once the last worker is gone.
        info!("Stopping the liquidation workers and the rebalancer...");
        self.queue.stop();
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                error!("A liquidation worker or the rebalancer panicked.");
            }
        }
        info!("The Liquidator loop is stopped.");

        Ok(())
    }

    /// Starts the execution workers and the rebalancer. Both outlive the evaluation cycles, so a
    /// send waiting for confirmation or a sweep selling seized collateral never holds up the loop
    /// that applies Geyser updates.
    fn spawn_threads(&mut self, rebalancer: Rebalancer) -> Result<()> {
        // Nothing is ever sent: the channel disconnects once every worker holding a sender has
        // exited, which stops the rebalancer.
        let (workers_alive, workers_gone) = bounded::<()>(1);
        for index in 0..MAX_CONCURRENT_LIQUIDATIONS {
            let queue = Arc::clone(&self.queue);
            let evaluator = Arc::clone(&self.evaluator);
            let executor = Arc::clone(&self.executor);
            let strategy = Arc::clone(&self.strategy);
            let activity = Arc::clone(&self.activity);
            let alive = workers_alive.clone();
            self.threads.push(
                thread::Builder::new()
                    .name(format!("liquidation-worker-{index}"))
                    .spawn(move || {
                        let _alive = alive;
                        run_worker(&queue, &evaluator, &executor, &strategy, &activity)
                    })?,
            );
        }
        drop(workers_alive);

        let executor = Arc::clone(&self.executor);
        let activity = Arc::clone(&self.activity);
        self.threads.push(
            thread::Builder::new()
                .name("rebalancer".to_string())
                .spawn(move || run_rebalancer(rebalancer, &executor, &activity, &workers_gone))?,
        );
        Ok(())
    }

    fn run_liquidation_cycle(&mut self, accounts_to_check: Vec<Pubkey>) {
        let started = Instant::now();
        match self.evaluator.check_accounts(&accounts_to_check) {
            Ok((liquidations, unhealthy)) => {
                let liquidatable = liquidations.len();
                self.queue.offer(&accounts_to_check, liquidations);
                self.handle_unhealthy(&accounts_to_check, &unhealthy);
                info!(
                    "Evaluated {} account(s) in {} ms: {} queued for liquidation, {} unhealthy but not worth liquidating.",
                    accounts_to_check.len(),
                    started.elapsed().as_millis(),
                    liquidatable,
                    unhealthy.len()
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
            slot,
        } = update;

        debug!("Processing the {:?} {:?} update.", account_type, address);

        match account_type {
            AccountType::Oracle | AccountType::HermesPrice => {
                if matches!(account_type, AccountType::Oracle) {
                    self.cache.pyth_prices.record_onchain(address, &account);
                }
                let banks = self.cache.banks.get_banks_for_oracle(&address)?;
                self.advance_clock_to_oracle(&address, &account, &banks);
                // An older price than the one cached changes nothing to evaluate.
                if !self.cache.oracles.try_update(&address, account)? {
                    return Ok(());
                }

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
                self.cache
                    .tokens
                    .try_update_account(address, account, slot)?;
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

    /// Reports the unhealthy accounts we are not liquidating and tags the ones the tag policy says
    /// are due.
    fn handle_unhealthy(&mut self, checked: &[Pubkey], unhealthy: &[UnhealthyAccount]) {
        let now = Instant::now();
        self.skip_reports.prune(now);
        for account in unhealthy {
            if self.skip_reports.allow(account.address, now) {
                self.report_skipped(account);
            }
        }

        let candidates = unhealthy.iter().map(|account| Candidate {
            address: account.address,
            taggable: account.taggable,
            balances: &account.balances,
        });
        let due = self.tag_policy.observe(checked, candidates, now);
        if due.is_empty() {
            return;
        }
        let by_address: HashMap<Pubkey, &UnhealthyAccount> =
            unhealthy.iter().map(|a| (a.address, a)).collect();
        for (address, waited) in due {
            let observation_accounts = &by_address[&address].observation_accounts;
            match self
                .liquidator_account
                .send_tag_tx(address, observation_accounts)
            {
                Ok(signature) => info!(
                    "Sent a tag for {} after {} min unhealthy with nobody liquidating it, so its premium grows: {}",
                    address,
                    waited.as_secs() / 60,
                    signature
                ),
                Err(err) => error!("Failed to tag {}: {:?}", address, err),
            }
        }
    }

    fn report_skipped(&self, account: &UnhealthyAccount) {
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
}

/// Takes intents off the queue and lands them until the queue stops.
fn run_worker(
    queue: &ExecutionQueue,
    evaluator: &Evaluator,
    executor: &Executor,
    strategy: &InventoryStrategy,
    activity: &LiquidationActivity,
) {
    while let Some(queued) = queue.take() {
        let liquidatee = queued.intent.liquidatee_account.address;
        // A panic would otherwise end the worker and leave the account marked in flight for good.
        let attempt = panic::catch_unwind(AssertUnwindSafe(|| {
            attempt_liquidation(queued, evaluator, executor, strategy, activity)
        }));
        if attempt.is_err() {
            error!("The liquidation attempt on {} panicked.", liquidatee);
        }
        queue.release(liquidatee);
    }
}

fn attempt_liquidation(
    QueuedIntent { intent, queued_at }: QueuedIntent,
    evaluator: &Evaluator,
    executor: &Executor,
    strategy: &InventoryStrategy,
    activity: &LiquidationActivity,
) {
    let liquidatee = intent.liquidatee_account.address;
    // The intent was sized on the state at evaluation time. By now another liquidator may have
    // taken a chunk of the account or prices may have moved, so it is re-sized on the current
    // state and dropped if it is no longer worth it, instead of sending a transaction the program
    // would reject.
    match evaluator.re_evaluate(&liquidatee) {
        Ok(Some(fresh)) if executor.is_on_hold(&fresh).unwrap_or(false) => {
            debug!(
                "Skipping {}: it or its liability mint is backed off after a recent failure",
                liquidatee
            );
        }
        Ok(Some(fresh)) => {
            info!(
                "Liquidating {}: seizing {} for {} (est. profit ${:.4}, ${:.4} when queued {} ms ago){}",
                liquidatee,
                fresh.asset_bank,
                fresh.liab_bank,
                fresh.profit.to_num::<f64>(),
                intent.profit.to_num::<f64>(),
                queued_at.elapsed().as_millis(),
                if fresh.needs_price_push {
                    "; healthy on the chain's prices, so behind a price push"
                } else {
                    ""
                }
            );
            if let Some(pair) = &fresh.sandwich_pair {
                debug!(
                    "{}: a sandwich would seize {} for {} instead (est. profit ${:.4})",
                    liquidatee,
                    pair.asset_bank,
                    pair.liab_bank,
                    pair.profit.to_num::<f64>()
                );
            }
            let mut attempt = activity.attempt(liquidatee);
            match executor.try_execute(strategy, &fresh) {
                Ok(()) => attempt.succeeded(),
                Err(e) => error!(
                    "Failed to execute liquidation for {:?}: {:?}",
                    liquidatee, e
                ),
            }
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
}

/// Sells leftover tokens back to the swap mint and refills the inventory whenever a pass is due
/// (see `LiquidationActivity`), until every worker is gone.
fn run_rebalancer(
    mut rebalancer: Rebalancer,
    executor: &Executor,
    activity: &LiquidationActivity,
    workers_gone: &Receiver<()>,
) {
    // The startup pass just ran.
    let mut last_pass = Instant::now();
    while let Err(RecvTimeoutError::Timeout) = workers_gone.recv_timeout(REBALANCE_CHECK_INTERVAL) {
        if !activity.rebalance_due(last_pass, Instant::now()) {
            continue;
        }
        last_pass = Instant::now();
        match panic::catch_unwind(AssertUnwindSafe(|| rebalancer.run())) {
            // Liquidations size from the cached balances; the sold tokens stay claimed until the
            // balances show them gone.
            Ok(Ok(pass)) => {
                if pass.trades > 0 {
                    executor.refresh_wallet_balances();
                }
                drop(pass.claim);
            }
            Ok(Err(error)) => error!("Rebalancing failed: {:?}", error),
            Err(_) => {
                error!("Rebalancing panicked.");
                executor.refresh_wallet_balances();
            }
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
            sandwich,
            maintenance_health,
            needs_price_push,
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

        let profit = liquidation_amounts.liquidator_profit * LIQUIDATION_SIZE_HAIRCUT;
        if self
            .max_profit
            .is_some_and(|max| profit > I80F48::from_num(max))
        {
            debug!(
                "Account {}: ${:.4} profit is above this instance's ${:?} cap; left to another instance",
                account.address,
                profit.to_num::<f64>(),
                self.max_profit
            );
            return Ok(None);
        }

        let slippage_adjusted_asset_amount =
            liquidation_amounts.max_liquidatable_asset_amount * LIQUIDATION_SIZE_HAIRCUT;
        let slippage_adjusted_liab_amount =
            liquidation_amounts.max_liquidatable_liab_amount * LIQUIDATION_SIZE_HAIRCUT;

        debug!(
                "asset_amount_to_liquidate: {:?}, slippage_adjusted_asset_amount: {:?}, slippage_adjusted_liab_amount: {:?}",
                liquidation_amounts.max_liquidatable_asset_amount, slippage_adjusted_asset_amount, slippage_adjusted_liab_amount
            );

        // Sized like the main pair; a pair the program won't let liquidate is no alternative.
        let sandwich_pair = sandwich
            .filter(|(_, _, amounts)| amounts.is_liquidatable())
            .map(|(asset_bank, liab_bank, amounts)| LiquidationPair {
                asset_bank,
                liab_bank,
                asset_amount: amounts.max_liquidatable_asset_amount * LIQUIDATION_SIZE_HAIRCUT,
                liab_amount: amounts.max_liquidatable_liab_amount * LIQUIDATION_SIZE_HAIRCUT,
                profit: amounts.liquidator_profit * LIQUIDATION_SIZE_HAIRCUT,
            });

        Ok(Some(Evaluation::Liquidate(PreparedLiquidatableAccount {
            liquidatee_account: account.clone(),
            observation_accounts,
            asset_bank: asset_bank_pk,
            liab_bank: liab_bank_pk,
            asset_amount: slippage_adjusted_asset_amount,
            liab_amount: slippage_adjusted_liab_amount,
            profit,
            needs_price_push,
            sandwich_pair,
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
        let needs_price_push = match self.maintenance_health_on_chain_prices(
            cache,
            &clock,
            account,
            group,
            banks_and_oracles,
        ) {
            Ok(health) => health.is_some_and(|health| health >= I80F48::ZERO),
            // The chain's price is unusable as it stands (too old for the program, say): only a
            // pushed one can make this liquidation pass.
            Err(e) => {
                debug!(
                    "Account {}: can't price it on the chain's Pyth prices ({e}); a push is needed",
                    account.address
                );
                true
            }
        };

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
        // liquidation take, so every pair is sized. The most profitable one wins for a repay from
        // the wallet; an in-tx swap needs room for its route, so for it a lighter pair wins unless
        // the heavier earns much more.
        let mut candidates: Vec<(Pubkey, Pubkey, LiquidationAmounts, u8)> = Vec::new();
        let mut any_taggable = false;
        for asset_bank in asset_banks {
            let weight = cache
                .banks
                .try_get_bank(asset_bank)
                .map(|bank| seize_weight(bank.bank.config.asset_tag))
                .unwrap_or(0);
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
                candidates.push((*asset_bank, *liab_bank, amounts, weight));
            }
        }
        let Some((main, sandwich)) = choose_pairs(
            candidates
                .iter()
                .map(|(_, _, amounts, weight)| (amounts.liquidator_profit, *weight)),
        ) else {
            return Ok(None);
        };
        let (asset_bank, liab_bank, amounts, _) = candidates[main];
        let sandwich = sandwich.map(|index| {
            let (asset_bank, liab_bank, amounts, _) = candidates[index];
            (asset_bank, liab_bank, amounts)
        });
        // Tagging helps every pair, so it is worth it when any pair would profit from it.
        Ok(Some(Sizing {
            asset_bank,
            liab_bank,
            amounts: LiquidationAmounts {
                taggable: any_taggable,
                ..amounts
            },
            sandwich,
            maintenance_health,
            needs_price_push,
        }))
    }

    /// The account's maintenance health on the Pyth prices the chain holds, where eva prices any
    /// of its feeds ahead of them; `None` when every price it reads is the chain's already. This is
    /// what the program would compute without a push, so it tells a liquidation that needs one
    /// apart before anything is built.
    fn maintenance_health_on_chain_prices(
        &self,
        cache: &Cache,
        clock: &Clock,
        account: &MarginfiAccountWrapper,
        group: &MarginfiGroup,
        banks_and_oracles: &[Pubkey],
    ) -> Result<Option<I80F48>> {
        let mut behind = false;
        let mut accounts = Vec::with_capacity(banks_and_oracles.len());
        for pk in banks_and_oracles {
            let account = match cache.banks.try_get_bank(pk) {
                Ok(wrapper) => wrapper.account,
                Err(_) => {
                    let cached = cache.oracles.try_get_account(pk)?;
                    match cache.pyth_prices.onchain_if_behind(pk, &cached) {
                        Some(onchain) => {
                            behind = true;
                            onchain
                        }
                        None => cached,
                    }
                }
            };
            accounts.push(account);
        }
        if !behind {
            return Ok(None);
        }
        let remaining_ais: Vec<_> = accounts
            .iter_mut()
            .zip(banks_and_oracles.iter())
            .map(|(account, pk)| (pk, account).into_account_info())
            .collect();
        let (assets, liabilities) = get_health_components(
            &account.account,
            group,
            &remaining_ais,
            RequirementType::Maintenance,
            &mut None,
            HealthPriceMode::Client(clock.clone()),
            &mut None,
        )?;
        Ok(Some(assets - liabilities))
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

            if !bank_covers_its_borrows(&bank_wrapper.bank).unwrap_or(false) {
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

#[cfg(test)]
mod bank_solvency_tests {
    use super::*;

    fn bank(assets: i32, liabilities: i32) -> Bank {
        let mut bank: Bank = bytemuck::Zeroable::zeroed();
        bank.asset_share_value = I80F48::ONE.into();
        bank.liability_share_value = I80F48::ONE.into();
        bank.total_asset_shares = I80F48::from_num(assets).into();
        bank.total_liability_shares = I80F48::from_num(liabilities).into();
        bank
    }

    #[test]
    fn a_bank_whose_deposits_cover_its_borrows_is_solvent() {
        assert!(bank_covers_its_borrows(&bank(100, 90)).unwrap());
        assert!(bank_covers_its_borrows(&bank(100, 100)).unwrap());
    }

    #[test]
    fn a_bank_borrowed_beyond_its_deposits_is_not() {
        assert!(!bank_covers_its_borrows(&bank(100, 101)).unwrap());
    }

    #[test]
    fn share_values_are_applied() {
        let mut in_bad_debt = bank(100, 90);
        in_bad_debt.liability_share_value = I80F48::from_num(1.2).into();
        assert!(!bank_covers_its_borrows(&in_bad_debt).unwrap());
    }
}

#[cfg(test)]
mod pair_preference_tests {
    use super::*;

    #[test]
    fn a_lighter_pair_wins_within_the_margin() {
        // Plain SOL earning 20% less than Kamino SOL is still the better pair.
        assert!(is_better_pair(I80F48!(0.8), 0, I80F48!(1), 1));
        // Kamino SOL needs more than 30% more to beat plain SOL.
        assert!(!is_better_pair(I80F48!(1.25), 1, I80F48!(1), 0));
        assert!(is_better_pair(I80F48!(1.31), 1, I80F48!(1), 0));
    }

    #[test]
    fn a_much_better_heavier_pair_still_wins() {
        assert!(!is_better_pair(I80F48!(0.5), 0, I80F48!(1), 1));
    }

    #[test]
    fn equal_weights_compare_on_profit() {
        assert!(is_better_pair(I80F48!(1.001), 0, I80F48!(1), 0));
        assert!(!is_better_pair(I80F48!(1), 0, I80F48!(1), 0));
    }

    #[test]
    fn a_heavy_most_profitable_pair_gets_a_native_one_for_the_sandwich() {
        let pairs =
            |list: &[(f64, u8)]| choose_pairs(list.iter().map(|(p, w)| (I80F48::from_num(*p), *w)));
        // A Kamino pair earning 1.2 against a native one earning 1: the wallet repay takes the
        // Kamino one, the sandwich the native one (within the 30% margin).
        assert_eq!(pairs(&[(1.0, 0), (1.2, 1)]), Some((1, Some(0))));
        // A native pair is the most profitable: one pair suffices.
        assert_eq!(pairs(&[(1.2, 0), (1.0, 1)]), Some((0, None)));
        // A Kamino pair earning far more: the sandwich takes it too.
        assert_eq!(pairs(&[(1.0, 0), (2.0, 1)]), Some((1, None)));
        // Only heavy pairs: the sandwich has nothing lighter.
        assert_eq!(pairs(&[(1.0, 1), (1.1, 1)]), Some((1, None)));
        assert_eq!(pairs(&[]), None);
    }

    #[test]
    fn integration_banks_are_heavier() {
        assert_eq!(seize_weight(ASSET_TAG_KAMINO), 1);
        assert_eq!(seize_weight(ASSET_TAG_JUPLEND), 1);
        assert_eq!(seize_weight(0), 0);
    }
}
