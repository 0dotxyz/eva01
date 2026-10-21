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
        reconcile_emode_configs, BalanceSide, BankOperationalState, EmodeConfig, HealthPriceMode,
        MarginfiAccount, MarginfiGroup, OracleSetup, RequirementType,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
    sync::{atomic::AtomicBool, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

/// Max liquidations executed concurrently. Bounds in-flight RPC/sim/bundle load while still
/// letting a burst of liquidatable accounts (e.g. a price crash) be worked in parallel rather
/// than one-at-a-time.
const MAX_CONCURRENT_LIQUIDATIONS: usize = 8;

/// How often the rebalancing sweep runs when Geyser has delivered no updates at all.
const IDLE_REBALANCE_INTERVAL: Duration = Duration::from_secs(60);

/// Taken off the premium `end_liquidation` allows, so price moves between evaluation and landing
/// cannot push the seizure over the cap.
const PREMIUM_SAFETY_MARGIN: I80F48 = I80F48!(0.005);

/// Minimum time between tag attempts on the same account, so a failing tag is not resent on every
/// update.
const TAG_RETRY_INTERVAL: Duration = Duration::from_secs(600);

pub struct Liquidator {
    rebalancer: Rebalancer,
    executor: Executor,
    strategy: InventoryStrategy,
    liquidator_account: Arc<LiquidatorAccount>,
    tag_attempts: Mutex<HashMap<Pubkey, Instant>>,
    min_profit: f64,
    geyser_rx: Receiver<GeyserUpdate>,
    stop_liquidator: Arc<AtomicBool>,
    cache: Arc<Cache>,
}

#[derive(Debug, Clone, Copy)]
struct LiquidationAmounts {
    max_liquidatable_asset_amount: I80F48,
    max_liquidatable_liab_amount: I80F48,
    liquidator_profit: I80F48,
    /// Unhealthy, untagged, solvent and not e-mode boosted: tagging would let the premium grow.
    taggable: bool,
}

impl LiquidationAmounts {
    fn none() -> Self {
        Self {
            max_liquidatable_asset_amount: I80F48::ZERO,
            max_liquidatable_liab_amount: I80F48::ZERO,
            liquidator_profit: I80F48::ZERO,
            taggable: false,
        }
    }

    fn is_liquidatable(&self) -> bool {
        !self.max_liquidatable_asset_amount.is_zero()
    }
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
    above_closeout: bool,
}

impl RepayBounds {
    /// Largest repayment (USD) `end_liquidation` accepts when seizing `repaid * (1 + premium)`:
    /// * the seizure fits the collateral and the repayment fits the debt;
    /// * an account that started solvent must stay solvent, i.e. the premium taken cannot exceed
    ///   its equity;
    /// * above the closeout threshold the account must stay unhealthy. Each dollar repaid raises
    ///   maintenance health by `l - w * (1 + premium)`; once that is no longer positive the
    ///   liquidation cannot heal the account and this bound no longer applies.
    fn max_repay(&self, premium: I80F48) -> I80F48 {
        let one_plus_premium = I80F48::ONE + premium;
        let mut max_repay = min(self.liab_value, self.asset_value / one_plus_premium);
        if self.equity >= I80F48::ZERO && premium > I80F48::ZERO {
            max_repay = min(max_repay, self.equity / premium);
        }
        let health_gain_per_repaid =
            self.liab_weight_maint - self.asset_weight_maint * one_plus_premium;
        if self.above_closeout && health_gain_per_repaid > I80F48::ZERO {
            max_repay = min(max_repay, -self.maintenance_health / health_gain_per_repaid);
        }
        I80F48::max(I80F48::ZERO, max_repay)
    }
}

#[allow(clippy::large_enum_variant)]
enum Evaluation {
    Liquidate(PreparedLiquidatableAccount),
    /// Unhealthy but not worth liquidating yet: tag it so the allowed premium grows.
    Tag(TagIntent),
}

struct TagIntent {
    address: Pubkey,
    observation_accounts: ObservationAccounts,
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
            executor,
            strategy,
            liquidator_account,
            tag_attempts: Mutex::new(HashMap::new()),
            min_profit: config.min_profit,
            geyser_rx,
            stop_liquidator,
            cache,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.rebalancer.run()?;

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
            let has_updates = !accounts_to_check.is_empty();
            if has_updates {
                self.run_liquidation_cycle(accounts_to_check);
            }

            // Sell-only sweep: convert any seized collateral / JIT-buy overshoot back to USDC.
            // Idle cycles only sweep on a heartbeat — nothing can have changed since the last one.
            if has_updates || last_rebalance.elapsed() >= IDLE_REBALANCE_INTERVAL {
                last_rebalance = Instant::now();
                if let Err(error) = self.rebalancer.run() {
                    error!("Rebalancing failed: {:?}", error);
                }
            }
        }
        info!("The Liquidator loop is stopped.");

        Ok(())
    }

    fn run_liquidation_cycle(&mut self, accounts_to_check: Vec<Pubkey>) {
        info!(
            "Running the Liquidation process for {} account(s)...",
            accounts_to_check.len()
        );

        let checked_accounts = self.check_accounts(accounts_to_check);

        match checked_accounts {
            Ok((mut accounts, tags)) => {
                // Rank highest-profit first.
                accounts.sort_by(|a, b| a.profit.cmp(&b.profit));
                accounts.reverse();

                // Drop below-min-profit intents up front so the parallel section only runs
                // real work.
                let intents: Vec<PreparedLiquidatableAccount> = accounts
                    .into_iter()
                    .filter(|acc| {
                        let worth_it = acc.profit >= I80F48::from_num(self.min_profit);
                        if !worth_it {
                            info!(
                                "Skipping {}: est profit ${:.6} < min ${}",
                                acc.liquidatee_account.address,
                                acc.profit.to_num::<f64>(),
                                self.min_profit
                            );
                        }
                        worth_it
                    })
                    .collect();

                // Hand ranked intents to the execution layer concurrently, bounded to
                // MAX_CONCURRENT_LIQUIDATIONS in-flight. Each try_execute sells the collateral
                // for the liability in-tx (or JIT-buys the liability), simulates first, and lands
                // a Jito bundle (with a sequential RPC fallback). Workers pull the next-highest-
                // profit intent from the shared queue, so a burst of liquidatable accounts is
                // worked in parallel.
                if !intents.is_empty() {
                    let worker_count = MAX_CONCURRENT_LIQUIDATIONS.min(intents.len());
                    let queue = Mutex::new(intents.into_iter());
                    let executor = &self.executor;
                    let strategy = &self.strategy;
                    thread::scope(|scope| {
                        for _ in 0..worker_count {
                            scope.spawn(|| loop {
                                let next = queue.lock().unwrap().next();
                                let Some(acc) = next else { break };
                                let liquidatee = acc.liquidatee_account.address;
                                if let Err(e) = executor.try_execute(strategy, &acc) {
                                    error!(
                                        "Failed to execute liquidation for {:?}: {:?}",
                                        liquidatee, e
                                    );
                                }
                            });
                        }
                    });
                }

                self.tag_accounts(tags);
            }
            Err(err) => {
                error!("Failed to evaluate accounts in liquidation: {}", err);
            }
        }

        info!("The Liquidation process is complete.");
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

    fn check_accounts(
        &self,
        account_addresses: Vec<Pubkey>,
    ) -> Result<(Vec<PreparedLiquidatableAccount>, Vec<TagIntent>)> {
        let clock = clock_manager::get_clock(&self.cache.clock)?;

        // Evaluation is CPU-bound (oracle parsing + health math) and independent per account, so a
        // full scan of every borrower parallelises cleanly across cores.
        let evaluations: Vec<Evaluation> = account_addresses
            .into_par_iter()
            .filter_map(|account_address| {
                let account = self
                    .cache
                    .marginfi_accounts
                    .try_get_account(&account_address)
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
        let mut tags = Vec::new();
        for evaluation in evaluations {
            match evaluation {
                Evaluation::Liquidate(acc) => liquidations.push(acc),
                Evaluation::Tag(tag) => tags.push(tag),
            }
        }
        Ok((liquidations, tags))
    }

    /// Tags accounts that are unhealthy but not yet worth liquidating, so the receivership premium
    /// they pay grows until they are.
    fn tag_accounts(&self, tags: Vec<TagIntent>) {
        for tag in tags {
            {
                let mut attempts = self.tag_attempts.lock().unwrap();
                if attempts
                    .get(&tag.address)
                    .is_some_and(|at| at.elapsed() < TAG_RETRY_INTERVAL)
                {
                    continue;
                }
                attempts.insert(tag.address, Instant::now());
            }
            match self
                .liquidator_account
                .send_tag_tx(tag.address, &tag.observation_accounts)
            {
                Ok(signature) => info!(
                    "Tagged {} so its liquidation premium grows: {}",
                    tag.address, signature
                ),
                Err(err) => error!("Failed to tag {}: {:?}", tag.address, err),
            }
        }
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

        let (asset_bank_pk, liab_bank_pk) =
            match self.find_liquidation_bank_candidates(deposit_values, liab_values)? {
                Some(banks) => banks,
                None => return Ok(None),
            };

        let asset_bank_wrapper = self.cache.banks.try_get_bank(&asset_bank_pk)?;
        let liab_bank_wrapper = self.cache.banks.try_get_bank(&liab_bank_pk)?;

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
        let liquidation_amounts = self.compute_max_liquidatable_amounts_with_banks(
            &self.cache,
            clock,
            account,
            &self.cache.marginfi_group,
            observation_accounts.observation_accounts.as_slice(),
            &asset_bank_wrapper,
            &liab_bank_wrapper,
        )?;

        if !liquidation_amounts.is_liquidatable() {
            if liquidation_amounts.taggable {
                return Ok(Some(Evaluation::Tag(TagIntent {
                    address: account.address,
                    observation_accounts,
                })));
            }
            return Ok(None);
        }

        let slippage_adjusted_asset_amount =
            liquidation_amounts.max_liquidatable_asset_amount * I80F48!(0.90);
        let slippage_adjusted_liab_amount =
            liquidation_amounts.max_liquidatable_liab_amount * I80F48!(0.90);

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
            profit: liquidation_amounts.liquidator_profit * I80F48!(0.90),
        })))
    }

    fn find_liquidation_bank_candidates(
        &self,
        deposit_values: Vec<(I80F48, Pubkey)>,
        liab_values: Vec<(I80F48, Pubkey)>,
    ) -> Result<Option<(Pubkey, Pubkey)>> {
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

        // Largest deposit we can seize and largest liability we can buy, so an unsupported or
        // excluded position doesn't block an otherwise liquidatable account.
        let usable = |bank_pk: &Pubkey, collateral: bool| {
            self.cache.banks.try_get_bank(bank_pk).is_ok_and(|bank| {
                !self.cache.mints.is_excluded(&bank.bank.mint)
                    && (!collateral || is_seizable_asset_tag(bank.bank.config.asset_tag))
            })
        };
        let asset_bank = deposit_values
            .iter()
            .filter(|(_, bank_pk)| usable(bank_pk, true))
            .max_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, bank_pk)| *bank_pk);
        let liab_bank = liab_values
            .iter()
            .filter(|(_, bank_pk)| usable(bank_pk, false))
            .max_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, bank_pk)| *bank_pk);

        match (asset_bank, liab_bank) {
            (Some(asset_bank), Some(liab_bank)) => Ok(Some((asset_bank, liab_bank))),
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
        asset_bank_wrapper: &BankWrapper,
        liab_bank_wrapper: &BankWrapper,
    ) -> Result<LiquidationAmounts> {
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
            return Ok(LiquidationAmounts::none());
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
        let in_bad_debt = total_assets_equity < total_liabs_equity;

        let asset_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &asset_bank_wrapper.address)?;

        let liab_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &liab_bank_wrapper.address)?;

        let asset_weight_maint: I80F48 = asset_bank_wrapper
            .bank
            .get_asset_weight(RequirementType::Maintenance, &reconciled_emode_config);
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
            above_closeout: total_assets_equity >= LIQUIDATION_CLOSEOUT_DOLLAR_THRESHOLD,
        };
        let max_repay = bounds.max_repay(premium_taken);
        let one_plus_premium = I80F48::ONE + premium_taken;
        let liquidator_profit = max_repay * premium_taken;
        // Tag only when the premium a tag can grow to would make the liquidation worth it.
        let taggable = tagged_at == 0 && !in_bad_debt && !emode_boosted && {
            let max_premium = LIQUIDATION_TAG_MAX_PREMIUM - PREMIUM_SAFETY_MARGIN;
            bounds.max_repay(max_premium) * max_premium > self.min_profit
        };

        if liquidator_profit <= self.min_profit {
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

        debug!("Account {:?} liquidability evaluation:\nTotal weighted Assets {:?}\nTotal weighted Liabilities {:?}\nMaintenance health {:?}\n\
            Equity assets {:?}\nEquity liabilities {:?}\nPremium {:?} (base {:?}, tagged_at {}, emode boosted {})\n\
            Asset Bank {:?}\nAsset maint weight: {:?}\nAsset Amount {:?}\nAsset Value (USD) {:?}\n\
            Liab Bank {:?}\nLiab maint weight: {:?}\nLiab Amount {:?}\nLiab Value (USD) {:?}\n\
            Max Repay (USD) {:?}\nMax Liquidatable Asset Amount {:?}\nMax Liquidatable Liab Amount {:?}\nLiquidator profit (USD) {:?}",
            account.address, total_weighted_assets, total_weighted_liabilities, maintenance_health,
            total_assets_equity, total_liabs_equity, premium_taken, base_premium, tagged_at, emode_boosted,
            asset_bank_wrapper.address, asset_weight_maint, asset_amount, asset_value,
            liab_bank_wrapper.address, liab_weight_maint, liab_amount, liab_value,
            max_repay, max_liquidatable_asset_amount, max_liquidatable_liab_amount, liquidator_profit);

        Ok(LiquidationAmounts {
            max_liquidatable_asset_amount,
            max_liquidatable_liab_amount,
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

            if bank_wrapper.bank.check_utilization_ratio().is_err() {
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
            above_closeout: true,
        }
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
