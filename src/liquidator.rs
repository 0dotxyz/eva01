use crate::{
    cache::Cache,
    clock_manager,
    config::Eva01Config,
    execution::{executor::Executor, inventory::InventoryStrategy},
    geyser::{AccountType, GeyserUpdate},
    rebalancer::Rebalancer,
    utils::{
        jito::JitoClient,
        log_genuine_error,
        swb_cranker::{SwbCranker, SWB_STALE_HANDLED_ERROR, SWB_STALE_PRICE_ERROR_CODE_NUMBER},
    },
    wrappers::{
        bank::BankWrapper,
        liquidator_account::{LiquidatorAccount, PreparedLiquidatableAccount, PROFIT_SHARE},
        marginfi_account::MarginfiAccountWrapper,
        oracle::{OracleWrapper, OracleWrapperTrait},
    },
};
use anchor_lang::AccountDeserialize;
use anyhow::{anyhow, Result};
use crossbeam::channel::{Receiver, RecvTimeoutError};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{debug, error, info, warn};
use marginfi::state::{bank::BankImpl, marginfi_account::get_health_components};
use marginfi_type_crate::{
    constants::BANKRUPT_THRESHOLD,
    types::{
        reconcile_emode_configs, BalanceSide, Bank, BankOperationalState, EmodeConfig,
        HealthPriceMode, MarginfiAccount, MarginfiGroup, RequirementType,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use solana_program::pubkey::Pubkey;
use solana_sdk::{account_info::IntoAccountInfo, clock::Clock, signature::Keypair};
use std::sync::atomic::Ordering;
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc, Mutex, RwLock},
    thread,
    time::{Duration, Instant},
};

/// Max liquidations executed concurrently. Bounds in-flight RPC/sim/bundle load while still
/// letting a burst of liquidatable accounts (e.g. a price crash) be worked in parallel rather
/// than one-at-a-time.
const MAX_CONCURRENT_LIQUIDATIONS: usize = 8;

/// How often the feeds that are blocking accounts are reported.
const STALE_ORACLE_REPORT_INTERVAL: Duration = Duration::from_secs(300);

/// How often the rebalancing sweep runs when Geyser has delivered no updates at all.
const IDLE_REBALANCE_INTERVAL: Duration = Duration::from_secs(60);

pub struct Liquidator {
    rebalancer: Rebalancer,
    executor: Executor,
    strategy: InventoryStrategy,
    min_profit: f64,
    geyser_rx: Receiver<GeyserUpdate>,
    stop_liquidator: Arc<AtomicBool>,
    cache: Arc<Cache>,
    stale_reported_at: Instant,
}

#[derive(Debug, Clone, Copy)]
struct LiquidationAmounts {
    max_liquidatable_asset_amount: I80F48,
    max_liquidatable_liab_amount: I80F48,
    liquidator_profit: I80F48,
}

impl LiquidationAmounts {
    fn none() -> Self {
        Self {
            max_liquidatable_asset_amount: I80F48::ZERO,
            max_liquidatable_liab_amount: I80F48::ZERO,
            liquidator_profit: I80F48::ZERO,
        }
    }

    fn is_liquidatable(&self) -> bool {
        !self.max_liquidatable_asset_amount.is_zero()
    }
}

impl Liquidator {
    pub fn new(
        config: Eva01Config,
        liquidator_account: Arc<LiquidatorAccount>,
        geyser_rx: Receiver<GeyserUpdate>,
        stop_liquidator: Arc<AtomicBool>,
        cache: Arc<Cache>,
    ) -> Result<Self> {
        // The cranker is owned by the executor (for simulate-first cranking), behind an Arc.
        let swb_cranker = Arc::new(SwbCranker::new(&config, cache.as_ref())?);

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
            swb_cranker,
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
        )?;

        Ok(Liquidator {
            rebalancer,
            executor,
            strategy,
            min_profit: config.min_profit,
            geyser_rx,
            stop_liquidator,
            cache,
            stale_reported_at: Instant::now() - STALE_ORACLE_REPORT_INTERVAL,
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

        let account_count = accounts_to_check.len();
        let stale_swb_oracles: RwLock<HashSet<Pubkey>> = RwLock::new(HashSet::new());
        let checked_accounts = self.check_accounts(accounts_to_check, &stale_swb_oracles);
        self.report_stale_oracles(&stale_swb_oracles, account_count);

        match checked_accounts {
            Ok(mut accounts) => {
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
                // MAX_CONCURRENT_LIQUIDATIONS in-flight. Each try_execute JIT-buys the liability
                // shortfall, simulate-first cranks stale oracles, and lands a Jito bundle (with a
                // sequential RPC fallback). Workers pull the next-highest-profit intent from the
                // shared queue, so a burst of liquidatable accounts is worked in parallel.
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
                self.cache.oracles.try_update(&address, account)?;

                for bank in self.cache.banks.get_banks_for_oracle(&address)? {
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
                let mut data = account.data.as_slice();
                let bank = Bank::try_deserialize(&mut data)?;
                self.cache.banks.try_insert(address, bank, account)?;

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

    /// Switchboard feeds that neither the price fetcher nor Crossbar could price block every
    /// account touching them, silently. Surface them at most once per interval, since the same
    /// feeds recur on every cycle.
    fn report_stale_oracles(&mut self, stale: &RwLock<HashSet<Pubkey>>, accounts: usize) {
        let stale = match stale.read() {
            Ok(stale) if !stale.is_empty() => stale,
            _ => return,
        };

        if self.stale_reported_at.elapsed() < STALE_ORACLE_REPORT_INTERVAL {
            return;
        }
        self.stale_reported_at = Instant::now();

        warn!(
            "{} Switchboard oracle(s) have no usable price and block accounts from being \
             evaluated (of {} scanned): {:?}",
            stale.len(),
            accounts,
            stale.iter().collect::<Vec<_>>()
        );
    }

    fn check_accounts(
        &self,
        account_addresses: Vec<Pubkey>,
        stale_swb_oracles: &RwLock<HashSet<Pubkey>>,
    ) -> Result<Vec<PreparedLiquidatableAccount>> {
        let clock = clock_manager::get_clock(&self.cache.clock)?;

        // Evaluation is CPU-bound (oracle parsing + health math) and independent per account, so a
        // full scan of every borrower parallelises cleanly across cores.
        Ok(account_addresses
            .into_par_iter()
            .filter_map(|account_address| {
                let account = self
                    .cache
                    .marginfi_accounts
                    .try_get_account(&account_address)
                    .ok()?;

                match self.process_account(&account, clock.clone(), stale_swb_oracles) {
                    Ok(acc_opt) => acc_opt,
                    Err(err) => {
                        if !err.to_string().contains(SWB_STALE_HANDLED_ERROR) {
                            debug!("Failed to process account {:?}: {:?}", account.address, err);
                        }
                        None
                    }
                }
            })
            .collect())
    }

    fn process_account(
        &self,
        account: &MarginfiAccountWrapper,
        clock: Clock,
        stale_swb_oracles: &RwLock<HashSet<Pubkey>>,
    ) -> Result<Option<PreparedLiquidatableAccount>> {
        let (deposit_shares, liab_shares) = account.get_deposits_and_liabilities_shares();
        if deposit_shares.is_empty() || liab_shares.is_empty() {
            return Ok(None);
        }

        let deposit_values = self.get_value_of_shares(
            &clock,
            deposit_shares,
            &BalanceSide::Assets,
            RequirementType::Maintenance,
            stale_swb_oracles,
        )?;

        let liab_values = self.get_value_of_shares(
            &clock,
            liab_shares,
            &BalanceSide::Liabilities,
            RequirementType::Maintenance,
            stale_swb_oracles,
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

        Ok(Some(PreparedLiquidatableAccount {
            liquidatee_account: account.clone(),
            observation_accounts,
            asset_bank: asset_bank_pk,
            liab_bank: liab_bank_pk,
            asset_amount: slippage_adjusted_asset_amount,
            liab_amount: slippage_adjusted_liab_amount,
            profit: liquidation_amounts.liquidator_profit,
        }))
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

        let (_, asset_bank) = deposit_values
            .iter()
            .max_by(|a, b| {
                //debug!("Asset Bank {:?} value: {:?}", a.1, a.0);
                a.0.cmp(&b.0)
            })
            .ok_or_else(|| anyhow!("No asset bank found"))?;

        let (_, liab_bank) = liab_values
            .iter()
            .max_by(|a, b| {
                //debug!("Liab Bank {:?} value: {:?}", a.1, a.0);
                a.0.cmp(&b.0)
            })
            .ok_or_else(|| anyhow!("No liability bank found"))?;

        Ok(Some((*asset_bank, *liab_bank)))
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

        let asset_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &asset_bank_wrapper.address)?;

        let liab_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &liab_bank_wrapper.address)?;

        let asset_weight_maint: I80F48 = asset_bank_wrapper
            .bank
            .get_asset_weight(RequirementType::Maintenance, &reconciled_emode_config);
        let liab_weight_maint: I80F48 = liab_bank_wrapper.bank.config.liability_weight_maint.into();

        let liquidation_discount = fixed_macro::types::I80F48!(0.95);

        let all = asset_weight_maint - liab_weight_maint * liquidation_discount;

        if all >= I80F48::ZERO {
            debug!("Account {:?} has no liquidatable amount: {:?}, asset_weight_maint: {:?}, liab_weight_maint: {:?}", account.address, all, asset_weight_maint, liab_weight_maint);
            return Ok(LiquidationAmounts::none());
        }

        let underwater_value =
            maintenance_health / (asset_weight_maint - liab_weight_maint * liquidation_discount);

        let (asset_amount, _) = account.get_balance_for_bank(asset_bank_wrapper)?;
        let (_, liab_amount) = account.get_balance_for_bank(liab_bank_wrapper)?;

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

        let max_liquidatable_value = min(min(asset_value, liab_value), underwater_value);
        let liquidator_profit = max_liquidatable_value
            .checked_mul(I80F48::from_num(PROFIT_SHARE))
            .ok_or_else(|| anyhow!("Liquidator profit overflow"))?;

        if liquidator_profit <= self.min_profit {
            debug!(
                "Account {} profit ${:.6} <= min profit ${}",
                account.address,
                liquidator_profit.to_num::<f64>(),
                self.min_profit
            );
            return Ok(LiquidationAmounts::none());
        }

        let max_liquidatable_asset_amount = asset_bank_wrapper.calc_amount(
            &asset_oracle_wrapper,
            max_liquidatable_value,
            BalanceSide::Assets,
            RequirementType::Equity,
        )?;

        let max_liquidatable_liab_amount = liab_bank_wrapper.calc_amount(
            &liab_oracle_wrapper,
            max_liquidatable_value,
            BalanceSide::Liabilities,
            RequirementType::Equity,
        )?;

        debug!("Account {:?} liquidability evaluation:\nTotal weighted Assets {:?}\nTotal weighted Liabilities {:?}\nMaintenance health {:?}\n\
            Asset Bank {:?}\nAsset maint weight: {:?}\nAsset Amount {:?}\nAsset Value (USD) {:?}\n\
            Liab Bank {:?}\nLiab maint weight: {:?}\nLiab Amount {:?}\nLiab Value (USD) {:?}\n\
            Max Liquidatable Value {:?}\nMax Liquidatable Asset Amount {:?}\nMax Liquidatable Liab Amount {:?}\nLiquidator profit (USD) {:?}", 
            account.address, total_weighted_assets, total_weighted_liabilities, maintenance_health,
            asset_bank_wrapper.address, asset_bank_wrapper.bank.config.asset_weight_maint, asset_amount, asset_value,
            liab_bank_wrapper.address, liab_bank_wrapper.bank.config.liability_weight_maint, liab_amount, liab_value,
            max_liquidatable_value, max_liquidatable_asset_amount, max_liquidatable_liab_amount, liquidator_profit);

        Ok(LiquidationAmounts {
            max_liquidatable_asset_amount,
            max_liquidatable_liab_amount,
            liquidator_profit,
        })
    }

    fn get_value_of_shares(
        &self,
        clock: &Clock,
        shares: Vec<(I80F48, Pubkey)>,
        balance_side: &BalanceSide,
        requirement_type: RequirementType,
        stale_swb_oracles: &RwLock<HashSet<Pubkey>>,
    ) -> Result<Vec<(I80F48, Pubkey)>> {
        let mut values: Vec<(I80F48, Pubkey)> = Vec::new();

        for (shares_amount, bank_pk) in shares {
            let bank_wrapper = self.cache.banks.try_get_bank(&bank_pk)?;
            if stale_swb_oracles
                .read()
                .map_err(|_| anyhow!("stale oracle set is poisoned"))?
                .contains(&bank_wrapper.bank.config.oracle_keys[0])
            {
                return Err(anyhow!(SWB_STALE_HANDLED_ERROR));
            }

            let oracle_wrapper = match OracleWrapper::build(&self.cache, clock, &bank_pk) {
                Ok(oracle_wrapper) => oracle_wrapper,
                Err(err) => {
                    if err
                        .downcast_ref::<anchor_lang::error::Error>()
                        .is_some_and(|e| {
                            if let anchor_lang::error::Error::AnchorError(anchor_error) = e {
                                return anchor_error.error_code_number
                                    == SWB_STALE_PRICE_ERROR_CODE_NUMBER;
                            }
                            false
                        })
                    {
                        // If it's Switchboard, then it's always at position 0
                        stale_swb_oracles
                            .write()
                            .map_err(|_| anyhow!("stale oracle set is poisoned"))?
                            .insert(bank_wrapper.bank.config.oracle_keys[0]);
                        return Err(anyhow!(SWB_STALE_HANDLED_ERROR));
                    }

                    return Err(err);
                }
            };

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
