use crate::{
    cache::Cache,
    clock_manager,
    config::Eva01Config,
    metrics::{
        record_liquidation_failure, ACCOUNTS_SCANNED_TOTAL, ACCOUNT_SCAN_DURATION_SECONDS,
        ERROR_COUNT, FAILURE_REASON_INTERNAL, FAILURE_REASON_NOT_ENOUGH_FUNDS,
        FAILURE_REASON_RPC_ERROR, FAILURE_REASON_STALE_ORACLES, LIQUIDATABLE_ACCOUNTS_FOUND,
        LIQUIDATABLE_ACCOUNTS_FOUND_TOTAL, LIQUIDATION_SCAN_IN_PROGRESS,
    },
    rebalancer::Rebalancer,
    utils::swb_cranker::{SwbCranker, SWB_STALE_HANDLED_ERROR, SWB_STALE_PRICE_ERROR_CODE_NUMBER},
    wrappers::{
        bank::BankWrapper,
        liquidator_account::{
            LiquidationError, LiquidatorAccount, PreparedLiquidatableAccount, PROFIT_SHARE,
        },
        marginfi_account::MarginfiAccountWrapper,
        oracle::{OracleWrapper, OracleWrapperTrait},
    },
};
use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use fixed_macro::types::I80F48;
use log::{debug, error, info, warn};
use marginfi::state::{
    bank::BankImpl, marginfi_account::get_health_components,
};
use marginfi_type_crate::{
    constants::BANKRUPT_THRESHOLD,
    types::{BalanceSide, BankOperationalState, HealthPriceMode, RequirementType},
};
use solana_client::client_error::ClientError;
use solana_program::pubkey::Pubkey;
use solana_sdk::{account_info::IntoAccountInfo, clock::Clock};
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};
use std::{sync::atomic::Ordering, thread};

const ACCOUNT_SCAN_BATCH_SIZE: usize = 4_096;

pub struct Liquidator {
    liquidator_account: Arc<LiquidatorAccount>,
    rebalancer: Rebalancer,
    min_profit: f64,
    run_liquidation: Arc<AtomicBool>,
    stop_liquidator: Arc<AtomicBool>,
    cache: Arc<Cache>,
    swb_cranker: SwbCranker,
    token_dust_threshold: I80F48,
}

#[derive(Debug, Clone, Copy)]
struct LiquidationAmounts {
    max_liquidatable_asset_amount: I80F48,
    max_liquidatable_liab_amount: I80F48,
    liquidator_profit: I80F48,
    dust_liab_threshold: I80F48,
}

impl LiquidationAmounts {
    fn none() -> Self {
        Self {
            max_liquidatable_asset_amount: I80F48::ZERO,
            max_liquidatable_liab_amount: I80F48::ZERO,
            liquidator_profit: I80F48::ZERO,
            dust_liab_threshold: I80F48::ZERO,
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
        run_liquidation: Arc<AtomicBool>,
        stop_liquidator: Arc<AtomicBool>,
        cache: Arc<Cache>,
    ) -> Result<Self> {
        let swb_cranker = SwbCranker::new(&config, cache.as_ref())?;

        let rebalancer =
            Rebalancer::new(config.clone(), liquidator_account.clone(), cache.clone())?;

        Ok(Liquidator {
            liquidator_account,
            rebalancer,
            min_profit: config.min_profit,
            run_liquidation,
            stop_liquidator,
            cache,
            swb_cranker,
            token_dust_threshold: config.token_dust_threshold,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.rebalancer.run(HashMap::new())?;

        info!("Staring the Liquidator loop.");
        while !self.stop_liquidator.load(Ordering::Relaxed) {
            debug!("Waiting for any data change...");
            if !self.run_liquidation.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));
                continue;
            }

            info!("Running the Liquidation process...");
            self.run_liquidation.store(false, Ordering::Relaxed);

            let mut missing_tokens: HashMap<Pubkey, I80F48> = HashMap::new();
            let mut stale_swb_oracles: HashSet<Pubkey> = HashSet::new();
            let evaluated_accounts = self.evaluate_all_accounts(&mut stale_swb_oracles);

            if let Ok(mut accounts) = evaluated_accounts {
                // Accounts are sorted from the highest profit to the lowest
                accounts.sort_by(|a, b| a.profit.cmp(&b.profit));
                accounts.reverse();

                let mut tokens_in_shortage: HashSet<Pubkey> = HashSet::new();
                for acc in accounts {
                    let liquidatee = acc.liquidatee_account.address;
                    let liab_bank = acc.liab_bank;
                    let liab_amount = acc.liab_amount;
                    let liab_mint = self
                        .cache
                        .banks
                        .try_get_bank(&liab_bank)
                        .ok()
                        .map(|bank| bank.bank.mint);

                    if let Err(e) = self.liquidator_account.liquidate(acc, &mut tokens_in_shortage) {
                        match e {
                            LiquidationError::Anyhow(e) => {
                                error!("Failed to liquidate account {:?}: {:?}", liquidatee, e);
                                let reason = if e.downcast_ref::<ClientError>().is_some() {
                                    FAILURE_REASON_RPC_ERROR
                                } else {
                                    FAILURE_REASON_INTERNAL
                                };
                                record_liquidation_failure(reason, liab_mint, None);
                                ERROR_COUNT.inc();
                            }
                            LiquidationError::StaleOracles(swb_oracles) => {
                                info!("Cranking stale SWB oracles for next cycle: {:?}", swb_oracles);
                                if let Err(crank_err) = self.swb_cranker.crank_oracles(swb_oracles.clone()) {
                                    warn!("Failed to crank stale SWB oracles: {}", crank_err);
                                }
                                let oracle = swb_oracles.first().copied();
                                record_liquidation_failure(FAILURE_REASON_STALE_ORACLES, None, oracle);
                            }
                            LiquidationError::NotEnoughFunds => {
                                missing_tokens
                                    .entry(liab_bank)
                                    .and_modify(|m| *m += liab_amount)
                                    .or_insert(liab_amount);
                                record_liquidation_failure(
                                    FAILURE_REASON_NOT_ENOUGH_FUNDS,
                                    liab_mint,
                                    None,
                                );
                            }
                        }
                    }
                }
            } else if let Err(err) = evaluated_accounts {
                error!("Failed to evaluate accounts in liquidation: {}", err);
                ERROR_COUNT.inc();
            }

            info!("The Liquidation process is complete.");

            if let Err(error) = self.rebalancer.run(missing_tokens) {
                error!("Rebalancing failed: {:?}", error);
                ERROR_COUNT.inc();
            }
        }
        info!("The Liquidator loop is stopped.");

        Ok(())
    }

    /// Checks if liquidation is needed, for each account one by one
    fn evaluate_all_accounts(
        &mut self,
        stale_swb_oracles: &mut HashSet<Pubkey>,
    ) -> Result<Vec<PreparedLiquidatableAccount>> {
        LIQUIDATION_SCAN_IN_PROGRESS.set(1);
        let scan_started = Instant::now();
        let mut total_scanned: u64 = 0;
        let clock = clock_manager::get_clock(&self.cache.clock)?;

        let evaluation_result = {
            let mut next_index: usize = 0;
            let mut result: Vec<PreparedLiquidatableAccount> = vec![];
            loop {
                let accounts_batch = self
                    .cache
                    .marginfi_accounts
                    .try_get_account_batch(next_index, ACCOUNT_SCAN_BATCH_SIZE)?;
                if accounts_batch.is_empty() {
                    break;
                }
                next_index += accounts_batch.len();

                for account in accounts_batch {
                    total_scanned += 1;
                    if account.address == self.liquidator_account.liquidator_address {
                        continue;
                    }
                    match self.process_account(&account, clock.clone(), stale_swb_oracles) {
                        Ok(acc_opt) => {
                            if let Some(acc) = acc_opt {
                                result.push(acc);
                            }
                        }
                        Err(err) => {
                            if !err.to_string().contains(SWB_STALE_HANDLED_ERROR) {
                                debug!(
                                    "Failed to process account {:?}: {:?}",
                                    account.address, err
                                );
                                ERROR_COUNT.inc();
                            }
                        }
                    }
                }
            }

            Ok(result)
        };

        LIQUIDATION_SCAN_IN_PROGRESS.set(0);
        ACCOUNT_SCAN_DURATION_SECONDS.observe(scan_started.elapsed().as_secs_f64());
        ACCOUNTS_SCANNED_TOTAL.inc_by(total_scanned);

        match &evaluation_result {
            Ok(accounts) => {
                LIQUIDATABLE_ACCOUNTS_FOUND.set(accounts.len() as i64);
                LIQUIDATABLE_ACCOUNTS_FOUND_TOTAL.inc_by(accounts.len() as u64);
            }
            Err(_) => {
                LIQUIDATABLE_ACCOUNTS_FOUND.set(0);
            }
        }

        evaluation_result
    }

    fn process_account(
        &self,
        account: &MarginfiAccountWrapper,
        clock: Clock,
        stale_swb_oracles: &mut HashSet<Pubkey>,
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
            profit: liquidation_amounts.liquidator_profit.to_num(),
            dust_liab_threshold: liquidation_amounts.dust_liab_threshold,
        }))
    }

    fn find_liquidation_bank_candidates(
        &self,
        deposit_values: Vec<(I80F48, Pubkey)>,
        liab_values: Vec<(I80F48, Pubkey)>,
    ) -> Result<Option<(Pubkey, Pubkey)>> {
        if deposit_values.is_empty() || liab_values.is_empty() {
            return Ok(None);
        }

        if deposit_values
            .iter()
            .map(|(v, _)| v.to_num::<f64>())
            .sum::<f64>()
            < BANKRUPT_THRESHOLD
        {
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
        banks_and_oracles: &[Pubkey],
        asset_bank_wrapper: &BankWrapper,
        liab_bank_wrapper: &BankWrapper,
    ) -> Result<LiquidationAmounts> {
        let mut accounts: Vec<_> = vec![];
        for pk in banks_and_oracles.iter() {
            let bank_account = cache.banks.try_get_bank(pk);
            let account = match bank_account {
                Ok(wrapper) => wrapper.account,
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
            &remaining_ais,
            RequirementType::Maintenance,
            &mut None,
            HealthPriceMode::Client(clock.clone()),
        )?;

        let maintenance_health = total_weighted_assets - total_weighted_liabilities;
        debug!(
            "Account {} maintenance_health = {:?} (assets {:?}, liabilities {:?})",
            account.address, maintenance_health, total_weighted_assets, total_weighted_liabilities
        );
        if maintenance_health >= I80F48::ZERO {
            return Ok(LiquidationAmounts::none());
        }

        let asset_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &asset_bank_wrapper.address)?;

        let liab_oracle_wrapper =
            OracleWrapper::build(&self.cache, &clock, &liab_bank_wrapper.address)?;

        let asset_weight_maint: I80F48 = asset_bank_wrapper.bank.config.asset_weight_maint.into();
        let liab_weight_maint: I80F48 = liab_bank_wrapper.bank.config.liability_weight_maint.into();

        let liquidation_discount = fixed_macro::types::I80F48!(0.95);

        let all = asset_weight_maint - liab_weight_maint * liquidation_discount;

        if all >= I80F48::ZERO {
            debug!("Account {:?} has no liquidatable amount: {:?}, asset_weight_maint: {:?}, liab_weight_maint: {:?}", account.address, all, asset_weight_maint, liab_weight_maint);
            return Ok(LiquidationAmounts::none());
        }

        let underwater_maint_value =
            maintenance_health / (asset_weight_maint - liab_weight_maint * liquidation_discount);

        let (asset_amount, _) = account.get_balance_for_bank(asset_bank_wrapper)?;
        let (_, liab_amount) = account.get_balance_for_bank(liab_bank_wrapper)?;

        let asset_value = asset_bank_wrapper.calc_value(
            &asset_oracle_wrapper,
            asset_amount,
            BalanceSide::Assets,
            RequirementType::Maintenance,
        )?;

        let liab_value = liab_bank_wrapper.calc_value(
            &liab_oracle_wrapper,
            liab_amount,
            BalanceSide::Liabilities,
            RequirementType::Maintenance,
        )?;

        let max_liquidatable_value = min(min(asset_value, liab_value), underwater_maint_value);
        let liquidator_profit = max_liquidatable_value
            .checked_mul(I80F48::from_num(PROFIT_SHARE))
            .unwrap();

        if liquidator_profit <= self.min_profit {
            return Ok(LiquidationAmounts::none());
        }

        let max_liquidatable_asset_amount = asset_bank_wrapper.calc_amount(
            &asset_oracle_wrapper,
            max_liquidatable_value,
            BalanceSide::Assets,
            RequirementType::Maintenance,
        )?;

        let max_liquidatable_liab_amount = liab_bank_wrapper.calc_amount(
            &liab_oracle_wrapper,
            max_liquidatable_value,
            BalanceSide::Liabilities,
            RequirementType::Maintenance,
        )?;

        let dust_liab_threshold = liab_bank_wrapper.calc_amount(
            &liab_oracle_wrapper,
            self.token_dust_threshold, // in USD
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
            dust_liab_threshold,
        })
    }

    fn get_value_of_shares(
        &self,
        clock: &Clock,
        shares: Vec<(I80F48, Pubkey)>,
        balance_side: &BalanceSide,
        requirement_type: RequirementType,
        stale_swb_oracles: &mut HashSet<Pubkey>,
    ) -> Result<Vec<(I80F48, Pubkey)>> {
        let mut values: Vec<(I80F48, Pubkey)> = Vec::new();

        for (shares_amount, bank_pk) in shares {
            let bank_wrapper = self.cache.banks.try_get_bank(&bank_pk)?;
            if stale_swb_oracles.contains(&bank_wrapper.bank.config.oracle_keys[0]) {
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
                        stale_swb_oracles.insert(bank_wrapper.bank.config.oracle_keys[0]);
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
