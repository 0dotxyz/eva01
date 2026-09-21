use crate::cache::Cache;

use crate::wrappers::bank::BankWrapper;
use marginfi::state::bank::BankImpl;
use solana_sdk::clock::Clock;

use super::oracle::{oracle_account_keys, OracleWrapperTrait};
use anyhow::Result;
use fixed::types::I80F48;
use marginfi_type_crate::types::{BalanceSide, LendingAccount, MarginfiAccount, OracleSetup};
use solana_program::pubkey::Pubkey;
use std::collections::HashSet;

#[derive(Clone)]
pub struct MarginfiAccountWrapper {
    pub address: Pubkey,
    pub account: MarginfiAccount,
}

type Shares = Vec<(I80F48, Pubkey)>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationAccounts {
    pub observation_accounts: Vec<Pubkey>,
    pub bank_pks: Vec<Pubkey>,
    pub kamino_reserves: HashSet<Pubkey>,
    pub drift_spot_markets: HashSet<Pubkey>,
    pub juplend_states: HashSet<Pubkey>,
}

impl MarginfiAccountWrapper {
    pub fn new(address: Pubkey, account: MarginfiAccount) -> Self {
        MarginfiAccountWrapper { address, account }
    }

    pub fn get_balance_for_bank(&self, bank_wrapper: &BankWrapper) -> Result<(I80F48, I80F48)> {
        let balance = self
            .account
            .lending_account
            .balances
            .iter()
            .find(|b| b.bank_pk == bank_wrapper.address && b.is_active())
            .map(|b| match b.get_side()? {
                BalanceSide::Assets => {
                    let amount = bank_wrapper
                        .bank
                        .get_asset_amount(b.asset_shares.into())
                        .ok()?;
                    Some((amount, I80F48::ZERO))
                }
                BalanceSide::Liabilities => {
                    let amount = bank_wrapper
                        .bank
                        .get_liability_amount(b.liability_shares.into())
                        .ok()?;
                    Some((I80F48::ZERO, amount))
                }
            })
            .map(|e| e.unwrap_or_default())
            .unwrap_or_default();

        Ok(balance)
    }

    pub fn get_deposits_and_liabilities_shares(&self) -> (Shares, Shares) {
        let mut liabilities = Vec::new();
        let mut deposits = Vec::new();

        for balance in &self.account.lending_account.balances {
            if balance.is_active() {
                match balance.get_side() {
                    Some(BalanceSide::Liabilities) => {
                        liabilities.push((balance.liability_shares.into(), balance.bank_pk))
                    }
                    Some(BalanceSide::Assets) => {
                        deposits.push((balance.asset_shares.into(), balance.bank_pk))
                    }
                    _ => {}
                }
            }
        }

        (deposits, liabilities)
    }

    pub fn get_active_banks(lending_account: &LendingAccount) -> Vec<Pubkey> {
        lending_account
            .balances
            .iter()
            .filter_map(|balance| balance.is_active().then_some(balance.bank_pk))
            .collect::<Vec<_>>()
    }

    pub fn get_observation_accounts<T: OracleWrapperTrait>(
        lending_account: &LendingAccount,
        include_banks: &[Pubkey],
        exclude_banks: &[Pubkey],
        cache: &Cache,
        clock: &Clock,
    ) -> Result<ObservationAccounts> {
        let mut bank_pks: HashSet<Pubkey> =
            MarginfiAccountWrapper::get_active_banks(lending_account)
                .into_iter()
                .collect();

        bank_pks.extend(include_banks.iter());

        let mut bank_pks = bank_pks
            .into_iter()
            .filter(|bank_pk| !exclude_banks.contains(bank_pk))
            .collect::<Vec<_>>();

        // Sort all bank_pks in descending order
        bank_pks.sort_by(|a, b| b.cmp(a));

        let mut observation_accounts: Vec<Pubkey> = vec![];
        let mut kamino_reserves = HashSet::new();
        let mut drift_spot_markets = HashSet::new();
        let mut juplend_states = HashSet::new();

        for bank_pk in bank_pks.iter() {
            let bank_wrapper = cache.banks.try_get_bank(bank_pk)?;
            T::build(cache, clock, bank_pk)?;

            let setup = bank_wrapper.bank.config.oracle_setup;
            let oracle_keys = oracle_account_keys(&bank_wrapper.bank, bank_pk)?;

            match setup {
                OracleSetup::KaminoPythPush
                | OracleSetup::FixedKamino
                | OracleSetup::KaminoMSOL
                | OracleSetup::KaminoLST => {
                    kamino_reserves.insert(bank_wrapper.bank.integration_acc_1);
                }
                OracleSetup::DriftPythPull | OracleSetup::FixedDrift => {
                    drift_spot_markets.insert(bank_wrapper.bank.integration_acc_1);
                }
                OracleSetup::JuplendPythPull
                | OracleSetup::FixedJuplend
                | OracleSetup::JuplendMSOL
                | OracleSetup::JuplendLST => {
                    juplend_states.insert(bank_wrapper.bank.integration_acc_1);
                }
                _ => {}
            }

            observation_accounts.push(*bank_pk);
            observation_accounts.extend(oracle_keys);
        }

        Ok(ObservationAccounts {
            observation_accounts,
            bank_pks,
            kamino_reserves,
            drift_spot_markets,
            juplend_states,
        })
    }
}
