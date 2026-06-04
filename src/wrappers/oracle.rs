use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::error;
use marginfi::state::price::{OraclePriceFeedAdapter, PriceAdapter};
use marginfi_type_crate::types::{OnRampTransition, OraclePriceType, OracleSetup, PriceBias};
use solana_program::pubkey::Pubkey;
use solana_sdk::{account_info::IntoAccountInfo, clock::Clock};

use crate::{cache::Cache, utils::find_oracle_keys};

pub trait OracleWrapperTrait {
    fn build(cache: &Cache, clock: &Clock, bank_address: &Pubkey) -> Result<Self>
    where
        Self: Sized;
    fn get_price_of_type(
        &self,
        oracle_type: OraclePriceType,
        price_bias: Option<PriceBias>,
        oracle_max_confidence: u32,
    ) -> anyhow::Result<I80F48>;
    fn get_address(&self) -> Pubkey;
}

#[derive(Clone)]
pub struct OracleWrapper {
    pub addresses: Vec<Pubkey>,
    source: OraclePriceFeedAdapter,
}

impl OracleWrapper {
    /// Builds an OracleWrapper without enforcing oracle staleness — suitable for
    /// use cases where an approximate price is acceptable (e.g. the rebalancer's
    /// threshold comparisons) and a stale feed should not block execution.
    pub fn build_lenient(cache: &Cache, clock: &Clock, bank_address: &Pubkey) -> Result<Self> {
        Self::build_inner(cache, clock, bank_address, Some(u16::MAX))
    }

    fn build_inner(
        cache: &Cache,
        clock: &Clock,
        bank_address: &Pubkey,
        max_age_override: Option<u16>,
    ) -> Result<Self> {
        let bank_wrapper = cache.banks.try_get_bank(bank_address)?;
        let oracle_addresses = find_oracle_keys(&bank_wrapper.bank.config);

        let mut patched;
        let bank = match max_age_override {
            Some(max_age) => {
                patched = bank_wrapper.bank;
                patched.config.oracle_max_age = max_age;
                &patched
            }
            None => &bank_wrapper.bank,
        };

        let mut result: Option<Self> = None;
        match bank.config.oracle_setup {
            OracleSetup::PythPushOracle | OracleSetup::SwitchboardPull => {
                if oracle_addresses.len() != 1 {
                    return Err(anyhow!(
                        "PythPull/SwitchboardPull setup requires exactly 1 oracle key, but found {} for the Bank {:?} (setup: {:?})",
                        oracle_addresses.len(), bank_address, bank.config.oracle_setup
                    ));
                }

                let bank_oracle_address = *oracle_addresses.first().unwrap();
                let mut bank_oracle = cache.oracles.try_get_account(&bank_oracle_address)?;
                let bank_oracle_account_info =
                    (&bank_oracle_address, &mut bank_oracle).into_account_info();

                let price_adapter = OraclePriceFeedAdapter::try_from_bank(
                    bank,
                    &[bank_oracle_account_info],
                    clock,
                    OnRampTransition::PreTransition,
                )?;

                result = Some(Self {
                    addresses: [bank_oracle_address].to_vec(),
                    source: price_adapter,
                });
            }
            OracleSetup::StakedWithPythPush => {
                if oracle_addresses.len() != 3 {
                    return Err(anyhow!(
                        "StakedWithPythPush setup requires exactly 3 oracle keys, but found {} for the Bank {:?}",
                        oracle_addresses.len(), bank_address
                    ));
                }

                let bank_oracle_address = *oracle_addresses.first().unwrap();
                let mut bank_oracle = cache.oracles.try_get_account(&bank_oracle_address)?;
                let bank_oracle_account_info =
                    (&bank_oracle_address, &mut bank_oracle).into_account_info();

                let mint_oracle_address = *oracle_addresses.get(1).unwrap();
                let mut mint_oracle = cache.oracles.try_get_account(&mint_oracle_address)?;
                let mint_oracle_account_info =
                    (&mint_oracle_address, &mut mint_oracle).into_account_info();

                let sol_pool_oracle_address = *oracle_addresses.get(2).unwrap();
                let mut sol_pool_oracle =
                    cache.oracles.try_get_account(&sol_pool_oracle_address)?;
                let sol_pool_account_info =
                    (&sol_pool_oracle_address, &mut sol_pool_oracle).into_account_info();

                let price_adapter = OraclePriceFeedAdapter::try_from_bank(
                    bank,
                    &[
                        bank_oracle_account_info,
                        mint_oracle_account_info,
                        sol_pool_account_info,
                    ],
                    clock,
                    OnRampTransition::PreTransition,
                )?;
                result = Some(Self {
                    addresses: vec![
                        bank_oracle_address,
                        mint_oracle_address,
                        sol_pool_oracle_address,
                    ],
                    source: price_adapter,
                });
            }
            OracleSetup::Fixed => {
                let price_adapter = OraclePriceFeedAdapter::try_from_bank(
                    bank,
                    &[],
                    clock,
                    OnRampTransition::PreTransition,
                )?;
                result = Some(Self {
                    addresses: vec![],
                    source: price_adapter,
                });
            }
            OracleSetup::KaminoPythPush
            | OracleSetup::KaminoSwitchboardPull
            | OracleSetup::DriftPythPull
            | OracleSetup::DriftSwitchboardPull
            | OracleSetup::JuplendPythPull
            | OracleSetup::JuplendSwitchboardPull => {
                if oracle_addresses.len() != 2 {
                    return Err(anyhow!(
                        "Integration PythPush/SwitchboardPull setup requires exactly 2 oracle keys, but found {} for the Bank {:?} (setup: {:?})",
                        oracle_addresses.len(), bank_address, bank.config.oracle_setup
                    ));
                }

                let bank_oracle_address = *oracle_addresses.first().unwrap();
                let mut bank_oracle = cache.oracles.try_get_account(&bank_oracle_address)?;
                let bank_oracle_account_info =
                    (&bank_oracle_address, &mut bank_oracle).into_account_info();

                let integration_oracle_address = *oracle_addresses.get(1).unwrap();
                let mut integration_oracle =
                    cache.oracles.try_get_account(&integration_oracle_address)?;
                let integration_oracle_account_info =
                    (&integration_oracle_address, &mut integration_oracle).into_account_info();

                let price_adapter = OraclePriceFeedAdapter::try_from_bank(
                    bank,
                    &[bank_oracle_account_info, integration_oracle_account_info],
                    clock,
                    OnRampTransition::PreTransition,
                )?;
                result = Some(Self {
                    addresses: [bank_oracle_address, integration_oracle_address].to_vec(),
                    source: price_adapter,
                });
            }
            OracleSetup::FixedKamino | OracleSetup::FixedDrift | OracleSetup::FixedJuplend => {
                if oracle_addresses.len() != 1 {
                    return Err(anyhow!(
                        "Integration Fixed setup requires exactly 1 oracle key, but found {} for the Bank {:?} (setup: {:?})",
                        oracle_addresses.len(), bank_address, bank.config.oracle_setup
                    ));
                }

                let integration_oracle_address = *oracle_addresses.first().unwrap();
                let mut integration_oracle =
                    cache.oracles.try_get_account(&integration_oracle_address)?;
                let integration_oracle_account_info =
                    (&integration_oracle_address, &mut integration_oracle).into_account_info();

                let price_adapter = OraclePriceFeedAdapter::try_from_bank(
                    bank,
                    &[integration_oracle_account_info],
                    clock,
                    OnRampTransition::PreTransition,
                )?;
                result = Some(Self {
                    addresses: vec![integration_oracle_address],
                    source: price_adapter,
                });
            }
            _ => {
                error!(
                    "Unsupported Oracle setup for the Bank {:?} : {:?}",
                    bank_address, bank.config.oracle_setup
                )
            }
        }

        match result {
            Some(wrapper) => Ok(wrapper),
            None => Err(anyhow!(
                "No valid oracle wrapper found for the Bank {:?}",
                bank_address
            )),
        }
    }
}

impl OracleWrapperTrait for OracleWrapper {
    fn get_price_of_type(
        &self,
        oracle_type: OraclePriceType,
        price_bias: Option<PriceBias>,
        oracle_max_confidence: u32,
    ) -> anyhow::Result<I80F48> {
        Ok(self
            .source
            .get_price_of_type(oracle_type, price_bias, oracle_max_confidence)?)
    }

    fn get_address(&self) -> Pubkey {
        *self.addresses.first().unwrap_or(&Pubkey::default())
    }

    fn build(cache: &Cache, clock: &Clock, bank_address: &Pubkey) -> Result<Self> {
        Self::build_inner(cache, clock, bank_address, None)
    }
}
