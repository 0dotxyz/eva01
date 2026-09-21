use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use marginfi::state::{
    marginfi_account::get_remaining_accounts_per_bank,
    price::{OraclePriceFeedAdapter, PriceAdapter},
};
use marginfi_type_crate::types::{Bank, OraclePriceType, OracleSetup, PriceBias};
use solana_program::pubkey::Pubkey;
use solana_sdk::{account::Account, account_info::IntoAccountInfo, clock::Clock};

use crate::{cache::Cache, utils::staked_onramp};

const SOL_POOL_INDEX: usize = 2;
const ONRAMP_INDEX: usize = 3;

pub fn oracle_account_keys(bank: &Bank, bank_address: &Pubkey) -> Result<Vec<Pubkey>> {
    let key = |index: usize| -> Result<Pubkey> {
        let key = bank.config.oracle_keys[index];
        if key == Pubkey::default() {
            return Err(anyhow!(
                "Bank {} ({:?}) has no oracle_keys[{}], which this setup requires",
                bank_address,
                bank.config.oracle_setup,
                index
            ));
        }
        Ok(key)
    };

    let addresses = match bank.config.oracle_setup {
        // `oracle_keys[0]` alone: a plain feed (Pyth/Scope), or the Exponent vault that PTFixed
        // prices from directly.
        OracleSetup::PythPushOracle | OracleSetup::Scope | OracleSetup::PTFixed => vec![key(0)?],
        OracleSetup::StakedWithPythPush => {
            let sol_pool = key(SOL_POOL_INDEX)?;
            let onramp = staked_onramp(bank).unwrap_or(sol_pool);
            vec![key(0)?, key(1)?, sol_pool, onramp]
        }
        // Feed + one multiplier account: the venue's reserve/market/lending state, the Marinade
        // state (mSOL/SOL), the SPL stake pool (LST/SOL), or the Exponent vault (PT).
        OracleSetup::KaminoPythPush
        | OracleSetup::DriftPythPull
        | OracleSetup::JuplendPythPull
        | OracleSetup::PythMSOL
        | OracleSetup::PythLST
        | OracleSetup::PTPyth => vec![key(0)?, key(1)?],
        // Feed + venue multiplier (`oracle_keys[1]`) + native multiplier (`oracle_keys[2]`).
        OracleSetup::KaminoMSOL
        | OracleSetup::JuplendMSOL
        | OracleSetup::KaminoLST
        | OracleSetup::JuplendLST => vec![key(0)?, key(1)?, key(2)?],
        OracleSetup::FixedKamino | OracleSetup::FixedDrift | OracleSetup::FixedJuplend => {
            vec![key(1)?]
        }
        OracleSetup::Fixed => vec![],
        setup => {
            return Err(anyhow!(
                "Unsupported oracle setup {:?} for the Bank {}",
                setup,
                bank_address
            ))
        }
    };

    let expected = get_remaining_accounts_per_bank(bank)?.saturating_sub(1);
    if addresses.len() != expected {
        return Err(anyhow!(
            "Bank {} ({:?}) needs {} oracle accounts, resolved {}",
            bank_address,
            bank.config.oracle_setup,
            expected,
            addresses.len()
        ));
    }

    Ok(addresses)
}

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
}

#[derive(Clone)]
pub struct OracleWrapper {
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

        let mut patched;
        let bank = match max_age_override {
            Some(max_age) => {
                patched = bank_wrapper.bank;
                patched.config.oracle_max_age = max_age;
                &patched
            }
            None => &bank_wrapper.bank,
        };

        let addresses = oracle_account_keys(bank, bank_address)?;
        let staked = bank.config.oracle_setup == OracleSetup::StakedWithPythPush;

        let mut accounts: Vec<Account> = Vec::with_capacity(addresses.len());
        for (index, address) in addresses.iter().enumerate() {
            let account = match cache.oracles.try_get_account(address) {
                Ok(account) => account,
                Err(_) if staked && index == ONRAMP_INDEX => accounts[SOL_POOL_INDEX].clone(),
                Err(e) => return Err(e),
            };
            accounts.push(account);
        }

        let remaining_ais: Vec<_> = addresses
            .iter()
            .zip(accounts.iter_mut())
            .map(|(address, account)| (address, account).into_account_info())
            .collect();

        let source = OraclePriceFeedAdapter::try_from_bank(bank, &remaining_ais, clock)?;

        Ok(Self { source })
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

    fn build(cache: &Cache, clock: &Clock, bank_address: &Pubkey) -> Result<Self> {
        Self::build_inner(cache, clock, bank_address, None)
    }
}
