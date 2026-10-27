use anchor_lang::AccountDeserialize;
use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use marginfi::state::{
    marginfi_account::get_remaining_accounts_per_bank,
    price::{OraclePriceFeedAdapter, PriceAdapter},
};
use marginfi_type_crate::{
    pdas::SCOPE_PROGRAM_ID,
    types::{Bank, OraclePriceType, OracleSetup, PriceBias},
};
use pyth_solana_receiver_sdk::price_update::PriceUpdateV2;
use scope_mocks::state::{ScopeOraclePrices, SCOPE_ORACLE_PRICES_DISCRIMINATOR};
use solana_account::Account;
use solana_account_info::IntoAccountInfo;
use solana_clock::Clock;
use solana_pubkey::Pubkey;

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
        | OracleSetup::JuplendPythPull
        | OracleSetup::ScopeKamino
        | OracleSetup::ScopeJuplend
        | OracleSetup::PythMSOL
        | OracleSetup::PythLST
        | OracleSetup::PTPyth => vec![key(0)?, key(1)?],
        // Feed + venue multiplier (`oracle_keys[1]`) + native multiplier (`oracle_keys[2]`).
        OracleSetup::KaminoMSOL
        | OracleSetup::JuplendMSOL
        | OracleSetup::KaminoLST
        | OracleSetup::JuplendLST => vec![key(0)?, key(1)?, key(2)?],
        OracleSetup::FixedKamino | OracleSetup::FixedJuplend => {
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

/// Publish time of a Pyth price, or the newest of the given Scope entries (only the ones our banks
/// use). `None` for other accounts and never-refreshed entries.
pub fn oracle_publish_time(account: &Account, scope_entries: &[u16]) -> Option<i64> {
    if account.owner == SCOPE_PROGRAM_ID {
        let data = account.data.as_slice();
        if data.get(..8)? != SCOPE_ORACLE_PRICES_DISCRIMINATOR {
            return None;
        }
        let prices: &ScopeOraclePrices = bytemuck::try_from_bytes(data.get(8..)?).ok()?;
        return scope_entries
            .iter()
            .filter_map(|&index| prices.prices.get(index as usize))
            .filter_map(|entry| i64::try_from(entry.unix_timestamp).ok())
            .filter(|&timestamp| timestamp > 0)
            .max();
    }

    PriceUpdateV2::try_deserialize(&mut account.data.as_slice())
        .ok()
        .map(|update| update.price_message.publish_time)
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

        let source = OraclePriceFeedAdapter::try_from_bank(bank, &remaining_ais, clock, false)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use anchor_lang::AccountSerialize;
    use pyth_solana_receiver_sdk::price_update::{PriceFeedMessage, VerificationLevel};
    use scope_mocks::state::ScopeDatedPrice;

    fn scope_account(entries: &[(usize, u64)]) -> Account {
        let mut prices: ScopeOraclePrices = bytemuck::Zeroable::zeroed();
        for &(index, unix_timestamp) in entries {
            prices.prices[index] = ScopeDatedPrice {
                unix_timestamp,
                ..ScopeDatedPrice::default()
            };
        }
        let mut data = SCOPE_ORACLE_PRICES_DISCRIMINATOR.to_vec();
        data.extend_from_slice(bytemuck::bytes_of(&prices));
        Account {
            owner: SCOPE_PROGRAM_ID,
            data,
            ..Account::default()
        }
    }

    #[test]
    fn scope_publish_time_reads_only_the_priced_entries() {
        let account = scope_account(&[(3, 1_000), (12, 1_050), (40, 9_999)]);

        assert_eq!(oracle_publish_time(&account, &[12]), Some(1_050));
        assert_eq!(oracle_publish_time(&account, &[3, 12]), Some(1_050));
        // Entries no bank uses are ignored.
        assert_eq!(oracle_publish_time(&account, &[3]), Some(1_000));
        // Never refreshed, out of range, or no entries.
        assert_eq!(oracle_publish_time(&account, &[7]), None);
        assert_eq!(oracle_publish_time(&account, &[600]), None);
        assert_eq!(oracle_publish_time(&account, &[]), None);
        // Not a valid i64 timestamp.
        let account = scope_account(&[(12, u64::MAX)]);
        assert_eq!(oracle_publish_time(&account, &[12]), None);
    }

    #[test]
    fn scope_publish_time_rejects_foreign_accounts() {
        let mut account = scope_account(&[(12, 1_050)]);
        account.data[0] ^= 0xff;
        assert_eq!(oracle_publish_time(&account, &[12]), None);

        let mut account = scope_account(&[(12, 1_050)]);
        account.owner = Pubkey::new_unique();
        assert_eq!(oracle_publish_time(&account, &[12]), None);
    }

    #[test]
    fn pyth_publish_time_is_read_from_the_price_message() {
        let update = PriceUpdateV2 {
            write_authority: Pubkey::new_unique(),
            verification_level: VerificationLevel::Full,
            price_message: PriceFeedMessage {
                feed_id: [7; 32],
                price: 100,
                conf: 1,
                exponent: -2,
                publish_time: 1_234,
                prev_publish_time: 1_233,
                ema_price: 100,
                ema_conf: 1,
            },
            posted_slot: 5,
        };
        let mut data = vec![];
        update.try_serialize(&mut data).unwrap();
        let account = Account {
            data,
            ..Account::default()
        };

        assert_eq!(oracle_publish_time(&account, &[]), Some(1_234));
        assert_eq!(oracle_publish_time(&Account::default(), &[]), None);
    }
}
