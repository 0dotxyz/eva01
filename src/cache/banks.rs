use crate::{
    utils::{find_oracle_keys, staked_onramp},
    wrappers::bank::BankWrapper,
};
use anyhow::{anyhow, Result};
use marginfi::state::bank_config::BankConfigImpl;
use marginfi_type_crate::{
    constants::{ASSET_TAG_DRIFT, ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO},
    types::{is_marginfi_asset_tag, Bank, OracleSetup},
};
use solana_sdk::{account::Account, pubkey::Pubkey};
use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};

#[derive(Default)]
struct BanksCacheInner {
    banks: HashMap<Pubkey, BankWrapper>,
    mint_to_p0_bank: HashMap<Pubkey, Pubkey>,
}

#[derive(Default)]
pub struct BanksCache {
    inner: RwLock<BanksCacheInner>,
}

impl BanksCache {
    pub fn try_insert(&self, bank_address: Pubkey, bank: Bank, account: Account) -> Result<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| anyhow!("Failed to lock the banks cache for insert! {}", e))?;

        inner
            .banks
            .insert(bank_address, BankWrapper::new(bank_address, bank, account));
        if is_marginfi_asset_tag(bank.config.asset_tag) {
            inner.mint_to_p0_bank.insert(bank.mint, bank_address);
        }
        Ok(())
    }

    pub fn try_get_bank(&self, address: &Pubkey) -> Result<BankWrapper> {
        self.inner
            .read()
            .map_err(|e| anyhow!("Failed to lock the banks cache for search! {}", e))?
            .banks
            .get(address)
            .ok_or(anyhow!("Failed to find the Bank {} in Cache!", address))
            .cloned()
    }

    pub fn get_oracles(&self) -> HashSet<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .iter()
            .flat_map(|(_, bank)| {
                let mut keys = find_oracle_keys(&bank.bank.config);
                // StakedWithPythPush needs a 4th "on-ramp" account that isn't in oracle_keys when
                // derived from the vote account — load it too, or pricing fails with 6051.
                if let Some(onramp) = staked_onramp(&bank.bank) {
                    keys.push(onramp);
                }
                keys
            })
            .collect()
    }

    /// Derived on-ramp accounts for every StakedWithPythPush bank.
    ///
    /// marginfi 0.1.9 consumes a 4th "on-ramp" account for staked pricing. In PreTransition mode
    /// this account may not exist on-chain yet (the program doesn't read it then), so the loader
    /// inserts an empty placeholder for any that are missing — keeping the 4-oracle count aligned
    /// for both in-process pricing and the on-chain observation list.
    pub fn get_staked_onramps(&self) -> HashSet<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .iter()
            .filter_map(|(_, bank)| staked_onramp(&bank.bank))
            .collect()
    }

    pub fn get_banks_for_oracle(&self, oracle: &Pubkey) -> Result<Vec<Pubkey>> {
        Ok(self
            .inner
            .read()
            .map_err(|e| anyhow!("Failed to lock the banks cache for oracle lookup! {}", e))?
            .banks
            .iter()
            .filter_map(|(bank_address, bank)| {
                find_oracle_keys(&bank.bank.config)
                    .contains(oracle)
                    .then_some(*bank_address)
            })
            .collect())
    }

    /// `oracle_keys[0] -> oracle_max_age` for every bank priced from a Pyth push account.
    pub fn get_pyth_push_oracles(&self) -> HashMap<Pubkey, u64> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .values()
            .filter(|bank| {
                matches!(
                    bank.bank.config.oracle_setup,
                    OracleSetup::PythPushOracle
                        | OracleSetup::StakedWithPythPush
                        | OracleSetup::KaminoPythPush
                        | OracleSetup::DriftPythPull
                        | OracleSetup::JuplendPythPull
                        | OracleSetup::PythMSOL
                        | OracleSetup::KaminoMSOL
                        | OracleSetup::JuplendMSOL
                        | OracleSetup::PythLST
                        | OracleSetup::KaminoLST
                        | OracleSetup::JuplendLST
                        | OracleSetup::PTPyth
                )
            })
            .map(|bank| {
                (
                    bank.bank.config.oracle_keys[0],
                    bank.bank.config.get_oracle_max_age(),
                )
            })
            .collect()
    }

    pub fn get_kamino_reserves(&self) -> HashSet<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .iter()
            .filter_map(|(_, bank)| {
                if bank.bank.config.asset_tag == ASSET_TAG_KAMINO {
                    Some(bank.bank.integration_acc_1)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_drift_users(&self) -> HashSet<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .iter()
            .filter_map(|(_, bank)| {
                if bank.bank.config.asset_tag == ASSET_TAG_DRIFT {
                    Some(bank.bank.integration_acc_2)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_juplend_lending_states(&self) -> HashSet<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .iter()
            .filter_map(|(_, bank)| {
                if bank.bank.config.asset_tag == ASSET_TAG_JUPLEND {
                    Some(bank.bank.integration_acc_1)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn try_get_account_for_mint(&self, mint_address: &Pubkey) -> Result<Pubkey> {
        self.inner
            .read()
            .map_err(|e| anyhow!("Failed to lock the banks cache for mint lookup! {}", e))?
            .mint_to_p0_bank
            .get(mint_address)
            .ok_or(anyhow!(
                "Failed to find Bank for the Mint {} in Cache!",
                &mint_address
            ))
            .copied()
    }

    pub fn get_mints(&self) -> Vec<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .values()
            .map(|bank| bank.bank.mint)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    }

    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .len()
    }
}
