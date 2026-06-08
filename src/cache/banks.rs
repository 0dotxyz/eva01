use crate::{utils::find_oracle_keys, wrappers::bank::BankWrapper};
use anyhow::{anyhow, Result};
use marginfi_type_crate::{
    constants::{
        ASSET_TAG_DEFAULT, ASSET_TAG_DRIFT, ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO, ASSET_TAG_SOL,
        ASSET_TAG_STAKED,
    },
    types::{Bank, OracleSetup},
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
        if matches!(
            bank.config.asset_tag,
            ASSET_TAG_DEFAULT | ASSET_TAG_SOL | ASSET_TAG_STAKED
        ) {
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
            .flat_map(|(_, bank)| find_oracle_keys(&bank.bank.config))
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

    pub fn get_swb_oracles(&self) -> HashSet<Pubkey> {
        self.inner
            .read()
            .expect("banks cache lock poisoned")
            .banks
            .iter()
            .filter_map(|(_, bank)| {
                if matches!(
                    bank.bank.config.oracle_setup,
                    OracleSetup::SwitchboardPull
                        | OracleSetup::KaminoSwitchboardPull
                        | OracleSetup::DriftSwitchboardPull
                        | OracleSetup::JuplendSwitchboardPull
                ) {
                    Some(bank.bank.config.oracle_keys[0])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns a map of SwitchboardPull oracle pubkey → bank addresses.
    /// Multiple banks can share the same oracle key, so each entry is a Vec.
    pub fn get_swb_oracle_to_bank_map(&self) -> HashMap<Pubkey, Vec<Pubkey>> {
        let mut map: HashMap<Pubkey, Vec<Pubkey>> = HashMap::new();
        let inner = self.inner.read().expect("banks cache lock poisoned");
        for (bank_addr, bank) in &inner.banks {
            if matches!(bank.bank.config.oracle_setup, OracleSetup::SwitchboardPull) {
                map.entry(bank.bank.config.oracle_keys[0])
                    .or_default()
                    .push(*bank_addr);
            }
        }
        map
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
