use anyhow::{anyhow, Result};

use super::pyth_prices::price_update;
use solana_account::Account;
use solana_pubkey::Pubkey;
use std::{collections::HashMap, sync::RwLock};

#[derive(Default)]
pub struct OraclesCache {
    accounts: RwLock<HashMap<Pubkey, Account>>,
}

impl OraclesCache {
    pub fn try_insert(&mut self, address: Pubkey, account: Account) -> Result<()> {
        self.accounts
            .write()
            .map_err(|e| anyhow!("Failed to lock the Oracle accounts map for insert! {}", e))?
            .insert(address, account);

        Ok(())
    }

    /// Stores `account`, and returns whether it was stored: a Pyth price update older than the
    /// one cached is ignored, since prices arrive from the chain and from Hermes in no fixed
    /// order and an older one must never replace a newer one.
    pub fn try_update(&self, address: &Pubkey, account: Account) -> Result<bool> {
        let mut accounts = self.accounts.write().map_err(|e| {
            anyhow!(
                "Failed to lock the Oracle accounts map for for update! {}",
                e
            )
        })?;
        if let (Some(new), Some(cached)) = (
            price_update(&account),
            accounts.get(address).and_then(price_update),
        ) {
            if new.price_message.publish_time < cached.price_message.publish_time {
                return Ok(false);
            }
        }
        accounts.insert(*address, account);
        Ok(true)
    }

    pub fn try_get_account(&self, address: &Pubkey) -> Result<Account> {
        self.accounts
            .read()
            .map_err(|e| {
                anyhow!(
                    "Failed to lock the Oracle accounts map for for search! {}",
                    e
                )
            })?
            .get(address)
            .ok_or(anyhow!(
                "Failed to find Oracle account with the Address {}!",
                &address
            ))
            .cloned()
    }

    pub fn try_get_addresses(&self) -> Result<Vec<Pubkey>> {
        let accounts_guard = self.accounts.read().map_err(|e| {
            anyhow!(
                "Failed to lock the Oracle accounts map for returning all addresses! {}",
                e
            )
        })?;

        Ok(accounts_guard.keys().cloned().collect())
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self
            .accounts
            .read()
            .map_err(|e| {
                anyhow!(
                    "Failed to lock the Oracle accounts map checking the size! {}",
                    e
                )
            })?
            .len())
    }
}
