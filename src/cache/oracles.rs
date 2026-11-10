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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::pyth_prices::test_price_update_account;

    fn cached_price(cache: &OraclesCache, address: &Pubkey) -> i64 {
        price_update(&cache.try_get_account(address).unwrap())
            .unwrap()
            .price_message
            .price
    }

    #[test]
    fn an_older_pyth_price_never_replaces_a_newer_one() {
        let mut cache = OraclesCache::default();
        let oracle = Pubkey::new_unique();
        let feed = [1u8; 32];
        cache
            .try_insert(oracle, test_price_update_account(feed, 100, 5))
            .unwrap();
        // Geyser delivering the chain's older price after a Hermes one, or the reverse.
        assert!(!cache
            .try_update(&oracle, test_price_update_account(feed, 99, 6))
            .unwrap());
        assert_eq!(cached_price(&cache, &oracle), 5);
        assert!(cache
            .try_update(&oracle, test_price_update_account(feed, 101, 7))
            .unwrap());
        assert_eq!(cached_price(&cache, &oracle), 7);
    }

    #[test]
    fn other_oracle_accounts_always_update() {
        let mut cache = OraclesCache::default();
        let oracle = Pubkey::new_unique();
        cache.try_insert(oracle, Account::default()).unwrap();
        let newer = Account {
            lamports: 42,
            ..Account::default()
        };
        assert!(cache.try_update(&oracle, newer).unwrap());
        assert_eq!(cache.try_get_account(&oracle).unwrap().lamports, 42);
    }
}
