use std::{collections::HashMap, sync::RwLock};

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use log::error;
use solana_account::Account;
use solana_pubkey::Pubkey;

/// The liquidator's token accounts, each with the slot it was read at. Writes arrive from Geyser
/// and from RPC refreshes in no particular order, so an older read never replaces a newer one:
/// otherwise a balance from before a liquidation spent it could reappear after the liquidation's
/// claim on it was released.
#[derive(Default)]
pub struct TokensCache {
    tokens: RwLock<IndexMap<Pubkey, (Account, u64)>>,
    mint_to_token: HashMap<Pubkey, Pubkey>,
}

impl TokensCache {
    pub fn try_insert(
        &mut self,
        token_address: Pubkey,
        token: Account,
        mint_address: Pubkey,
    ) -> anyhow::Result<()> {
        self.tokens
            .write()
            .map_err(|e| anyhow!("Failed to lock the token map for insert: {}", e))?
            .insert(token_address, (token, 0));
        self.mint_to_token.insert(mint_address, token_address);
        Ok(())
    }

    pub fn try_get_account(&self, address: &Pubkey) -> Result<Account> {
        self.tokens
            .read()
            .map_err(|err| anyhow!("Failed to lock the tokens map for search! {}", err))?
            .get(address)
            .map(|(account, _)| account.clone())
            .ok_or(anyhow!("Failed to find Token {}!", &address))
    }

    /// Stores `account` as read at `slot`, unless the cache already holds a newer read. A slot of
    /// 0 means unknown and only fills an account never read with a slot.
    pub fn try_update_account(&self, address: Pubkey, account: Account, slot: u64) -> Result<()> {
        let mut tokens = self
            .tokens
            .write()
            .map_err(|e| anyhow!("Failed to lock the token map for update! {}", e))?;
        match tokens.get(&address) {
            Some((_, cached_slot)) if *cached_slot > slot => {}
            _ => {
                tokens.insert(address, (account, slot));
            }
        }
        Ok(())
    }

    pub fn try_get_token_for_mint(&self, mint_address: &Pubkey) -> Result<Pubkey> {
        self.mint_to_token
            .get(mint_address)
            .ok_or(anyhow!(
                "Failed to find Token for the Mint: {}",
                &mint_address
            ))
            .copied()
    }

    pub fn get_token_for_mint(&self, mint_address: &Pubkey) -> Option<Pubkey> {
        self.try_get_token_for_mint(mint_address)
            .map_err(|err| error!("{}", err))
            .ok()
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self
            .tokens
            .read()
            .map_err(|e| anyhow!("Failed to lock the tokens map for size! {}", e))?
            .len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache() {
        let cache = TokensCache::default();
        assert_eq!(cache.len().unwrap(), 0);
    }

    #[test]
    fn test_try_insert_and_get_account() {
        let mut cache = TokensCache::default();
        let token_address = Pubkey::new_unique();
        let mint_address = Pubkey::new_unique();
        let account = Account::default();

        cache
            .try_insert(token_address, account.clone(), mint_address)
            .unwrap();

        let retrieved_account = cache.try_get_account(&token_address).unwrap();
        assert_eq!(retrieved_account, account);
    }

    #[test]
    fn test_try_get_account_not_found() {
        let cache = TokensCache::default();
        let token_address = Pubkey::new_unique();

        let result = cache.try_get_account(&token_address);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_update_account() {
        let mut cache = TokensCache::default();
        let token_address = Pubkey::new_unique();
        let mint_address = Pubkey::new_unique();
        let account = Account::default();
        let updated_account = Account {
            lamports: 100,
            ..Account::default()
        };

        cache
            .try_insert(token_address, account, mint_address)
            .unwrap();
        cache
            .try_update_account(token_address, updated_account.clone(), 1)
            .unwrap();

        let retrieved_account = cache.try_get_account(&token_address).unwrap();
        assert_eq!(retrieved_account, updated_account);
    }

    #[test]
    fn test_get_token_for_mint() {
        let mut cache = TokensCache::default();
        let token_address = Pubkey::new_unique();
        let mint_address = Pubkey::new_unique();
        let account = Account::default();

        cache
            .try_insert(token_address, account, mint_address)
            .unwrap();

        let retrieved_token = cache.get_token_for_mint(&mint_address).unwrap();
        assert_eq!(retrieved_token, token_address);
    }

    #[test]
    fn test_try_get_token_for_mint_not_found() {
        let cache = TokensCache::default();
        let mint_address = Pubkey::new_unique();

        let result = cache.try_get_token_for_mint(&mint_address);
        assert!(result.is_err());
    }

    #[test]
    fn test_len() {
        let mut cache = TokensCache::default();
        let token_address = Pubkey::new_unique();
        let mint_address = Pubkey::new_unique();
        let account = Account::default();

        assert_eq!(cache.len().unwrap(), 0);

        cache
            .try_insert(token_address, account, mint_address)
            .unwrap();

        assert_eq!(cache.len().unwrap(), 1);
    }

    fn token(lamports: u64) -> Account {
        Account {
            lamports,
            ..Default::default()
        }
    }

    #[test]
    fn an_older_read_never_replaces_a_newer_one() {
        let cache = TokensCache::default();
        let address = Pubkey::new_unique();
        cache.try_update_account(address, token(1), 100).unwrap();
        cache.try_update_account(address, token(2), 99).unwrap();
        assert_eq!(cache.try_get_account(&address).unwrap().lamports, 1);
        cache.try_update_account(address, token(3), 100).unwrap();
        assert_eq!(cache.try_get_account(&address).unwrap().lamports, 3);
        cache.try_update_account(address, token(4), 0).unwrap();
        assert_eq!(cache.try_get_account(&address).unwrap().lamports, 3);
    }
}
