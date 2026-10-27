use std::{collections::HashMap, sync::RwLock};

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use log::error;
use solana_account::Account;
use solana_pubkey::Pubkey;

//The Liquidator Token accounts
#[derive(Default)]
pub struct TokensCache {
    tokens: RwLock<IndexMap<Pubkey, Account>>,
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
            .insert(token_address, token);
        self.mint_to_token.insert(mint_address, token_address);
        Ok(())
    }

    pub fn try_get_account(&self, address: &Pubkey) -> Result<Account> {
        self.tokens
            .read()
            .map_err(|err| anyhow!("Failed to lock the tokens map for search! {}", err))?
            .get(address)
            .cloned()
            .ok_or(anyhow!("Failed to find Token {}!", &address))
    }

    pub fn try_update_account(&self, address: Pubkey, account: Account) -> Result<()> {
        self.tokens
            .write()
            .map_err(|e| anyhow!("Failed to lock the token map for update! {}", e))?
            .insert(address, account);
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
            .try_update_account(token_address, updated_account.clone())
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
}
