use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use solana_sdk::{account::Account, pubkey::Pubkey};

use crate::wrappers::mint::MintWrapper;

#[derive(Default)]
pub struct MintsCache {
    mints: HashMap<Pubkey, MintWrapper>,
    token_to_mint: HashMap<Pubkey, Pubkey>,
    excluded_mints: HashSet<Pubkey>,
}

impl MintsCache {
    pub fn new(excluded_mints: HashSet<Pubkey>) -> Self {
        Self {
            excluded_mints,
            ..Default::default()
        }
    }

    pub fn insert(&mut self, mint_address: Pubkey, mint: Account, token_address: Pubkey) {
        self.mints
            .insert(mint_address, MintWrapper::new(token_address, mint));
        self.token_to_mint.insert(token_address, mint_address);
    }

    pub fn try_get_account(&self, address: &Pubkey) -> Result<MintWrapper> {
        self.mints
            .get(address)
            .ok_or(anyhow!("Failed to find Mint for the Address {}!", &address))
            .cloned()
    }

    pub fn get_mints(&self) -> Vec<Pubkey> {
        self.mints
            .keys()
            .filter(|mint| !self.excluded_mints.contains(mint))
            .cloned()
            .collect()
    }

    /// Whether `mint` is in `EXCLUDED_LIQUIDATION_MINTS`: neither seized, bought, nor sold.
    pub fn is_excluded(&self, mint: &Pubkey) -> bool {
        self.excluded_mints.contains(mint)
    }

    pub fn get_tokens(&self) -> Vec<Pubkey> {
        self.mints.values().map(|mint| mint.token).collect()
    }

    pub fn try_get_mint_for_token(&self, token_address: &Pubkey) -> Result<Pubkey> {
        self.token_to_mint
            .get(token_address)
            .ok_or(anyhow!(
                "Failed to find Mint for the Token {}!",
                &token_address
            ))
            .copied()
    }

    pub fn len(&self) -> usize {
        self.mints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_sdk::{account::Account, pubkey::Pubkey};

    #[test]
    fn test_mints_cache() {
        let mut mints_cache = MintsCache::default();
        let mint_address = Pubkey::new_unique();
        let token_address = Pubkey::new_unique();
        let mint_account = Account::new(0, 0, &Pubkey::new_unique());

        mints_cache.insert(mint_address, mint_account.clone(), token_address);

        assert_eq!(mints_cache.len(), 1);
        assert!(mints_cache.try_get_account(&mint_address).is_ok());
        assert!(mints_cache.try_get_mint_for_token(&token_address).is_ok());
    }

    #[test]
    fn test_try_get_account_not_found() {
        let mints_cache = MintsCache::default();
        let mint_address = Pubkey::new_unique();

        let result = mints_cache.try_get_account(&mint_address);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_get_mint_for_token_not_found() {
        let mints_cache = MintsCache::default();
        let token_address = Pubkey::new_unique();

        let result = mints_cache.try_get_mint_for_token(&token_address);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_token() {
        let mut mints_cache = MintsCache::default();
        let mint_address = Pubkey::new_unique();
        let token_address = Pubkey::new_unique();
        let mint_account = Account::new(0, 0, &Pubkey::new_unique());

        mints_cache.insert(mint_address, mint_account.clone(), token_address);

        let tokens = mints_cache.get_tokens();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], token_address);
    }

    #[test]
    fn test_len() {
        let mut mints_cache = MintsCache::default();
        assert_eq!(mints_cache.len(), 0);

        let mint_address = Pubkey::new_unique();
        let token_address = Pubkey::new_unique();
        let mint_account = Account::new(0, 0, &Pubkey::new_unique());

        mints_cache.insert(mint_address, mint_account.clone(), token_address);

        assert_eq!(mints_cache.len(), 1);
    }

    #[test]
    fn test_insert_duplicate() {
        let mut mints_cache = MintsCache::default();
        let mint_address = Pubkey::new_unique();
        let token_address = Pubkey::new_unique();
        let mint_account = Account::new(0, 0, &Pubkey::new_unique());

        mints_cache.insert(mint_address, mint_account.clone(), token_address);
        mints_cache.insert(mint_address, mint_account.clone(), token_address);

        // Ensure the length remains 1
        assert_eq!(mints_cache.len(), 1);
    }
}
