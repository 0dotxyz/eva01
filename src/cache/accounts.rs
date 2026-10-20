use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};

use indexmap::IndexMap;
use solana_pubkey::Pubkey;

use crate::wrappers::marginfi_account::MarginfiAccountWrapper;
use anyhow::{anyhow, Result};

#[derive(Default)]
struct MarginfiAccountsCacheInner {
    accounts: IndexMap<Pubkey, MarginfiAccountWrapper>,
    bank_to_accounts: HashMap<Pubkey, HashSet<Pubkey>>,
    accounts_without_liabilities: HashSet<Pubkey>,
}

#[derive(Default)]
pub struct MarginfiAccountsCache {
    inner: RwLock<MarginfiAccountsCacheInner>,
}

impl MarginfiAccountsCache {
    /// Inserts (or replaces) an account and refreshes its bank / liability indexes. Returns whether
    /// the account carries any liability, so callers can skip queuing debt-free accounts for the
    /// liquidatability check (they can't be liquidated until they borrow).
    pub fn try_insert(&self, account: MarginfiAccountWrapper) -> Result<bool> {
        let mut inner = self.inner.write().map_err(|e| {
            anyhow!(
                "Failed to lock the marginfi accounts cache for insert! {}",
                e
            )
        })?;

        if let Some(existing_account) = inner.accounts.get(&account.address).cloned() {
            Self::remove_account_indexes(&mut inner, &existing_account);
        }

        let has_liabilities = Self::has_liabilities(&account);
        Self::insert_account_indexes(&mut inner, &account);
        inner.accounts.insert(account.address, account);
        Ok(has_liabilities)
    }

    pub fn try_get_account(&self, address: &Pubkey) -> Result<MarginfiAccountWrapper> {
        self.inner
            .read()
            .map_err(|e| {
                anyhow!(
                    "Failed to lock the Marginfi accounts map for for search! {}",
                    e
                )
            })?
            .accounts
            .get(address)
            .ok_or(anyhow!("Failed to find the Marginfi account: {}", &address))
            .cloned()
    }

    pub fn try_get_accounts_with_liabilities(&self) -> Result<Vec<Pubkey>> {
        let inner = self.inner.read().map_err(|e| {
            anyhow!(
                "Failed to lock the marginfi accounts cache for liability lookup: {}",
                e
            )
        })?;

        Ok(inner
            .accounts
            .keys()
            .filter(|account| !inner.accounts_without_liabilities.contains(account))
            .copied()
            .collect())
    }

    pub fn try_get_accounts_for_bank_with_liabilities(
        &self,
        bank_address: &Pubkey,
    ) -> Result<Vec<Pubkey>> {
        let inner = self.inner.read().map_err(|e| {
            anyhow!(
                "Failed to lock the marginfi accounts cache for bank lookup: {}",
                e
            )
        })?;

        Ok(inner
            .bank_to_accounts
            .get(bank_address)
            .into_iter()
            .flat_map(|accounts| accounts.iter())
            .filter(|account| !inner.accounts_without_liabilities.contains(account))
            .copied()
            .collect())
    }

    fn insert_account_indexes(
        inner: &mut MarginfiAccountsCacheInner,
        account: &MarginfiAccountWrapper,
    ) {
        for bank_address in Self::active_bank_addresses(account) {
            inner
                .bank_to_accounts
                .entry(bank_address)
                .or_default()
                .insert(account.address);
        }

        if Self::has_liabilities(account) {
            inner.accounts_without_liabilities.remove(&account.address);
        } else {
            inner.accounts_without_liabilities.insert(account.address);
        }
    }

    fn remove_account_indexes(
        inner: &mut MarginfiAccountsCacheInner,
        account: &MarginfiAccountWrapper,
    ) {
        for bank_address in Self::active_bank_addresses(account) {
            if let Some(accounts) = inner.bank_to_accounts.get_mut(&bank_address) {
                accounts.remove(&account.address);
                if accounts.is_empty() {
                    inner.bank_to_accounts.remove(&bank_address);
                }
            }
        }
        inner.accounts_without_liabilities.remove(&account.address);
    }

    fn active_bank_addresses(account: &MarginfiAccountWrapper) -> HashSet<Pubkey> {
        account
            .account
            .lending_account
            .balances
            .iter()
            .filter_map(|balance| balance.is_active().then_some(balance.bank_pk))
            .collect()
    }

    /// Whether the account carries any liability (i.e. it can be liquidated). Reads marginfi's
    /// balance-derived indexer flags rather than scanning balances: they're synced on every
    /// balance-mutating instruction, so they're accurate the instant an account borrows or repays.
    ///
    /// `is_lending_only` is 1 only when the account has balances but none are liabilities;
    /// `is_empty` is 1 when it has no balances at all — so it has a liability iff neither is set.
    /// A borrower is never flagged lending-only (synced accounts → 0, and un-synced legacy accounts
    /// default to 0), so this can never wrongly skip a liquidatable account.
    fn has_liabilities(account: &MarginfiAccountWrapper) -> bool {
        let flags = &account.account.indexer_flags;
        flags.is_lending_only == 0 && flags.is_empty == 0
    }
}
