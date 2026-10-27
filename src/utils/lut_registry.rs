//! Eva's own address lookup tables.
//!
//! Every liquidation references the group's banks, vaults, oracles and venue accounts, far more
//! than fit in a 1232-byte transaction as 32-byte keys. The registry keeps those addresses in
//! lookup tables the wallet owns, so the transaction compiler can shrink each one to an index:
//!
//! - at startup the tables already on-chain (authority = the wallet) are discovered, and every
//!   address the liquidation instructions can reference is added to them if missing;
//! - a bank that appears later has its addresses added in the background;
//! - when a transaction still doesn't fit (a swap route touching accounts outside its venue's
//!   tables), the uncovered addresses are learned in the background and the liquidation falls back
//!   to the plan without the swap; next time the route fits.
//!
//! Tables only ever grow: nothing is deactivated or closed, and a table is created only when the
//! current one is full. Referencing a table costs no account lock and ~34 bytes, and the compiler
//! references only the tables it draws from, so every table is passed on every compile.
//!
//! Optional, read-only extra tables (the per-group tables) are used the same way but never written.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use solana_account_decoder_client_types::UiAccountEncoding;
use solana_address_lookup_table_interface::{
    instruction::{create_lookup_table, extend_lookup_table},
    program::ID as LOOKUP_TABLE_PROGRAM_ID,
    state::AddressLookupTable,
};
use solana_client::{
    rpc_client::RpcClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig, RpcSendTransactionConfig},
    rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType},
};
use solana_commitment_config::CommitmentConfig;
use solana_keypair::Keypair;
use solana_message::AddressLookupTableAccount;
use solana_pubkey::Pubkey;
use solana_signer::Signer;

/// Lookup table account layout: 4-byte state tag, `deactivation_slot: u64`, `last_extended_slot:
/// u64`, `last_extended_slot_start_index: u8`, then `authority: Option<Pubkey>` (1-byte tag + key).
const AUTHORITY_OFFSET: usize = 22;
/// Addresses one table can hold.
const TABLE_CAPACITY: usize = 256;
/// Addresses one `extend` instruction can add.
const EXTEND_CHUNK: usize = 20;

#[derive(Default)]
struct Tables {
    /// Tables the wallet owns, oldest first. Only the last one can have room.
    owned: Vec<AddressLookupTableAccount>,
    /// Read-only tables from the configuration.
    extra: Vec<AddressLookupTableAccount>,
    /// Every address in any table above.
    covered: HashSet<Pubkey>,
}

#[derive(Default)]
pub struct LutRegistry {
    tables: Mutex<Tables>,
    /// Serializes on-chain writes, so two callers never create or extend at the same time.
    writer: Mutex<()>,
    /// Addresses queued for a background `ensure_covered`, so a burst of misses extends once.
    pending: Mutex<HashSet<Pubkey>>,
}

impl LutRegistry {
    /// Loads the active tables owned by `authority` from the chain. Returns how many were found.
    pub fn discover(&self, rpc_client: &RpcClient, authority: &Pubkey) -> Result<usize> {
        let accounts = rpc_client.get_program_ui_accounts_with_config(
            &LOOKUP_TABLE_PROGRAM_ID,
            RpcProgramAccountsConfig {
                filters: Some(vec![RpcFilterType::Memcmp(Memcmp::new(
                    AUTHORITY_OFFSET,
                    MemcmpEncodedBytes::Base58(authority.to_string()),
                ))]),
                account_config: RpcAccountInfoConfig {
                    encoding: Some(UiAccountEncoding::Base64),
                    commitment: Some(CommitmentConfig::confirmed()),
                    ..Default::default()
                },
                with_context: None,
                sort_results: None,
            },
        )?;

        let mut owned = Vec::new();
        for (key, ui_account) in accounts {
            let Some(data) = ui_account.data.decode() else {
                warn!("Skipping the undecodable lookup table {key}");
                continue;
            };
            match AddressLookupTable::deserialize(&data) {
                // A deactivated table can't take new addresses; leave it to expire.
                Ok(table) if table.meta.deactivation_slot != u64::MAX => {
                    debug!("Ignoring the deactivated lookup table {key}")
                }
                Ok(table) => owned.push(AddressLookupTableAccount {
                    key,
                    addresses: table.addresses.to_vec(),
                }),
                Err(e) => warn!("Skipping the unreadable lookup table {key}: {e:?}"),
            }
        }
        // Fill order: the fullest first, so appends go to the one table with room.
        owned.sort_by_key(|table| std::cmp::Reverse(table.addresses.len()));
        let found = owned.len();
        let mut tables = self.lock_tables()?;
        tables.owned = owned;
        tables.covered = Self::all_addresses(&tables);
        Ok(found)
    }

    /// Read-only tables from the configuration.
    pub fn set_extra(&self, extra: Vec<AddressLookupTableAccount>) -> Result<()> {
        let mut tables = self.lock_tables()?;
        tables.extra = extra;
        tables.covered = Self::all_addresses(&tables);
        Ok(())
    }

    /// Every table, to pass to the transaction compiler (it references only the ones it uses).
    pub fn tables(&self) -> Vec<AddressLookupTableAccount> {
        match self.lock_tables() {
            Ok(tables) => tables
                .owned
                .iter()
                .chain(tables.extra.iter())
                .cloned()
                .collect(),
            Err(e) => {
                warn!("{e}");
                Vec::new()
            }
        }
    }

    /// Of `keys`, those no table holds (deduplicated, in order).
    pub fn uncovered(&self, keys: &[Pubkey]) -> Vec<Pubkey> {
        let covered = match self.lock_tables() {
            Ok(tables) => tables.covered.clone(),
            Err(_) => HashSet::new(),
        };
        let mut seen = HashSet::new();
        keys.iter()
            .filter(|key| !covered.contains(key) && seen.insert(**key))
            .copied()
            .collect()
    }

    /// Adds every address of `keys` that no table holds yet to the owned tables, creating a table
    /// when the last one is full. Blocks until the chain confirms. Returns how many were added.
    pub fn ensure_covered(
        &self,
        rpc_client: &RpcClient,
        signer: &Keypair,
        keys: &[Pubkey],
    ) -> Result<usize> {
        let _writer = self
            .writer
            .lock()
            .map_err(|_| anyhow!("The lookup table writer lock is poisoned"))?;

        let missing = self.uncovered(keys);
        if missing.is_empty() {
            return Ok(0);
        }
        info!(
            "Adding {} address(es) to the owned lookup tables...",
            missing.len()
        );

        let mut remaining = missing.as_slice();
        while !remaining.is_empty() {
            let (key, room) = match self.table_with_room()? {
                Some(table) => table,
                None => (self.create_table(rpc_client, signer)?, TABLE_CAPACITY),
            };
            let (now, rest) = remaining.split_at(remaining.len().min(room));
            for chunk in now.chunks(EXTEND_CHUNK) {
                let ix = extend_lookup_table(
                    key,
                    signer.pubkey(),
                    Some(signer.pubkey()),
                    chunk.to_vec(),
                );
                send_confirmed(rpc_client, signer, ix)?;
            }
            self.reload_table(rpc_client, key)?;
            remaining = rest;
        }

        info!(
            "Added {} address(es) to the owned lookup tables.",
            missing.len()
        );
        Ok(missing.len())
    }

    /// `ensure_covered` on a background thread; addresses already queued are not queued again.
    pub fn cover_in_background(
        self: &Arc<Self>,
        rpc_url: String,
        signer_bytes: Vec<u8>,
        keys: Vec<Pubkey>,
    ) {
        let keys: Vec<Pubkey> = {
            let Ok(mut pending) = self.pending.lock() else {
                return;
            };
            self.uncovered(&keys)
                .into_iter()
                .filter(|key| pending.insert(*key))
                .collect()
        };
        if keys.is_empty() {
            return;
        }

        let registry = Arc::clone(self);
        thread::spawn(move || {
            let outcome = Keypair::try_from(signer_bytes.as_slice())
                .map_err(|e| anyhow!("{e}"))
                .and_then(|signer| {
                    let rpc_client =
                        RpcClient::new_with_commitment(rpc_url, CommitmentConfig::confirmed());
                    registry.ensure_covered(&rpc_client, &signer, &keys)
                });
            if let Err(e) = outcome {
                warn!("Failed to extend the owned lookup tables: {e}");
            }
            if let Ok(mut pending) = registry.pending.lock() {
                for key in &keys {
                    pending.remove(key);
                }
            }
        });
    }

    fn lock_tables(&self) -> Result<std::sync::MutexGuard<'_, Tables>> {
        self.tables
            .lock()
            .map_err(|_| anyhow!("The lookup table registry lock is poisoned"))
    }

    fn all_addresses(tables: &Tables) -> HashSet<Pubkey> {
        tables
            .owned
            .iter()
            .chain(tables.extra.iter())
            .flat_map(|table| table.addresses.iter().copied())
            .collect()
    }

    /// The owned table that still has room, with that room.
    fn table_with_room(&self) -> Result<Option<(Pubkey, usize)>> {
        Ok(self
            .lock_tables()?
            .owned
            .iter()
            .find(|table| table.addresses.len() < TABLE_CAPACITY)
            .map(|table| (table.key, TABLE_CAPACITY - table.addresses.len())))
    }

    fn create_table(&self, rpc_client: &RpcClient, signer: &Keypair) -> Result<Pubkey> {
        // The table address derives from `recent_slot`, which must still be in `SlotHashes` when
        // the create executes. A finalized slot is in every node's past and well inside the window;
        // the latest confirmed slot can be a node's future and fail with InvalidArgument.
        let recent_slot = rpc_client.get_slot_with_commitment(CommitmentConfig::finalized())?;
        let (ix, key) = create_lookup_table(signer.pubkey(), signer.pubkey(), recent_slot);
        send_confirmed(rpc_client, signer, ix)?;
        info!("Created the owned lookup table {key}.");
        self.lock_tables()?.owned.push(AddressLookupTableAccount {
            key,
            addresses: Vec::new(),
        });
        Ok(key)
    }

    /// Re-reads `key` from the chain, so the registry holds exactly what transactions can use.
    fn reload_table(&self, rpc_client: &RpcClient, key: Pubkey) -> Result<()> {
        let account = rpc_client.get_account(&key)?;
        let addresses = AddressLookupTable::deserialize(&account.data)
            .map_err(|e| anyhow!("Failed to read the lookup table {key}: {e:?}"))?
            .addresses
            .to_vec();
        let mut tables = self.lock_tables()?;
        match tables.owned.iter_mut().find(|table| table.key == key) {
            Some(table) => table.addresses = addresses,
            None => tables
                .owned
                .push(AddressLookupTableAccount { key, addresses }),
        }
        tables.covered = Self::all_addresses(&tables);
        debug!(
            "Lookup table {key} now holds {} addresses.",
            tables
                .owned
                .iter()
                .find(|table| table.key == key)
                .map_or(0, |table| table.addresses.len())
        );
        Ok(())
    }
}

fn send_confirmed(
    rpc_client: &RpcClient,
    signer: &Keypair,
    ix: solana_instruction::Instruction,
) -> Result<()> {
    let blockhash = rpc_client.get_latest_blockhash()?;
    let tx = solana_transaction::Transaction::new_signed_with_payer(
        &[ix],
        Some(&signer.pubkey()),
        &[signer],
        blockhash,
    );
    rpc_client.send_and_confirm_transaction_with_spinner_and_config(
        &tx,
        CommitmentConfig::confirmed(),
        RpcSendTransactionConfig {
            skip_preflight: true,
            ..Default::default()
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(n: usize) -> AddressLookupTableAccount {
        AddressLookupTableAccount {
            key: Pubkey::new_unique(),
            addresses: (0..n).map(|_| Pubkey::new_unique()).collect(),
        }
    }

    #[test]
    fn uncovered_dedupes_and_skips_known_addresses() {
        let registry = LutRegistry::default();
        let extra = table(3);
        let known = extra.addresses[1];
        registry.set_extra(vec![extra]).unwrap();

        let new = Pubkey::new_unique();
        assert_eq!(registry.uncovered(&[known, new, new, known]), vec![new]);
        assert!(registry.uncovered(&[known]).is_empty());
    }

    #[test]
    fn only_the_table_with_room_receives_appends() {
        let registry = LutRegistry::default();
        {
            let mut tables = registry.lock_tables().unwrap();
            tables.owned = vec![table(TABLE_CAPACITY), table(250)];
        }
        let (key, room) = registry.table_with_room().unwrap().unwrap();
        assert_eq!(key, registry.lock_tables().unwrap().owned[1].key);
        assert_eq!(room, 6);

        registry.lock_tables().unwrap().owned[1] = table(TABLE_CAPACITY);
        assert!(registry.table_with_room().unwrap().is_none());
    }

    #[test]
    fn tables_lists_owned_before_extra() {
        let registry = LutRegistry::default();
        let owned = table(1);
        let extra = table(1);
        registry.lock_tables().unwrap().owned = vec![owned.clone()];
        registry.set_extra(vec![extra.clone()]).unwrap();

        let keys: Vec<Pubkey> = registry.tables().iter().map(|t| t.key).collect();
        assert_eq!(keys, vec![owned.key, extra.key]);
    }
}
