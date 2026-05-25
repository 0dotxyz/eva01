pub mod healthcheck;
pub mod simulation_cache;
pub mod swb_cranker;
pub mod swb_price_fetcher;

use anyhow::{anyhow, Error, Result};
use backoff::ExponentialBackoff;
use log::{debug, error};
use marginfi::{bank_authority_seed, errors::MarginfiError, state::bank::BankVaultType};
use marginfi_type_crate::types::BankConfig;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use solana_account_decoder::{UiAccountEncoding, UiDataSliceConfig};
use solana_client::{
    rpc_client::RpcClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig},
    rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType},
};
use solana_program::pubkey::Pubkey;
use solana_sdk::account::Account;
use std::sync::{atomic::AtomicUsize, Arc};
use yellowstone_grpc_proto::geyser::SubscribeUpdateAccountInfo;

pub struct BatchLoadingConfig {
    pub max_batch_size: usize,
    pub max_concurrent_calls: usize,
}

impl BatchLoadingConfig {
    pub const DEFAULT: Self = Self {
        max_batch_size: 100,
        max_concurrent_calls: 16,
    };
}

/// Batch load accounts from the RPC client using the getMultipleAccounts RPC call.
///
/// - `max_batch_size`: The maximum number of accounts to load in a single RPC call.
/// - `max_concurrent_calls`: The maximum number of concurrent RPC calls.
///
/// This function will perform multiple RPC calls concurrently, up to `max_concurrent_calls`.
/// If the number of pending RPC calls exceeds `max_concurrent_calls`, the function will
/// await until some calls complete before initiating more, to respect the concurrency limit.
/// Additionally, logs progress information including the number of accounts being fetched,
/// the size of each chunk, and the current progress using trace and debug logs.
pub fn batch_get_multiple_accounts(
    rpc_client: &solana_client::rpc_client::RpcClient,
    addresses: &[Pubkey],
    BatchLoadingConfig {
        max_batch_size,
        max_concurrent_calls,
    }: BatchLoadingConfig,
) -> Result<Vec<Option<Account>>> {
    let batched_addresses = addresses.chunks(max_batch_size * max_concurrent_calls);
    let total_addresses = addresses.len();
    let total_batches = batched_addresses.len();

    let mut accounts = Vec::new();
    let fetched_accounts = Arc::new(AtomicUsize::new(0));

    for (batch_index, batch) in batched_addresses.enumerate() {
        let batch_size = batch.len();

        log::debug!(
            "Fetching batch {} / {} with {} addresses.",
            batch_index + 1,
            total_batches,
            batch_size
        );

        let mut batched_accounts = batch
            .par_chunks(max_batch_size)
            .map(|chunk| -> Result<Vec<_>> {
                let chunk = chunk.to_vec();
                let chunk_size = chunk.len();

                log::trace!(" - Fetching chunk of size {}", chunk_size);

                let chunk_res = backoff::retry(ExponentialBackoff::default(), move || {
                    let chunk = chunk.clone();

                    rpc_client
                        .get_multiple_accounts_with_config(
                            &chunk,
                            RpcAccountInfoConfig {
                                encoding: Some(UiAccountEncoding::Base64Zstd),
                                ..Default::default()
                            },
                        )
                        .map_err(backoff::Error::transient)
                })?
                .value;

                let fetched_chunk_size = chunk_res.len();

                fetched_accounts
                    .fetch_add(fetched_chunk_size, std::sync::atomic::Ordering::Relaxed);

                log::trace!(
                    " - Fetched chunk with {} accounts. Progress: {} / {}",
                    fetched_chunk_size,
                    fetched_accounts.load(std::sync::atomic::Ordering::Relaxed),
                    total_addresses
                );

                Ok(chunk_res)
            })
            .collect::<Result<Vec<_>>>()?
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();

        accounts.append(&mut batched_accounts);
    }

    log::debug!(
        "Finished fetching all batches. Total entries fetched: {}",
        fetched_accounts.load(std::sync::atomic::Ordering::Relaxed)
    );

    Ok(accounts)
}

pub mod accessor {
    use super::*;

    pub fn amount(bytes: &[u8]) -> Result<u64> {
        if bytes.len() < 72 {
            return Err(anyhow!("Invalid data length: {}", bytes.len()));
        }

        let mut amount_bytes = [0u8; 8];
        amount_bytes.copy_from_slice(&bytes[64..72]);
        Ok(u64::from_le_bytes(amount_bytes))
    }

    #[cfg(test)]
    pub fn mint(bytes: &[u8]) -> Pubkey {
        let mut mint_bytes = [0u8; 32];
        mint_bytes.copy_from_slice(&bytes[..32]);
        Pubkey::new_from_array(mint_bytes)
    }
}

pub fn account_update_to_account(account_update: &SubscribeUpdateAccountInfo) -> Result<Account> {
    let SubscribeUpdateAccountInfo {
        lamports,
        owner,
        executable,
        rent_epoch,
        data,
        ..
    } = account_update;

    let owner = Pubkey::try_from(owner.clone())
        .map_err(|e| anyhow!("Invalid pubkey: {:?}, error: {:?}", owner, e))?;

    let account = Account {
        lamports: *lamports,
        data: data.clone(),
        owner,
        executable: *executable,
        rent_epoch: *rent_epoch,
    };

    Ok(account)
}

pub fn find_bank_liquidity_vault_authority(bank_pk: &Pubkey, program_id: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(
        bank_authority_seed!(BankVaultType::Liquidity, bank_pk),
        program_id,
    )
    .0
}

pub fn find_oracle_keys(bank_config: &BankConfig) -> Vec<Pubkey> {
    bank_config
        .oracle_keys
        .iter()
        .filter_map(|key| {
            if *key != Pubkey::default() {
                Some(*key)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

#[macro_export]
macro_rules! ward {
    ($res:expr) => {
        match $res {
            Some(value) => value,
            None => return,
        }
    };
    ($res:expr, break) => {
        match $res {
            Some(value) => value,
            None => break,
        }
    };
    ($res:expr, continue) => {
        match $res {
            Some(value) => value,
            None => continue,
        }
    };
}

pub fn log_genuine_error(prefix: &str, error: Error) {
    match error.downcast::<anchor_lang::error::Error>() {
        Ok(error) => match error {
            anchor_lang::error::Error::AnchorError(anchor_error) => {
                match MarginfiError::from(anchor_error.error_code_number) {
                    MarginfiError::SwitchboardStalePrice | MarginfiError::PythPushStalePrice => {
                        debug!("Discarding the oracle stale price error");
                    }
                    MarginfiError::MathError => {
                        debug!("Discarding the empty staked bank error");
                    }
                    _ => {
                        error!("{}: MarginfiError - {}", prefix, anchor_error.error_msg);
                    }
                }
            }

            anchor_lang::error::Error::ProgramError(program_error) => {
                error!("{}: ProgramError - {}", prefix, program_error);
            }
        },
        Err(err) => error!("{}: {}", prefix, err),
    }
}

pub fn format_error_chain(err: &Error) -> String {
    let mut chain = err.chain();
    let primary = chain
        .next()
        .map(|cause| cause.to_string())
        .unwrap_or_else(|| "unknown error".to_string());

    let causes = chain.map(ToString::to_string).collect::<Vec<_>>();
    if causes.is_empty() {
        return primary;
    }

    format!("{primary} | caused by: {}", causes.join(" -> "))
}

pub fn marginfi_account_by_authority(
    authority: Pubkey,
    rpc_client: &RpcClient,
    marginfi_group_id: Pubkey,
) -> anyhow::Result<Vec<Pubkey>> {
    let marginfi_account_address = rpc_client.get_program_accounts_with_config(
        &marginfi_type_crate::ID,
        RpcProgramAccountsConfig {
            account_config: RpcAccountInfoConfig {
                encoding: Some(UiAccountEncoding::Base64),
                data_slice: Some(UiDataSliceConfig {
                    offset: 0,
                    length: 0,
                }),
                ..Default::default()
            },
            filters: Some(vec![
                RpcFilterType::Memcmp(Memcmp::new(
                    8,
                    MemcmpEncodedBytes::Base58(marginfi_group_id.to_string()),
                )),
                RpcFilterType::Memcmp(Memcmp::new(
                    8 + 32,
                    MemcmpEncodedBytes::Base58(authority.to_string()),
                )),
            ]),
            with_context: Some(false),
            sort_results: None,
        },
    )?;

    let marginfi_account_pubkeys: Vec<Pubkey> = marginfi_account_address
        .iter()
        .map(|(pubkey, _)| *pubkey)
        .collect();

    Ok(marginfi_account_pubkeys)
}

#[cfg(test)]
mod tests {

    use crate::utils::find_oracle_keys;
    use anyhow::anyhow;

    use super::{accessor, format_error_chain};
    use marginfi_type_crate::types::{BankConfig, OracleSetup};
    use solana_program::pubkey::Pubkey;

    #[test]
    fn test_accessor_amount_valid() {
        // 72 bytes, with bytes 64..72 set to a known u64 value (e.g., 0x0102030405060708)
        let mut data = vec![0u8; 72];
        let value: u64 = 0x0102030405060708;
        data[64..72].copy_from_slice(&value.to_le_bytes());
        let result = accessor::amount(&data).unwrap();
        assert_eq!(result, value);
    }

    #[test]
    fn test_accessor_amount_invalid_length() {
        // Less than 72 bytes should error
        let data = vec![0u8; 50];
        let result = accessor::amount(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_accessor_amount_all_zeros() {
        // 72 bytes, all zeros, should return 0
        let data = vec![0u8; 72];
        let result = accessor::amount(&data).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_accessor_mint() {
        // 32 bytes for mint, rest can be anything
        let mut data = vec![0u8; 40];
        let mint_bytes: [u8; 32] = [1; 32];
        data[..32].copy_from_slice(&mint_bytes);
        let mint = accessor::mint(&data);
        assert_eq!(mint, Pubkey::new_from_array(mint_bytes));
    }

    #[test]
    fn test_find_oracle_keys_pyth_pull() {
        let mut config = BankConfig::default();
        let mut keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 0);

        config.oracle_setup = OracleSetup::PythPushOracle;

        let feed_id = Pubkey::new_unique();
        config.oracle_keys[0] = feed_id;

        keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 1);

        assert_eq!(keys[0], feed_id);

        // Migrate the bank and check again
        config.config_flags = 1;

        keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], feed_id);
    }

    #[test]
    fn test_find_oracle_keys_staked_pyth_pull() {
        let mut config = BankConfig::default();

        config.oracle_setup = OracleSetup::StakedWithPythPush;

        let feed_id = Pubkey::new_unique();
        config.oracle_keys[0] = feed_id;
        let spl_mint = Pubkey::new_unique();
        config.oracle_keys[1] = spl_mint;
        let spl_sol_pool = Pubkey::new_unique();
        config.oracle_keys[2] = spl_sol_pool;

        let mut keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 3);

        assert_eq!(keys[0], feed_id);
        assert_eq!(keys[1], spl_mint);
        assert_eq!(keys[2], spl_sol_pool);

        // Migrate the bank and check again
        config.config_flags = 1;

        keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], feed_id);
        assert_eq!(keys[1], spl_mint);
        assert_eq!(keys[2], spl_sol_pool);
    }

    #[test]
    fn test_find_oracle_keys_swb() {
        let mut config = BankConfig::default();
        let mut keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 0);

        config.oracle_setup = OracleSetup::SwitchboardPull;

        let feed_id = Pubkey::new_unique();
        config.oracle_keys[0] = feed_id;

        keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], feed_id);

        // "Migrate" (no-op for Swb oracles) the bank and check again
        config.config_flags = 1;

        keys = find_oracle_keys(&config);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], feed_id);
    }

    #[test]
    fn test_format_error_chain_includes_all_causes() {
        let err = anyhow!("root failure")
            .context("middle failure")
            .context("top-level failure");

        let formatted = format_error_chain(&err);

        assert!(formatted.contains("top-level failure"));
        assert!(formatted.contains("middle failure"));
        assert!(formatted.contains("root failure"));
        assert!(formatted.contains("caused by:"));
    }
}
