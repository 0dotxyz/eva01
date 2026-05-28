mod accounts;
mod banks;
pub mod mints;
mod oracles;
mod tokens;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use accounts::MarginfiAccountsCache;
use anchor_lang::AccountDeserialize;
use anyhow::{anyhow, Result};
use banks::BanksCache;
use marginfi_type_crate::{
    constants::{ASSET_TAG_DEFAULT, ASSET_TAG_SOL, ASSET_TAG_STAKED, FEE_STATE_SEED},
    pdas::derive_kamino_lending_market_authority,
};
use mints::MintsCache;
use oracles::OraclesCache;
use solana_client::{rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig};
use solana_sdk::{
    address_lookup_table::{self, state::AddressLookupTable, AddressLookupTableAccount},
    clock::Clock,
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
};
use tokens::TokensCache;

use crate::{
    clock_manager,
    drift::accounts::{SpotMarket, User as DriftUser},
    juplend_earn::accounts::Lending,
    kamino_lending::accounts::Reserve,
    utils::accessor,
    wrappers::{oracle::OracleWrapper, token_account::TokenAccountWrapper},
};

#[derive(Default, Clone)]
pub struct GroupedLuts {
    pub group1: Vec<AddressLookupTableAccount>,
    pub group2: Vec<AddressLookupTableAccount>,
    pub group3: Vec<AddressLookupTableAccount>,
    /// On-the-fly LUTs created during retry_with_new_luts for edge-case liquidations.
    /// Persists across liquidations so the same edge case doesn't re-trigger creation.
    pub overflow: Vec<AddressLookupTableAccount>,
}

impl GroupedLuts {
    /// Select LUT groups covering all asset tags present in a liquidation position.
    /// - group1 included if any tag is DEFAULT(0) or SOL(1)
    /// - group2 included if any tag is SOL(1) or STAKED(2)
    /// - group3 included if any tag >= 3 (integration protocol)
    /// - overflow always included
    pub fn select_for_tags(&self, tags: &[u8]) -> Vec<AddressLookupTableAccount> {
        let has_staked = tags.iter().any(|&t| t == ASSET_TAG_STAKED);
        let needs_group1 = !has_staked
            && tags
                .iter()
                .any(|&t| t == ASSET_TAG_DEFAULT || t == ASSET_TAG_SOL);
        let needs_group2 = tags
            .iter()
            .any(|&t| t == ASSET_TAG_SOL || t == ASSET_TAG_STAKED);
        let needs_group3 = tags.iter().any(|&t| t >= 3);

        let mut result = Vec::new();
        if needs_group1 {
            result.extend_from_slice(&self.group1);
        }
        if needs_group2 {
            result.extend_from_slice(&self.group2);
        }
        if needs_group3 {
            result.extend_from_slice(&self.group3);
        }
        result.extend_from_slice(&self.overflow);
        result
    }
}

pub struct Cache {
    pub signer_pk: Pubkey,
    pub marginfi_group_address: Pubkey,
    pub marginfi_accounts: MarginfiAccountsCache,
    pub banks: BanksCache,
    pub mints: MintsCache,
    pub oracles: OraclesCache,
    pub tokens: TokensCache,
    pub clock: Arc<Mutex<Clock>>,
    pub luts: Arc<Mutex<GroupedLuts>>,
    pub global_fee_state_key: Pubkey,
    pub global_fee_wallet: Pubkey,
    pub drift_users: HashMap<Pubkey, DriftUser>,
}

#[derive(Clone)]
pub struct KaminoReserve {
    pub address: Pubkey,
    pub reserve: Reserve,
    pub lending_market_authority: Pubkey,
}

#[derive(Clone)]
pub struct DriftSpotMarket {
    pub address: Pubkey,
    pub market: SpotMarket,
}

impl Cache {
    pub fn new(
        signer_pk: Pubkey,
        marginfi_group_address: Pubkey,
        clock: Arc<Mutex<Clock>>,
    ) -> Self {
        let (global_fee_state_key, _) =
            Pubkey::find_program_address(&[FEE_STATE_SEED.as_bytes()], &marginfi_type_crate::ID);
        Self {
            signer_pk,
            marginfi_group_address,
            marginfi_accounts: MarginfiAccountsCache::default(),
            banks: BanksCache::default(),
            mints: MintsCache::default(),
            oracles: OraclesCache::default(),
            tokens: TokensCache::default(),
            clock,
            luts: Arc::new(Mutex::new(GroupedLuts::default())),
            global_fee_state_key,
            global_fee_wallet: Pubkey::default(),
            drift_users: HashMap::new(),
        }
    }

    fn build_kamino_reserve(address: Pubkey, reserve: Reserve) -> KaminoReserve {
        let lending_market_authority =
            derive_kamino_lending_market_authority(&reserve.lending_market).0;
        KaminoReserve {
            address,
            reserve,
            lending_market_authority,
        }
    }

    pub fn try_get_kamino_reserve(&self, address: &Pubkey) -> Result<KaminoReserve> {
        let account = self.oracles.try_get_account(address)?;
        let mut data: &[u8] = &account.data;
        let reserve = Reserve::try_deserialize(&mut data).map_err(|e| {
            anyhow!(
                "Failed to deserialize Kamino reserve {} from OracleCache: {}",
                address,
                e
            )
        })?;

        Ok(Self::build_kamino_reserve(*address, reserve))
    }

    pub fn try_get_drift_market(&self, address: &Pubkey) -> Result<DriftSpotMarket> {
        let account = self.oracles.try_get_account(address)?;
        let mut data: &[u8] = &account.data;
        let market = SpotMarket::try_deserialize(&mut data).map_err(|e| {
            anyhow!(
                "Failed to deserialize Drift spot market {} from OracleCache: {}",
                address,
                e
            )
        })?;

        Ok(DriftSpotMarket {
            address: *address,
            market,
        })
    }

    pub fn try_get_juplend_lending_state(&self, address: &Pubkey) -> Result<Lending> {
        let account = self.oracles.try_get_account(address)?;
        let mut data: &[u8] = &account.data;
        Lending::try_deserialize(&mut data).map_err(|e| {
            anyhow!(
                "Failed to deserialize Juplend lending state {} from OracleCache: {}",
                address,
                e
            )
        })
    }

    pub fn try_get_token_wrapper_lenient(
        &self,
        mint_address: &Pubkey,
        token_address: &Pubkey,
    ) -> Result<TokenAccountWrapper<OracleWrapper>> {
        let token_account = self.tokens.try_get_account(token_address)?;
        let bank_address = self.banks.try_get_account_for_mint(mint_address)?;
        let bank_wrapper = self.banks.try_get_bank(&bank_address)?;
        let clock = clock_manager::get_clock(&self.clock)?;
        let oracle_wrapper = OracleWrapper::build_lenient(self, &clock, &bank_address)?;

        Ok(TokenAccountWrapper {
            balance: accessor::amount(&token_account.data)?,
            bank_wrapper,
            oracle_wrapper,
        })
    }

    /// Creates a single targeted LUT containing exactly `accounts`, adds it to overflow,
    /// and returns it. Used by the tx-too-large retry path so the retry uses only this
    /// one tight LUT with no unrelated group-LUT header overhead.
    pub fn create_targeted_lut(
        &self,
        rpc_client: &RpcClient,
        signer_keypair: &Keypair,
        accounts: Vec<Pubkey>,
    ) -> anyhow::Result<AddressLookupTableAccount> {
        let lut = create_lut(rpc_client, signer_keypair, accounts)?;
        self.luts.lock().unwrap().overflow.push(lut.clone());
        Ok(lut)
    }
}

fn create_lut(
    rpc_client: &RpcClient,
    signer_keypair: &Keypair,
    addresses: Vec<Pubkey>,
) -> anyhow::Result<AddressLookupTableAccount> {
    let recent_slot = rpc_client.get_slot_with_commitment(CommitmentConfig::confirmed())?;
    let (create_ix, lut_address) = address_lookup_table::instruction::create_lookup_table(
        signer_keypair.pubkey(),
        signer_keypair.pubkey(),
        recent_slot,
    );

    let recent_blockhash = rpc_client.get_latest_blockhash()?;
    let tx = solana_sdk::transaction::Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&signer_keypair.pubkey()),
        &[signer_keypair],
        recent_blockhash,
    );
    rpc_client.send_and_confirm_transaction_with_spinner_and_config(
        &tx,
        CommitmentConfig::confirmed(),
        RpcSendTransactionConfig {
            skip_preflight: true,
            ..Default::default()
        },
    )?;

    let updated_addresses = extend_lut(rpc_client, signer_keypair, lut_address, addresses)?;
    Ok(AddressLookupTableAccount {
        key: lut_address,
        addresses: updated_addresses,
    })
}

fn extend_lut(
    rpc_client: &RpcClient,
    signer_keypair: &Keypair,
    lut_address: Pubkey,
    addresses: Vec<Pubkey>,
) -> anyhow::Result<Vec<Pubkey>> {
    const NEW_ADDRESSES_MAX: usize = 20;

    for chunk in addresses.chunks(NEW_ADDRESSES_MAX) {
        let ix = address_lookup_table::instruction::extend_lookup_table(
            lut_address,
            signer_keypair.pubkey(),
            Some(signer_keypair.pubkey()),
            chunk.to_vec(),
        );

        let recent_blockhash = rpc_client.get_latest_blockhash()?;
        let tx = solana_sdk::transaction::Transaction::new_signed_with_payer(
            &[ix],
            Some(&signer_keypair.pubkey()),
            &[signer_keypair],
            recent_blockhash,
        );
        rpc_client.send_and_confirm_transaction_with_spinner_and_config(
            &tx,
            CommitmentConfig::confirmed(),
            RpcSendTransactionConfig {
                skip_preflight: true,
                ..Default::default()
            },
        )?;
    }

    let lut_account = rpc_client.get_account(&lut_address)?;
    let lut = AddressLookupTable::deserialize(&lut_account.data).unwrap();
    Ok(lut.addresses.to_vec())
}
