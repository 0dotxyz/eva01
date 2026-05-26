mod accounts;
mod banks;
pub mod mints;
mod oracles;
mod tokens;

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use accounts::MarginfiAccountsCache;
use anchor_lang::AccountDeserialize;
use anyhow::{anyhow, Result};
use banks::BanksCache;
use log::info;
use marginfi_type_crate::{
    constants::FEE_STATE_SEED, pdas::derive_kamino_lending_market_authority,
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
    wrappers::{oracle::OracleWrapperTrait, token_account::TokenAccountWrapper},
};

const LUT_CAPACITY: usize = 265usize;
const NEW_ADDRESSES_MAX: usize = 20usize;

pub struct Cache {
    pub signer_pk: Pubkey,
    pub marginfi_group_address: Pubkey,
    pub marginfi_accounts: MarginfiAccountsCache,
    pub banks: BanksCache,
    pub mints: MintsCache,
    pub oracles: OraclesCache,
    pub tokens: TokensCache,
    pub clock: Arc<Mutex<Clock>>,
    pub luts: Arc<Mutex<Vec<AddressLookupTableAccount>>>,
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
        let luts = Arc::new(Mutex::new(vec![]));
        Self {
            signer_pk,
            marginfi_group_address,
            marginfi_accounts: MarginfiAccountsCache::default(),
            banks: BanksCache::default(),
            mints: MintsCache::default(),
            oracles: OraclesCache::default(),
            tokens: TokensCache::default(),
            clock,
            luts,
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

    pub fn try_get_kamino_reserves(&self) -> Result<Vec<(Pubkey, KaminoReserve)>> {
        self.try_get_kamino_reserve_addresses()?
            .into_iter()
            .map(|address| Ok((address, self.try_get_kamino_reserve(&address)?)))
            .collect()
    }

    pub fn try_get_kamino_reserve_addresses(&self) -> Result<Vec<Pubkey>> {
        Ok(self.banks.get_kamino_reserves().into_iter().collect())
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

    pub fn try_get_juplend_lending_states(&self) -> Result<Vec<(Pubkey, Lending)>> {
        self.try_get_juplend_lending_state_addresses()?
            .into_iter()
            .map(|address| Ok((address, self.try_get_juplend_lending_state(&address)?)))
            .collect()
    }

    pub fn try_get_juplend_lending_state_addresses(&self) -> Result<Vec<Pubkey>> {
        Ok(self
            .banks
            .get_juplend_lending_states()
            .into_iter()
            .collect())
    }

    pub fn add_lut(&mut self, lut: AddressLookupTableAccount) {
        self.luts.lock().unwrap().push(lut)
    }

    pub fn try_get_token_wrapper<T: OracleWrapperTrait>(
        &self,
        mint_address: &Pubkey,
        token_address: &Pubkey,
    ) -> Result<TokenAccountWrapper<T>> {
        let token_account = self.tokens.try_get_account(token_address)?;
        let bank_address = self.banks.try_get_account_for_mint(mint_address)?;
        let bank_wrapper = self.banks.try_get_bank(&bank_address)?;
        let clock = clock_manager::get_clock(&self.clock)?;
        let oracle_wrapper = T::build(self, &clock, &bank_address)?;

        Ok(TokenAccountWrapper {
            balance: accessor::amount(&token_account.data)?,
            bank_wrapper,
            oracle_wrapper,
        })
    }

    pub fn add_addresses_to_lut(
        &self,
        rpc_client: &RpcClient,
        signer_keypair: &Keypair,
        addresses: HashSet<Pubkey>,
    ) -> anyhow::Result<()> {
        let mut luts = self.luts.lock().unwrap();
        if luts.is_empty() {
            let new_lut = create_lut(rpc_client, signer_keypair, addresses.into_iter().collect())?;
            luts.push(new_lut);
            return Ok(());
        }

        let lut = luts.last_mut().unwrap();
        let existing: HashSet<Pubkey> = lut.addresses.iter().cloned().collect();
        let addresses_to_add: Vec<Pubkey> = addresses
            .into_iter()
            .filter(|k| !existing.contains(k))
            .collect();
        if existing.len() + addresses_to_add.len() > LUT_CAPACITY {
            let new_lut = create_lut(rpc_client, signer_keypair, addresses_to_add)?;
            luts.push(new_lut);
            return Ok(());
        }

        lut.addresses = extend_lut(rpc_client, signer_keypair, lut.key, addresses_to_add)?;

        Ok(())
    }
}

fn create_lut(
    rpc_client: &RpcClient,
    signer_keypair: &Keypair,
    addresses: Vec<Pubkey>,
) -> anyhow::Result<AddressLookupTableAccount> {
    let recent_slot = rpc_client.get_slot()?;
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

    info!("Initialized new LUT: {:?}", lut_address);

    let updated_addresses = extend_lut(rpc_client, signer_keypair, lut_address, addresses)?;
    Ok(AddressLookupTableAccount {
        key: lut_address,
        addresses: updated_addresses.to_vec(),
    })
}

fn extend_lut(
    rpc_client: &RpcClient,
    signer_keypair: &Keypair,
    lut_address: Pubkey,
    addresses: Vec<Pubkey>,
) -> anyhow::Result<Vec<Pubkey>> {
    // Send multiple extend txs, each with at most NEW_ADDRESSES_MAX new addresses.
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

        info!(
            "Extended LUT: {:?} with the new addresses: {:?}",
            lut_address, chunk
        );
    }

    // Refresh LUT addresses once at the very end.
    let lut_account = rpc_client.get_account(&lut_address)?;
    let lut = AddressLookupTable::deserialize(&lut_account.data).unwrap();

    Ok(lut.addresses.to_vec())
}
