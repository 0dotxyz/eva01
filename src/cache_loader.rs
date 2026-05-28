use std::{collections::HashMap, sync::Arc};

use anchor_lang::AccountDeserialize;
use anchor_spl::associated_token;
use anyhow::Ok;
use log::{debug, error, info, warn};
use marginfi_type_crate::types::{Bank, MarginfiAccount, MarginfiGroup};
use solana_account_decoder::{UiAccountEncoding, UiDataSliceConfig};
use solana_client::{
    rpc_client::RpcClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig, RpcSendTransactionConfig},
    rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType},
};
use solana_sdk::{
    account::Account,
    address_lookup_table::{state::AddressLookupTable, AddressLookupTableAccount},
    bs58,
    commitment_config::{CommitmentConfig, CommitmentLevel},
    instruction::Instruction,
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
};

use crate::{
    cache::{Cache, GroupedLuts},
    drift,
    geyser::AccountType,
    utils::{batch_get_multiple_accounts, BatchLoadingConfig},
    wrappers::marginfi_account::MarginfiAccountWrapper,
};
use anchor_client::Program;
use anyhow::{anyhow, Result};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const BANK_GROUP_PK_OFFSET: usize = 32 + 1 + 8;
const MAX_INIT_TA_IXS: usize = 10;

pub struct CacheLoader {
    signer: Keypair,
    rpc_url: String,
    rpc_client: RpcClient,
    lut_addresses_group1: Vec<Pubkey>,
    lut_addresses_group2: Vec<Pubkey>,
    lut_addresses_group3: Vec<Pubkey>,
}

impl CacheLoader {
    pub fn new(
        wallet_keypair: &[u8],
        rpc_url: String,
        lut_addresses_group1: Vec<Pubkey>,
        lut_addresses_group2: Vec<Pubkey>,
        lut_addresses_group3: Vec<Pubkey>,
    ) -> Result<Self> {
        let signer = Keypair::from_bytes(wallet_keypair)?;
        let rpc_client = RpcClient::new(&rpc_url);

        Ok(Self {
            signer,
            rpc_url,
            rpc_client,
            lut_addresses_group1,
            lut_addresses_group2,
            lut_addresses_group3,
        })
    }

    pub fn load_cache(&self, cache: &mut Cache) -> anyhow::Result<()> {
        let marginfi_group_account = self.rpc_client.get_account(&cache.marginfi_group_address)?;
        let marginfi_group =
            bytemuck::from_bytes::<MarginfiGroup>(&marginfi_group_account.data[8..]);
        cache.global_fee_wallet = marginfi_group.fee_state_cache.global_fee_wallet;

        self.load_luts(cache)?;
        self.load_marginfi_accounts(cache)?;
        self.load_banks(cache)?;
        self.load_mints(cache)?;
        self.load_oracles(cache)?;
        self.load_tokens(cache)?;
        self.load_drift_users(cache)?;

        Ok(())
    }

    fn load_luts(&self, cache: &mut Cache) -> anyhow::Result<()> {
        info!("Loading LUTs.");

        let group1 = self.load_lut_vec(&self.lut_addresses_group1, "group1")?;
        let group2 = self.load_lut_vec(&self.lut_addresses_group2, "group2")?;
        let group3 = self.load_lut_vec(&self.lut_addresses_group3, "group3")?;

        info!(
            "Loaded LUTs: group1={}, group2={}, group3={}",
            group1.len(),
            group2.len(),
            group3.len()
        );

        *cache.luts.lock().unwrap() = GroupedLuts {
            group1,
            group2,
            group3,
            overflow: vec![],
        };

        Ok(())
    }

    fn load_lut_vec(
        &self,
        addresses: &[Pubkey],
        label: &str,
    ) -> anyhow::Result<Vec<AddressLookupTableAccount>> {
        if addresses.is_empty() {
            return Ok(vec![]);
        }

        let lut_accounts = self.rpc_client.get_multiple_accounts(addresses)?;

        addresses
            .iter()
            .zip(lut_accounts)
            .map(|(lut_address, lut_account_opt)| {
                let lut_account = lut_account_opt
                    .ok_or_else(|| anyhow!("Failed to find {} LUT {}.", label, lut_address))?;
                let lut = AddressLookupTable::deserialize(&lut_account.data).map_err(|e| {
                    anyhow!(
                        "Failed to deserialize {} LUT {}: {:?}",
                        label,
                        lut_address,
                        e
                    )
                })?;
                Ok(AddressLookupTableAccount {
                    key: *lut_address,
                    addresses: lut.addresses.to_vec(),
                })
            })
            .collect()
    }

    fn load_marginfi_accounts(&self, cache: &mut Cache) -> anyhow::Result<()> {
        info!("Loading marginfi accounts, this may take a few minutes, please be patient!");
        let start = std::time::Instant::now();
        let marginfi_accounts_pubkeys =
            self.load_marginfi_account_addresses(&cache.marginfi_group_address)?;

        info!("Loading marginfi accounts...");
        let marginfi_accounts = batch_get_multiple_accounts(
            &self.rpc_client,
            &marginfi_accounts_pubkeys,
            BatchLoadingConfig {
                max_batch_size: BatchLoadingConfig::DEFAULT.max_batch_size,
                max_concurrent_calls: 32,
            },
        )?;
        let marginfi_accounts_length = marginfi_accounts.len();

        for (address, account_opt) in marginfi_accounts_pubkeys
            .into_iter()
            .zip(marginfi_accounts.into_iter())
        {
            if let Some(account) = account_opt {
                let marginfi_account = bytemuck::from_bytes::<MarginfiAccount>(&account.data[8..]);
                let maw = MarginfiAccountWrapper::new(address, *marginfi_account);
                cache.marginfi_accounts.try_insert(maw)?;
            } else {
                warn!("Couldn't load Marginfi account for key: {}", address);
            }
        }

        info!(
            "Loaded {} marginfi accounts in {:?}",
            marginfi_accounts_length,
            start.elapsed()
        );

        Ok(())
    }

    fn load_marginfi_account_addresses(
        &self,
        marginfi_group_address: &Pubkey,
    ) -> anyhow::Result<Vec<Pubkey>> {
        info!("Loading marginfi account addresses...");
        let marginfi_account_addresses = &self.rpc_client.get_program_accounts_with_config(
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
                        MemcmpEncodedBytes::Base58(marginfi_group_address.to_string()),
                    )),
                    RpcFilterType::Memcmp(Memcmp::new(
                        0,
                        MemcmpEncodedBytes::Base58(
                            bs58::encode(MarginfiAccount::DISCRIMINATOR).into_string(),
                        ),
                    )),
                ]),
                with_context: Some(false),
                sort_results: None,
            },
        )?;

        let marginfi_account_pubkeys: Vec<Pubkey> = marginfi_account_addresses
            .iter()
            .map(|(pubkey, _)| *pubkey)
            .collect();

        info!(
            "Loaded {} marginfi account addresses.",
            marginfi_account_pubkeys.len()
        );
        Ok(marginfi_account_pubkeys)
    }

    fn load_banks(&self, cache: &mut Cache) -> anyhow::Result<()> {
        let anchor_client = anchor_client::Client::new(
            anchor_client::Cluster::Custom(self.rpc_url.clone(), String::from("")),
            Arc::new(Keypair::new()),
        );

        let program: Program<Arc<Keypair>> = anchor_client.program(marginfi_type_crate::ID)?;

        info!("Loading banks...");
        let bank_accounts = program.rpc().get_program_accounts_with_config(
            &marginfi_type_crate::ID,
            RpcProgramAccountsConfig {
                account_config: RpcAccountInfoConfig {
                    encoding: Some(UiAccountEncoding::Base64),
                    ..Default::default()
                },
                filters: Some(vec![
                    RpcFilterType::Memcmp(Memcmp::new(
                        0,
                        MemcmpEncodedBytes::Base58(bs58::encode(Bank::DISCRIMINATOR).into_string()),
                    )),
                    RpcFilterType::Memcmp(Memcmp::new_base58_encoded(
                        BANK_GROUP_PK_OFFSET,
                        cache.marginfi_group_address.as_ref(),
                    )),
                ]),
                with_context: Some(false),
                sort_results: None,
            },
        )?;

        for (bank_address, bank_account) in bank_accounts.into_iter() {
            let mut data = bank_account.data.as_slice();
            let bank = Bank::try_deserialize(&mut data)
                .map_err(|e| anyhow!("Failed to deserialize Bank {}: {:?}", bank_address, e))?;
            cache.banks.insert(bank_address, bank, bank_account);
            debug!("Loaded the Bank {:?}.", bank_address);
        }

        info!("Loaded {} banks", cache.banks.len());

        Ok(())
    }

    fn load_oracles(&self, cache: &mut Cache) -> anyhow::Result<()> {
        info!("Loading Oracles...");

        let oracle_addresses = cache.banks.get_oracles().into_iter().collect::<Vec<_>>();
        let oracle_accounts = batch_get_multiple_accounts(
            &self.rpc_client,
            &oracle_addresses,
            BatchLoadingConfig::DEFAULT,
        )?;
        let oracle_map: HashMap<Pubkey, Account> = oracle_addresses
            .iter()
            .zip(oracle_accounts.iter())
            .filter_map(|(pk, account)| account.as_ref().map(|acc| (*pk, acc.clone())))
            .collect();

        for (oracle_address, oracle_account) in oracle_map.iter() {
            debug!("Loaded the Oracle {:?} .", oracle_address);
            cache
                .oracles
                .try_insert(*oracle_address, oracle_account.clone())?;
        }

        info!("Loaded {} Oracles.", cache.oracles.len()?);

        Ok(())
    }

    fn load_mints(&self, cache: &mut Cache) -> anyhow::Result<()> {
        info!("Loading Mints");
        let mint_addresses = cache.banks.get_mints();
        let mint_accounts = batch_get_multiple_accounts(
            &self.rpc_client,
            &mint_addresses,
            BatchLoadingConfig::DEFAULT,
        )?;

        for (mint_address, mint_account_opt) in mint_addresses.iter().zip(mint_accounts.iter()) {
            if let Some(mint_account) = mint_account_opt {
                let token_address = associated_token::get_associated_token_address_with_program_id(
                    &cache.signer_pk,
                    mint_address,
                    &mint_account.owner,
                );

                cache
                    .mints
                    .insert(*mint_address, mint_account.clone(), token_address);
                debug!("Loaded the Mint {:?}.", mint_address);
            } else {
                error!("Mint account {:?} not found.", mint_address);
            }
        }

        info!("Loaded {} mints.", cache.mints.len());
        Ok(())
    }

    fn load_tokens(&self, cache: &mut Cache) -> anyhow::Result<()> {
        info!("Loading Token accounts...");

        let token_addresses = cache.mints.get_tokens();
        let token_accounts = batch_get_multiple_accounts(
            &self.rpc_client,
            &token_addresses,
            BatchLoadingConfig::DEFAULT,
        )?;

        let mut new_token_addresses: Vec<Pubkey> = vec![];
        for (token_address, token_account_opt) in token_addresses.iter().zip(token_accounts.iter())
        {
            match token_account_opt {
                Some(token_account) => {
                    let mint_address = cache.mints.try_get_mint_for_token(token_address)?;
                    cache
                        .tokens
                        .try_insert(*token_address, token_account.clone(), mint_address)?;
                    debug!("Loaded the Token account {:?}.", token_address);
                }
                None => {
                    new_token_addresses.push(*token_address);
                }
            }
        }
        info!("Loaded {}  Token accounts.", cache.tokens.len()?);

        if !new_token_addresses.is_empty() {
            info!("Creating {} new Token accounts", new_token_addresses.len());

            let mut instructions: Vec<Instruction> = vec![];
            for token_address in &new_token_addresses {
                let mint_account = cache.mints.try_get_mint_for_token(token_address)?;
                let mint = cache.mints.try_get_account(&mint_account)?;
                let signer_pk = cache.signer_pk;
                let ix = spl_associated_token_account::instruction::create_associated_token_account_idempotent(
                    &signer_pk,
                    &signer_pk,
                    &mint_account,
                    &mint.account.owner
                );
                instructions.push(ix);
            }

            self.create_token_accounts(instructions)?;

            info!("Fetching newly created Token accounts...");
            let new_token_accounts = batch_get_multiple_accounts(
                &self.rpc_client,
                &new_token_addresses,
                BatchLoadingConfig::DEFAULT,
            )?;
            for (new_token_address, new_token_account_opt) in
                new_token_addresses.iter().zip(new_token_accounts.iter())
            {
                if let Some(new_token_account) = new_token_account_opt {
                    let mint_address = cache.mints.try_get_mint_for_token(new_token_address)?;
                    cache.tokens.try_insert(
                        *new_token_address,
                        new_token_account.clone(),
                        mint_address,
                    )?;
                    debug!(
                        "Fetched the newly created Token account {:?}.",
                        new_token_address
                    );
                }
            }
            info!(
                "Fetched {} newly created Token accounts.",
                new_token_addresses.len()
            );
        }

        Ok(())
    }

    fn create_token_accounts(&self, instructions: Vec<Instruction>) -> Result<()> {
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;

        instructions
            .par_iter()
            .chunks(MAX_INIT_TA_IXS)
            .try_for_each(|chunk| {
                let ixs = chunk.iter().map(|ix| (*ix).clone()).collect::<Vec<_>>();

                let tx = solana_sdk::transaction::Transaction::new_signed_with_payer(
                    &ixs,
                    Some(&self.signer.pubkey()),
                    &[&self.signer],
                    recent_blockhash,
                );

                info!("Registering {} new Token accounts...", chunk.len());

                let sig = self
                    .rpc_client
                    .send_and_confirm_transaction_with_spinner_and_config(
                        &tx,
                        CommitmentConfig::finalized(),
                        RpcSendTransactionConfig {
                            skip_preflight: false,
                            preflight_commitment: Some(CommitmentLevel::Processed),
                            ..Default::default()
                        },
                    )?;

                info!(
                    "{} new Token accounts were successfully registered. Txn sig: {:?}",
                    chunk.len(),
                    sig
                );

                Ok(())
            })?;

        Ok(())
    }

    fn load_drift_users(&self, cache: &mut Cache) -> anyhow::Result<()> {
        info!("Loading Drift Users...");

        let user_addresses = cache
            .banks
            .get_drift_users()
            .into_iter()
            .collect::<Vec<_>>();
        let user_accounts = batch_get_multiple_accounts(
            &self.rpc_client,
            &user_addresses,
            BatchLoadingConfig::DEFAULT,
        )?;

        for (&address, account) in user_addresses.iter().zip(user_accounts.iter()) {
            if let Some(account) = account {
                debug!("Loaded the Drift User: {:?}", address);
                let mut data: &[u8] = &account.data;
                let user = drift::accounts::User::try_deserialize(&mut data)?;

                cache.drift_users.insert(address, user);
            } else {
                error!("User account {:?} not found.", address);
            }
        }

        info!("Loaded {} Drift Users.", cache.drift_users.len());

        Ok(())
    }
}

pub fn get_accounts_to_track(cache: &Cache) -> Result<HashMap<Pubkey, AccountType>> {
    let mut accounts: HashMap<Pubkey, AccountType> = HashMap::new();

    let swb_oracles = cache.banks.get_swb_oracles();
    for oracle_pk in cache.oracles.try_get_addresses()? {
        if !swb_oracles.contains(&oracle_pk) {
            accounts.insert(oracle_pk, AccountType::Oracle);
        }
    }

    for token in cache.mints.get_tokens() {
        accounts.insert(token, AccountType::Token);
    }

    Ok(accounts)
}
