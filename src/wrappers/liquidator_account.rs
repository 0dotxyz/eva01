use super::marginfi_account::{MarginfiAccountWrapper, ObservationAccounts};
use crate::{
    cache::Cache,
    config::Eva01Config,
    juplend_ixs::make_update_lending_rate_ix,
    kamino_ixs::{make_refresh_obligation_ix, make_refresh_reserve_ix},
    marginfi_ixs::{
        make_end_liquidate_ix, make_init_liquidation_record_ix, make_juplend_withdraw_ix,
        make_kamino_withdraw_ix, make_repay_ix, make_start_liquidate_ix, make_tag_liq_record_ix,
        make_withdraw_ix,
    },
    utils::{self, marginfi_account_by_authority},
};
use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use log::{debug, error, info, warn};
use marginfi_type_crate::constants::{
    ASSET_TAG_DEFAULT, ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO, ASSET_TAG_SOL, ASSET_TAG_STAKED,
};
use solana_account::ReadableAccount;
use solana_address_lookup_table_interface::state::AddressLookupTable;
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_dex_superagg::SwapInstructions;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::{v0::Message, AddressLookupTableAccount, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;
use std::{collections::HashSet, sync::Arc};

/// Collateral tags `build_liquidate_tx` can withdraw. Staked banks hold a plain SPL LST, so they go
/// through the standard withdraw like DEFAULT/SOL.
pub fn is_seizable_asset_tag(tag: u8) -> bool {
    matches!(
        tag,
        ASSET_TAG_DEFAULT | ASSET_TAG_SOL | ASSET_TAG_STAKED | ASSET_TAG_KAMINO | ASSET_TAG_JUPLEND
    )
}

/// Max serialized Solana transaction size on the wire.
const MAX_TX_SIZE: usize = 1232;

pub struct PreparedLiquidatableAccount {
    pub liquidatee_account: MarginfiAccountWrapper,
    pub observation_accounts: ObservationAccounts,
    pub asset_bank: Pubkey,
    pub liab_bank: Pubkey,
    pub asset_amount: I80F48,
    pub liab_amount: I80F48,
    pub profit: I80F48,
}

pub struct LiquidatorAccount {
    pub signer: Keypair,
    group: Pubkey,
    rpc_client: RpcClient,
    cu_limit_ix: Instruction,
    /// Priority-fee instruction (from `COMPUTE_UNIT_PRICE_MICRO_LAMPORTS`). Without it the heavy
    /// liquidation tx carries a CU limit but no price, so it confirms slowly under load — the
    /// sequential-send path especially needs this to land promptly.
    cu_price_ix: Instruction,
    pub cache: Arc<Cache>,
}

impl LiquidatorAccount {
    pub fn new(config: &Eva01Config, marginfi_group_id: Pubkey, cache: Arc<Cache>) -> Result<Self> {
        let signer = Keypair::try_from(config.wallet_keypair.as_slice())?;
        let rpc_client =
            RpcClient::new_with_commitment(config.rpc_url.clone(), CommitmentConfig::confirmed());

        let accounts =
            marginfi_account_by_authority(signer.pubkey(), &rpc_client, marginfi_group_id)?;
        info!(
            "Found {} MarginFi accounts for the provided signer: {:?}",
            accounts.len(),
            accounts
        );

        Ok(Self {
            signer,
            group: marginfi_group_id,
            rpc_client,
            cu_limit_ix: ComputeBudgetInstruction::set_compute_unit_limit(1400000),
            cu_price_ix: ComputeBudgetInstruction::set_compute_unit_price(
                config.compute_unit_price_micro_lamports,
            ),
            cache,
        })
    }

    /// Compiles and signs the liquidation transaction. With `swap`, its instructions (selling the
    /// seized collateral for the liability) go between the withdraw and the repay. Returns the
    /// transaction and the key of a temporary LUT, if one had to be created to fit it.
    pub fn build_liquidate_tx(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
        swap: Option<&SwapInstructions>,
    ) -> Result<(VersionedTransaction, Option<Pubkey>)> {
        let PreparedLiquidatableAccount {
            liquidatee_account,
            observation_accounts,
            asset_bank,
            liab_bank,
            ..
        } = account;

        let liquidatee_account_address = liquidatee_account.address;
        let signer_pk = self.signer.pubkey();

        let asset_bank_wrapper = self.cache.banks.try_get_bank(asset_bank)?;
        let liab_bank_wrapper = self.cache.banks.try_get_bank(liab_bank)?;
        let asset_mint = asset_bank_wrapper.bank.mint;
        let liab_mint = liab_bank_wrapper.bank.mint;

        let ObservationAccounts {
            observation_accounts: liquidatee_observation_accounts,
            bank_pks: liquidatee_banks,
            kamino_reserves,
            juplend_states,
            ..
        } = observation_accounts;

        debug!(
            "The Liquidatee {:?} observation accounts: {:?}",
            liquidatee_account_address, liquidatee_observation_accounts
        );

        // A first-time liquidatee has no record yet: create it in this same transaction, ahead of
        // `start_liquidation` (the program allows exactly that). A record created in its own
        // transaction can be closed by anyone before we use it, since a never-used record has no
        // inactivity lock, and waiting for it to finalize would stall the liquidation.
        let (init_record_ix, liquidation_record) =
            if liquidatee_account.account.liquidation_record == Pubkey::default() {
                let (ix, record) =
                    make_init_liquidation_record_ix(liquidatee_account_address, signer_pk);
                (Some(ix), record)
            } else {
                (None, liquidatee_account.account.liquidation_record)
            };

        let all_tags: Vec<u8> = liquidatee_banks
            .iter()
            .filter_map(|pk| self.cache.banks.try_get_bank(pk).ok())
            .map(|b| b.bank.config.asset_tag)
            .collect();
        let mut luts = self.cache.luts.lock().unwrap().select_for_tags(&all_tags);

        let mut ixs = Vec::new();
        ixs.push(self.cu_limit_ix.clone());
        ixs.push(self.cu_price_ix.clone());

        let start_ix = make_start_liquidate_ix(
            self.group,
            liquidatee_account_address,
            signer_pk,
            liquidation_record,
            liquidatee_observation_accounts.as_ref(),
            liquidatee_banks,
        );

        ixs.extend(self.venue_refresh_ixs(kamino_reserves, juplend_states)?);

        let asset_mint_wrapper = self.cache.mints.try_get_account(&asset_mint)?;

        let withdraw_ix = match asset_bank_wrapper.bank.config.asset_tag {
            ASSET_TAG_DEFAULT | ASSET_TAG_SOL | ASSET_TAG_STAKED => make_withdraw_ix(
                self.group,
                liquidatee_account_address,
                signer_pk,
                &asset_bank_wrapper,
                &asset_mint_wrapper,
                liquidatee_observation_accounts.as_ref(),
                asset_amount.to_num(),
                false,
            ),
            ASSET_TAG_KAMINO => {
                let kamino_reserve = self
                    .cache
                    .try_get_kamino_reserve(&asset_bank_wrapper.bank.integration_acc_1)?;

                let refresh_obligation_ix = make_refresh_obligation_ix(
                    asset_bank_wrapper.bank.integration_acc_2,
                    kamino_reserve.reserve.lending_market,
                    &[asset_bank_wrapper.bank.integration_acc_1],
                );
                ixs.push(refresh_obligation_ix);

                make_kamino_withdraw_ix(
                    self.group,
                    liquidatee_account_address,
                    signer_pk,
                    &asset_bank_wrapper,
                    &asset_mint_wrapper,
                    asset_bank_wrapper.bank.integration_acc_2,
                    &kamino_reserve,
                    liquidatee_observation_accounts.as_ref(),
                    asset_amount.to_num(),
                    false,
                )
            }
            ASSET_TAG_JUPLEND => {
                let lending_state = self
                    .cache
                    .try_get_juplend_lending_state(&asset_bank_wrapper.bank.integration_acc_1)?;

                make_juplend_withdraw_ix(
                    self.group,
                    liquidatee_account_address,
                    signer_pk,
                    &asset_bank_wrapper,
                    &asset_mint_wrapper,
                    &lending_state,
                    liquidatee_observation_accounts.as_ref(),
                    asset_amount.to_num(),
                    false,
                )
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported asset tag: {}",
                    asset_bank_wrapper.bank.config.asset_tag
                ));
            }
        };

        ixs.extend(init_record_ix);
        ixs.push(start_ix);
        ixs.push(withdraw_ix);
        if let Some(swap) = swap {
            ixs.extend(swap.instructions.iter().cloned());
            luts.extend(self.fetch_luts(&swap.address_lookup_tables)?);
        }

        let liab_mint_wrapper = self.cache.mints.try_get_account(&liab_mint)?;
        let repay_ix = make_repay_ix(
            self.group,
            liquidatee_account_address,
            signer_pk,
            &liab_bank_wrapper,
            &liab_mint_wrapper,
            liab_amount.to_num(),
            false,
        );
        ixs.push(repay_ix);

        let end_ix = make_end_liquidate_ix(
            self.group,
            liquidatee_account_address,
            signer_pk,
            liquidation_record,
            self.cache.global_fee_state_key,
            self.cache.global_fee_wallet,
            liquidatee_banks,
        );
        ixs.push(end_ix);

        self.compile_tx_with_lut_fit(&signer_pk, &ixs, &mut luts)
    }

    /// Kamino and JupLend venue accounts must be refreshed in the same slot as any health
    /// check that reads them.
    fn venue_refresh_ixs(
        &self,
        kamino_reserves: &HashSet<Pubkey>,
        juplend_states: &HashSet<Pubkey>,
    ) -> Result<Vec<Instruction>> {
        let mut ixs = Vec::new();
        for kamino_reserve_address in kamino_reserves {
            let kamino_reserve = self.cache.try_get_kamino_reserve(kamino_reserve_address)?;

            debug!(
                "Putting a refresh ix for Kamino Reserve: {}",
                kamino_reserve_address
            );

            let refresh_reserve_ix =
                make_refresh_reserve_ix(*kamino_reserve_address, &kamino_reserve);
            ixs.push(refresh_reserve_ix);
        }

        for lending_state_address in juplend_states {
            let lending_state = self
                .cache
                .try_get_juplend_lending_state(lending_state_address)?;

            let update_lending_rate_ix =
                make_update_lending_rate_ix(*lending_state_address, &lending_state);
            ixs.push(update_lending_rate_ix);
        }
        Ok(ixs)
    }

    /// Tags an unhealthy account so the receivership premium it pays grows over time. Sent over RPC
    /// and confirmed; there is no MEV to race for, since anyone may tag and the first tag wins.
    pub fn send_tag_tx(
        &self,
        liquidatee: Pubkey,
        observation_accounts: &ObservationAccounts,
    ) -> Result<Signature> {
        let ObservationAccounts {
            observation_accounts: liquidatee_observation_accounts,
            bank_pks: liquidatee_banks,
            kamino_reserves,
            juplend_states,
            ..
        } = observation_accounts;

        let mut ixs = vec![self.cu_limit_ix.clone(), self.cu_price_ix.clone()];
        ixs.extend(self.venue_refresh_ixs(kamino_reserves, juplend_states)?);
        ixs.push(make_tag_liq_record_ix(
            self.group,
            liquidatee,
            liquidatee_observation_accounts,
        ));

        let all_tags: Vec<u8> = liquidatee_banks
            .iter()
            .filter_map(|pk| self.cache.banks.try_get_bank(pk).ok())
            .map(|b| b.bank.config.asset_tag)
            .collect();
        let luts = self.cache.luts.lock().unwrap().select_for_tags(&all_tags);
        let tx = self.compile_tx(&self.signer.pubkey(), &ixs, &luts)?;
        Ok(self.rpc_client.send_and_confirm_transaction(&tx)?)
    }

    /// Simulates `tx`. On failure returns the program error with the last log lines.
    pub fn simulate_tx(&self, tx: &VersionedTransaction) -> Result<()> {
        let result = self.rpc_client.simulate_transaction(tx)?.value;
        match result.err {
            None => Ok(()),
            Some(err) => {
                let logs = result.logs.unwrap_or_default();
                let tail = logs.len().saturating_sub(5);
                Err(anyhow!(
                    "simulation failed: {:?}; logs: {:?}",
                    err,
                    &logs[tail..]
                ))
            }
        }
    }

    fn fetch_luts(&self, keys: &[Pubkey]) -> Result<Vec<AddressLookupTableAccount>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }
        self.rpc_client
            .get_multiple_accounts(keys)?
            .into_iter()
            .zip(keys)
            .map(|(account, key)| {
                let account = account.ok_or_else(|| anyhow!("Lookup table {} not found", key))?;
                Ok(AddressLookupTableAccount {
                    key: *key,
                    addresses: AddressLookupTable::deserialize(&account.data)?
                        .addresses
                        .to_vec(),
                })
            })
            .collect()
    }

    fn compile_tx_with_lut_fit(
        &self,
        signer_pk: &Pubkey,
        ixs: &[Instruction],
        luts: &mut Vec<AddressLookupTableAccount>,
    ) -> Result<(VersionedTransaction, Option<Pubkey>)> {
        let tx = self.compile_tx(signer_pk, ixs, luts)?;
        if bincode::serialize(&tx)?.len() <= MAX_TX_SIZE {
            return Ok((tx, None));
        }

        // Too large: put the transaction's accounts into a new lookup table and recompile. The
        // caller deactivates that table once the transaction lands. Measure with a stand-in table
        // first, so a transaction that wouldn't fit even then costs no table. (A table doesn't
        // help with the account-lock limit; that still fails.)
        let all_accounts: Vec<Pubkey> = ixs
            .iter()
            .flat_map(|ix| ix.accounts.iter().map(|a| a.pubkey))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        luts.push(AddressLookupTableAccount {
            key: Pubkey::new_unique(),
            addresses: all_accounts.clone(),
        });
        let size = bincode::serialize(&self.compile_tx(signer_pk, ixs, luts)?)?.len();
        luts.pop();
        if size > MAX_TX_SIZE {
            return Err(anyhow!(
                "transaction is {} bytes even with a lookup table (max {})",
                size,
                MAX_TX_SIZE
            ));
        }

        let lut = self
            .cache
            .get_targeted_lut(&self.rpc_client, &self.signer, all_accounts)?;
        let temp_lut = lut.key;
        luts.push(lut);
        match self.compile_tx(signer_pk, ixs, luts) {
            Ok(tx) => Ok((tx, Some(temp_lut))),
            Err(e) => {
                self.deactivate_luts(vec![temp_lut]);
                Err(e)
            }
        }
    }

    /// Deactivates temporary lookup tables whose transaction was abandoned, so their rent can be
    /// reclaimed later.
    pub fn deactivate_luts(&self, keys: Vec<Pubkey>) {
        for key in keys {
            if let Err(e) = self
                .cache
                .deactivate_lut(&self.rpc_client, &self.signer, key)
            {
                warn!("Failed to deactivate temporary LUT {}: {}", key, e);
            }
        }
    }

    fn compile_tx(
        &self,
        signer_pk: &Pubkey,
        ixs: &[Instruction],
        luts: &[AddressLookupTableAccount],
    ) -> Result<VersionedTransaction> {
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let msg = Message::try_compile(signer_pk, ixs, luts, recent_blockhash)?;
        VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])
            .map_err(Into::into)
    }

    pub fn get_token_balance_for_mint(&self, mint_address: &Pubkey) -> Option<u64> {
        let token_account_address = self.cache.tokens.get_token_for_mint(mint_address)?;
        match self.cache.tokens.try_get_account(&token_account_address) {
            Ok(account) => match utils::accessor::amount(account.data()) {
                Ok(amount) => Some(amount),
                Err(error) => {
                    error!(
                        "Failed to obtain balance amount for the Token {}: {}",
                        token_account_address, error
                    );
                    None
                }
            },
            Err(error) => {
                error!(
                    "Failed to get the Token account {}: {}",
                    token_account_address, error
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_stale_oracles(stale_oracles: &HashSet<Pubkey>, account_oracles: &[Pubkey]) -> bool {
        account_oracles
            .iter()
            .any(|oracle| stale_oracles.contains(oracle))
    }

    #[test]
    fn test_contains_stale_oracles_with_stale() {
        let stale_oracle = Pubkey::new_unique();
        let mut stale_oracles = HashSet::new();
        stale_oracles.insert(stale_oracle);
        let account_oracles = vec![Pubkey::new_unique(), stale_oracle, Pubkey::new_unique()];

        assert!(contains_stale_oracles(&stale_oracles, &account_oracles));
    }

    #[test]
    fn test_contains_stale_oracles_without_stale() {
        let stale_oracle = Pubkey::new_unique();
        let mut stale_oracles = HashSet::new();
        stale_oracles.insert(stale_oracle);
        let account_oracles = vec![Pubkey::new_unique(), Pubkey::new_unique()];

        assert!(!contains_stale_oracles(&stale_oracles, &account_oracles));
    }

    #[test]
    fn test_contains_stale_oracles_empty_account_oracles() {
        let stale_oracle = Pubkey::new_unique();
        let mut stale_oracles = HashSet::new();
        stale_oracles.insert(stale_oracle);
        let account_oracles = vec![];

        assert!(!contains_stale_oracles(&stale_oracles, &account_oracles));
    }

    #[test]
    fn test_contains_stale_oracles_empty_stale_oracles() {
        let account_oracles = vec![Pubkey::new_unique()];
        let stale_oracles = HashSet::new();

        assert!(!contains_stale_oracles(&stale_oracles, &account_oracles));
    }

    #[test]
    fn test_contains_stale_oracles_multiple_stale() {
        let stale1 = Pubkey::new_unique();
        let stale2 = Pubkey::new_unique();
        let mut stale_oracles = HashSet::new();
        stale_oracles.insert(stale1);
        stale_oracles.insert(stale2);
        let account_oracles = vec![stale2, Pubkey::new_unique()];

        assert!(contains_stale_oracles(&stale_oracles, &account_oracles));
    }
}
