use super::marginfi_account::{MarginfiAccountWrapper, ObservationAccounts};
use crate::{
    cache::{Cache, DriftSpotMarket},
    config::Eva01Config,
    drift_ixs::make_refresh_spot_market_ix,
    juplend_ixs::make_update_lending_rate_ix,
    kamino_ixs::{make_refresh_obligation_ix, make_refresh_reserve_ix},
    marginfi_ixs::{
        initialize_marginfi_account, make_drift_withdraw_ix, make_end_liquidate_ix,
        make_init_liquidation_record_ix, make_juplend_withdraw_ix, make_kamino_withdraw_ix,
        make_repay_ix, make_start_liquidate_ix, make_withdraw_ix,
    },
    utils::{self, marginfi_account_by_authority},
};
use anyhow::{anyhow, Context, Result};
use fixed::types::I80F48;
use log::{debug, error, info};
use marginfi_type_crate::{
    constants::{
        ASSET_TAG_DEFAULT, ASSET_TAG_DRIFT, ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO, ASSET_TAG_SOL,
    },
    pdas::derive_drift_spot_market,
    types::Bank,
};
use solana_client::{rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig};

use solana_commitment_config::{CommitmentConfig, CommitmentLevel};
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_program::pubkey::Pubkey;
use solana_sdk::{
    account::ReadableAccount,
    instruction::Instruction,
    message::{v0::Message, VersionedMessage},
    signature::Keypair,
    signer::Signer,
    transaction::VersionedTransaction,
};
use std::{collections::HashSet, sync::Arc, thread, time::Duration};

pub const PROFIT_SHARE: f64 = 0.085;

/// Max serialized Solana transaction size on the wire.
const MAX_TX_SIZE: usize = 1232;

pub struct PreparedLiquidatableAccount {
    pub liquidatee_account: MarginfiAccountWrapper,
    pub observation_accounts: ObservationAccounts,
    pub asset_bank: Pubkey,
    pub liab_bank: Pubkey,
    pub asset_amount: I80F48,
    pub liab_amount: I80F48,
    pub profit: u64,
}

pub struct LiquidatorAccount {
    pub liquidator_address: Pubkey,
    pub signer: Keypair,
    group: Pubkey,
    rpc_client: RpcClient,
    cu_limit_ix: Instruction,
    pub cache: Arc<Cache>,
}

impl LiquidatorAccount {
    pub fn new(
        config: &Eva01Config,
        marginfi_group_id: Pubkey,
        cache: Arc<Cache>,
    ) -> Result<Self> {
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

        let liquidator_address = if accounts.is_empty() {
            info!("No MarginFi account found for the provided signer. Creating it...");
            let liquidator_marginfi_account =
                initialize_marginfi_account(&rpc_client, marginfi_group_id, &signer)?;

            while cache
                .marginfi_accounts
                .try_get_account(&liquidator_marginfi_account)
                .is_err()
            {
                info!("Waiting for the new account info to arrive...");
                thread::sleep(Duration::from_secs(5));
            }

            liquidator_marginfi_account
        } else {
            accounts[0]
        };

        Ok(Self {
            liquidator_address,
            signer,
            group: marginfi_group_id,
            rpc_client,
            cu_limit_ix: ComputeBudgetInstruction::set_compute_unit_limit(1400000),
            cache,
        })
    }

    pub fn init_liq_record(&self, liquidatee_account: &MarginfiAccountWrapper) -> Result<Pubkey> {
        info!(
            "Initializing liquidation record for account {:?} with liquidator account {:?}.",
            liquidatee_account.address, self.liquidator_address
        );

        let signer_pk = self.signer.pubkey();
        let (init_ix, liquidation_record) =
            make_init_liquidation_record_ix(liquidatee_account.address, signer_pk);

        let recent_blockhash = self
            .rpc_client
            .get_latest_blockhash()
            .map_err(|e| anyhow!(e))?;

        let msg = Message::try_compile(&signer_pk, &[init_ix], &[], recent_blockhash)
            .map_err(|e| anyhow!(e))?;

        let txn = VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])
            .map_err(|e| anyhow!(e))?;

        info!(
            "Sending liquidation tx for the Account {} .",
            liquidatee_account.address
        );
        match self
            .rpc_client
            .send_and_confirm_transaction_with_spinner_and_config(
                &txn,
                CommitmentConfig::finalized(),
                RpcSendTransactionConfig {
                    skip_preflight: false,
                    preflight_commitment: Some(CommitmentLevel::Processed),
                    ..Default::default()
                },
            ) {
            Ok(signature) => {
                info!(
                    "Liquidation record init tx for the Account {} was finalized. Signature: {}",
                    liquidatee_account.address, signature,
                );
                Ok(liquidation_record)
            }
            Err(err) => Err(anyhow!(
                "Liquidation record init tx for the Account {} failed: {} ",
                liquidatee_account.address,
                err
            )),
        }
    }

    /// Assemble (compile + sign) the liquidation transaction for the given account and amounts.
    /// Pure tx construction used by the execution-layer `InventoryStrategy`. Returns the signed tx,
    /// its instructions (for the LUT-retry path), and an optional temporary-LUT key created to fit
    /// an oversized tx (the caller must deactivate it once the tx lands).
    pub fn build_liquidate_tx(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
    ) -> Result<(VersionedTransaction, Vec<Instruction>, Option<Pubkey>)> {
        let liquidatee_account_address = account.liquidatee_account.address;
        let signer_pk = self.signer.pubkey();

        let asset_bank_wrapper = self.cache.banks.try_get_bank(&account.asset_bank)?;
        let liab_bank_wrapper = self.cache.banks.try_get_bank(&account.liab_bank)?;
        let asset_mint = asset_bank_wrapper.bank.mint;
        let liab_mint = liab_bank_wrapper.bank.mint;

        let ObservationAccounts {
            observation_accounts: liquidatee_observation_accounts,
            bank_pks: liquidatee_banks,
            kamino_reserves,
            drift_spot_markets,
            juplend_states,
            ..
        } = &account.observation_accounts;

        debug!(
            "The Liquidatee {:?} observation accounts: {:?}",
            liquidatee_account_address, liquidatee_observation_accounts
        );

        let liquidation_record =
            if account.liquidatee_account.account.liquidation_record == Pubkey::default() {
                self.init_liq_record(&account.liquidatee_account)?
            } else {
                account.liquidatee_account.account.liquidation_record
            };

        let all_tags: Vec<u8> = liquidatee_banks
            .iter()
            .filter_map(|pk| self.cache.banks.try_get_bank(pk).ok())
            .map(|b| b.bank.config.asset_tag)
            .collect();
        let luts = self.cache.luts.lock().unwrap().select_for_tags(&all_tags);

        let mut ixs = Vec::new();
        ixs.push(self.cu_limit_ix.clone());

        let start_ix = make_start_liquidate_ix(
            self.group,
            liquidatee_account_address,
            signer_pk,
            liquidation_record,
            liquidatee_observation_accounts.as_ref(),
            liquidatee_banks,
        );

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

        for spot_market_address in drift_spot_markets {
            let spot_market = self.cache.try_get_drift_market(spot_market_address)?;

            let refresh_spot_market_ix = make_refresh_spot_market_ix(
                *spot_market_address,
                spot_market.market.vault,
                spot_market.market.oracle,
            );
            ixs.push(refresh_spot_market_ix);
        }

        for lending_state_address in juplend_states {
            let lending_state = self
                .cache
                .try_get_juplend_lending_state(lending_state_address)?;

            let update_lending_rate_ix =
                make_update_lending_rate_ix(*lending_state_address, &lending_state);
            ixs.push(update_lending_rate_ix);
        }

        let asset_mint_wrapper = self.cache.mints.try_get_account(&asset_mint)?;

        let withdraw_ix = match asset_bank_wrapper.bank.config.asset_tag {
            ASSET_TAG_DEFAULT | ASSET_TAG_SOL => make_withdraw_ix(
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
            ASSET_TAG_DRIFT => {
                let (drift_spot_market, reward_spot_market, reward_spot_market_2) =
                    self.get_drift_spot_markets_for_bank(&asset_bank_wrapper.bank)?;

                make_drift_withdraw_ix(
                    self.group,
                    liquidatee_account_address,
                    signer_pk,
                    &asset_bank_wrapper,
                    &asset_mint_wrapper,
                    &drift_spot_market,
                    reward_spot_market.as_ref(),
                    reward_spot_market_2.as_ref(),
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

        ixs.push(start_ix);
        ixs.push(withdraw_ix);

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

        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let msg = Message::try_compile(&signer_pk, &ixs, &luts, recent_blockhash)?;
        let mut tx = VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])?;

        // Proactive LUT fit: if the tx exceeds the wire-size limit, pack its accounts into a
        // freshly-created targeted LUT and recompile. The caller deactivates the temp LUT once
        // the tx lands. (The 64-account-lock cap can't be fixed by a LUT, so that still fails.)
        let mut temp_lut: Option<Pubkey> = None;
        if bincode::serialize(&tx)?.len() > MAX_TX_SIZE {
            let all_accounts: Vec<Pubkey> = ixs
                .iter()
                .flat_map(|ix| ix.accounts.iter().map(|a| a.pubkey))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let lut = self
                .cache
                .get_targeted_lut(&self.rpc_client, &self.signer, all_accounts)?;
            temp_lut = Some(lut.key);

            let mut fitted_luts = luts;
            fitted_luts.push(lut);
            let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
            let msg = Message::try_compile(&signer_pk, &ixs, &fitted_luts, recent_blockhash)?;
            tx = VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])?;
        }

        Ok((tx, ixs, temp_lut))
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

    fn get_drift_spot_markets_for_bank(
        &self,
        bank: &Bank,
    ) -> Result<(
        DriftSpotMarket,
        Option<DriftSpotMarket>,
        Option<DriftSpotMarket>,
    )> {
        let drift_spot_market = self.cache.try_get_drift_market(&bank.integration_acc_1)?;

        let drift_user = self
            .cache
            .drift_users
            .get(&bank.integration_acc_2)
            .context(format!(
                "Couldn't find the data for Drift user: {}",
                bank.integration_acc_2
            ))?;

        // Note: rewards can take up to 2 positions, at indexes 2 and 3 (0 and 1 are for deposits).
        let (reward_spot_market, reward_spot_market_2) =
            if drift_user.spot_positions[2].scaled_balance > 0 {
                let reward_spot_market_address =
                    derive_drift_spot_market(drift_user.spot_positions[2].market_index).0;

                let reward_spot_market = self
                    .cache
                    .try_get_drift_market(&reward_spot_market_address)?;

                if drift_user.spot_positions[3].scaled_balance > 0 {
                    let reward_spot_market_2_address =
                        derive_drift_spot_market(drift_user.spot_positions[3].market_index).0;

                    let reward_spot_market_2 = self
                        .cache
                        .try_get_drift_market(&reward_spot_market_2_address)?;

                    (Some(reward_spot_market), Some(reward_spot_market_2))
                } else {
                    (Some(reward_spot_market), None)
                }
            } else {
                (None, None)
            };

        Ok((drift_spot_market, reward_spot_market, reward_spot_market_2))
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
