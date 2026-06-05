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
    metrics::{LIQUIDATION_ATTEMPTS, LIQUIDATION_LATENCY_SECONDS, LIQUIDATION_SUCCESSES},
    utils::{
        self, marginfi_account_by_authority, simulation_cache::is_tx_too_large_client,
        swb_cranker::is_stale_swb_price_error,
    },
};
use anyhow::{anyhow, Context, Result};
use fixed::types::I80F48;
use log::{debug, error, info, warn};
use marginfi_type_crate::{
    constants::{
        ASSET_TAG_DEFAULT, ASSET_TAG_DRIFT, ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO, ASSET_TAG_SOL,
    },
    pdas::derive_drift_spot_market,
    types::{validate_asset_tags, Bank},
};
use solana_client::{
    client_error::ClientError, rpc_client::RpcClient, rpc_config::RpcSendTransactionConfig,
};

use solana_program::pubkey::Pubkey;
use solana_sdk::{
    account::ReadableAccount,
    commitment_config::{CommitmentConfig, CommitmentLevel},
    compute_budget::ComputeBudgetInstruction,
    instruction::Instruction,
    message::{v0::Message, CompileError, VersionedMessage},
    signature::{Keypair, Signature},
    signer::{Signer, SignerError},
    transaction::VersionedTransaction,
};
use std::{collections::HashSet, sync::Arc, thread, time::Duration};

pub const PROFIT_SHARE: f64 = 0.085;

#[derive(Debug)]
pub enum LiquidationError {
    Anyhow(anyhow::Error),
    StaleOracles(Vec<Pubkey>),
    NotEnoughFunds,
}

impl LiquidationError {
    fn from_anyhow(e: anyhow::Error) -> Self {
        Self::Anyhow(e)
    }

    fn from_compile(e: CompileError) -> Self {
        Self::Anyhow(e.into())
    }

    fn from_signer(e: SignerError) -> Self {
        Self::Anyhow(e.into())
    }

    fn from_client(e: ClientError) -> Self {
        Self::Anyhow(e.into())
    }
}

pub struct PreparedLiquidatableAccount {
    pub liquidatee_account: MarginfiAccountWrapper,
    pub observation_accounts: ObservationAccounts,
    pub asset_bank: Pubkey,
    pub liab_bank: Pubkey,
    pub asset_amount: I80F48,
    pub liab_amount: I80F48,
    pub profit: u64,
    pub dust_liab_threshold: I80F48,
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
    pub fn new(config: &Eva01Config, marginfi_group_id: Pubkey, cache: Arc<Cache>) -> Result<Self> {
        let signer = Keypair::from_bytes(&config.wallet_keypair)?;
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

        let msg = Message::try_compile(&signer_pk, &[init_ix.clone()], &[], recent_blockhash)
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

    pub fn liquidate(
        &self,
        account: PreparedLiquidatableAccount,
        tokens_in_shortage: &mut HashSet<Pubkey>,
    ) -> Result<(), LiquidationError> {
        LIQUIDATION_ATTEMPTS.inc();

        // Borrow fields off `account` so it stays whole for `build_liquidate_tx(&account, ..)`.
        let liquidatee_account = &account.liquidatee_account;
        let asset_amount = account.asset_amount;
        let liab_amount = account.liab_amount;
        let dust_liab_threshold = account.dust_liab_threshold;

        let liquidatee_account_address = liquidatee_account.address;
        let asset_bank_wrapper = self
            .cache
            .banks
            .try_get_bank(&account.asset_bank)
            .map_err(LiquidationError::from_anyhow)?;

        let liab_bank_wrapper = self
            .cache
            .banks
            .try_get_bank(&account.liab_bank)
            .map_err(LiquidationError::from_anyhow)?;

        let liab_mint = liab_bank_wrapper.bank.mint;
        if tokens_in_shortage.contains(&liab_mint) {
            debug!(
                "Skipping liquidation since the liab token is in shortage: {}",
                liab_mint
            );
            return Ok(());
        }

        let liquidator_account = &self
            .cache
            .marginfi_accounts
            .try_get_account(&self.liquidator_address)
            .map_err(LiquidationError::from_anyhow)?;

        for bank_to_validate_against in [&asset_bank_wrapper, &liab_bank_wrapper] {
            if !validate_asset_tags(&bank_to_validate_against.bank, &liquidator_account.account) {
                // This is a precaution to not attempt to liquidate staked collateral positions when liquidator has non-SOL positions open.
                // Expected to happen quite often for now. Later on, we can add a more sophisticated filtering logic on the higher level.
                debug!("Bank {:?} does not match the asset tags of the lending account -> skipping liquidation attempt", bank_to_validate_against.address);
                return Ok(());
            }
        }

        let liab_token_balance =
            I80F48::from_num(self.get_token_balance_for_mint(&liab_mint).unwrap_or(0));

        if liab_token_balance < dust_liab_threshold {
            if tokens_in_shortage.insert(liab_mint) {
                warn!("No tokens: {}", liab_mint);
            }
            return Err(LiquidationError::NotEnoughFunds);
        }

        let (asset_amount, liab_amount) = if liab_token_balance < liab_amount {
            tokens_in_shortage.insert(liab_mint);
            info!(
                "Not enough {} tokens: liquidating for: {} (of {})",
                liab_mint, liab_token_balance, liab_amount
            );
            let proportion = liab_token_balance.checked_div(liab_amount).unwrap();
            (
                asset_amount.checked_mul(proportion).unwrap(),
                liab_token_balance,
            )
        } else {
            (
                asset_amount,
                liab_amount
                    .checked_mul(I80F48::from_num(1.0 - PROFIT_SHARE))
                    .unwrap(),
            )
        };

        let (tx, ixs) = self
            .build_liquidate_tx(&account, asset_amount, liab_amount)
            .map_err(LiquidationError::from_anyhow)?;

        info!(
            "Liquidating account {:?} with liquidator account {:?}. Amount: {} (liab amount: {})",
            liquidatee_account_address,
            self.liquidator_address,
            asset_amount.to_num::<u64>(),
            liab_amount.to_num::<u64>()
        );

        let _liquidation_timer = LIQUIDATION_LATENCY_SECONDS.start_timer();

        match self
            .rpc_client
            .send_and_confirm_transaction_with_spinner_and_config(
                &tx,
                CommitmentConfig::confirmed(),
                RpcSendTransactionConfig {
                    skip_preflight: false,
                    preflight_commitment: Some(CommitmentLevel::Processed),
                    ..Default::default()
                },
            ) {
            Ok(signature) => {
                info!(
                    "Liquidation tx for the Account {} was confirmed. Signature: {}",
                    liquidatee_account_address, signature,
                );
                LIQUIDATION_SUCCESSES.inc();
                Ok(())
            }
            Err(err) => {
                if is_tx_too_large_client(&err) {
                    warn!("The liquidation tx was too large: adding the observation accounts to a LUT and retrying");
                    match self.retry_with_new_luts(ixs) {
                        Ok(signature) => {
                            info!(
                                "Liquidation tx for the Account {} was confirmed. Signature: {}",
                                liquidatee_account_address, signature,
                            );
                            LIQUIDATION_SUCCESSES.inc();
                            Ok(())
                        }
                        Err(err) => {
                            if err
                                .downcast_ref::<ClientError>()
                                .is_some_and(is_stale_swb_price_error)
                            {
                                Err(LiquidationError::StaleOracles(
                                    account.observation_accounts.swb_oracles.clone(),
                                ))
                            } else {
                                Err(LiquidationError::Anyhow(err))
                            }
                        }
                    }
                } else if is_stale_swb_price_error(&err) {
                    Err(LiquidationError::StaleOracles(
                        account.observation_accounts.swb_oracles.clone(),
                    ))
                } else {
                    Err(LiquidationError::from_client(err))
                }
            }
        }
    }

    /// Assemble (compile + sign) the liquidation transaction for the given account and amounts.
    /// Pure tx construction shared by the legacy `liquidate` path and the execution-layer
    /// `InventoryStrategy`; returns the signed tx plus its instructions (for the LUT-retry path).
    pub fn build_liquidate_tx(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
    ) -> Result<(VersionedTransaction, Vec<Instruction>)> {
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
            let lending_state = self.cache.try_get_juplend_lending_state(lending_state_address)?;

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
        let tx = VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])?;

        Ok((tx, ixs))
    }

    fn retry_with_new_luts(&self, ixs: Vec<Instruction>) -> Result<Signature> {
        self.cache
            .try_close_deactivated_luts(&self.rpc_client, &self.signer);

        let all_accounts: Vec<Pubkey> = ixs
            .iter()
            .flat_map(|ix| ix.accounts.iter().map(|a| a.pubkey))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let targeted_lut =
            self.cache
                .create_targeted_lut(&self.rpc_client, &self.signer, all_accounts)?;
        let lut_key = targeted_lut.key;
        let luts = vec![targeted_lut];

        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let msg = Message::try_compile(&self.signer.pubkey(), &ixs, &luts, recent_blockhash)?;
        let tx = VersionedTransaction::try_new(VersionedMessage::V0(msg), &[&self.signer])?;

        let result = self
            .rpc_client
            .send_and_confirm_transaction_with_spinner_and_config(
                &tx,
                CommitmentConfig::confirmed(),
                RpcSendTransactionConfig {
                    skip_preflight: false,
                    preflight_commitment: Some(CommitmentLevel::Processed),
                    ..Default::default()
                },
            );

        if let Err(e) = self
            .cache
            .deactivate_targeted_lut(&self.rpc_client, &self.signer, lut_key)
        {
            warn!("Failed to deactivate targeted LUT {lut_key}: {e}");
        }

        Ok(result?)
    }

    fn get_token_balance_for_mint(&self, mint_address: &Pubkey) -> Option<u64> {
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
        if let Some(oracle) = account_oracles
            .iter()
            .find(|oracle| stale_oracles.contains(*oracle))
        {
            true
        } else {
            false
        }
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
