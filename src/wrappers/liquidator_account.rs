use super::{
    bank::BankWrapper,
    marginfi_account::{MarginfiAccountWrapper, ObservationAccounts},
};
use crate::{
    cache::{pyth_prices::PushedPrices, Cache},
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
use log::{debug, error, info};
use marginfi_type_crate::constants::{
    ASSET_TAG_DEFAULT, ASSET_TAG_JUPLEND, ASSET_TAG_KAMINO, ASSET_TAG_SOL, ASSET_TAG_STAKED,
};
use solana_account::ReadableAccount;
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use solana_dex_superagg::SwapInstructions;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;
use std::{collections::HashSet, sync::Arc};

use crate::execution::Landing;
use crate::juplend_ixs::juplend_underlying_amount;
use crate::transaction_limits::{priority_fee, v1_transaction, MAX_COMPUTE_UNIT_LIMIT};
use crate::utils::quick_rpc::QuickRpc;

/// The transaction ran in simulation and failed, as opposed to the simulation not running.
#[derive(Debug)]
pub struct SimulationFailed(pub String);

impl std::fmt::Display for SimulationFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SimulationFailed {}

/// Collateral tags `build_liquidate_tx` can withdraw. Staked banks hold a plain SPL LST, so they go
/// through the standard withdraw like DEFAULT/SOL.
pub fn is_seizable_asset_tag(tag: u8) -> bool {
    matches!(
        tag,
        ASSET_TAG_DEFAULT | ASSET_TAG_SOL | ASSET_TAG_STAKED | ASSET_TAG_KAMINO | ASSET_TAG_JUPLEND
    )
}

/// Unique accounts the compiled transaction will lock: the payer, every program and every account
/// meta. (The blockhash is not an account.)
pub fn locked_accounts(signer_pk: &Pubkey, ixs: &[Instruction]) -> HashSet<Pubkey> {
    ixs.iter()
        .flat_map(|ix| {
            std::iter::once(ix.program_id).chain(ix.accounts.iter().map(|meta| meta.pubkey))
        })
        .chain(std::iter::once(*signer_pk))
        .collect()
}

/// A bank pair to liquidate, sized.
#[derive(Clone, Debug, PartialEq)]
pub struct LiquidationPair {
    pub asset_bank: Pubkey,
    pub liab_bank: Pubkey,
    pub asset_amount: I80F48,
    pub liab_amount: I80F48,
    pub profit: I80F48,
    pub real_time_profit: I80F48,
}

#[derive(Clone)]
pub struct PreparedLiquidatableAccount {
    /// Correlates sizing, queueing, execution, and terminal outcome logs for this evaluation.
    pub attempt_id: u64,
    pub liquidatee_account: MarginfiAccountWrapper,
    pub observation_accounts: ObservationAccounts,
    pub asset_bank: Pubkey,
    pub liab_bank: Pubkey,
    pub asset_amount: I80F48,
    pub liab_amount: I80F48,
    pub profit: I80F48,
    /// The profit at real-time prices, which the seized collateral sells near: below `profit`
    /// after a fast move, and negative once the move exceeds the premium. Only a swap that pays
    /// checks a liquidation against the market; one repaid from the wallet relies on this.
    pub real_time_profit: I80F48,
    pub estimated_repay_usd: f64,
    pub binding_cap: &'static str,
    /// The push the liquidation is sized on, which goes ahead of it; `None` when the chain's
    /// prices suffice.
    pub push: Option<PushedPrices>,
    /// The smaller liquidation the chain's prices already allow, taken instead when no bundle
    /// can carry the push right away.
    pub without_push: Option<Box<PreparedLiquidatableAccount>>,
    /// The pair for an in-tx swap, when it differs from the most profitable one above: that one
    /// seizes from a Kamino or JupLend bank, whose withdrawal leaves less room for the route, and a
    /// native pair earns nearly as much (`is_better_pair`).
    pub sandwich_pair: Option<LiquidationPair>,
}

impl PreparedLiquidatableAccount {
    /// This liquidation, and the smaller one kept beside it, logged under `attempt_id`.
    pub fn for_attempt(mut self, attempt_id: u64) -> Self {
        self.attempt_id = attempt_id;
        if let Some(unpushed) = self.without_push.as_mut() {
            unpushed.attempt_id = attempt_id;
        }
        self
    }

    /// What the liquidation is expected to earn: the lower of its profit at equity and at real-time
    /// prices. It ranks the queue, and gates and tips a wallet repay, which no swap checks against
    /// the market; a tip on it is no larger than what is earned.
    pub fn expected_profit(&self) -> I80F48 {
        self.profit.min(self.real_time_profit)
    }

    /// This liquidation on `sandwich_pair`, when there is one.
    pub fn for_sandwich(&self) -> std::borrow::Cow<'_, Self> {
        match &self.sandwich_pair {
            None => std::borrow::Cow::Borrowed(self),
            Some(pair) => std::borrow::Cow::Owned(Self {
                asset_bank: pair.asset_bank,
                liab_bank: pair.liab_bank,
                asset_amount: pair.asset_amount,
                liab_amount: pair.liab_amount,
                profit: pair.profit,
                real_time_profit: pair.real_time_profit,
                sandwich_pair: None,
                ..self.clone()
            }),
        }
    }
}

pub struct LiquidatorAccount {
    pub signer: Keypair,
    group: Pubkey,
    rpc_client: RpcClient,
    /// The priority fee of a liquidation, in lamports: the configured price per compute unit
    /// (`COMPUTE_UNIT_PRICE_MICRO_LAMPORTS`) over [`MAX_COMPUTE_UNIT_LIMIT`], and at least what a
    /// bundle relay asks.
    priority_fee: u64,
    pub cache: Arc<Cache>,
    /// Reads and simulations on the liquidation path: each fails within 2 s, never retried.
    quick: Arc<QuickRpc>,
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
            priority_fee: priority_fee(
                MAX_COMPUTE_UNIT_LIMIT,
                config.compute_unit_price_micro_lamports,
            ),
            cache,
            quick: QuickRpc::shared(config.rpc_url.clone()),
        })
    }

    /// Compiles and signs the liquidation transaction. With `swap`, its instructions (selling the
    /// seized collateral for the liability) go between the withdraw and the repay. A v1
    /// transaction holds up to 4096 bytes with every address in full, so a route's accounts fit
    /// without lookup tables (the venue's are ignored), within the 64-account lock limit; a route
    /// too large in bytes fails with `DoesNotFit`.
    pub fn build_liquidate_tx(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
        swap: Option<&SwapInstructions>,
        landing: Landing,
    ) -> Result<VersionedTransaction> {
        let ixs = self.liquidation_ixs(
            account,
            asset_amount,
            liab_amount,
            swap,
            landing.refreshes_in_tx(),
        )?;
        self.compile(&ixs)
    }

    /// The liquidation's instructions, in order. Without `refresh` the Kamino and JupLend refreshes
    /// are left out: an earlier transaction in the same bundle runs them (`refresh_ixs`).
    pub fn liquidation_ixs(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
        swap: Option<&SwapInstructions>,
        refresh: bool,
    ) -> Result<Vec<Instruction>> {
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
        // transaction can be closed by anyone before it is used, since a never-used record has no
        // inactivity lock, and waiting for it to finalize would stall the liquidation.
        let (init_record_ix, liquidation_record) =
            if liquidatee_account.account.liquidation_record == Pubkey::default() {
                let (ix, record) =
                    make_init_liquidation_record_ix(liquidatee_account_address, signer_pk);
                (Some(ix), record)
            } else {
                (None, liquidatee_account.account.liquidation_record)
            };

        // The compute budget is in the transaction's header.
        let mut ixs = Vec::new();

        let start_ix = make_start_liquidate_ix(
            self.group,
            liquidatee_account_address,
            signer_pk,
            liquidation_record,
            liquidatee_observation_accounts.as_ref(),
            liquidatee_banks,
        );

        if refresh {
            ixs.extend(self.venue_refresh_ixs(kamino_reserves, juplend_states)?);
        }

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

                if refresh {
                    ixs.push(make_refresh_obligation_ix(
                        asset_bank_wrapper.bank.integration_acc_2,
                        kamino_reserve.reserve.lending_market,
                        &[asset_bank_wrapper.bank.integration_acc_1],
                    ));
                }

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
                    juplend_underlying_amount(asset_amount, lending_state.token_exchange_price),
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

        Ok(ixs)
    }

    /// The underlying tokens the wallet receives for seizing `bank_units` of `bank`'s collateral:
    /// the amount a swap of that collateral can sell. Bank balances are venue shares: a JupLend
    /// balance is fTokens, redeemed at the lending state's exchange price (and its withdraw takes
    /// the underlying amount). A Kamino balance is collateral tokens, which its withdraw takes
    /// directly and redeems at a rate of at least one, so the share count is a safe lower bound.
    pub fn seized_underlying_amount(&self, bank: &BankWrapper, bank_units: I80F48) -> Result<u64> {
        if bank.bank.config.asset_tag == ASSET_TAG_JUPLEND {
            let lending = self
                .cache
                .try_get_juplend_lending_state(&bank.bank.integration_acc_1)?;
            return Ok(juplend_underlying_amount(
                bank_units,
                lending.token_exchange_price,
            ));
        }
        Ok(bank_units.to_num())
    }

    /// What the liquidation of `account` leaves to an earlier transaction in its bundle: every Kamino
    /// reserve and JupLend state its health reads, then the obligation of each Kamino bank it may
    /// seize from (the main pair's and the sandwich pair's).
    pub fn refresh_ixs(&self, account: &PreparedLiquidatableAccount) -> Result<Vec<Instruction>> {
        let ObservationAccounts {
            kamino_reserves,
            juplend_states,
            ..
        } = &account.observation_accounts;
        let mut ixs = self.venue_refresh_ixs(kamino_reserves, juplend_states)?;
        let seized = std::iter::once(account.asset_bank)
            .chain(account.sandwich_pair.as_ref().map(|pair| pair.asset_bank));
        let mut obligations = HashSet::new();
        for bank in seized {
            let bank = self.cache.banks.try_get_bank(&bank)?.bank;
            if bank.config.asset_tag != ASSET_TAG_KAMINO
                || !obligations.insert(bank.integration_acc_2)
            {
                continue;
            }
            let reserve = self.cache.try_get_kamino_reserve(&bank.integration_acc_1)?;
            ixs.push(make_refresh_obligation_ix(
                bank.integration_acc_2,
                reserve.reserve.lending_market,
                &[bank.integration_acc_1],
            ));
        }
        Ok(ixs)
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
    /// without waiting for confirmation; there is no MEV to race for, since anyone may tag and the
    /// first tag wins.
    pub fn send_tag_tx(
        &self,
        liquidatee: Pubkey,
        observation_accounts: &ObservationAccounts,
    ) -> Result<Signature> {
        let ObservationAccounts {
            observation_accounts: liquidatee_observation_accounts,
            kamino_reserves,
            juplend_states,
            ..
        } = observation_accounts;

        let mut ixs = self.venue_refresh_ixs(kamino_reserves, juplend_states)?;
        ixs.push(make_tag_liq_record_ix(
            self.group,
            liquidatee,
            liquidatee_observation_accounts,
        ));

        let tx = self.compile(&ixs)?;
        // Sent, not confirmed: a tag that never lands would hold up the tags behind it. Preflight
        // still rejects a tag the program would refuse.
        Ok(self.rpc_client.send_transaction(&tx)?)
    }

    /// Simulates `tx`. On failure returns the program error with the full simulation logs.
    pub fn simulate_tx(&self, tx: &VersionedTransaction) -> Result<()> {
        match self.quick.simulate(tx)? {
            None => Ok(()),
            Some((err, logs)) => Err(SimulationFailed(format!(
                "simulation failed: {err}; full_logs: {logs:?}"
            ))
            .into()),
        }
    }

    /// `ixs` signed as a v1 transaction asking for all the compute a transaction may, at the
    /// liquidation's priority fee.
    fn compile(&self, ixs: &[Instruction]) -> Result<VersionedTransaction> {
        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        v1_transaction(
            &[&self.signer],
            ixs,
            blockhash,
            MAX_COMPUTE_UNIT_LIMIT,
            self.priority_fee,
        )
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

    use solana_instruction::AccountMeta;

    #[test]
    fn locked_accounts_counts_unique_payer_programs_and_metas() {
        let payer = Pubkey::new_unique();
        let program = Pubkey::new_unique();
        let shared = Pubkey::new_unique();
        let ixs = [
            Instruction::new_with_bytes(
                program,
                &[],
                vec![
                    AccountMeta::new(payer, true),
                    AccountMeta::new(shared, false),
                ],
            ),
            Instruction::new_with_bytes(
                program,
                &[],
                vec![AccountMeta::new_readonly(shared, false)],
            ),
        ];
        // payer + program + shared: duplicates and the payer meta collapse.
        assert_eq!(locked_accounts(&payer, &ixs).len(), 3);
    }

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
