use super::{
    bank::BankWrapper,
    marginfi_account::{MarginfiAccountWrapper, ObservationAccounts},
};
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
use solana_message::{v0::Message, v1, AddressLookupTableAccount, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::execution::Landing;
use crate::juplend_ixs::juplend_underlying_amount;
use crate::transaction_limits::{
    priority_fee_lamports, MAX_COMPUTE_UNIT_LIMIT, MAX_LOADED_ACCOUNTS_DATA, MAX_TX_ACCOUNT_LOCKS,
    MAX_V0_TX_SIZE, MAX_V1_TX_SIZE,
};

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

/// Account metas of `ixs` that the compiler may resolve through a lookup table: everything but
/// the invoked programs (and the payer, which must stay static).
fn lookupable_accounts(signer_pk: &Pubkey, ixs: &[Instruction]) -> Vec<Pubkey> {
    let programs: HashSet<Pubkey> = ixs.iter().map(|ix| ix.program_id).collect();
    let mut seen = HashSet::new();
    ixs.iter()
        .flat_map(|ix| ix.accounts.iter().map(|meta| meta.pubkey))
        .filter(|key| key != signer_pk && !programs.contains(key) && seen.insert(*key))
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
    pub estimated_repay_usd: f64,
    pub binding_cap: &'static str,
    /// The Pyth oracles whose newer price (from Hermes) the liquidation is sized on, so it goes
    /// behind a push of them; empty when the chain's prices suffice.
    pub push_oracles: Vec<Pubkey>,
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
    rpc_url: String,
    cu_limit_ix: Instruction,
    /// The priority fee of a v1 liquidation transaction, in lamports: the configured price per
    /// compute unit times [`MAX_COMPUTE_UNIT_LIMIT`], as a v1 header states a total.
    v1_priority_fee: u64,
    /// Priority-fee instruction (from `COMPUTE_UNIT_PRICE_MICRO_LAMPORTS`). Without it the heavy
    /// liquidation tx carries a CU limit but no price, so it confirms slowly under load.
    cu_price_ix: Instruction,
    pub cache: Arc<Cache>,
    /// The swap venues' lookup tables, as last fetched. Routes keep reusing the same few.
    venue_luts: Mutex<HashMap<Pubkey, (AddressLookupTableAccount, Instant)>>,
}

/// How long a venue's lookup table is used as fetched. Venues extend their tables now and then;
/// a stale copy only leaves the new addresses outside it, in the transaction itself.
const VENUE_LUT_TTL: Duration = Duration::from_secs(60);

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
            rpc_url: config.rpc_url.clone(),
            cu_limit_ix: ComputeBudgetInstruction::set_compute_unit_limit(MAX_COMPUTE_UNIT_LIMIT),
            v1_priority_fee: priority_fee_lamports(config.compute_unit_price_micro_lamports),
            cu_price_ix: ComputeBudgetInstruction::set_compute_unit_price(
                config.compute_unit_price_micro_lamports,
            ),
            cache,
            venue_luts: Mutex::new(HashMap::new()),
        })
    }

    /// Compiles and signs the liquidation transaction. With `swap`, its instructions (selling the
    /// seized collateral for the liability) go between the withdraw and the repay.
    pub fn build_liquidate_tx(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
        swap: Option<&SwapInstructions>,
        landing: Landing,
    ) -> Result<VersionedTransaction> {
        let (ixs, mut luts, record) =
            self.liquidation_ixs(account, asset_amount, liab_amount, swap)?;

        // Over RPC, always v1: up to 4096 bytes with every address in full, so it needs no lookup
        // table and a route's pool accounts always fit, and its header carries the compute budget,
        // which saves the compute-budget program's account. Both formats lock at most 64 accounts,
        // so whatever fits a v0 transaction fits a v1 one. The Jito block engine cannot decode v1,
        // so only a bundled liquidation is v0, with lookup tables.
        if landing == Landing::Rpc {
            return self.compile_v1(&ixs);
        }

        if let Some(swap) = swap {
            luts.extend(self.fetch_luts(&swap.address_lookup_tables)?);
        }
        let mut v0_ixs = vec![self.cu_limit_ix.clone(), self.cu_price_ix.clone()];
        v0_ixs.extend(ixs);
        // Whatever doesn't fit gets learned for next time, except the liquidatee's own account and
        // record, which no other liquidation references.
        let liquidatee = account.liquidatee_account.address;
        self.compile_tx_fit(&v0_ixs, &luts, &[liquidatee, record])
    }

    /// The liquidation's instructions, in order, the lookup tables that cover them (the venue's
    /// tables for a swap are added by the caller) and the liquidation record they use.
    pub fn liquidation_ixs(
        &self,
        account: &PreparedLiquidatableAccount,
        asset_amount: I80F48,
        liab_amount: I80F48,
        swap: Option<&SwapInstructions>,
    ) -> Result<(Vec<Instruction>, Vec<AddressLookupTableAccount>, Pubkey)> {
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

        // The compute budget is added when compiling: a v1 header carries it, a v0 transaction
        // needs its instructions.
        let mut ixs = Vec::new();

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

        Ok((ixs, self.cache.luts.tables(), liquidation_record))
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

    /// Every address a liquidation on `bank` can reference, independent of the liquidatee: the
    /// real liquidation instructions built for a placeholder liquidatee, minus the placeholder's
    /// own account and record.
    pub fn static_accounts_for_bank(&self, bank: &Pubkey) -> Result<Vec<Pubkey>> {
        let liquidatee =
            MarginfiAccountWrapper::new(Pubkey::new_unique(), bytemuck::Zeroable::zeroed());
        let liquidatee_address = liquidatee.address;
        let intent = PreparedLiquidatableAccount {
            attempt_id: 0,
            liquidatee_account: liquidatee,
            observation_accounts: ObservationAccounts::for_banks(&self.cache, vec![*bank])?,
            asset_bank: *bank,
            liab_bank: *bank,
            asset_amount: I80F48::ONE,
            liab_amount: I80F48::ONE,
            profit: I80F48::ZERO,
            estimated_repay_usd: 0.0,
            binding_cap: "test",
            push_oracles: Vec::new(),
            without_push: None,
            sandwich_pair: None,
        };
        let (ixs, _, record) = self.liquidation_ixs(&intent, I80F48::ONE, I80F48::ONE, None)?;
        Ok(lookupable_accounts(&self.signer.pubkey(), &ixs)
            .into_iter()
            .filter(|key| *key != liquidatee_address && *key != record)
            .collect())
    }

    /// [`Self::static_accounts_for_bank`] over every bank in the cache; a bank whose instructions
    /// can't be built yet is logged and skipped.
    pub fn static_accounts(&self) -> Vec<Pubkey> {
        let mut seen = HashSet::new();
        self.cache
            .banks
            .get_addresses()
            .iter()
            .flat_map(|bank| {
                self.static_accounts_for_bank(bank).unwrap_or_else(|e| {
                    warn!("Skipping the bank {bank} for the lookup tables: {e}");
                    Vec::new()
                })
            })
            .filter(|key| seen.insert(*key))
            .collect()
    }

    /// Adds `bank`'s addresses to the owned lookup tables on a background thread.
    pub fn cover_bank_in_background(&self, bank: &Pubkey) {
        match self.static_accounts_for_bank(bank) {
            Ok(keys) => self.cache.luts.cover_in_background(
                self.rpc_url.clone(),
                self.signer.to_bytes().to_vec(),
                keys,
            ),
            Err(e) => warn!("Skipping the bank {bank} for the lookup tables: {e}"),
        }
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

        let tx = self.compile_v1(&ixs)?;
        // Sent, not confirmed: the liquidator loop runs tags inline, and waiting out a tag that
        // never lands would hold up every liquidation behind it. Preflight still rejects a tag the
        // program would refuse.
        Ok(self.rpc_client.send_transaction(&tx)?)
    }

    /// Simulates `tx`. On failure returns the program error with the full simulation logs.
    pub fn simulate_tx(&self, tx: &VersionedTransaction) -> Result<()> {
        let result = self.rpc_client.simulate_transaction(tx)?.value;
        match result.err {
            None => Ok(()),
            Some(err) => {
                let logs = result.logs.unwrap_or_default();
                Err(anyhow!(
                    "simulation failed: {:?}; full_logs: {:#?}",
                    err,
                    logs
                ))
            }
        }
    }

    /// The venue lookup tables `keys`, fetched in one call for those not fetched recently.
    fn fetch_luts(&self, keys: &[Pubkey]) -> Result<Vec<AddressLookupTableAccount>> {
        let now = Instant::now();
        let missing: Vec<Pubkey> = {
            let cached = self.venue_luts.lock().unwrap_or_else(|e| e.into_inner());
            keys.iter()
                .filter(|key| {
                    cached
                        .get(key)
                        .is_none_or(|(_, at)| now.duration_since(*at) > VENUE_LUT_TTL)
                })
                .copied()
                .collect()
        };
        if !missing.is_empty() {
            let fetched = self
                .rpc_client
                .get_multiple_accounts(&missing)?
                .into_iter()
                .zip(&missing)
                .map(|(account, key)| {
                    let account =
                        account.ok_or_else(|| anyhow!("Lookup table {} not found", key))?;
                    Ok(AddressLookupTableAccount {
                        key: *key,
                        addresses: AddressLookupTable::deserialize(&account.data)?
                            .addresses
                            .to_vec(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut cached = self.venue_luts.lock().unwrap_or_else(|e| e.into_inner());
            for table in fetched {
                cached.insert(table.key, (table, now));
            }
        }
        let cached = self.venue_luts.lock().unwrap_or_else(|e| e.into_inner());
        keys.iter()
            .map(|key| {
                cached
                    .get(key)
                    .map(|(table, _)| table.clone())
                    .ok_or_else(|| anyhow!("Lookup table {} not found", key))
            })
            .collect()
    }

    /// Compiles and signs `ixs` with every lookup table that applies. Over the account-lock limit
    /// nothing helps, so that fails first. Over the byte limit, the transaction fails too, and the
    /// addresses no table holds (except `per_target`, which only this liquidation references) are
    /// added to the owned tables in the background, so the next attempt fits.
    fn compile_tx_fit(
        &self,
        ixs: &[Instruction],
        luts: &[AddressLookupTableAccount],
        per_target: &[Pubkey],
    ) -> Result<VersionedTransaction> {
        let signer_pk = self.signer.pubkey();
        let locks = locked_accounts(&signer_pk, ixs).len();
        if locks > MAX_TX_ACCOUNT_LOCKS {
            return Err(anyhow!(
                "transaction loads {} accounts (max {})",
                locks,
                MAX_TX_ACCOUNT_LOCKS
            ));
        }

        let tx = self.compile_tx(&signer_pk, ixs, luts)?;
        let size = bincode::serialize(&tx)?.len();
        if size <= MAX_V0_TX_SIZE {
            return Ok(tx);
        }

        let in_tables: HashSet<Pubkey> = luts
            .iter()
            .flat_map(|table| table.addresses.iter().copied())
            .collect();
        let uncovered: Vec<Pubkey> = lookupable_accounts(&signer_pk, ixs)
            .into_iter()
            .filter(|key| !in_tables.contains(key))
            .collect();
        let learnable: Vec<Pubkey> = uncovered
            .iter()
            .filter(|key| !per_target.contains(key))
            .copied()
            .collect();
        if !learnable.is_empty() {
            self.cache.luts.cover_in_background(
                self.rpc_url.clone(),
                self.signer.to_bytes().to_vec(),
                learnable.clone(),
            );
        }
        Err(anyhow!(
            "transaction is {} bytes (max {}) with {} account(s) no lookup table holds{}",
            size,
            MAX_V0_TX_SIZE,
            uncovered.len(),
            if learnable.is_empty() {
                String::new()
            } else {
                format!("; adding {} of them", learnable.len())
            }
        ))
    }

    /// Compiles and signs `ixs` as a v1 transaction, with the liquidation's compute budget and
    /// priority fee in its header.
    fn compile_v1(&self, ixs: &[Instruction]) -> Result<VersionedTransaction> {
        let signer_pk = self.signer.pubkey();
        let locks = locked_accounts(&signer_pk, ixs).len();
        if locks > MAX_TX_ACCOUNT_LOCKS {
            return Err(anyhow!(
                "transaction loads {} accounts (max {})",
                locks,
                MAX_TX_ACCOUNT_LOCKS
            ));
        }
        let config = v1::TransactionConfig {
            priority_fee: Some(self.v1_priority_fee),
            compute_unit_limit: Some(MAX_COMPUTE_UNIT_LIMIT),
            loaded_accounts_data_size_limit: Some(MAX_LOADED_ACCOUNTS_DATA),
            ..v1::TransactionConfig::empty()
        };
        let recent_blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        let message =
            v1::Message::try_compile_with_config(&signer_pk, ixs, recent_blockhash, config)
                .map_err(|e| anyhow!("{e:?}"))?;
        let tx = VersionedTransaction::try_new(VersionedMessage::V1(message), &[&self.signer])?;
        let size = bincode::serialize(&tx)?.len();
        if size > MAX_V1_TX_SIZE {
            return Err(anyhow!(
                "transaction is {} bytes (max {})",
                size,
                MAX_V1_TX_SIZE
            ));
        }
        Ok(tx)
    }

    fn compile_tx(
        &self,
        signer_pk: &Pubkey,
        ixs: &[Instruction],
        luts: &[AddressLookupTableAccount],
    ) -> Result<VersionedTransaction> {
        let recent_blockhash = self.cache.blockhash.get(&self.rpc_client)?;
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
