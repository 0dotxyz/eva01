//! Pushes the latest Pyth prices on chain right before a liquidation that relies on them.
//!
//! eva evaluates accounts on the newest Pyth price it knows, which is usually Hermes's, ahead of
//! the sponsored push feed marginfi reads. A liquidation sized on such a price only passes once
//! the feed holds it, so the price is posted in the same bundle, ahead of the liquidation: post
//! the update's VAA to Wormhole, verify it, write each feed with `update_price_feed`, and close the
//! VAA account to reclaim its rent. A push carries one VAA for every feed, the one the liquidation
//! was sized on (`PushedPrices`). Nothing is fetched from Hermes on the way.

use anyhow::{anyhow, Result};
use log::debug;
use pyth_solana_receiver_sdk::config::Config;
use solana_client::rpc_client::RpcClient;
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::{v0, Hash, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_signer::Signer;
use solana_system_interface::instruction as system_instruction;
use solana_transaction::versioned::VersionedTransaction;

use anchor_lang::AccountDeserialize;
use pyth_solana_receiver_sdk::pda::get_config_address;

use std::sync::Arc;

use crate::cache::{pyth_prices::PushedPrices, Cache};
use crate::config::Eva01Config;
use crate::transaction_limits::{MAX_BUNDLE_TXS, MAX_COMPUTE_UNIT_LIMIT, MAX_V0_TX_SIZE};
use crate::utils::pyth_cranker::{
    close_encoded_vaa_ix, guardian_set_index, init_encoded_vaa_ix, update_price_feed_ix,
    verify_encoded_vaa_ix, write_encoded_vaa_ix, VAA_HEADER_SIZE, VAA_SPLIT_INDEX,
};

/// The push transactions a bundle has room for: the liquidation and the tip take two. The tip
/// stays the bundle's last transaction, after the liquidation: in a push transaction it would pay
/// for a push that lands without the liquidation if the bundle were split.
const MAX_PUSH_TXS: usize = MAX_BUNDLE_TXS - 2;

/// The most feeds a liquidation is sized on pushing. Posting and verifying the VAA take two
/// transactions and each further one holds about two updates; how many fit is measured
/// (`PythPusher::check`).
pub const MAX_PUSHED_FEEDS: usize = 4;

/// The most Kamino and JupLend refreshes in one transaction: a reserve refresh priced by Scope or
/// Switchboard takes up to ~90k compute units, so eight stay well inside the limit.
const MAX_REFRESHES_PER_TX: usize = 8;

/// The most taken off a transaction's compute limit to make it unique to its attempt: two
/// liquidations pushing or refreshing the same accounts in the same blockhash would otherwise send
/// the same transaction, which only one of them could land.
const ATTEMPT_CU_JITTER: u64 = 10_000;

/// The compute limit of a push or refresh transaction for `attempt`: all but a little of the
/// most a transaction may ask for, so neither VAA verification nor many reserve refreshes run out.
fn cu_limit(attempt: u64) -> u32 {
    MAX_COMPUTE_UNIT_LIMIT - (attempt % ATTEMPT_CU_JITTER) as u32
}

pub struct PythPusher {
    cache: Arc<Cache>,
    wormhole_program_id: Pubkey,
    rpc_client: RpcClient,
    payer: Keypair,
    cu_price_micro_lamports: u64,
}

/// The transactions that post one Hermes update, in the order they must land. They only ever
/// land in a bundle with the liquidation behind them, so the VAA account the first one creates is
/// always closed by the last.
pub struct PricePush {
    pub txs: Vec<VersionedTransaction>,
}

impl PythPusher {
    pub fn new(config: &Eva01Config, cache: Arc<Cache>) -> Result<Self> {
        let rpc_client = RpcClient::new(config.rpc_url.clone());
        let receiver_config = rpc_client.get_account(&get_config_address())?;
        let wormhole_program_id = Config::try_deserialize(&mut receiver_config.data.as_slice())
            .map_err(|e| anyhow!("Failed to read the Pyth receiver config: {e:?}"))?
            .wormhole;
        Ok(Self {
            cache,
            wormhole_program_id,
            rpc_client,
            payer: Keypair::try_from(config.wallet_keypair.as_slice())?,
            cu_price_micro_lamports: config.compute_unit_price_micro_lamports,
        })
    }

    /// The transactions that post `push` ahead of a liquidation, with `trailing` (the
    /// liquidation's Kamino and JupLend refreshes) after them.
    pub fn push_transactions(
        &self,
        push: &PushedPrices,
        trailing: &[Instruction],
        attempt: u64,
    ) -> Result<PricePush> {
        let (chunks, encoded_vaa) = self.layout(push, trailing)?;
        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        let mut txs = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            // The first transaction creates the VAA account, which signs for itself.
            let signers: &[&Keypair] = if i == 0 {
                &[&self.payer, &encoded_vaa]
            } else {
                &[&self.payer]
            };
            txs.push(self.sign(chunk, signers, blockhash, cu_limit(attempt))?);
        }
        debug!(
            "PythPusher: {} transaction(s) push {} feed(s)",
            txs.len(),
            push.feeds.len()
        );
        Ok(PricePush { txs })
    }

    /// Whether `push` fits a bundle. Nothing is signed or fetched: only the transactions' sizes
    /// are measured.
    pub fn check(&self, push: &PushedPrices) -> Result<()> {
        self.layout(push, &[]).map(|_| ())
    }

    /// `ixs` alone in one signed transaction, ahead of a liquidation in its bundle: its Kamino and
    /// JupLend refreshes when no push carries them. `None` when they don't fit one.
    pub fn transaction(
        &self,
        ixs: &[Instruction],
        attempt: u64,
    ) -> Result<Option<VersionedTransaction>> {
        if ixs.len() > MAX_REFRESHES_PER_TX || !self.fits(ixs)? {
            return Ok(None);
        }
        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        Ok(Some(self.sign(
            ixs,
            &[&self.payer],
            blockhash,
            cu_limit(attempt),
        )?))
    }

    /// The push's instructions split into the transactions that fit a bundle, and the VAA
    /// account's keypair, which signs the first.
    fn layout(
        &self,
        push: &PushedPrices,
        trailing: &[Instruction],
    ) -> Result<(Vec<Vec<Instruction>>, Keypair)> {
        let encoded_vaa = Keypair::new();
        let (post, rest) = self.instructions(push, &encoded_vaa.pubkey())?;
        if !self.fits(&post)? {
            return Err(anyhow!("posting the VAA doesn't fit a transaction"));
        }
        let mut chunks = vec![post];
        chunks.extend(self.pack(&rest, usize::MAX)?);
        // The refreshes in transactions of their own, never beside the VAA's verification, which
        // takes most of a transaction's compute.
        chunks.extend(self.pack(trailing, MAX_REFRESHES_PER_TX)?);
        if chunks.len() > MAX_PUSH_TXS {
            return Err(anyhow!(
                "pushing {} feeds takes {} transactions (max {})",
                push.feeds.len(),
                chunks.len(),
                MAX_PUSH_TXS
            ));
        }
        Ok((chunks, encoded_vaa))
    }

    /// The push's instructions: the first transaction's, which create the VAA account at
    /// `vaa_account` and write its head, then the rest in order.
    fn instructions(
        &self,
        push: &PushedPrices,
        vaa_account: &Pubkey,
    ) -> Result<(Vec<Instruction>, Vec<Instruction>)> {
        let vaa = push.vaa.as_slice();

        let payer = self.payer.pubkey();
        let wormhole = &self.wormhole_program_id;
        let vaa_account_size = VAA_HEADER_SIZE + vaa.len();
        // The chain's rent, served at startup (see `rent_sysvar`); the RPC only if it isn't.
        let lamports = match <solana_sysvar::rent::Rent as solana_sysvar::Sysvar>::get() {
            Ok(rent) => rent.minimum_balance(vaa_account_size),
            Err(_) => self
                .rpc_client
                .get_minimum_balance_for_rent_exemption(vaa_account_size)?,
        };

        let split = VAA_SPLIT_INDEX.min(vaa.len());
        let post = vec![
            system_instruction::create_account(
                &payer,
                vaa_account,
                lamports,
                vaa_account_size as u64,
                wormhole,
            ),
            init_encoded_vaa_ix(wormhole, &payer, vaa_account),
            write_encoded_vaa_ix(wormhole, &payer, vaa_account, 0, &vaa[..split]),
        ];

        // The VAA's tail, its verification, one update per feed, then the close that reclaims the
        // VAA account's rent.
        let mut rest = Vec::new();
        if split < vaa.len() {
            rest.push(write_encoded_vaa_ix(
                wormhole,
                &payer,
                vaa_account,
                split as u32,
                &vaa[split..],
            ));
        }
        rest.push(verify_encoded_vaa_ix(
            wormhole,
            &payer,
            vaa_account,
            guardian_set_index(vaa)?,
        ));
        for (_, signed) in &push.feeds {
            rest.push(update_price_feed_ix(
                &payer,
                vaa_account,
                signed.message.feed_id,
                signed.update.clone(),
            )?);
        }
        rest.push(close_encoded_vaa_ix(wormhole, &payer, vaa_account));
        Ok((post, rest))
    }

    /// Splits `ixs`, in order, into as few transactions as fit, at most `max_per_tx` each.
    fn pack(&self, ixs: &[Instruction], max_per_tx: usize) -> Result<Vec<Vec<Instruction>>> {
        let mut chunks: Vec<Vec<Instruction>> = Vec::new();
        let mut current: Vec<Instruction> = Vec::new();
        for ix in ixs {
            current.push(ix.clone());
            if current.len() > max_per_tx || !self.fits(&current)? {
                current.pop();
                if current.is_empty() {
                    return Err(anyhow!(
                        "a price push instruction doesn't fit a transaction"
                    ));
                }
                chunks.push(std::mem::take(&mut current));
                current.push(ix.clone());
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        Ok(chunks)
    }

    /// The message for `ixs` behind the compute budget, paid by the payer.
    fn message(&self, ixs: &[Instruction], blockhash: Hash, cu_limit: u32) -> Result<v0::Message> {
        let mut all = vec![
            ComputeBudgetInstruction::set_compute_unit_limit(cu_limit),
            ComputeBudgetInstruction::set_compute_unit_price(self.cu_price_micro_lamports),
        ];
        all.extend_from_slice(ixs);
        Ok(v0::Message::try_compile(
            &self.payer.pubkey(),
            &all,
            &[],
            blockhash,
        )?)
    }

    /// Whether `ixs` fit one transaction. The size depends on neither the blockhash nor the
    /// signatures' values, so nothing is fetched or signed.
    fn fits(&self, ixs: &[Instruction]) -> Result<bool> {
        // The limit's value doesn't change the size.
        let message = self.message(ixs, Hash::default(), MAX_COMPUTE_UNIT_LIMIT)?;
        let signatures = usize::from(message.header.num_required_signatures);
        let tx = VersionedTransaction {
            signatures: vec![Signature::default(); signatures],
            message: VersionedMessage::V0(message),
        };
        Ok(bincode::serialize(&tx)?.len() <= MAX_V0_TX_SIZE)
    }

    fn sign(
        &self,
        ixs: &[Instruction],
        signers: &[&Keypair],
        blockhash: Hash,
        cu_limit: u32,
    ) -> Result<VersionedTransaction> {
        let message = self.message(ixs, blockhash, cu_limit)?;
        Ok(VersionedTransaction::try_new(
            VersionedMessage::V0(message),
            signers,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_instruction::AccountMeta;

    #[test]
    fn a_push_is_packed_in_order_into_transactions_that_fit() {
        let pusher = PythPusher {
            cache: Arc::new(Cache::new(
                Pubkey::new_unique(),
                Pubkey::new_unique(),
                Arc::new(std::sync::Mutex::new(solana_clock::Clock::default())),
                std::collections::HashSet::new(),
            )),
            wormhole_program_id: Pubkey::new_unique(),
            rpc_client: RpcClient::new("http://127.0.0.1:1".to_string()),
            payer: Keypair::new(),
            cu_price_micro_lamports: 1_000,
        };
        let program = Pubkey::new_unique();
        // About the size of a VAA tail write and per-feed updates.
        let ixs: Vec<Instruction> = (0..6u8)
            .map(|i| Instruction {
                program_id: program,
                accounts: vec![AccountMeta::new(Pubkey::new_unique(), false)],
                data: vec![i; 400],
            })
            .collect();
        let chunks = pusher.pack(&ixs, usize::MAX).unwrap();
        // A count cap splits further.
        assert_eq!(pusher.pack(&ixs, 2).unwrap().len(), 3);

        assert!(chunks.len() > 1);
        let order: Vec<u8> = chunks.iter().flatten().map(|ix| ix.data[0]).collect();
        assert_eq!(order, (0..6).collect::<Vec<u8>>());
        // Measured unsigned, each still fits once signed on a real blockhash.
        for chunk in &chunks {
            let tx = pusher
                .sign(
                    chunk,
                    &[&pusher.payer],
                    Hash::new_unique(),
                    MAX_COMPUTE_UNIT_LIMIT,
                )
                .unwrap();
            assert!(bincode::serialize(&tx).unwrap().len() <= MAX_V0_TX_SIZE);
        }
    }
}
