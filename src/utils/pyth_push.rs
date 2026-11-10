//! Pushes the latest Pyth prices on chain right before a liquidation that relies on them.
//!
//! eva evaluates accounts on the newest Pyth price it knows, which is usually Hermes's, ahead of
//! the sponsored push feed marginfi reads. A liquidation sized on such a price only passes once
//! the feed holds it, so the price is posted in the same bundle, ahead of the liquidation: post
//! the update's VAA to Wormhole, verify it, write each feed with `update_price_feed`, and close the
//! VAA account to reclaim its rent. A push carries one VAA for every feed: the newest one the Hermes
//! stream delivered under which every feed has an update the chain doesn't hold yet. Nothing is
//! fetched from Hermes on the way.

use anyhow::{anyhow, Result};
use log::debug;
use pyth_solana_receiver_sdk::config::Config;
use solana_client::rpc_client::RpcClient;
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::{v0, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_system_interface::instruction as system_instruction;
use solana_transaction::versioned::VersionedTransaction;

use anchor_lang::AccountDeserialize;
use pyth_solana_receiver_sdk::pda::get_config_address;

use std::sync::Arc;

use crate::cache::Cache;
use crate::config::Eva01Config;
use crate::transaction_limits::{MAX_BUNDLE_TXS, MAX_COMPUTE_UNIT_LIMIT, MAX_V0_TX_SIZE};
use crate::utils::pyth_cranker::{
    close_encoded_vaa_ix, guardian_set_index, init_encoded_vaa_ix, price_feed_address,
    update_price_feed_ix, verify_encoded_vaa_ix, write_encoded_vaa_ix, VAA_HEADER_SIZE,
    VAA_SPLIT_INDEX,
};

/// The push transactions a bundle has room for: the liquidation and the tip take two.
pub const MAX_PUSH_TXS: usize = MAX_BUNDLE_TXS - 2;

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

    /// Builds the transactions that post the latest Hermes prices of `feed_ids`.
    pub fn push_transactions(&self, feed_ids: &[[u8; 32]]) -> Result<PricePush> {
        // A streamed update is pushed only while it is still ahead of the chain.
        let prices = &self.cache.pyth_prices;
        let (vaa, updates) = prices
            .signed_together(feed_ids, |feed_id, publish_time| {
                prices
                    .onchain(&price_feed_address(feed_id))
                    .is_none_or(|chain| publish_time > chain)
            })
            .ok_or_else(|| {
                anyhow!(
                    "the stream holds no signed update covering all {} feed(s) ahead of the chain",
                    feed_ids.len()
                )
            })?;
        let vaa = vaa.as_slice();

        let payer = self.payer.pubkey();
        let wormhole = &self.wormhole_program_id;
        let encoded_vaa = Keypair::new();
        let vaa_account = encoded_vaa.pubkey();
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
                &vaa_account,
                lamports,
                vaa_account_size as u64,
                wormhole,
            ),
            init_encoded_vaa_ix(wormhole, &payer, &vaa_account),
            write_encoded_vaa_ix(wormhole, &payer, &vaa_account, 0, &vaa[..split]),
        ];

        // Everything after the first transaction, in order: the VAA's tail, its verification, one
        // update per feed, then the close that reclaims the VAA account's rent.
        let mut rest = Vec::new();
        if split < vaa.len() {
            rest.push(write_encoded_vaa_ix(
                wormhole,
                &payer,
                &vaa_account,
                split as u32,
                &vaa[split..],
            ));
        }
        rest.push(verify_encoded_vaa_ix(
            wormhole,
            &payer,
            &vaa_account,
            guardian_set_index(vaa)?,
        ));
        for (feed_id, merkle_price_update) in feed_ids.iter().zip(updates) {
            rest.push(update_price_feed_ix(
                &payer,
                &vaa_account,
                *feed_id,
                merkle_price_update,
            )?);
        }
        rest.push(close_encoded_vaa_ix(wormhole, &payer, &vaa_account));

        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        let mut txs = vec![self.compile(&post, &[&self.payer, &encoded_vaa], blockhash)?];
        for chunk in self.pack(&rest, blockhash)? {
            txs.push(self.compile(&chunk, &[&self.payer], blockhash)?);
        }
        if txs.len() > MAX_PUSH_TXS {
            return Err(anyhow!(
                "pushing {} feeds takes {} transactions (max {})",
                feed_ids.len(),
                txs.len(),
                MAX_PUSH_TXS
            ));
        }
        debug!(
            "PythPusher: {} transaction(s) push {} feed(s)",
            txs.len(),
            feed_ids.len()
        );
        Ok(PricePush { txs })
    }

    /// Splits `ixs`, in order, into as few transactions as fit.
    fn pack(
        &self,
        ixs: &[Instruction],
        blockhash: solana_message::Hash,
    ) -> Result<Vec<Vec<Instruction>>> {
        let mut chunks: Vec<Vec<Instruction>> = Vec::new();
        let mut current: Vec<Instruction> = Vec::new();
        for ix in ixs {
            current.push(ix.clone());
            if self.compile(&current, &[&self.payer], blockhash).is_err() {
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

    fn compile(
        &self,
        ixs: &[Instruction],
        signers: &[&Keypair],
        blockhash: solana_message::Hash,
    ) -> Result<VersionedTransaction> {
        let mut all = vec![
            // VAA verification is the heaviest step.
            ComputeBudgetInstruction::set_compute_unit_limit(MAX_COMPUTE_UNIT_LIMIT),
            ComputeBudgetInstruction::set_compute_unit_price(self.cu_price_micro_lamports),
        ];
        all.extend_from_slice(ixs);
        let message = v0::Message::try_compile(&self.payer.pubkey(), &all, &[], blockhash)?;
        let tx = VersionedTransaction::try_new(VersionedMessage::V0(message), signers)?;
        let size = bincode::serialize(&tx)?.len();
        if size > MAX_V0_TX_SIZE {
            return Err(anyhow!(
                "transaction is {size} bytes (max {MAX_V0_TX_SIZE})"
            ));
        }
        Ok(tx)
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
        let blockhash = solana_message::Hash::default();
        let chunks = pusher.pack(&ixs, blockhash).unwrap();

        assert!(chunks.len() > 1);
        let order: Vec<u8> = chunks.iter().flatten().map(|ix| ix.data[0]).collect();
        assert_eq!(order, (0..6).collect::<Vec<u8>>());
        for chunk in &chunks {
            assert!(pusher.compile(chunk, &[&pusher.payer], blockhash).is_ok());
        }
    }
}
