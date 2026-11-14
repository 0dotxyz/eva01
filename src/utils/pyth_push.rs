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
use crate::transaction_limits::{
    bundle_cu_price, MAX_BUNDLE_TXS, MAX_COMPUTE_UNIT_LIMIT, MAX_V0_TX_SIZE,
};
use crate::utils::pyth_cranker::{
    close_encoded_vaa_ix, guardian_set_index, init_encoded_vaa_ix, price_feed_bump,
    update_price_feed_ix, verify_encoded_vaa_ix, write_encoded_vaa_ix, VAA_HEADER_SIZE,
    VAA_SPLIT_INDEX,
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

/// The most added to a transaction's compute limit to make it unique to its attempt: two
/// liquidations pushing or refreshing the same accounts in the same blockhash would otherwise send
/// the same transaction, which only one of them could land.
const ATTEMPT_CU_JITTER: u64 = 10_000;

/// Compute each instruction takes, measured on mainnet (in parentheses) with about 30% over it.
/// A bundle's requested compute weighs against its tip in the block engine's auction, so a push
/// asks for what it uses rather than the most a transaction may.
const CU_CREATE_ACCOUNT: u32 = 1_000; // (150)
const CU_INIT_VAA: u32 = 2_000; // (1,363)
const CU_WRITE_VAA: u32 = 2_000; // (1,375)
/// Verifying a VAA, per guardian signature it carries (83,598 measured for Hermes's three).
const CU_VERIFY_VAA_BASE: u32 = 5_000;
const CU_VERIFY_VAA_PER_SIGNATURE: u32 = 33_000;
/// Updating a feed whose account's bump is 255: 27,406–27,422 writing a newer price, 7,709 when
/// not newer. The push oracle derives the address on every call, each bump below 255 adding ~1,500
/// (USDT's, bump 250, measured 34,923).
const CU_UPDATE_FEED: u32 = 36_000;
const CU_UPDATE_FEED_PER_BUMP: u32 = 2_000;
const CU_CLOSE_VAA: u32 = 2_000; // (1,265)
const CU_TRANSFER: u32 = 1_000; // (150)
/// Venue refreshes, measured on every Kamino reserve and obligation and JupLend lending state of
/// the prod group (worst in parentheses) with about 30% over it. A reserve's cost grows ~5k per
/// link of its Scope chain (1 link ~24k, 2 links 29,209), so a chain of four (Kamino's most) is
/// ~39k; this keeps ~28% over that.
const CU_KAMINO_REFRESH_RESERVE: u32 = 50_000;
/// One deposit, no borrows, its reserve as the only remaining account: marginfi's shape (19,753).
const CU_KAMINO_REFRESH_OBLIGATION: u32 = 26_000;
/// JupLend `UpdateRate` (15,515).
const CU_JUPLEND_UPDATE_RATE: u32 = 20_000;
/// Any other instruction among the refreshes: unmeasured, so generous.
const CU_REFRESH_UNKNOWN: u32 = 150_000;
/// The compute-budget instructions and what else a transaction is charged around its own.
const CU_TX_OVERHEAD: u32 = 1_000;

/// The compute verifying `vaa` takes: its signature count is its sixth byte.
fn verify_cu(vaa: &[u8]) -> u32 {
    let signatures = u32::from(vaa.get(5).copied().unwrap_or(19)).max(1);
    CU_VERIFY_VAA_BASE + CU_VERIFY_VAA_PER_SIGNATURE * signatures
}

/// The compute updating `feed_id`'s push feed takes.
fn update_cu(feed_id: &[u8; 32]) -> u32 {
    CU_UPDATE_FEED + CU_UPDATE_FEED_PER_BUMP * u32::from(255 - price_feed_bump(feed_id))
}

/// An instruction and the compute it is given.
type Budgeted = (Instruction, u32);

/// The compute limit of a transaction running `ixs`, for `attempt`.
fn cu_limit(ixs: &[Budgeted], attempt: u64) -> u32 {
    let needed = ixs
        .iter()
        .fold(CU_TX_OVERHEAD, |sum, (_, cu)| sum.saturating_add(*cu));
    needed
        .saturating_add((attempt % ATTEMPT_CU_JITTER) as u32)
        .min(MAX_COMPUTE_UNIT_LIMIT)
}

fn refreshes(ixs: &[Instruction]) -> Vec<Budgeted> {
    ixs.iter().map(|ix| (ix.clone(), refresh_cu(ix))).collect()
}

/// The compute a venue refresh takes, by what it is.
fn refresh_cu(ix: &Instruction) -> u32 {
    use crate::juplend_earn::{client as juplend, program::Lending};
    use crate::kamino_lending::{client as kamino, program::KaminoLending};
    use anchor_lang::{Id, InstructionData};
    if ix.program_id == KaminoLending::id() {
        if ix.data == kamino::args::RefreshReserve.data() {
            return CU_KAMINO_REFRESH_RESERVE;
        }
        // Lending market, obligation, and its one deposit's reserve.
        if ix.data == kamino::args::RefreshObligation.data() && ix.accounts.len() <= 3 {
            return CU_KAMINO_REFRESH_OBLIGATION;
        }
    }
    if ix.program_id == Lending::id() && ix.data == juplend::args::UpdateRate.data() {
        return CU_JUPLEND_UPDATE_RATE;
    }
    CU_REFRESH_UNKNOWN
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
    /// What was pushed, and the refreshes carried behind it, to build the push again with a tip.
    pub prices: PushedPrices,
    pub trailing: Vec<Instruction>,
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
    /// liquidation's Kamino and JupLend refreshes) after them, and `tip` in whichever has room.
    pub fn push_transactions(
        &self,
        push: &PushedPrices,
        trailing: &[Instruction],
        tip: Option<&Instruction>,
        attempt: u64,
    ) -> Result<PricePush> {
        let (chunks, encoded_vaa) = self.layout(push, trailing, tip)?;
        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        let mut txs = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            // The first transaction creates the VAA account, which signs for itself.
            let signers: &[&Keypair] = if i == 0 {
                &[&self.payer, &encoded_vaa]
            } else {
                &[&self.payer]
            };
            txs.push(self.sign(chunk, signers, blockhash, attempt)?);
        }
        debug!(
            "PythPusher: {} transaction(s) push {} feed(s)",
            txs.len(),
            push.feeds.len()
        );
        Ok(PricePush {
            txs,
            prices: push.clone(),
            trailing: trailing.to_vec(),
        })
    }

    /// Whether `push` fits a bundle. Nothing is signed or fetched: only the transactions' sizes
    /// are measured.
    pub fn check(&self, push: &PushedPrices) -> Result<()> {
        self.layout(push, &[], None).map(|_| ())
    }

    /// `ixs` alone in one signed transaction, ahead of a liquidation in its bundle: its Kamino and
    /// JupLend refreshes when no push carries them. `None` when they don't fit one.
    pub fn transaction(
        &self,
        ixs: &[Instruction],
        attempt: u64,
    ) -> Result<Option<VersionedTransaction>> {
        let ixs = refreshes(ixs);
        if ixs.len() > MAX_REFRESHES_PER_TX || !self.fits(&ixs)? {
            return Ok(None);
        }
        let blockhash = self.cache.blockhash.get(&self.rpc_client)?;
        Ok(Some(self.sign(&ixs, &[&self.payer], blockhash, attempt)?))
    }

    /// The push's instructions split into the transactions that fit a bundle, and the VAA
    /// account's keypair, which signs the first.
    fn layout(
        &self,
        push: &PushedPrices,
        trailing: &[Instruction],
        tip: Option<&Instruction>,
    ) -> Result<(Vec<Vec<Budgeted>>, Keypair)> {
        let encoded_vaa = Keypair::new();
        let (post, rest) = self.instructions(push, &encoded_vaa.pubkey())?;
        if !self.fits(&post)? {
            return Err(anyhow!("posting the VAA doesn't fit a transaction"));
        }
        let mut chunks = vec![post];
        chunks.extend(self.pack(&rest, usize::MAX)?);
        // The refreshes in transactions of their own, never beside the VAA's verification, which
        // takes most of a transaction's compute.
        chunks.extend(self.pack(&refreshes(trailing), MAX_REFRESHES_PER_TX)?);
        if let Some(tip) = tip {
            let tip = (tip.clone(), CU_TRANSFER);
            let mut placed = false;
            for chunk in chunks.iter_mut().rev() {
                chunk.push(tip.clone());
                if self.fits(chunk)? {
                    placed = true;
                    break;
                }
                chunk.pop();
            }
            if !placed {
                chunks.push(vec![tip]);
            }
        }
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
    ) -> Result<(Vec<Budgeted>, Vec<Budgeted>)> {
        let vaa = push.vaa.as_slice();
        let split = VAA_SPLIT_INDEX.min(vaa.len());
        let mut post = self.post_instructions(push, vaa_account, split)?;
        let mut rest = post.split_off(3);
        rest.extend(self.update_instructions(push, vaa_account)?);
        Ok((post, rest))
    }

    /// Create the VAA account at `vaa_account`, initialize it, write the VAA up to `split`, then the
    /// tail, if any.
    fn post_instructions(
        &self,
        push: &PushedPrices,
        vaa_account: &Pubkey,
        split: usize,
    ) -> Result<Vec<Budgeted>> {
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

        let mut ixs = vec![
            (
                system_instruction::create_account(
                    &payer,
                    vaa_account,
                    lamports,
                    vaa_account_size as u64,
                    wormhole,
                ),
                CU_CREATE_ACCOUNT,
            ),
            (
                init_encoded_vaa_ix(wormhole, &payer, vaa_account),
                CU_INIT_VAA,
            ),
            (
                write_encoded_vaa_ix(wormhole, &payer, vaa_account, 0, &vaa[..split]),
                CU_WRITE_VAA,
            ),
        ];
        if split < vaa.len() {
            ixs.push((
                write_encoded_vaa_ix(wormhole, &payer, vaa_account, split as u32, &vaa[split..]),
                CU_WRITE_VAA,
            ));
        }
        Ok(ixs)
    }

    /// The VAA's verification, one update per feed, then the close that reclaims the VAA
    /// account's rent.
    fn update_instructions(
        &self,
        push: &PushedPrices,
        vaa_account: &Pubkey,
    ) -> Result<Vec<Budgeted>> {
        let payer = self.payer.pubkey();
        let wormhole = &self.wormhole_program_id;
        let mut ixs = vec![(
            verify_encoded_vaa_ix(
                wormhole,
                &payer,
                vaa_account,
                guardian_set_index(push.vaa.as_slice())?,
            ),
            verify_cu(push.vaa.as_slice()),
        )];
        for (_, signed) in &push.feeds {
            ixs.push((
                update_price_feed_ix(
                    &payer,
                    vaa_account,
                    signed.message.feed_id,
                    signed.update.clone(),
                )?,
                update_cu(&signed.message.feed_id),
            ));
        }
        ixs.push((
            close_encoded_vaa_ix(wormhole, &payer, vaa_account),
            CU_CLOSE_VAA,
        ));
        Ok(ixs)
    }

    /// Splits `ixs`, in order, into as few transactions as fit, at most `max_per_tx` each.
    fn pack(&self, ixs: &[Budgeted], max_per_tx: usize) -> Result<Vec<Vec<Budgeted>>> {
        let mut chunks: Vec<Vec<Budgeted>> = Vec::new();
        let mut current: Vec<Budgeted> = Vec::new();
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

    /// The message for `ixs` behind their compute budget, paid by the payer.
    fn message(&self, ixs: &[Budgeted], blockhash: Hash, attempt: u64) -> Result<v0::Message> {
        let limit = cu_limit(ixs, attempt);
        let mut all = vec![
            ComputeBudgetInstruction::set_compute_unit_limit(limit),
            ComputeBudgetInstruction::set_compute_unit_price(bundle_cu_price(
                limit,
                self.cu_price_micro_lamports,
            )),
        ];
        all.extend(ixs.iter().map(|(ix, _)| ix.clone()));
        Ok(v0::Message::try_compile(
            &self.payer.pubkey(),
            &all,
            &[],
            blockhash,
        )?)
    }

    /// Whether `ixs` fit one transaction. The size depends on neither the blockhash nor the
    /// signatures' values, so nothing is fetched or signed.
    fn fits(&self, ixs: &[Budgeted]) -> Result<bool> {
        // The limit's value doesn't change the size.
        let message = self.message(ixs, Hash::default(), 0)?;
        let signatures = usize::from(message.header.num_required_signatures);
        let tx = VersionedTransaction {
            signatures: vec![Signature::default(); signatures],
            message: VersionedMessage::V0(message),
        };
        Ok(bincode::serialize(&tx)?.len() <= MAX_V0_TX_SIZE)
    }

    fn sign(
        &self,
        ixs: &[Budgeted],
        signers: &[&Keypair],
        blockhash: Hash,
        attempt: u64,
    ) -> Result<VersionedTransaction> {
        let message = self.message(ixs, blockhash, attempt)?;
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
        let ixs: Vec<Budgeted> = (0..6u8)
            .map(|i| {
                let ix = Instruction {
                    program_id: program,
                    accounts: vec![AccountMeta::new(Pubkey::new_unique(), false)],
                    data: vec![i; 400],
                };
                (ix, CU_UPDATE_FEED)
            })
            .collect();
        let chunks = pusher.pack(&ixs, usize::MAX).unwrap();
        // A count cap splits further.
        assert_eq!(pusher.pack(&ixs, 2).unwrap().len(), 3);

        assert!(chunks.len() > 1);
        let order: Vec<u8> = chunks.iter().flatten().map(|(ix, _)| ix.data[0]).collect();
        assert_eq!(order, (0..6).collect::<Vec<u8>>());
        // Measured unsigned, each still fits once signed on a real blockhash.
        for chunk in &chunks {
            let tx = pusher
                .sign(chunk, &[&pusher.payer], Hash::new_unique(), 9_999)
                .unwrap();
            assert!(bincode::serialize(&tx).unwrap().len() <= MAX_V0_TX_SIZE);
        }
    }

    #[test]
    fn each_refresh_is_budgeted_by_its_kind() {
        use crate::juplend_earn::{client as juplend, program::Lending};
        use crate::kamino_lending::{client as kamino, program::KaminoLending};
        use anchor_lang::{Id, InstructionData};
        let ix = |program_id, data: Vec<u8>, accounts: usize| Instruction {
            program_id,
            accounts: (0..accounts)
                .map(|_| AccountMeta::new(Pubkey::new_unique(), false))
                .collect(),
            data,
        };
        let kamino = KaminoLending::id();
        assert_eq!(
            refresh_cu(&ix(kamino, kamino::args::RefreshReserve.data(), 6)),
            CU_KAMINO_REFRESH_RESERVE
        );
        assert_eq!(
            refresh_cu(&ix(kamino, kamino::args::RefreshObligation.data(), 3)),
            CU_KAMINO_REFRESH_OBLIGATION
        );
        assert_eq!(
            refresh_cu(&ix(kamino, kamino::args::RefreshObligation.data(), 5)),
            CU_REFRESH_UNKNOWN,
            "more positions than marginfi's obligations hold"
        );
        assert_eq!(
            refresh_cu(&ix(Lending::id(), juplend::args::UpdateRate.data(), 5)),
            CU_JUPLEND_UPDATE_RATE
        );
        assert_eq!(
            refresh_cu(&ix(Pubkey::new_unique(), vec![1], 1)),
            CU_REFRESH_UNKNOWN
        );
    }

    #[test]
    fn a_push_asks_for_the_compute_it_uses() {
        let ix = |cu| {
            (
                Instruction {
                    program_id: Pubkey::new_unique(),
                    accounts: vec![],
                    data: vec![],
                },
                cu,
            )
        };
        // Verifying the VAA and two updates.
        let verify = verify_cu(&[1, 0, 0, 0, 1, 3]);
        assert_eq!(verify, 104_000, "83,598 measured for three signatures");
        assert_eq!(verify_cu(&[1, 0, 0, 0, 4, 13]), 434_000);
        let ixs = [ix(verify), ix(CU_UPDATE_FEED), ix(CU_UPDATE_FEED)];
        let base = CU_TX_OVERHEAD + verify + 2 * CU_UPDATE_FEED;
        assert_eq!(cu_limit(&ixs, 0), base);
        assert_eq!(cu_limit(&ixs, 7), base + 7, "unique to its attempt");
        assert_eq!(cu_limit(&ixs, ATTEMPT_CU_JITTER), base);
        assert!(base < 200_000);
        // USDT's feed (bump 250) measured 34,923.
        let usdt: [u8; 32] = [
            0x2b, 0x89, 0xb9, 0xdc, 0x8f, 0xdf, 0x9f, 0x34, 0x70, 0x9a, 0x5b, 0x10, 0x6b, 0x47,
            0x2f, 0x0f, 0x39, 0xbb, 0x6c, 0xa9, 0xce, 0x04, 0xb0, 0xfd, 0x7f, 0x2e, 0x97, 0x16,
            0x88, 0xe2, 0xe5, 0x3b,
        ];
        assert_eq!(crate::utils::pyth_cranker::price_feed_bump(&usdt), 250);
        assert_eq!(update_cu(&usdt), 46_000);
        assert!(
            verify_cu(&[1, 0, 0, 0, 4, 19]) < MAX_COMPUTE_UNIT_LIMIT,
            "a full guardian set still fits"
        );
        let heavy: Vec<Budgeted> = (0..20).map(|_| ix(CU_REFRESH_UNKNOWN)).collect();
        assert_eq!(
            cu_limit(&heavy, 5),
            MAX_COMPUTE_UNIT_LIMIT,
            "never over the most"
        );
    }

    /// Pushes `PUSH_FEEDS` (default 2) feeds of a live Hermes payload (`PUSH_SIM_PAYLOAD_HEX`)
    /// through Helius Sender with the minimum tip inside the push, and waits for it on chain;
    /// prints how many transactions 1..=5 feeds take with the tip. Costs ~0.00104 SOL.
    /// `HELIUS_SENDER_API_KEY`, `WALLET_KEYPAIR`.
    #[test]
    #[ignore = "sends a real bundle"]
    fn a_tipped_push_lands_through_sender() {
        use crate::utils::helius_sender::{self, HeliusSender, MIN_TIP_LAMPORTS};
        let rpc_url = "https://api.mainnet-beta.solana.com".to_string();
        let hex = std::env::var("PUSH_SIM_PAYLOAD_HEX").unwrap();
        let payload: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let signed = crate::utils::hermes_stream::signed_updates(&payload).unwrap();
        let feeds: Vec<_> = signed
            .into_iter()
            .map(|(feed, signed)| {
                (
                    crate::utils::pyth_cranker::price_feed_address(&feed),
                    signed,
                )
            })
            .collect();
        let rpc = RpcClient::new(rpc_url.clone());
        let receiver_config = rpc.get_account(&get_config_address()).unwrap();
        let keypair: Vec<u8> =
            serde_json::from_str(&std::env::var("WALLET_KEYPAIR").unwrap()).unwrap();
        let pusher = PythPusher {
            cache: Arc::new(Cache::new(
                Pubkey::new_unique(),
                Pubkey::new_unique(),
                Arc::new(std::sync::Mutex::new(solana_clock::Clock::default())),
                std::collections::HashSet::new(),
            )),
            wormhole_program_id: Config::try_deserialize(&mut receiver_config.data.as_slice())
                .unwrap()
                .wormhole,
            rpc_client: RpcClient::new(rpc_url),
            payer: Keypair::try_from(keypair.as_slice()).unwrap(),
            cu_price_micro_lamports: 10_000,
        };
        let tip = helius_sender::tip_instruction(&pusher.payer.pubkey(), MIN_TIP_LAMPORTS).unwrap();
        let push_of = |n: usize| PushedPrices {
            vaa: Arc::clone(&feeds[0].1.vaa),
            feeds: feeds[..n].to_vec(),
        };
        for n in 1..=feeds.len() {
            let count = |tip| pusher.layout(&push_of(n), &[], tip).map(|(c, _)| c.len());
            println!(
                "{n} feed(s): {:?} txs, with the tip {:?}",
                count(None),
                count(Some(&tip))
            );
        }
        let n: usize = std::env::var("PUSH_FEEDS").map_or(2, |v| v.parse().unwrap());
        let push = pusher
            .push_transactions(&push_of(n), &[], Some(&tip), 7)
            .unwrap();
        let sender = HeliusSender::new(None, std::env::var("HELIUS_SENDER_API_KEY").ok());
        let sent_at = std::time::Instant::now();
        println!("answer {:?}", sender.send_bundle(&push.txs));
        let sigs: Vec<_> = push.txs.iter().map(|tx| tx.signatures[0]).collect();
        for _ in 0..60 {
            let statuses = rpc.get_signature_statuses(&sigs).unwrap().value;
            if statuses.iter().all(Option::is_some) {
                println!(
                    "{} txs on chain after {} ms: {:?}",
                    sigs.len(),
                    sent_at.elapsed().as_millis(),
                    statuses
                        .iter()
                        .map(|s| s.as_ref().map(|s| (s.slot, s.err.clone())))
                        .collect::<Vec<_>>()
                );
                println!("first signature {}", sigs[0]);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        panic!("not on chain after 30 s: {}", sigs[0]);
    }
}
