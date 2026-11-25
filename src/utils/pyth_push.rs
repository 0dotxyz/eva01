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
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::Hash;
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_system_interface::instruction as system_instruction;
use solana_transaction::versioned::VersionedTransaction;

use anchor_lang::AccountDeserialize;
use pyth_solana_receiver_sdk::pda::get_config_address;

use std::sync::Arc;

use crate::cache::{pyth_prices::PushedPrices, Cache};
use crate::config::Eva01Config;
use crate::transaction_limits::{
    priority_fee, signed_size, v1_message, v1_transaction, DoesNotFit, MAX_BUNDLE_TXS,
    MAX_COMPUTE_UNIT_LIMIT, MAX_TX_SIZE,
};
use crate::utils::pyth_cranker::{
    close_encoded_vaa_ix, guardian_set_index, init_encoded_vaa_ix, price_feed_bump,
    update_price_feed_ix, verify_encoded_vaa_ix, write_encoded_vaa_ix, VAA_HEADER_SIZE,
};

/// The push transactions a bundle has room for: the liquidation and the tip take two. Jito's tip
/// goes after the liquidation; Sender's rides in a push transaction, which a Sender bundle keeps
/// atomic with the liquidation.
const MAX_PUSH_TXS: usize = MAX_BUNDLE_TXS - 2;

/// The most feeds a liquidation is sized on pushing. Whether they fit the bundle is measured
/// (`PythPusher::check`).
pub const MAX_PUSHED_FEEDS: usize = 4;

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
/// a mainnet group (worst in parentheses) with about 30% over it. A reserve's cost grows ~5k per
/// link of its Scope chain (1 link ~24k, 2 links 29,209), so a chain of four (Kamino's most) is
/// ~39k; this keeps ~28% over that.
const CU_KAMINO_REFRESH_RESERVE: u32 = 50_000;
/// One deposit, no borrows, its reserve as the only remaining account: marginfi's shape (19,753).
const CU_KAMINO_REFRESH_OBLIGATION: u32 = 26_000;
/// JupLend `UpdateRate` (15,515).
const CU_JUPLEND_UPDATE_RATE: u32 = 20_000;
/// Any other instruction among the refreshes: unmeasured, so generous.
const CU_REFRESH_UNKNOWN: u32 = 150_000;
/// What a transaction is charged around its instructions.
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

/// The compute `ixs` take in one transaction.
fn cu_needed(ixs: &[Budgeted]) -> u32 {
    ixs.iter()
        .fold(CU_TX_OVERHEAD, |sum, (_, cu)| sum.saturating_add(*cu))
}

/// The compute limit of a transaction running `ixs`, for `attempt`.
fn cu_limit(ixs: &[Budgeted], attempt: u64) -> u32 {
    cu_needed(ixs)
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
/// always closed within it.
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
        if !self.fits(&ixs)? {
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
        // In order, so the first transaction creates the VAA account, and the refreshes read the
        // prices already written.
        let mut ixs = self.instructions(push, &encoded_vaa.pubkey())?;
        ixs.extend(refreshes(trailing));
        let mut chunks = self.pack(&ixs)?;
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

    /// The push's instructions, in order: create the VAA account at `vaa_account` and post the VAA,
    /// verify it, update the feeds and close it.
    fn instructions(&self, push: &PushedPrices, vaa_account: &Pubkey) -> Result<Vec<Budgeted>> {
        let mut ixs = self.post_instructions(push, vaa_account)?;
        ixs.extend(self.update_instructions(push, vaa_account)?);
        Ok(ixs)
    }

    /// Create the VAA account at `vaa_account`, initialize it, and write the VAA in one go.
    fn post_instructions(
        &self,
        push: &PushedPrices,
        vaa_account: &Pubkey,
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

        Ok(vec![
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
                write_encoded_vaa_ix(wormhole, &payer, vaa_account, 0, vaa),
                CU_WRITE_VAA,
            ),
        ])
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

    /// Splits `ixs`, in order, into as few transactions as fit.
    fn pack(&self, ixs: &[Budgeted]) -> Result<Vec<Vec<Budgeted>>> {
        let mut chunks: Vec<Vec<Budgeted>> = Vec::new();
        let mut current: Vec<Budgeted> = Vec::new();
        for ix in ixs {
            current.push(ix.clone());
            if !self.fits(&current)? {
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

    /// Whether `ixs` fit one transaction: its size, its accounts, and its compute with room for
    /// any attempt's jitter. Nothing is fetched or signed: neither the blockhash nor the signatures'
    /// values change the size.
    fn fits(&self, ixs: &[Budgeted]) -> Result<bool> {
        if cu_needed(ixs).saturating_add(ATTEMPT_CU_JITTER as u32) > MAX_COMPUTE_UNIT_LIMIT {
            return Ok(false);
        }
        let only: Vec<Instruction> = ixs.iter().map(|(ix, _)| ix.clone()).collect();
        let message = match v1_message(&self.payer.pubkey(), &only, Hash::default(), 0, 0) {
            Ok(message) => message,
            // Too many accounts or instructions for one transaction.
            Err(e) if e.is::<DoesNotFit>() => return Ok(false),
            Err(e) => return Err(e),
        };
        Ok(signed_size(&message)? <= MAX_TX_SIZE)
    }

    /// `ixs` signed by `signers`, the payer first, limited to the compute they take for `attempt`.
    fn sign(
        &self,
        ixs: &[Budgeted],
        signers: &[&Keypair],
        blockhash: Hash,
        attempt: u64,
    ) -> Result<VersionedTransaction> {
        let limit = cu_limit(ixs, attempt);
        let only: Vec<Instruction> = ixs.iter().map(|(ix, _)| ix.clone()).collect();
        v1_transaction(
            signers,
            &only,
            blockhash,
            limit,
            priority_fee(limit, self.cu_price_micro_lamports),
        )
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
        let ix = |i: u8, data: usize, cu: u32| {
            let ix = Instruction {
                program_id: program,
                accounts: vec![AccountMeta::new(Pubkey::new_unique(), false)],
                data: vec![i; data],
            };
            (ix, cu)
        };
        // About the size of a VAA write: four to a transaction.
        let ixs: Vec<Budgeted> = (0..6u8).map(|i| ix(i, 900, CU_UPDATE_FEED)).collect();
        let chunks = pusher.pack(&ixs).unwrap();
        assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), [4, 2]);
        let order: Vec<u8> = chunks.iter().flatten().map(|(ix, _)| ix.data[0]).collect();
        assert_eq!(order, (0..6).collect::<Vec<u8>>());
        // Measured unsigned, each still fits once signed on a real blockhash.
        for chunk in &chunks {
            let tx = pusher
                .sign(chunk, &[&pusher.payer], Hash::new_unique(), 9_999)
                .unwrap();
            assert!(crate::transaction_limits::wire_bytes(&tx).unwrap().len() <= MAX_TX_SIZE);
        }
        // Small, but too much compute for one.
        let heavy: Vec<Budgeted> = (0..3u8).map(|i| ix(i, 10, 700_000)).collect();
        assert_eq!(pusher.pack(&heavy).unwrap().len(), 3);
        // More than any transaction holds.
        assert!(pusher.pack(&[ix(0, MAX_TX_SIZE, 1)]).is_err());
        // Over the account limit together: split, not refused.
        let wide = |i: u8| {
            let ix = Instruction {
                program_id: program,
                accounts: (0..40)
                    .map(|_| AccountMeta::new(Pubkey::new_unique(), false))
                    .collect(),
                data: vec![i],
            };
            (ix, 1_000)
        };
        assert_eq!(pusher.pack(&[wide(0), wide(1)]).unwrap().len(), 2);
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

    /// How a live Hermes payload of 1..=5 feeds is laid out, with and without a tip inside: the
    /// transactions, their bytes and compute. Nothing is sent. `PYTH_API_KEY`.
    #[test]
    #[ignore = "reads Hermes and the RPC"]
    fn a_live_push_layout() {
        let ids = [
            "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
            "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
            "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
            "eaa020c61cc479712813461ce153894a96a6c00b21ed0cfc2798d1f9a9e9c94a",
            "2b89b9dc8fdf9f34709a5b106b472f0f39bb6ca9ce04b0fd7f2e971688e2e53b",
        ];
        let query: String = ids.iter().map(|id| format!("ids[]={id}&")).collect();
        let resp: serde_json::Value = reqwest::blocking::Client::new()
            .get(format!(
                "{}/v2/updates/price/latest?{query}encoding=hex",
                crate::utils::pyth_cranker::DEFAULT_HERMES_URL.trim_end_matches('/')
            ))
            .bearer_auth(std::env::var("PYTH_API_KEY").unwrap())
            .send()
            .unwrap()
            .json()
            .unwrap();
        let hex = resp["binary"]["data"][0].as_str().unwrap();
        let payload: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let feeds: Vec<_> = crate::utils::hermes_stream::signed_updates(&payload)
            .unwrap()
            .into_iter()
            .map(|(feed, signed)| {
                (
                    crate::utils::pyth_cranker::price_feed_address(&feed),
                    signed,
                )
            })
            .collect();
        let rpc = RpcClient::new("https://api.mainnet-beta.solana.com".to_string());
        let receiver_config = rpc.get_account(&get_config_address()).unwrap();
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
            rpc_client: rpc,
            payer: Keypair::new(),
            cu_price_micro_lamports: 10_000,
        };
        let tip = crate::utils::helius_sender::tip_instruction(&pusher.payer.pubkey(), 1_000_000)
            .unwrap();
        println!("VAA {} bytes", feeds[0].1.vaa.len());
        for n in 1..=feeds.len() {
            let push = PushedPrices {
                vaa: Arc::clone(&feeds[0].1.vaa),
                feeds: feeds[..n].to_vec(),
            };
            for tipped in [None, Some(&tip)] {
                let (chunks, vaa) = pusher.layout(&push, &[], tipped).unwrap();
                let shape: Vec<String> = chunks
                    .iter()
                    .enumerate()
                    .map(|(i, chunk)| {
                        let signers: Vec<&Keypair> = if i == 0 {
                            vec![&pusher.payer, &vaa]
                        } else {
                            vec![&pusher.payer]
                        };
                        let tx = pusher.sign(chunk, &signers, Hash::new_unique(), 0).unwrap();
                        format!(
                            "{} B / {} CU",
                            crate::transaction_limits::wire_bytes(&tx).unwrap().len(),
                            cu_limit(chunk, 0)
                        )
                    })
                    .collect();
                println!("{n} feed(s), tip {}: {shape:?}", tipped.is_some());
            }
        }
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
