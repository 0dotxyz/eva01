//! Keeps Pyth push-oracle accounts fresh when their sponsor stops pushing.
//!
//! marginfi prices a Pyth bank from the Pyth-sponsored `PriceUpdateV2` account at
//! `PDA([0, feed_id], pyth-push-oracle)`, and rejects it once it is older than the bank's
//! `oracle_max_age`. If its sponsors stop pushing, this service posts the update itself: pull the
//! signed price from Hermes, post its VAA to the Wormhole program the receiver verifies against,
//! then `update_price_feed` on the push-oracle program.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use anchor_lang::{prelude::borsh, AccountDeserialize};
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use log::{debug, info, warn};
use marginfi_type_crate::constants::PYTH_SPONSORED_SHARD_ID;
use pyth_solana_receiver_sdk::{
    config::Config,
    pda::{get_config_address, get_treasury_address},
    price_update::PriceUpdateV2,
    PostUpdateParams, PYTH_PUSH_ORACLE_ID,
};
use pythnet_sdk::{
    messages::Message,
    wire::{
        from_slice,
        v1::{AccumulatorUpdateData, MerklePriceUpdate, Proof},
    },
};
use reqwest::blocking::Client;
use serde::Deserialize;
use solana_client::{
    client_error::ClientErrorKind,
    rpc_client::RpcClient,
    rpc_request::{RpcError, RpcResponseErrorData},
};
use solana_commitment_config::CommitmentConfig;
use solana_instruction::{AccountMeta, Instruction};
use solana_keypair::Keypair;
use solana_pubkey::Pubkey;
use solana_sdk_ids::system_program;
use solana_signer::Signer;
use solana_system_interface::instruction as system_instruction;
use solana_transaction::Transaction;

use crate::{cache::Cache, clock_manager, config::Eva01Config};

pub const DEFAULT_HERMES_URL: &str = "https://hermes.pyth.network";

/// How often the cached price accounts are checked for staleness (in-memory, no RPC).
const CHECK_INTERVAL: Duration = Duration::from_secs(5);
/// Two independent crankers keep these feeds fresh, so this one only adopts a feed once it is this
/// many times past its `oracle_max_age`, i.e. long after the others gave up on it.
const ADOPT_AGE_MULTIPLE: u64 = 3;
/// While a feed is adopted it is refreshed at this fraction of its `oracle_max_age`, which keeps it
/// inside the window marginfi accepts instead of letting it decay back to the adoption threshold.
const REFRESH_AGE_DIVISOR: u64 = 2;

/// Anchor `global:<name>` discriminators of the instructions this builds.
const IX_INIT_ENCODED_VAA: [u8; 8] = [209, 193, 173, 25, 91, 202, 181, 218];
const IX_WRITE_ENCODED_VAA: [u8; 8] = [199, 208, 110, 177, 150, 76, 118, 42];
const IX_VERIFY_ENCODED_VAA_V1: [u8; 8] = [103, 56, 177, 229, 240, 103, 68, 73];
const IX_CLOSE_ENCODED_VAA: [u8; 8] = [48, 221, 174, 198, 231, 7, 152, 38];
const IX_UPDATE_PRICE_FEED: [u8; 8] = [28, 9, 93, 150, 86, 153, 188, 115];

/// Header the Wormhole account keeps in front of the VAA bytes.
const VAA_HEADER_SIZE: usize = 46;
/// The VAA is written in two chunks; the first is as large as fits alongside create + init.
const VAA_SPLIT_INDEX: usize = 721;
const DEFAULT_TREASURY_ID: u8 = 0;

/// Seconds since `publish_time`, as of the cached clock. The cached clock can trail a fresh
/// update's publish time, so a negative age is clamped to 0 rather than wrapping when compared as
/// unsigned (which made fresh feeds look ancient and get re-cranked).
fn feed_age(now: i64, publish_time: i64) -> u64 {
    now.saturating_sub(publish_time).max(0) as u64
}

#[derive(Deserialize)]
struct HermesResponse {
    binary: HermesBinary,
}

#[derive(Deserialize)]
struct HermesBinary {
    data: Vec<String>,
}

/// A push-oracle account that needs a new price, and the coordinates to write it.
struct StaleFeed {
    oracle: Pubkey,
    feed_id: [u8; 32],
    age: u64,
    max_age: u64,
}

pub struct PythCranker {
    wormhole_program_id: Pubkey,
    hermes_url: String,
    hermes_api_key: Option<String>,
    http_client: Client,
    rpc_client: RpcClient,
    payer: Keypair,
    cache: Arc<Cache>,
    /// Feeds this cranker is keeping alive, and the publish time it last posted for each. A newer
    /// publish time on-chain means their sponsor resumed, so the feed is released.
    adopted: Mutex<HashMap<Pubkey, i64>>,
    stop: Arc<AtomicBool>,
}

impl PythCranker {
    pub fn new(config: &Eva01Config, cache: Arc<Cache>, stop: Arc<AtomicBool>) -> Result<Self> {
        let rpc_client =
            RpcClient::new_with_commitment(config.rpc_url.clone(), CommitmentConfig::confirmed());
        let receiver_config = rpc_client.get_account(&get_config_address())?;
        let wormhole_program_id = Config::try_deserialize(&mut receiver_config.data.as_slice())
            .map_err(|e| anyhow!("Failed to read the Pyth receiver config: {e:?}"))?
            .wormhole;

        Ok(Self {
            wormhole_program_id,
            hermes_url: config.pyth_hermes_url.clone(),
            hermes_api_key: config.pyth_api_key.clone(),
            http_client: Client::new(),
            rpc_client,
            payer: Keypair::try_from(config.wallet_keypair.as_slice())?,
            cache,
            adopted: Mutex::new(HashMap::new()),
            stop,
        })
    }

    pub fn start(&self) {
        info!("PythCranker starting.");
        while !self.stop.load(Ordering::Relaxed) {
            match self.stale_feeds() {
                Ok(feeds) => {
                    for feed in feeds {
                        debug!(
                            "PythCranker: cranking {} ({}s old, max {}s)",
                            feed.oracle, feed.age, feed.max_age
                        );
                        if let Err(e) = self.crank(&feed) {
                            warn!("PythCranker: failed to crank {}: {e}", feed.oracle);
                        }
                    }
                }
                Err(e) => warn!("PythCranker: staleness check failed: {e}"),
            }
            thread::sleep(CHECK_INTERVAL);
        }
        info!("PythCranker stopped.");
    }

    /// Feeds to crank this cycle.
    ///
    /// A feed is adopted once it is [`ADOPT_AGE_MULTIPLE`] times past its max age, and is then kept
    /// alive on the [`REFRESH_AGE_DIVISOR`] cadence until its sponsor comes back: a publish time
    /// newer than the one we posted means somebody else cranked it, so we release it.
    fn stale_feeds(&self) -> Result<Vec<StaleFeed>> {
        let now = clock_manager::get_clock(&self.cache.clock)?.unix_timestamp;
        let mut adopted = self
            .adopted
            .lock()
            .map_err(|_| anyhow!("The adopted feeds map is poisoned"))?;
        let mut feeds = Vec::new();

        for (oracle, max_age) in self.cache.banks.get_pyth_push_oracles() {
            if max_age == 0 {
                continue;
            }
            let account = match self.cache.oracles.try_get_account(&oracle) {
                Ok(account) => account,
                Err(_) => continue,
            };
            let price_update = match PriceUpdateV2::try_deserialize(&mut account.data.as_slice()) {
                Ok(price_update) => price_update,
                Err(_) => continue,
            };

            let published = price_update.price_message.publish_time;
            let threshold = match adopted.get(&oracle) {
                Some(&ours) if published > ours => {
                    info!(
                        "PythCranker: the Pyth feed {oracle} is being cranked again, releasing it"
                    );
                    adopted.remove(&oracle);
                    continue;
                }
                // Our own write may not have reached the cache yet, so age from whichever is newer.
                Some(&ours) => {
                    let age = feed_age(now, ours.max(published));
                    if age < max_age / REFRESH_AGE_DIVISOR {
                        continue;
                    }
                    age
                }
                None => {
                    let age = feed_age(now, published);
                    if age < max_age.saturating_mul(ADOPT_AGE_MULTIPLE) {
                        continue;
                    }
                    warn!(
                        "Pyth feed {oracle} is {age}s old (max {max_age}s): its sponsors stopped, \
                         adopting it until they resume."
                    );
                    age
                }
            };

            let feed_id = price_update.price_message.feed_id;
            if price_feed_address(&feed_id) != oracle {
                debug!("Pyth feed {oracle} is not a sponsored push feed, skipping the crank");
                continue;
            }
            feeds.push(StaleFeed {
                oracle,
                feed_id,
                age: threshold,
                max_age,
            });
        }

        Ok(feeds)
    }

    /// Latest signed update for a feed, as the accumulator payload Hermes serves.
    fn fetch_update(&self, feed_id: &[u8; 32]) -> Result<AccumulatorUpdateData> {
        let url = format!(
            "{}/v2/updates/price/latest?ids[]=0x{}&encoding=base64",
            self.hermes_url.trim_end_matches('/'),
            hex_encode(feed_id)
        );
        let mut request = self.http_client.get(&url);
        if let Some(api_key) = &self.hermes_api_key {
            request = request.bearer_auth(api_key);
        }
        let response: HermesResponse = request.send()?.error_for_status()?.json()?;
        let encoded = response
            .binary
            .data
            .first()
            .ok_or_else(|| anyhow!("Hermes returned no update for the feed"))?;

        AccumulatorUpdateData::try_from_slice(&BASE64.decode(encoded)?)
            .map_err(|e| anyhow!("Failed to parse the Hermes accumulator update: {e:?}"))
    }

    /// Post the VAA, update the feed account, then reclaim the VAA account's rent. Two transactions
    /// because the VAA doesn't fit in one, and the second depends on the first having landed.
    fn crank(&self, feed: &StaleFeed) -> Result<()> {
        let update = self.fetch_update(&feed.feed_id)?;
        let Proof::WormholeMerkle { vaa, updates } = update.proof;
        let vaa: Vec<u8> = vaa.into();
        let merkle_price_update = updates
            .first()
            .ok_or_else(|| anyhow!("Hermes update carried no merkle price update"))?
            .clone();
        let posted_publish_time = publish_time(&merkle_price_update)?;

        let payer = self.payer.pubkey();
        let encoded_vaa = Keypair::new();
        let vaa_account_size = VAA_HEADER_SIZE + vaa.len();
        let lamports = self
            .rpc_client
            .get_minimum_balance_for_rent_exemption(vaa_account_size)?;

        // A VAA carrying the full guardian set doesn't fit in one transaction alongside the
        // account creation, so the tail is written in the second one.
        let split = VAA_SPLIT_INDEX.min(vaa.len());
        let post = vec![
            system_instruction::create_account(
                &payer,
                &encoded_vaa.pubkey(),
                lamports,
                vaa_account_size as u64,
                &self.wormhole_program_id,
            ),
            self.init_encoded_vaa_ix(&payer, &encoded_vaa.pubkey()),
            self.write_encoded_vaa_ix(&payer, &encoded_vaa.pubkey(), 0, &vaa[..split]),
        ];
        self.send(&post, &[&self.payer, &encoded_vaa])?;

        let mut update = Vec::new();
        if split < vaa.len() {
            update.push(self.write_encoded_vaa_ix(
                &payer,
                &encoded_vaa.pubkey(),
                split as u32,
                &vaa[split..],
            ));
        }
        update.extend([
            self.verify_encoded_vaa_ix(&payer, &encoded_vaa.pubkey(), guardian_set_index(&vaa)?),
            update_price_feed_ix(
                &payer,
                &encoded_vaa.pubkey(),
                feed.feed_id,
                merkle_price_update,
            )?,
            self.close_encoded_vaa_ix(&payer, &encoded_vaa.pubkey()),
        ]);
        let signature = match self.send(&update, &[&self.payer]) {
            Ok(signature) => signature,
            Err(e) => {
                // The VAA account from the first transaction holds rent that only this payer can
                // reclaim; close it on the way out instead of stranding it.
                let close = self.close_encoded_vaa_ix(&payer, &encoded_vaa.pubkey());
                if let Err(close_err) = self.send(&[close], &[&self.payer]) {
                    warn!(
                        "PythCranker: failed to reclaim the VAA account {} after a failed update: \
                         {close_err}",
                        encoded_vaa.pubkey()
                    );
                }
                return Err(e);
            }
        };

        self.adopted
            .lock()
            .map_err(|_| anyhow!("The adopted feeds map is poisoned"))?
            .insert(feed.oracle, posted_publish_time);

        info!(
            "PythCranker: cranked the Pyth feed {}: {}",
            feed.oracle, signature
        );
        Ok(())
    }

    fn send(&self, instructions: &[Instruction], signers: &[&Keypair]) -> Result<String> {
        let blockhash = self.rpc_client.get_latest_blockhash()?;
        let tx = Transaction::new_signed_with_payer(
            instructions,
            Some(&self.payer.pubkey()),
            signers,
            blockhash,
        );
        match self.rpc_client.send_and_confirm_transaction(&tx) {
            Ok(signature) => Ok(signature.to_string()),
            Err(e) => {
                if let ClientErrorKind::RpcError(RpcError::RpcResponseError {
                    data: RpcResponseErrorData::SendTransactionPreflightFailure(result),
                    ..
                }) = e.kind()
                {
                    if let Some(logs) = &result.logs {
                        warn!("PythCranker: crank simulation failed:\n{}", logs.join("\n"));
                    }
                }
                Err(e.into())
            }
        }
    }

    fn guardian_set_address(&self, index: u32) -> Pubkey {
        Pubkey::find_program_address(
            &[b"GuardianSet", &index.to_be_bytes()],
            &self.wormhole_program_id,
        )
        .0
    }

    fn init_encoded_vaa_ix(&self, write_authority: &Pubkey, encoded_vaa: &Pubkey) -> Instruction {
        Instruction {
            program_id: self.wormhole_program_id,
            accounts: vec![
                AccountMeta::new_readonly(*write_authority, true),
                AccountMeta::new(*encoded_vaa, false),
            ],
            data: IX_INIT_ENCODED_VAA.to_vec(),
        }
    }

    fn write_encoded_vaa_ix(
        &self,
        write_authority: &Pubkey,
        draft_vaa: &Pubkey,
        index: u32,
        data: &[u8],
    ) -> Instruction {
        let mut ix_data = IX_WRITE_ENCODED_VAA.to_vec();
        ix_data.extend_from_slice(&index.to_le_bytes());
        ix_data.extend_from_slice(&(data.len() as u32).to_le_bytes());
        ix_data.extend_from_slice(data);

        Instruction {
            program_id: self.wormhole_program_id,
            accounts: vec![
                AccountMeta::new_readonly(*write_authority, true),
                AccountMeta::new(*draft_vaa, false),
            ],
            data: ix_data,
        }
    }

    fn verify_encoded_vaa_ix(
        &self,
        write_authority: &Pubkey,
        draft_vaa: &Pubkey,
        guardian_set_index: u32,
    ) -> Instruction {
        Instruction {
            program_id: self.wormhole_program_id,
            accounts: vec![
                AccountMeta::new_readonly(*write_authority, true),
                AccountMeta::new(*draft_vaa, false),
                AccountMeta::new_readonly(self.guardian_set_address(guardian_set_index), false),
            ],
            data: IX_VERIFY_ENCODED_VAA_V1.to_vec(),
        }
    }

    fn close_encoded_vaa_ix(&self, write_authority: &Pubkey, encoded_vaa: &Pubkey) -> Instruction {
        Instruction {
            program_id: self.wormhole_program_id,
            accounts: vec![
                AccountMeta::new(*write_authority, true),
                AccountMeta::new(*encoded_vaa, false),
            ],
            data: IX_CLOSE_ENCODED_VAA.to_vec(),
        }
    }
}

/// The Pyth-sponsored push feed account for `feed_id`. marginfi no longer sponsors a shard of its
/// own, so shard 0 is the only one banks point at.
fn price_feed_address(feed_id: &[u8; 32]) -> Pubkey {
    Pubkey::find_program_address(
        &[&PYTH_SPONSORED_SHARD_ID.to_le_bytes(), feed_id],
        &PYTH_PUSH_ORACLE_ID,
    )
    .0
}

/// Guardian set that signed the VAA: bytes 1..5 of the header, big-endian.
fn guardian_set_index(vaa: &[u8]) -> Result<u32> {
    let bytes = vaa
        .get(1..5)
        .ok_or_else(|| anyhow!("VAA is too short to carry a guardian set index"))?;
    Ok(u32::from_be_bytes(bytes.try_into()?))
}

fn update_price_feed_ix(
    payer: &Pubkey,
    encoded_vaa: &Pubkey,
    feed_id: [u8; 32],
    merkle_price_update: MerklePriceUpdate,
) -> Result<Instruction> {
    let params = PostUpdateParams {
        merkle_price_update,
        treasury_id: DEFAULT_TREASURY_ID,
    };

    let mut data = IX_UPDATE_PRICE_FEED.to_vec();
    data.extend_from_slice(&borsh::to_vec(&params)?);
    data.extend_from_slice(&PYTH_SPONSORED_SHARD_ID.to_le_bytes());
    data.extend_from_slice(&feed_id);

    Ok(Instruction {
        program_id: PYTH_PUSH_ORACLE_ID,
        accounts: vec![
            AccountMeta::new(*payer, true),
            AccountMeta::new_readonly(pyth_solana_receiver_sdk::ID, false),
            AccountMeta::new_readonly(*encoded_vaa, false),
            AccountMeta::new_readonly(get_config_address(), false),
            AccountMeta::new(get_treasury_address(DEFAULT_TREASURY_ID), false),
            AccountMeta::new(price_feed_address(&feed_id), false),
            AccountMeta::new_readonly(system_program::ID, false),
        ],
        data,
    })
}

/// Publish time carried by the update we post, used to recognise our own write later.
fn publish_time(update: &MerklePriceUpdate) -> Result<i64> {
    match from_slice::<byteorder::BE, Message>(update.message.as_ref())
        .map_err(|e| anyhow!("Failed to parse the price message: {e:?}"))?
    {
        Message::PriceFeedMessage(message) => Ok(message.publish_time),
        _ => Err(anyhow!("The Hermes update is not a price feed message")),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_age_counts_seconds_since_publish() {
        assert_eq!(feed_age(1_000, 940), 60);
        assert_eq!(feed_age(1_000, 1_000), 0);
    }

    #[test]
    fn feed_published_after_the_cached_clock_is_fresh() {
        // The cached clock trails a just-published update: that's age 0, not ~u64::MAX.
        assert_eq!(feed_age(1_000, 1_003), 0);
        assert!(feed_age(1_000, 1_003) < 60 / REFRESH_AGE_DIVISOR);
    }
}
