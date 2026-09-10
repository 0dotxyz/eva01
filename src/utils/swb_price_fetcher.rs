use std::{
    collections::HashSet,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::cache::is_switchboard_pull_setup;
use anyhow::Result;
use bytemuck::Zeroable;
use fixed::types::I80F48;
use log::{debug, info, warn};
use reqwest::blocking::Client;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Deserializer};
use solana_cluster_type::ClusterType;
use solana_sdk::{account::Account, pubkey, pubkey::Pubkey};
use switchboard_on_demand_client::{CrossbarClient, PullFeedAccountData};
use tokio::runtime::{Builder, Runtime};

use crate::{
    cache::Cache,
    geyser::{AccountType, GeyserUpdate},
};
use crossbeam::channel::Sender;

const SWITCHBOARD_PULL_PROGRAM_ID: Pubkey = pubkey!("SBondMDrcV3K4kxZR1HNVT7osZxAHVHgYXL5Ze1oMUv");
const SWB_PULL_FEED_DISCRIMINATOR: [u8; 8] = [196, 27, 108, 196, 10, 215, 219, 40];

fn build_synthetic_swb_account(price: I80F48, conf: I80F48) -> Account {
    let mut feed = PullFeedAccountData::zeroed();
    feed.last_update_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    const SCALE: f64 = 1e18;
    feed.result.value = (price.to_num::<f64>() * SCALE) as i128;
    feed.result.std_dev = (conf.to_num::<f64>() * SCALE) as i128;
    feed.result.mean = feed.result.value;
    feed.result.num_samples = 1;

    let mut data = Vec::with_capacity(8 + std::mem::size_of::<PullFeedAccountData>());
    data.extend_from_slice(&SWB_PULL_FEED_DISCRIMINATOR);
    data.extend_from_slice(bytemuck::bytes_of(&feed));

    Account {
        lamports: 1,
        data,
        owner: SWITCHBOARD_PULL_PROGRAM_ID,
        executable: false,
        rent_epoch: 0,
    }
}

const FETCH_INTERVAL: Duration = Duration::from_secs(30);
const FALLBACK_CROSSBAR_URL: &str = "https://crossbar.switchboard.xyz";

#[derive(Deserialize)]
struct ViewPriceResponse {
    prices: std::collections::HashMap<String, ViewPriceEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ViewPriceEntry {
    oracle_price: OraclePriceDto,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OraclePriceDto {
    /// Integration-adjusted final price (raw feed price * integration ratio). For banks that share
    /// a Switchboard feed across multiple LSTs (KaminoSwitchboardPull / JuplendSwitchboardPull),
    /// this differs per-bank and must NOT be written into the shared oracle account.
    price_realtime: PriceWithConfidenceDto,
    /// Raw underlying feed price BEFORE the integration ratio is applied. Present only for
    /// integration banks. This is what the on-chain Switchboard feed account actually holds; the
    /// program re-applies the per-bank exchange rate on read. All banks sharing a feed report the
    /// same `sourcePrice`, so writing it is collision-free.
    source_price: Option<SourcePriceDto>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourcePriceDto {
    price_realtime: PriceWithConfidenceDto,
}

#[derive(Deserialize)]
struct PriceWithConfidenceDto {
    #[serde(deserialize_with = "deserialize_f64_from_string")]
    price: f64,
    #[serde(deserialize_with = "deserialize_f64_from_string")]
    confidence: f64,
}

fn deserialize_f64_from_string<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<f64, D::Error> {
    String::deserialize(deserializer)?
        .parse::<f64>()
        .map_err(serde::de::Error::custom)
}

pub struct SwbPriceFetcher {
    api_url: Option<String>,
    http_client: Client,
    crossbar: CrossbarClient,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    geyser_tx: Sender<GeyserUpdate>,
    stop: Arc<AtomicBool>,
}

impl SwbPriceFetcher {
    pub fn new(
        api_url: Option<String>,
        crossbar_api_url: Option<String>,
        cache: Arc<Cache>,
        geyser_tx: Sender<GeyserUpdate>,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let crossbar_url = crossbar_api_url.as_deref().unwrap_or(FALLBACK_CROSSBAR_URL);
        let tokio_rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build SwbPriceFetcher tokio runtime");

        Self {
            api_url,
            http_client: Client::new(),
            crossbar: CrossbarClient::new(crossbar_url, false),
            tokio_rt,
            cache,
            geyser_tx,
            stop,
        }
    }

    /// These oracles are excluded from the Geyser subscription, so the liquidator never sees them
    /// change. Publish every write into the same channel, otherwise a price move on a Switchboard
    /// feed queues no accounts for the liquidatability check.
    fn publish(&self, address: Pubkey, account: Account) {
        if let Err(e) = self.geyser_tx.send(GeyserUpdate {
            account_type: AccountType::Oracle,
            address,
            account,
        }) {
            warn!("SwbPriceFetcher: failed to publish the oracle {address} update: {e}");
        }
    }

    pub fn start(&self) {
        info!("SwbPriceFetcher starting.");
        while !self.stop.load(Ordering::Relaxed) {
            if let Err(e) = self.fetch_and_update() {
                warn!("SwbPriceFetcher: fetch cycle failed: {e}");
            }
            thread::sleep(FETCH_INTERVAL);
        }
        info!("SwbPriceFetcher stopped.");
    }

    pub fn fetch_and_update(&self) -> Result<()> {
        // The API only serves the banks of its own group, so fall back to Crossbar whenever it
        // covered fewer than all of ours, not only when the request itself failed.
        let tracked = self.cache.banks.get_swb_oracles();
        if tracked.is_empty() {
            return Ok(());
        }

        let mut updated: HashSet<Pubkey> = HashSet::new();
        if let Some(api_url) = &self.api_url {
            match self.fetch_from_api(api_url) {
                Ok(from_api) => updated = from_api,
                Err(e) => {
                    warn!("SwbPriceFetcher: API fetch failed, falling back to Crossbar: {e}");
                }
            }
        }

        let missing: Vec<Pubkey> = tracked.difference(&updated).copied().collect();
        if missing.is_empty() {
            return Ok(());
        }

        debug!(
            "SwbPriceFetcher: querying Crossbar for {} oracle(s) the API didn't cover",
            missing.len()
        );
        let from_crossbar = self.fetch_from_crossbar()?;
        let still_missing: Vec<&Pubkey> = missing
            .iter()
            .filter(|oracle| !from_crossbar.contains(oracle))
            .collect();
        if !still_missing.is_empty() {
            warn!(
                "SwbPriceFetcher: no price for {:?}; accounts using them cannot be evaluated",
                still_missing
            );
        }
        Ok(())
    }

    fn fetch_from_api(&self, api_url: &str) -> Result<HashSet<Pubkey>> {
        let url = format!("{}/v0/realprice", api_url.trim_end_matches('/'));
        let resp: ViewPriceResponse = self.http_client.get(&url).send()?.json()?;

        let mut updated: HashSet<Pubkey> = HashSet::new();
        for (bank_addr_str, entry) in resp.prices {
            let bank_address = Pubkey::from_str(&bank_addr_str).map_err(|e| {
                anyhow::anyhow!("Invalid bank pubkey '{bank_addr_str}' in /v0/realprice: {e}")
            })?;

            // Write the RAW underlying feed price to the (potentially shared) Switchboard oracle
            // account. For integration banks (KaminoSwitchboardPull / JuplendSwitchboardPull) the
            // program re-applies the per-bank exchange rate on read, so writing the integration-
            // adjusted `price_realtime` here would both double-count and clobber every other bank
            // that shares the same feed. `source_price` is identical across all banks on a feed.
            let raw = entry
                .oracle_price
                .source_price
                .as_ref()
                .map(|sp| &sp.price_realtime)
                .unwrap_or(&entry.oracle_price.price_realtime);
            let price_rt = I80F48::from_num(raw.price);
            let conf_rt = I80F48::from_num(raw.confidence);
            if let Ok(bank) = self.cache.banks.try_get_bank(&bank_address) {
                if is_switchboard_pull_setup(bank.bank.config.oracle_setup) {
                    if raw.price <= 0.0 {
                        warn!("SwbPriceFetcher: the API reported a non-positive price for bank {bank_address}, skipping");
                        continue;
                    }
                    if let Some(&oracle_key) = bank.bank.config.oracle_keys.first() {
                        let synthetic = build_synthetic_swb_account(price_rt, conf_rt);
                        if let Err(e) = self
                            .cache
                            .oracles
                            .try_update(&oracle_key, synthetic.clone())
                        {
                            warn!("SwbPriceFetcher: failed to write synthetic oracle for {oracle_key}: {e}");
                        } else {
                            self.publish(oracle_key, synthetic);
                            updated.insert(oracle_key);
                        }
                    }
                }
            }
        }

        Ok(updated)
    }

    fn fetch_from_crossbar(&self) -> Result<HashSet<Pubkey>> {
        let oracle_to_bank = self.cache.banks.get_swb_oracle_to_bank_map();
        if oracle_to_bank.is_empty() {
            return Ok(HashSet::new());
        }

        let oracle_addresses: Vec<Pubkey> = oracle_to_bank.keys().cloned().collect();
        let responses = self.tokio_rt.block_on(
            self.crossbar
                .simulate_solana_feeds(ClusterType::MainnetBeta, &oracle_addresses),
        )?;

        let mut updated: HashSet<Pubkey> = HashSet::new();
        for response in responses {
            let oracle_pk: Pubkey = response.feed.parse().map_err(|e| {
                anyhow::anyhow!(
                    "Invalid oracle pubkey '{}' in Crossbar response: {e}",
                    response.feed
                )
            })?;

            let Some(bank_addresses) = oracle_to_bank.get(&oracle_pk) else {
                continue;
            };

            let Some(result) = response.result else {
                warn!(
                    "SwbPriceFetcher: no Crossbar result for oracle {}",
                    oracle_pk
                );
                continue;
            };

            let price: f64 = result
                .to_f64()
                .ok_or_else(|| anyhow::anyhow!("Decimal overflow converting price to f64"))?;

            // A feed Crossbar can't resolve reports 0, which would value the asset at nothing.
            if price <= 0.0 {
                warn!("SwbPriceFetcher: Crossbar reported a non-positive price for the oracle {oracle_pk}, skipping");
                continue;
            }

            // Half-range across oracle submissions as confidence interval
            let valid: Vec<f64> = response
                .results
                .iter()
                .filter_map(|opt| *opt)
                .filter_map(|d| d.to_f64())
                .collect();
            let conf = if valid.len() > 1 {
                let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
                (max - min) / 2.0
            } else {
                0.0
            };

            let price_rt = I80F48::from_num(price);
            let conf_rt = I80F48::from_num(conf);
            let synthetic = build_synthetic_swb_account(price_rt, conf_rt);
            for bank_address in bank_addresses {
                if let Ok(bank) = self.cache.banks.try_get_bank(bank_address) {
                    if let Some(&oracle_key) = bank.bank.config.oracle_keys.first() {
                        if let Err(e) = self
                            .cache
                            .oracles
                            .try_update(&oracle_key, synthetic.clone())
                        {
                            warn!("SwbPriceFetcher: failed to write synthetic oracle for {oracle_key}: {e}");
                        } else {
                            self.publish(oracle_key, synthetic.clone());
                            updated.insert(oracle_key);
                        }
                    }
                }
            }
        }
        Ok(updated)
    }
}
