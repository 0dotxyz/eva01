use std::{
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use bytemuck::Zeroable;
use fixed::types::I80F48;
use log::{debug, info, warn};
use reqwest::blocking::Client;
use serde::Deserialize;
use marginfi_type_crate::types::OracleSetup;
use solana_sdk::{account::Account, genesis_config::ClusterType, pubkey, pubkey::Pubkey};
use switchboard_on_demand_client::{CrossbarClient, PullFeedAccountData};
use tokio::runtime::{Builder, Runtime};

use crate::cache::Cache;

const SWITCHBOARD_PULL_PROGRAM_ID: Pubkey =
    pubkey!("SBondMDrcV3K4kxZR1HNVT7osZxAHVHgYXL5Ze1oMUv");
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
struct ViewPriceEntry {
    #[serde(rename = "oraclePrice")]
    oracle_price: OraclePriceDto,
}

#[derive(Deserialize)]
struct OraclePriceDto {
    #[serde(rename = "priceRealtime")]
    price_realtime: PriceWithConfidenceDto,
}

#[derive(Deserialize)]
struct PriceWithConfidenceDto {
    price: String,
    confidence: String,
}

pub struct SwbPriceFetcher {
    api_url: Option<String>,
    http_client: Client,
    crossbar: CrossbarClient,
    tokio_rt: Runtime,
    cache: Arc<Cache>,
    stop: Arc<AtomicBool>,
}

impl SwbPriceFetcher {
    pub fn new(
        api_url: Option<String>,
        crossbar_api_url: Option<String>,
        cache: Arc<Cache>,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let crossbar_url = crossbar_api_url
            .as_deref()
            .unwrap_or(FALLBACK_CROSSBAR_URL);
        let tokio_rt = Builder::new_multi_thread()
            .thread_name("SwbPriceFetcher")
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to build SwbPriceFetcher tokio runtime");

        Self {
            api_url,
            http_client: Client::new(),
            crossbar: CrossbarClient::new(crossbar_url, false),
            tokio_rt,
            cache,
            stop,
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

    fn fetch_and_update(&self) -> Result<()> {
        if let Some(api_url) = &self.api_url {
            match self.fetch_from_api(api_url) {
                Ok(count) => {
                    debug!("SwbPriceFetcher: updated {} bank prices from API", count);
                    return Ok(());
                }
                Err(e) => {
                    warn!("SwbPriceFetcher: API fetch failed, falling back to Crossbar: {e}");
                }
            }
        }
        match self.fetch_from_crossbar() {
            Ok(count) => {
                debug!(
                    "SwbPriceFetcher: updated {} bank prices from Crossbar",
                    count
                );
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn fetch_from_api(&self, api_url: &str) -> Result<usize> {
        let url = format!("{}/v0/realprice", api_url.trim_end_matches('/'));
        let resp: ViewPriceResponse = self.http_client.get(&url).send()?.json()?;

        let mut count = 0usize;
        for (bank_addr_str, entry) in resp.prices {
            let bank_address = Pubkey::from_str(&bank_addr_str).map_err(|e| {
                anyhow::anyhow!("Invalid bank pubkey '{bank_addr_str}' in /v0/realprice: {e}")
            })?;

            let price_realtime = entry.oracle_price.price_realtime.price.parse::<f64>()?;
            let conf_realtime = entry
                .oracle_price
                .price_realtime
                .confidence
                .parse::<f64>()?;

            let price_rt = I80F48::from_num(price_realtime);
            let conf_rt = I80F48::from_num(conf_realtime);
            if let Ok(bank) = self.cache.banks.try_get_bank(&bank_address) {
                if matches!(
                    bank.bank.config.oracle_setup,
                    OracleSetup::SwitchboardPull
                        | OracleSetup::KaminoSwitchboardPull
                        | OracleSetup::DriftSwitchboardPull
                        | OracleSetup::JuplendSwitchboardPull
                ) {
                    if let Some(&oracle_key) = bank.bank.config.oracle_keys.first() {
                        let synthetic = build_synthetic_swb_account(price_rt, conf_rt);
                        if let Err(e) = self.cache.oracles.try_update(&oracle_key, synthetic) {
                            warn!("SwbPriceFetcher: failed to write synthetic oracle for {oracle_key}: {e}");
                        }
                    }
                }
            }
            count += 1;
        }
        Ok(count)
    }

    fn fetch_from_crossbar(&self) -> Result<usize> {
        let oracle_to_bank = self.cache.banks.get_swb_oracle_to_bank_map();
        if oracle_to_bank.is_empty() {
            return Ok(0);
        }

        let oracle_addresses: Vec<Pubkey> = oracle_to_bank.keys().cloned().collect();
        let responses = self.tokio_rt.block_on(
            self.crossbar
                .simulate_solana_feeds(ClusterType::MainnetBeta, &oracle_addresses),
        )?;

        let mut count = 0usize;
        for response in responses {
            let oracle_pk: Pubkey = response.feed.parse().map_err(|e| {
                anyhow::anyhow!(
                    "Invalid oracle pubkey '{}' in Crossbar response: {e}",
                    response.feed
                )
            })?;

            let Some(&bank_address) = oracle_to_bank.get(&oracle_pk) else {
                continue;
            };

            let Some(result) = response.result else {
                warn!(
                    "SwbPriceFetcher: no Crossbar result for oracle {}",
                    oracle_pk
                );
                continue;
            };

            let price: f64 = result.to_string().parse()?;

            // Half-range across oracle submissions as confidence interval
            let valid: Vec<f64> = response
                .results
                .iter()
                .filter_map(|opt| opt.as_ref())
                .filter_map(|d| d.to_string().parse::<f64>().ok())
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
            if let Ok(bank) = self.cache.banks.try_get_bank(&bank_address) {
                if let Some(&oracle_key) = bank.bank.config.oracle_keys.first() {
                    let synthetic = build_synthetic_swb_account(price_rt, conf_rt);
                    if let Err(e) = self.cache.oracles.try_update(&oracle_key, synthetic) {
                        warn!("SwbPriceFetcher: failed to write synthetic oracle for {oracle_key}: {e}");
                    }
                }
            }
            count += 1;
        }
        Ok(count)
    }
}
