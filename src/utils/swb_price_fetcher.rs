use std::{
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use anyhow::Result;
use fixed::types::I80F48;
use log::{debug, info, warn};
use reqwest::blocking::Client;
use serde::Deserialize;
use solana_sdk::{genesis_config::ClusterType, pubkey::Pubkey};
use switchboard_on_demand_client::CrossbarClient;
use tokio::runtime::{Builder, Runtime};

use crate::cache::{Cache, SwbPrice};

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
    #[serde(rename = "priceWeighted")]
    price_weighted: PriceWithConfidenceDto,
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
            let price_weighted = entry.oracle_price.price_weighted.price.parse::<f64>()?;
            let conf_weighted = entry
                .oracle_price
                .price_weighted
                .confidence
                .parse::<f64>()?;

            self.cache.swb_prices.upsert(
                bank_address,
                SwbPrice {
                    price_realtime: I80F48::from_num(price_realtime),
                    conf_realtime: I80F48::from_num(conf_realtime),
                    price_weighted: I80F48::from_num(price_weighted),
                    conf_weighted: I80F48::from_num(conf_weighted),
                },
            )?;
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

            self.cache.swb_prices.upsert(
                bank_address,
                SwbPrice {
                    price_realtime: I80F48::from_num(price),
                    conf_realtime: I80F48::from_num(conf),
                    // Crossbar doesn't distinguish realtime vs weighted; use same value
                    price_weighted: I80F48::from_num(price),
                    conf_weighted: I80F48::from_num(conf),
                },
            )?;
            count += 1;
        }
        Ok(count)
    }
}
