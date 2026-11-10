//! Streams Pyth prices from Hermes, so eva evaluates accounts on prices the sponsored feeds have
//! not been pushed yet.
//!
//! Hermes serves Pythnet prices as they are published; the sponsored push feeds marginfi reads
//! receive them only when somebody pushes, about once a second for SOL and less often for thinner
//! feeds. Each streamed price newer than the one cached is written into the oracle cache as the
//! feed's `PriceUpdateV2`, through the same channel as Geyser updates, so the accounts priced by
//! it are re-evaluated. A liquidation that relies on such a price pushes it on chain first (see
//! `PythPusher`). The oracle cache keeps whichever of the chain's and Hermes's prices is newer, so
//! a lagging stream never rolls a price back.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anchor_lang::AccountSerialize;
use anyhow::{anyhow, Result};
use crossbeam::channel::Sender;
use log::{info, warn};
use reqwest::blocking::Client;
use serde::Deserialize;
use solana_pubkey::Pubkey;

use crate::{
    cache::{pyth_prices::price_update, Cache},
    config::Eva01Config,
    geyser::{AccountType, GeyserUpdate},
    utils::{
        pyth_cranker::{hex_encode, price_feed_address},
        INITIAL_RECONNECT_BACKOFF, MAX_RECONNECT_BACKOFF,
    },
};

/// A connection that streamed at least this long counts as healthy, resetting the backoff.
const HEALTHY_CONNECTION: Duration = Duration::from_secs(60);

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(default)]
    parsed: Vec<ParsedUpdate>,
}

#[derive(Deserialize)]
struct ParsedUpdate {
    id: String,
    price: HermesPrice,
    ema_price: HermesPrice,
    #[serde(default)]
    metadata: Metadata,
}

#[derive(Deserialize)]
struct HermesPrice {
    price: String,
    conf: String,
    expo: i32,
    publish_time: i64,
}

#[derive(Deserialize, Default)]
struct Metadata {
    prev_publish_time: Option<i64>,
}

pub struct HermesStream {
    hermes_url: String,
    api_key: Option<String>,
    http_client: Client,
    cache: Arc<Cache>,
    updates: Sender<GeyserUpdate>,
    stop: Arc<AtomicBool>,
    /// The sponsored push feed each streamed feed id is written to.
    feeds: HashMap<[u8; 32], Pubkey>,
}

impl HermesStream {
    pub fn new(
        config: &Eva01Config,
        cache: Arc<Cache>,
        updates: Sender<GeyserUpdate>,
        stop: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut feeds = HashMap::new();
        for (oracle, _) in cache.banks.get_pyth_push_oracles() {
            let Some(update) = cache
                .oracles
                .try_get_account(&oracle)
                .ok()
                .as_ref()
                .and_then(price_update)
            else {
                continue;
            };
            let feed_id = update.price_message.feed_id;
            // Only the sponsored feed is one anybody can push to.
            if price_feed_address(&feed_id) == oracle {
                feeds.insert(feed_id, oracle);
            }
        }
        Ok(Self {
            hermes_url: config.pyth_hermes_url.clone(),
            api_key: config.pyth_api_key.clone(),
            // No overall timeout on a stream that lives for hours; keepalive notices a dead
            // connection that would otherwise block the read forever.
            http_client: Client::builder()
                .timeout(None)
                .connect_timeout(Duration::from_secs(10))
                .tcp_keepalive(Duration::from_secs(15))
                .build()?,
            cache,
            updates,
            stop,
            feeds,
        })
    }

    pub fn start(&self) {
        if self.feeds.is_empty() {
            warn!("HermesStream: no sponsored Pyth feed to stream");
            return;
        }
        info!("HermesStream: streaming {} Pyth feeds", self.feeds.len());
        let mut backoff = INITIAL_RECONNECT_BACKOFF;
        while !self.stop.load(Ordering::Relaxed) {
            let connected = Instant::now();
            if let Err(e) = self.stream() {
                warn!("HermesStream: the stream ended ({e}); reconnecting");
            }
            if connected.elapsed() >= HEALTHY_CONNECTION {
                backoff = INITIAL_RECONNECT_BACKOFF;
            }
            thread::sleep(backoff);
            backoff = (backoff * 2).min(MAX_RECONNECT_BACKOFF);
        }
    }

    fn stream(&self) -> Result<()> {
        let ids: String = self
            .feeds
            .keys()
            .map(|id| format!("ids[]=0x{}&", hex_encode(id)))
            .collect();
        let url = format!(
            "{}/v2/updates/price/stream?{}parsed=true&encoding=base64",
            self.hermes_url.trim_end_matches('/'),
            ids
        );
        let mut request = self.http_client.get(&url);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.send()?.error_for_status()?;
        for line in BufReader::new(response).lines() {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let line = line?;
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let event: StreamEvent = match serde_json::from_str(data.trim()) {
                Ok(event) => event,
                Err(e) => {
                    warn!("HermesStream: unreadable event ({e})");
                    continue;
                }
            };
            for update in event.parsed {
                if let Err(e) = self.apply(update) {
                    warn!("HermesStream: {e}");
                }
            }
        }
        Err(anyhow!("Hermes closed the stream"))
    }

    /// Writes `update` over the feed's cached price, if it is newer.
    fn apply(&self, update: ParsedUpdate) -> Result<()> {
        let feed_id = parse_feed_id(&update.id)?;
        let Some(oracle) = self.feeds.get(&feed_id) else {
            return Ok(());
        };
        let mut account = self.cache.oracles.try_get_account(oracle)?;
        let mut price_update =
            price_update(&account).ok_or_else(|| anyhow!("{oracle} is not a Pyth price update"))?;
        let message = &mut price_update.price_message;
        if update.price.publish_time <= message.publish_time {
            return Ok(());
        }
        message.prev_publish_time = update
            .metadata
            .prev_publish_time
            .unwrap_or(message.publish_time);
        message.price = update.price.price.parse()?;
        message.conf = update.price.conf.parse()?;
        message.exponent = update.price.expo;
        message.publish_time = update.price.publish_time;
        message.ema_price = update.ema_price.price.parse()?;
        message.ema_conf = update.ema_price.conf.parse()?;

        let mut data = Vec::with_capacity(account.data.len());
        price_update.try_serialize(&mut data)?;
        data.resize(account.data.len(), 0);
        account.data = data;
        self.updates
            .send(GeyserUpdate {
                account_type: AccountType::HermesPrice,
                address: *oracle,
                account,
                slot: 0,
            })
            .map_err(|e| anyhow!("failed to publish the {oracle} price: {e}"))
    }
}

fn parse_feed_id(id: &str) -> Result<[u8; 32]> {
    let hex = id.trim_start_matches("0x");
    if hex.len() != 64 {
        return Err(anyhow!("unexpected Pyth feed id {id}"));
    }
    let mut feed_id = [0u8; 32];
    for (i, byte) in feed_id.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
    }
    Ok(feed_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::pyth_prices::test_price_update_account;
    use solana_clock::Clock;
    use std::{collections::HashSet, sync::Mutex};

    fn update(publish_time: i64, price: &str) -> ParsedUpdate {
        let quote = |value: &str| HermesPrice {
            price: value.to_string(),
            conf: "2".to_string(),
            expo: -8,
            publish_time,
        };
        ParsedUpdate {
            id: hex_encode(&[7u8; 32]),
            price: quote(price),
            ema_price: quote("1"),
            metadata: Metadata {
                prev_publish_time: Some(publish_time - 1),
            },
        }
    }

    #[test]
    fn only_a_newer_hermes_price_is_written_over_the_cached_one() {
        let feed = [7u8; 32];
        let oracle = price_feed_address(&feed);
        let mut cache = Cache::new(
            Pubkey::new_unique(),
            Pubkey::new_unique(),
            Arc::new(Mutex::new(Clock::default())),
            HashSet::new(),
        );
        cache
            .oracles
            .try_insert(oracle, test_price_update_account(feed, 100, 5))
            .unwrap();
        let (tx, rx) = crossbeam::channel::unbounded();
        let stream = HermesStream {
            hermes_url: String::new(),
            api_key: None,
            http_client: Client::new(),
            cache: Arc::new(cache),
            updates: tx,
            stop: Arc::new(AtomicBool::new(false)),
            feeds: HashMap::from([(feed, oracle)]),
        };

        stream.apply(update(100, "9")).unwrap();
        assert!(rx.try_recv().is_err(), "a price as old as the cached one");

        stream.apply(update(101, "9")).unwrap();
        let written = rx.try_recv().unwrap();
        assert!(matches!(written.account_type, AccountType::HermesPrice));
        assert_eq!(written.address, oracle);
        let price = price_update(&written.account).unwrap().price_message;
        assert_eq!((price.price, price.publish_time, price.conf), (9, 101, 2));
        assert_eq!(written.account.owner, pyth_solana_receiver_sdk::ID);
    }

    #[test]
    fn parses_feed_ids_with_or_without_the_prefix() {
        let hex = "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d";
        let id = parse_feed_id(hex).unwrap();
        assert_eq!(id[0], 0xef);
        assert_eq!(id[31], 0x6d);
        assert_eq!(parse_feed_id(&format!("0x{hex}")).unwrap(), id);
        assert!(parse_feed_id("0x12").is_err());
    }

    #[test]
    fn reads_a_stream_event() {
        let event: StreamEvent = serde_json::from_str(
            r#"{"binary":{"encoding":"base64","data":["AA=="]},"parsed":[{"id":"ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d","price":{"price":"8650000000","conf":"3467500","expo":-8,"publish_time":1794317000},"ema_price":{"price":"8640000000","conf":"3000000","expo":-8,"publish_time":1794317000},"metadata":{"slot":1,"proof_available_time":1794317001,"prev_publish_time":1794316999}}]}"#,
        )
        .unwrap();
        let update = &event.parsed[0];
        assert_eq!(update.price.price, "8650000000");
        assert_eq!(update.price.publish_time, 1794317000);
        assert_eq!(update.metadata.prev_publish_time, Some(1794316999));
    }
}
