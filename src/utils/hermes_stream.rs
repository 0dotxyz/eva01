//! Streams Pyth prices from Hermes, so eva evaluates accounts on prices the sponsored feeds have
//! not been pushed yet.
//!
//! Hermes serves Pythnet prices as they are published; the sponsored push feeds marginfi reads
//! receive them only when somebody pushes, about once a second for SOL and less often for thinner
//! feeds. Each streamed price newer than the one cached is written into the oracle cache as the
//! feed's `PriceUpdateV2`, through the same channel as Geyser updates, so the accounts priced by
//! it are re-evaluated. A liquidation that relies on such a price pushes it on chain first (see
//! `PythPusher`), from the signed update the same event carried, kept in `PythPrices`. The oracle
//! cache keeps whichever of the chain's and Hermes's prices is newer, so a lagging stream never
//! rolls a price back.
//!
//! Hermes cuts streams now and then, and every reconnect misses a second or more of prices, so
//! two connections stream the same feeds side by side. Each price is applied once, whichever
//! connection delivers it first.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use anchor_lang::AccountSerialize;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use crossbeam::channel::Sender;
use log::{debug, error, info, warn};
use pythnet_sdk::{
    messages::Message,
    wire::{
        from_slice,
        v1::{AccumulatorUpdateData, Proof},
    },
};
use reqwest::{blocking::Client, StatusCode};
use serde::Deserialize;
use solana_pubkey::Pubkey;

use crate::{
    cache::{
        pyth_prices::{price_update, SignedUpdate},
        Cache,
    },
    config::Eva01Config,
    geyser::{AccountType, GeyserUpdate},
    utils::{
        pyth_cranker::{hex_encode, price_feed_address},
        INITIAL_RECONNECT_BACKOFF, MAX_RECONNECT_BACKOFF,
    },
};

/// A connection that streamed at least this long counts as healthy, resetting the backoff.
const HEALTHY_CONNECTION: Duration = Duration::from_secs(60);

/// The connections streaming side by side, so one being cut leaves no gap.
const CONNECTIONS: usize = 2;

/// How long after the first connection the next one opens, so they don't run the same age.
const CONNECTION_STAGGER: Duration = Duration::from_secs(20);

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(default)]
    binary: Option<Binary>,
    #[serde(default)]
    parsed: Vec<ParsedUpdate>,
}

/// The event's prices as Hermes signed them: one accumulator update, base64.
#[derive(Deserialize)]
struct Binary {
    #[serde(default)]
    data: Vec<String>,
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

/// Why a stream ended.
enum StreamError {
    /// Hermes refused the API key.
    Rejected(StatusCode),
    Other(anyhow::Error),
}

impl From<anyhow::Error> for StreamError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
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
    /// The newest publish time applied per feed, so a price both connections deliver is applied
    /// (and its accounts re-evaluated) once.
    applied: Mutex<HashMap<[u8; 32], i64>>,
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
            applied: Mutex::new(HashMap::new()),
        })
    }

    pub fn start(&self) {
        if self.feeds.is_empty() {
            warn!("HermesStream: no sponsored Pyth feed to stream");
            return;
        }
        if self.api_key.is_none() {
            error!(
                "HermesStream: PYTH_API_KEY is not set and Hermes rejects requests without one; \
                 pricing on the chain's Pyth feeds only, and no price can be pushed"
            );
            return;
        }
        info!(
            "HermesStream: streaming {} Pyth feeds over {CONNECTIONS} connections",
            self.feeds.len()
        );
        thread::scope(|scope| {
            for connection in 1..=CONNECTIONS {
                scope.spawn(move || {
                    let stagger = CONNECTION_STAGGER * (connection as u32 - 1);
                    let opens_at = Instant::now() + stagger;
                    while Instant::now() < opens_at && !self.stop.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(500));
                    }
                    self.keep_streaming(connection);
                });
            }
        });
    }

    /// Streams on one connection, reconnecting whenever it ends, until `stop` is set.
    fn keep_streaming(&self, connection: usize) {
        let mut backoff = INITIAL_RECONNECT_BACKOFF;
        let mut reported_rejection = false;
        while !self.stop.load(Ordering::Relaxed) {
            let connected = Instant::now();
            let mut delivered = false;
            match self.stream(&mut delivered) {
                Err(StreamError::Rejected(status)) => {
                    // A key Hermes refuses won't start working on a retry; say so once and keep
                    // trying quietly, in case it is fixed without a restart.
                    if !reported_rejection && connection == 1 {
                        error!(
                            "HermesStream: Hermes rejected PYTH_API_KEY ({status}); pricing on the \
                             chain's Pyth feeds only until it is accepted"
                        );
                    }
                    reported_rejection = true;
                    backoff = MAX_RECONNECT_BACKOFF;
                }
                Err(StreamError::Other(e)) => {
                    warn!("HermesStream #{connection}: the stream ended ({e}); reconnecting")
                }
                Ok(()) => {}
            }
            // Hermes cuts streams every minute or so; a connection that delivered prices was
            // healthy, so the next one starts at once instead of after a growing wait.
            if delivered || connected.elapsed() >= HEALTHY_CONNECTION {
                backoff = INITIAL_RECONNECT_BACKOFF;
                reported_rejection = false;
            }
            thread::sleep(backoff);
            backoff = (backoff * 2).min(MAX_RECONNECT_BACKOFF);
        }
    }

    /// Streams until the connection ends; `delivered` is set once a price event was read.
    fn stream(&self, delivered: &mut bool) -> std::result::Result<(), StreamError> {
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
        // The error would otherwise carry the whole URL, one id per feed.
        let response = request.send().map_err(|e| anyhow!(e.without_url()))?;
        let status = response.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(StreamError::Rejected(status));
        }
        let response = response
            .error_for_status()
            .map_err(|e| anyhow!(e.without_url()))?;
        for line in BufReader::new(response).lines() {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let line = line.map_err(|e| anyhow!(e))?;
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
            *delivered = true;
            // Kept before the prices are applied, so a push for any price eva evaluates on has
            // its signed update at hand.
            if let Some(binary) = &event.binary {
                if let Err(e) = self.record_signed(binary) {
                    debug!("HermesStream: unreadable signed update ({e})");
                }
            }
            for update in event.parsed {
                if let Err(e) = self.apply(update) {
                    warn!("HermesStream: {e}");
                }
            }
        }
        Err(anyhow!("Hermes closed the stream").into())
    }

    /// Keeps the signed update of every streamed feed the event carries.
    fn record_signed(&self, binary: &Binary) -> Result<()> {
        for encoded in &binary.data {
            for (feed_id, signed) in signed_updates(&BASE64.decode(encoded)?)? {
                if self.feeds.contains_key(&feed_id) {
                    self.cache.pyth_prices.record_signed(feed_id, signed);
                }
            }
        }
        Ok(())
    }

    /// Whether `publish_time` is newer than any price applied for `feed_id` so far, which makes it
    /// this connection's to apply. The cache itself only moves once the update sent is drained,
    /// too late to stop the other connection sending the same price.
    fn first_to_apply(&self, feed_id: [u8; 32], publish_time: i64) -> bool {
        let mut applied = self.applied.lock().unwrap_or_else(|e| e.into_inner());
        let newest = applied.entry(feed_id).or_insert(i64::MIN);
        if publish_time <= *newest {
            return false;
        }
        *newest = publish_time;
        true
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
        if update.price.publish_time <= message.publish_time
            || !self.first_to_apply(feed_id, update.price.publish_time)
        {
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

/// Every feed's update in an accumulator payload, under the payload's single VAA.
fn signed_updates(payload: &[u8]) -> Result<Vec<([u8; 32], SignedUpdate)>> {
    let data = AccumulatorUpdateData::try_from_slice(payload)
        .map_err(|e| anyhow!("not an accumulator update: {e:?}"))?;
    let Proof::WormholeMerkle { vaa, updates } = data.proof;
    let vaa = Arc::new(Vec::<u8>::from(vaa));
    let mut signed = Vec::with_capacity(updates.len());
    for update in updates {
        let Message::PriceFeedMessage(message) =
            from_slice::<byteorder::BE, Message>(update.message.as_ref())
                .map_err(|e| anyhow!("unreadable price message: {e:?}"))?
        else {
            continue;
        };
        signed.push((
            message.feed_id,
            SignedUpdate {
                publish_time: message.publish_time,
                vaa: Arc::clone(&vaa),
                update,
            },
        ));
    }
    Ok(signed)
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
            applied: Mutex::new(HashMap::new()),
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

        // The other connection delivers the same price before the first one's is drained.
        stream.apply(update(101, "9")).unwrap();
        assert!(rx.try_recv().is_err(), "a price already applied");
    }

    #[test]
    fn a_streamed_payload_yields_each_feeds_update_under_its_vaa() {
        use pythnet_sdk::{
            accumulators::merkle::MerklePath,
            messages::PriceFeedMessage,
            wire::{to_vec, v1::MerklePriceUpdate},
        };
        let message = |feed_id: [u8; 32], publish_time: i64| MerklePriceUpdate {
            message: to_vec::<_, byteorder::BE>(&Message::PriceFeedMessage(PriceFeedMessage {
                feed_id,
                price: 1,
                conf: 1,
                exponent: -8,
                publish_time,
                prev_publish_time: publish_time - 1,
                ema_price: 1,
                ema_conf: 1,
            }))
            .unwrap()
            .into(),
            proof: MerklePath::new(vec![]),
        };
        let payload =
            to_vec::<_, byteorder::BE>(&AccumulatorUpdateData::new(Proof::WormholeMerkle {
                vaa: vec![9u8, 8, 7].into(),
                updates: vec![message([1u8; 32], 100), message([2u8; 32], 101)],
            }))
            .unwrap();

        let signed = signed_updates(&payload).unwrap();
        assert_eq!(signed.len(), 2);
        assert_eq!(signed[0].0, [1u8; 32]);
        assert_eq!(signed[1].0, [2u8; 32]);
        assert_eq!(signed[1].1.publish_time, 101);
        assert_eq!(signed[0].1.vaa.as_slice(), &[9, 8, 7]);
        assert!(Arc::ptr_eq(&signed[0].1.vaa, &signed[1].1.vaa));
        assert!(signed_updates(&[1, 2, 3]).is_err());
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
