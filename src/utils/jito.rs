//! Minimal Jito block-engine bundle client.
//!
//! Submits an ordered set of transactions as an atomic bundle via the public block-engine
//! JSON-RPC (`sendBundle`). Whether the bundle landed is read from the chain, by its last
//! transaction's signature, so landing it costs one block-engine request; the block engine is
//! asked about it again only to learn early that it won't land. No extra crates: plain `reqwest`
//! + `serde_json`.

use std::{
    str::FromStr,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use log::{info, warn};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::{v0, Hash, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_system_interface::instruction::transfer;
use solana_transaction::versioned::VersionedTransaction;

const DEFAULT_BUNDLE_ENDPOINT: &str = "https://mainnet.block-engine.jito.wtf/api/v1/bundles";
/// The well-known static Jito tip accounts. These rarely change; a bundle must tip one of them.
const JITO_TIP_ACCOUNTS: [&str; 8] = [
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
];
const TIP_FLOOR_URL: &str = "https://bundles.jito.wtf/api/v1/bundles/tip_floor";

/// The window the block engine's rate limit counts requests over (`sendBundle` and the status
/// methods alike); a request over the limit is rejected with HTTP 429.
const RATE_WINDOW: Duration = Duration::from_secs(1);

/// Requests per window without an auth uuid.
const UNAUTHENTICATED_REQUESTS: usize = 1;

/// Requests per window with an auth uuid. Ours admits about 37 a second; this leaves room for
/// bursts the window doesn't count evenly.
const AUTHENTICATED_REQUESTS: usize = 20;

pub struct JitoClient {
    http: Client,
    endpoint: String,
    api_key: Option<String>,
    slot: Mutex<SlotState>,
    outage: Mutex<Outage>,
}

/// How long a request to the block engine may take before the bundle is taken as not sent.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// The block engine is taken as down after this many failed requests within
/// [`OUTAGE_WINDOW`], for [`OUTAGE_PAUSE`].
const OUTAGE_FAILURES: usize = 3;
const OUTAGE_WINDOW: Duration = Duration::from_secs(30);
const OUTAGE_PAUSE: Duration = Duration::from_secs(30);

/// Recent requests the block engine failed, and until when it is taken as down.
#[derive(Default)]
struct Outage {
    failures: std::collections::VecDeque<Instant>,
    down_until: Option<Instant>,
}

/// The requests the block engine admits per [`RATE_WINDOW`]: a slot counts from its reservation,
/// and once its request goes out, until the window has passed.
struct SlotState {
    /// Requests admitted per window.
    capacity: usize,
    /// When the requests of the last window went out, oldest first.
    sent: std::collections::VecDeque<Instant>,
    /// Slots reserved and not yet used or given back.
    held: usize,
}

impl SlotState {
    fn prune(&mut self, now: Instant) {
        while self
            .sent
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) >= RATE_WINDOW)
        {
            self.sent.pop_front();
        }
    }

    fn is_full(&self) -> bool {
        self.held + self.sent.len() >= self.capacity
    }

    /// Whether more than half the window's requests are still free; with one request a window,
    /// whether it is.
    fn has_spare(&self) -> bool {
        let used = self.held + self.sent.len();
        self.capacity.saturating_sub(used) > self.capacity / 2
    }
}

/// How often a caller waiting for a slot looks again while the others are all held.
const HELD_SLOT_RECHECK: Duration = Duration::from_millis(20);

/// A request slot, reserved until its request goes out or it is dropped. A held slot counts
/// against the limit, so a caller that quotes a swap after reserving can't be overtaken into a
/// request over it; dropping it unused gives it back.
pub struct JitoSlot<'a> {
    jito: &'a JitoClient,
}

impl Drop for JitoSlot<'_> {
    fn drop(&mut self) {
        let mut state = self.jito.slot.lock().unwrap_or_else(|e| e.into_inner());
        state.held = state.held.saturating_sub(1);
    }
}

impl JitoSlot<'_> {
    /// Counts the slot as used from now.
    fn mark_sent(&self) {
        self.jito
            .slot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sent
            .push_back(Instant::now());
    }

    /// Submits a bundle in this slot. An `Err` is a bundle that couldn't be encoded, never sent. A
    /// request that failed or timed out is `Unavailable`: taken as not sent, though a timed-out one
    /// may still have arrived and land; the bundle is atomic, so at worst a later plan counting the
    /// same wallet tokens fails.
    pub fn send_bundle(self, txs: &[VersionedTransaction]) -> Result<BundleSend> {
        if txs.is_empty() {
            return Err(anyhow!("Cannot send an empty bundle"));
        }
        let encoded = encode_transactions(txs)?;
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [encoded, { "encoding": "base64" }],
        });
        self.mark_sent();
        let resp = match self.jito.http.post(self.jito.url()).json(&body).send() {
            Ok(resp) => resp,
            // Unreachable, no answer in time, or the connection dropped: taken as not sent.
            Err(e) => {
                self.jito.note_unavailable();
                return Ok(BundleSend::Unavailable(e.to_string()));
            }
        };
        // Our own rate limit: refused before the auction, so nothing waits for it.
        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Ok(BundleSend::Rejected("rate limited (HTTP 429)".to_string()));
        }
        // The block engine failing, as opposed to refusing this bundle: taken as not sent.
        if status.is_server_error()
            || status == reqwest::StatusCode::UNAUTHORIZED
            || status == reqwest::StatusCode::FORBIDDEN
            || status == reqwest::StatusCode::NOT_FOUND
        {
            self.jito.note_unavailable();
            return Ok(BundleSend::Unavailable(format!("HTTP {status}")));
        }
        // Anything else carries the block engine's JSON-RPC answer, an error included (HTTP 400):
        // an invalid bundle, or one already processed, which means it landed.
        // Not the block engine's JSON-RPC (a proxy's page, a wrong endpoint): as good as down.
        let resp: Value = match resp.json() {
            Ok(resp) => resp,
            Err(e) => {
                self.jito.note_unavailable();
                return Ok(BundleSend::Unavailable(format!("unreadable answer: {e}")));
            }
        };
        self.jito.note_reachable();
        Ok(match parse_send_bundle_result(&resp) {
            Ok(id) => BundleSend::Accepted(id),
            Err(e) => BundleSend::Rejected(e.to_string()),
        })
    }

    /// The block engine's view of a bundle sent in the last few minutes, asked in this slot.
    /// `None` when the response carries no status.
    pub fn inflight_bundle_status(self, bundle_id: &str) -> Result<Option<InflightStatus>> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getInflightBundleStatuses",
            "params": [[bundle_id]],
        });
        self.mark_sent();
        let resp: Value = self
            .jito
            .http
            .post(self.jito.url())
            .json(&body)
            .send()?
            .json()?;
        parse_inflight_status(&resp)
    }
}

impl JitoClient {
    pub fn new(endpoint: Option<String>, api_key: Option<String>) -> Self {
        let capacity = if api_key.is_some() {
            AUTHENTICATED_REQUESTS
        } else {
            UNAUTHENTICATED_REQUESTS
        };
        info!(
            "Jito: {capacity} request(s) per second ({})",
            if api_key.is_some() {
                "authenticated"
            } else {
                "no auth uuid"
            }
        );
        Self {
            http: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .unwrap_or_else(|_| Client::new()),
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_BUNDLE_ENDPOINT.to_string()),
            api_key,
            outage: Mutex::new(Outage::default()),
            slot: Mutex::new(SlotState {
                capacity,
                sent: std::collections::VecDeque::with_capacity(capacity),
                held: 0,
            }),
        }
    }

    /// Whether the block engine is taken as down: several requests failed lately.
    pub fn is_down(&self) -> bool {
        let outage = self.outage.lock().unwrap_or_else(|e| e.into_inner());
        outage
            .down_until
            .is_some_and(|until| Instant::now() < until)
    }

    fn note_unavailable(&self) {
        let now = Instant::now();
        let mut outage = self.outage.lock().unwrap_or_else(|e| e.into_inner());
        outage.failures.push_back(now);
        while outage
            .failures
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) > OUTAGE_WINDOW)
        {
            outage.failures.pop_front();
        }
        if outage.failures.len() >= OUTAGE_FAILURES {
            outage.failures.clear();
            outage.down_until = Some(now + OUTAGE_PAUSE);
            warn!(
                "Jito: the block engine failed {OUTAGE_FAILURES} requests in {}s; taken as down for {}s, liquidations that need no push go over RPC",
                OUTAGE_WINDOW.as_secs(),
                OUTAGE_PAUSE.as_secs()
            );
        }
    }

    fn note_reachable(&self) {
        let mut outage = self.outage.lock().unwrap_or_else(|e| e.into_inner());
        outage.failures.clear();
        outage.down_until = None;
    }

    /// A request slot, if one is free now, without waiting: a liquidation that finds none goes
    /// out another way instead of queueing behind the other workers. The lock is held only to
    /// read and update the state, never across a wait.
    pub fn try_reserve(&self) -> Option<JitoSlot<'_>> {
        let mut state = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        state.prune(Instant::now());
        if state.is_full() {
            return None;
        }
        state.held += 1;
        Some(JitoSlot { jito: self })
    }

    /// Requests admitted per window.
    pub fn capacity(&self) -> usize {
        self.slot.lock().unwrap_or_else(|e| e.into_inner()).capacity
    }

    /// A request slot only while more than half the window's are free, for a request that can
    /// wait (a status check), so sends always keep the other half.
    pub fn try_reserve_spare(&self) -> Option<JitoSlot<'_>> {
        let mut state = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        state.prune(Instant::now());
        if !state.has_spare() {
            return None;
        }
        state.held += 1;
        Some(JitoSlot { jito: self })
    }

    /// Waits for a request slot and takes it, for a bundle that has no other way out. Sleeps
    /// without the lock, so others' `try_reserve` never waits behind it.
    pub fn reserve(&self) -> JitoSlot<'_> {
        loop {
            let wait = {
                let mut state = self.slot.lock().unwrap_or_else(|e| e.into_inner());
                let now = Instant::now();
                state.prune(now);
                if !state.is_full() {
                    Duration::ZERO
                } else {
                    // Until the oldest request leaves the window, or sooner, in case a held slot
                    // is given back.
                    state.sent.front().map_or(HELD_SLOT_RECHECK, |at| {
                        (*at + RATE_WINDOW)
                            .saturating_duration_since(now)
                            .min(HELD_SLOT_RECHECK)
                    })
                }
            };
            if wait.is_zero() {
                if let Some(slot) = self.try_reserve() {
                    return slot;
                }
                continue;
            }
            thread::sleep(wait);
        }
    }

    /// The block-engine URL with the optional auth uuid appended (matches p0-app).
    fn url(&self) -> String {
        match &self.api_key {
            Some(key) => format!("{}?uuid={}", self.endpoint, key),
            None => self.endpoint.clone(),
        }
    }
}

/// The block engine's answer to `sendBundle`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleSend {
    /// Accepted for the auction, with its id ("0x0" when already processed).
    Accepted(String),
    /// Refused, such as over the rate limit: it will never land.
    Rejected(String),
    /// The block engine couldn't be reached or answered with an error: the bundle is taken as not
    /// sent, and repeated, as the block engine being down.
    Unavailable(String),
}

/// Serialize each transaction with bincode and base64-encode it for the bundle payload.
fn encode_transactions(txs: &[VersionedTransaction]) -> Result<Vec<String>> {
    txs.iter()
        .map(|tx| {
            let bytes =
                bincode::serialize(tx).map_err(|e| anyhow!("Failed to serialize tx: {e}"))?;
            Ok(BASE64.encode(bytes))
        })
        .collect()
}

/// Extract the bundle id from a `sendBundle` response, treating "already processed" as success.
fn parse_send_bundle_result(resp: &Value) -> Result<String> {
    if let Some(err) = resp.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or_default();
        if msg.contains("already processed") {
            return Ok("0x0".to_string());
        }
        return Err(anyhow!("Jito sendBundle error: {}", err));
    }
    resp.get("result")
        .and_then(|r| r.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("Jito sendBundle: missing result in response: {}", resp))
}

/// Where a recently sent bundle stands, per `getInflightBundleStatuses`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InflightStatus {
    /// Still being auctioned; it may land.
    Pending,
    Landed,
    /// Every region gave up on it: it won't land.
    Failed,
    /// Unknown to the block engine, which forgets bundles after a few minutes: it won't land.
    Invalid,
}

impl InflightStatus {
    /// Whether the bundle can no longer land.
    pub fn is_dead(self) -> bool {
        matches!(self, Self::Failed | Self::Invalid)
    }
}

fn parse_inflight_status(resp: &Value) -> Result<Option<InflightStatus>> {
    if let Some(err) = resp.get("error") {
        return Err(anyhow!("Jito getInflightBundleStatuses error: {}", err));
    }
    Ok(
        match resp
            .pointer("/result/value/0/status")
            .and_then(|s| s.as_str())
        {
            Some("Pending") => Some(InflightStatus::Pending),
            Some("Landed") => Some(InflightStatus::Landed),
            Some("Failed") => Some(InflightStatus::Failed),
            Some("Invalid") => Some(InflightStatus::Invalid),
            _ => None,
        },
    )
}

/// How often the tip floor is re-read, in the background.
const TIP_REFRESH_INTERVAL: Duration = Duration::from_secs(60 * 60);
/// The floor until the first read succeeds.
const TIP_FALLBACK_LAMPORTS: u64 = 10_000;

/// Jito tips: a share of the bundle's expected profit, between a floor and a cap.
///
/// The floor is the EMA of the 50th-percentile landed tip from Jito's `tip_floor` endpoint,
/// refreshed hourly on a background thread, so sizing a tip never waits on HTTP.
pub struct TipEstimator {
    floor: Arc<Mutex<u64>>,
    max_lamports: u64,
    profit_share: f64,
}

impl TipEstimator {
    pub fn new(max_lamports: u64, profit_share: f64) -> Self {
        let floor = Arc::new(Mutex::new(TIP_FALLBACK_LAMPORTS));
        let refreshed = Arc::clone(&floor);
        thread::spawn(move || {
            let http = Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_else(|_| Client::new());
            loop {
                match fetch_tip_floor(&http) {
                    Ok(lamports) => {
                        *refreshed.lock().unwrap_or_else(|e| e.into_inner()) = lamports;
                    }
                    Err(e) => warn!("Tip floor fetch failed ({e}); keeping the last value"),
                }
                thread::sleep(TIP_REFRESH_INTERVAL);
            }
        });
        Self {
            floor,
            max_lamports,
            profit_share,
        }
    }

    /// The tip for a bundle expected to earn `profit_lamports`, moved by `jitter` lamports (up, or
    /// down when that would pass the cap) so two bundles never send the same tip transaction.
    pub fn tip_for(&self, profit_lamports: u64, jitter: u64) -> u64 {
        let floor = *self.floor.lock().unwrap_or_else(|e| e.into_inner());
        let tip = tip(profit_lamports, self.profit_share, floor, self.max_lamports);
        if tip.saturating_add(jitter) <= self.max_lamports {
            tip + jitter
        } else {
            tip.saturating_sub(jitter)
        }
    }

    /// The tip transfer, the bundle's last transaction. It can't go inside the liquidation:
    /// marginfi accepts only swap programs between a receivership's start and end, and nothing
    /// after the end.
    pub fn build_tip_transaction(
        &self,
        signer: &Keypair,
        blockhash: Hash,
        lamports: u64,
    ) -> Result<VersionedTransaction> {
        let ix = tip_instruction(&signer.pubkey(), lamports)?;
        let msg = v0::Message::try_compile(&signer.pubkey(), &[ix], &[], blockhash)?;
        VersionedTransaction::try_new(VersionedMessage::V0(msg), &[signer]).map_err(Into::into)
    }
}

/// A transfer of `lamports` from `payer` to one of Jito's tip accounts.
fn tip_instruction(payer: &Pubkey, lamports: u64) -> Result<Instruction> {
    Ok(transfer(payer, &select_tip_account()?, lamports))
}

/// `share` of `profit_lamports`, at least `floor` and at most `cap` (the cap wins).
fn tip(profit_lamports: u64, share: f64, floor: u64, cap: u64) -> u64 {
    let wanted = (profit_lamports as f64 * share.max(0.0)) as u64;
    wanted.max(floor).min(cap)
}

fn fetch_tip_floor(http: &Client) -> Result<u64> {
    let resp: Value = http.get(TIP_FLOOR_URL).send()?.json()?;
    parse_tip_floor_lamports(&resp)
}

fn select_tip_account() -> Result<Pubkey> {
    let idx = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as usize)
        .unwrap_or(0)
        % JITO_TIP_ACCOUNTS.len();
    Pubkey::from_str(JITO_TIP_ACCOUNTS[idx])
        .map_err(|e| anyhow!("hardcoded Jito tip account is invalid: {e}"))
}

#[derive(Deserialize)]
struct TipFloorEntry {
    ema_landed_tips_50th_percentile: f64,
}

/// Parse `ema_landed_tips_50th_percentile` (SOL) from the tip_floor response into lamports.
fn parse_tip_floor_lamports(resp: &Value) -> Result<u64> {
    let entries: Vec<TipFloorEntry> = serde_json::from_value(resp.clone())
        .map_err(|e| anyhow!("invalid tip_floor response ({e}): {resp}"))?;
    let sol = entries
        .first()
        .ok_or_else(|| anyhow!("tip_floor response is empty: {resp}"))?
        .ema_landed_tips_50th_percentile;
    if sol < 0.0 {
        return Err(anyhow!("negative tip floor: {sol}"));
    }
    Ok((sol * solana_native_token::LAMPORTS_PER_SOL as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_inflight_status_tells_whether_the_bundle_can_still_land() {
        let status = |value: Value| {
            parse_inflight_status(
                &json!({ "result": { "context": { "slot": 1 }, "value": [value] } }),
            )
            .unwrap()
        };
        assert_eq!(
            status(json!({ "bundle_id": "x", "status": "Pending", "landed_slot": null })),
            Some(InflightStatus::Pending)
        );
        assert!(status(json!({ "status": "Failed" })).unwrap().is_dead());
        assert!(status(json!({ "status": "Invalid" })).unwrap().is_dead());
        assert!(!status(json!({ "status": "Landed", "landed_slot": 5 }))
            .unwrap()
            .is_dead());
        assert_eq!(status(json!({})), None);
        assert!(parse_inflight_status(&json!({ "error": { "code": -32602 } })).is_err());
    }

    #[test]
    fn a_request_slot_is_held_until_used_and_then_spaced_out() {
        let jito = JitoClient::new(None, None);
        let slot = jito.try_reserve().expect("free at start");
        assert!(jito.try_reserve().is_none(), "held by the first caller");
        drop(slot);
        let slot = jito.try_reserve().expect("given back unused");
        slot.mark_sent();
        drop(slot);
        assert!(
            jito.try_reserve().is_none(),
            "used: the next one is a second away"
        );
    }

    #[test]
    fn status_checks_leave_sends_half_the_window() {
        let jito = JitoClient::new(None, Some("uuid".to_string()));
        let half = AUTHENTICATED_REQUESTS / 2;
        let sends: Vec<_> = (0..half - 1)
            .map(|_| jito.try_reserve().expect("within the limit"))
            .collect();
        let check = jito.try_reserve_spare().expect("more than half free");
        assert!(jito.try_reserve_spare().is_none(), "only half left");
        assert!(jito.try_reserve().is_some(), "sends still get one");
        drop((sends, check));
    }

    #[test]
    fn an_auth_uuid_admits_more_requests_per_window() {
        let jito = JitoClient::new(None, Some("uuid".to_string()));
        let slots: Vec<_> = (0..AUTHENTICATED_REQUESTS)
            .map(|_| jito.try_reserve().expect("within the limit"))
            .collect();
        assert!(jito.try_reserve().is_none(), "every slot held");
        for slot in &slots {
            slot.mark_sent();
        }
        drop(slots);
        assert!(
            jito.try_reserve().is_none(),
            "all used: none until the window passes"
        );
    }

    #[test]
    fn waiting_for_the_slot_never_makes_a_try_wait() {
        let jito = std::sync::Arc::new(JitoClient::new(None, None));
        let held = jito.try_reserve().unwrap();
        let waiter = {
            let jito = std::sync::Arc::clone(&jito);
            thread::spawn(move || {
                let slot = jito.reserve();
                slot.mark_sent();
            })
        };
        thread::sleep(Duration::from_millis(50));
        let started = Instant::now();
        assert!(jito.try_reserve().is_none());
        assert!(
            started.elapsed() < Duration::from_millis(10),
            "the try didn't wait"
        );
        drop(held);
        waiter.join().unwrap();
        assert!(
            jito.try_reserve().is_none(),
            "the waiter took it and used it"
        );
    }

    #[test]
    fn the_tip_is_a_share_of_the_profit_between_the_floor_and_the_cap() {
        // 8% of 10M lamports of profit.
        assert_eq!(tip(10_000_000, 0.08, 5_000, 5_000_000), 800_000);
        assert_eq!(
            tip(10_000, 0.08, 5_000, 5_000_000),
            5_000,
            "at least the floor"
        );
        assert_eq!(
            tip(1_000_000_000, 0.08, 5_000, 5_000_000),
            5_000_000,
            "the cap"
        );
        assert_eq!(
            tip(10_000, 0.08, 9_000_000, 5_000_000),
            5_000_000,
            "the cap wins"
        );
    }

    #[test]
    fn test_parse_send_bundle_result_ok() {
        let resp = json!({ "jsonrpc": "2.0", "id": 1, "result": "abc123" });
        assert_eq!(parse_send_bundle_result(&resp).unwrap(), "abc123");
    }

    #[test]
    fn test_parse_send_bundle_result_already_processed() {
        let resp = json!({ "error": { "code": -32000, "message": "bundle already processed" } });
        assert_eq!(parse_send_bundle_result(&resp).unwrap(), "0x0");
    }

    #[test]
    fn test_parse_send_bundle_result_error() {
        let resp = json!({ "error": { "code": -32602, "message": "invalid bundle" } });
        assert!(parse_send_bundle_result(&resp).is_err());
    }

    #[test]
    fn test_parse_send_bundle_result_missing() {
        let resp = json!({ "jsonrpc": "2.0", "id": 1 });
        assert!(parse_send_bundle_result(&resp).is_err());
    }

    #[test]
    fn test_static_tip_accounts() {
        assert_eq!(JITO_TIP_ACCOUNTS.len(), 8);
        let accts: Vec<_> = JITO_TIP_ACCOUNTS
            .iter()
            .map(|account| Pubkey::from_str(account).unwrap())
            .collect();
        let unique: std::collections::HashSet<_> = accts.iter().collect();
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn test_parse_tip_floor_lamports() {
        let resp = json!([{ "ema_landed_tips_50th_percentile": 0.00005 }]);
        assert_eq!(parse_tip_floor_lamports(&resp).unwrap(), 50_000);
    }

    #[test]
    fn test_parse_tip_floor_lamports_missing_field() {
        let resp = json!([{ "landed_tips_50th_percentile": 0.00005 }]);
        assert!(parse_tip_floor_lamports(&resp).is_err());
    }

    #[test]
    fn test_parse_tip_floor_lamports_empty() {
        let resp = json!([]);
        assert!(parse_tip_floor_lamports(&resp).is_err());
    }
}
