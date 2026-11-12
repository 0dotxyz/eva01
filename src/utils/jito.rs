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
use log::warn;
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

/// The block engine admits one request per second per endpoint (`sendBundle` and the status
/// methods alike); anything faster is rejected with `-32097`.
const MIN_REQUEST_INTERVAL: Duration = Duration::from_secs(1);

pub struct JitoClient {
    http: Client,
    endpoint: String,
    api_key: Option<String>,
    slot: Mutex<SlotState>,
}

/// The one request the block engine admits per [`MIN_REQUEST_INTERVAL`].
struct SlotState {
    /// When the last request went out.
    last_request: Instant,
    /// Whether a caller holds the slot, between reserving it and sending.
    held: bool,
}

/// How often a caller waiting for the slot looks again while another caller holds it.
const HELD_SLOT_RECHECK: Duration = Duration::from_millis(20);

/// A request slot, reserved until its request goes out or it is dropped. While it is held, nobody
/// else gets one, so a caller that quotes a swap after reserving can't be overtaken into a second
/// request within the interval; dropping it unused gives it back.
pub struct JitoSlot<'a> {
    jito: &'a JitoClient,
}

impl Drop for JitoSlot<'_> {
    fn drop(&mut self) {
        self.jito
            .slot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .held = false;
    }
}

impl JitoSlot<'_> {
    /// Counts the slot as used from now.
    fn mark_sent(&self) {
        self.jito
            .slot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last_request = Instant::now();
    }

    /// Submits a bundle in this slot. An `Err` is a request that may or may not have arrived.
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
        let resp: Value = self
            .jito
            .http
            .post(self.jito.url())
            .json(&body)
            .send()?
            .json()?;
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
        Self {
            http: Client::new(),
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_BUNDLE_ENDPOINT.to_string()),
            api_key,
            slot: Mutex::new(SlotState {
                last_request: Instant::now() - MIN_REQUEST_INTERVAL,
                held: false,
            }),
        }
    }

    /// The request slot, if it is free now, without waiting: a liquidation that finds none goes
    /// out another way instead of queueing behind the other workers. The lock is held only to
    /// read and flip the state, never across a wait.
    pub fn try_reserve(&self) -> Option<JitoSlot<'_>> {
        let mut state = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if state.held || state.last_request.elapsed() < MIN_REQUEST_INTERVAL {
            return None;
        }
        state.held = true;
        Some(JitoSlot { jito: self })
    }

    /// Waits for the request slot and takes it, for a bundle that has no other way out. Sleeps
    /// without the lock, so others' `try_reserve` never waits behind it.
    pub fn reserve(&self) -> JitoSlot<'_> {
        loop {
            let wait = {
                let state = self.slot.lock().unwrap_or_else(|e| e.into_inner());
                let since = state.last_request.elapsed();
                if state.held {
                    HELD_SLOT_RECHECK
                } else {
                    MIN_REQUEST_INTERVAL.saturating_sub(since)
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

    /// The tip for a bundle expected to earn `profit_lamports`.
    pub fn tip_for(&self, profit_lamports: u64) -> u64 {
        let floor = *self.floor.lock().unwrap_or_else(|e| e.into_inner());
        tip(profit_lamports, self.profit_share, floor, self.max_lamports)
    }

    /// The tip transfer as the bundle's last transaction, unless a price push carries it. It can't
    /// go inside the liquidation: marginfi accepts only swap programs between a receivership's
    /// start and end, and nothing after the end.
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
pub fn tip_instruction(payer: &Pubkey, lamports: u64) -> Result<Instruction> {
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
