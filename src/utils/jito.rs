//! Minimal Jito block-engine bundle client.
//!
//! Submits an ordered set of transactions as an atomic bundle via the public block-engine
//! JSON-RPC (`sendBundle`) and polls `getBundleStatuses` for confirmation. Tip accounts are
//! fetched from the Jito REST endpoint. No extra crates: plain `reqwest` + `serde_json`,
//! mirroring p0-app's `send-bundle` route.

use std::{
    str::FromStr,
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use log::{debug, warn};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use solana_program::pubkey::Pubkey;
use solana_sdk::transaction::VersionedTransaction;

const DEFAULT_BUNDLE_ENDPOINT: &str = "https://mainnet.block-engine.jito.wtf/api/v1/bundles";
const TIP_ACCOUNTS_URL: &str = "https://bundles.jito.wtf/api/v1/bundles/tip_accounts";
const TIP_FLOOR_URL: &str = "https://bundles.jito.wtf/api/v1/bundles/tip_floor";

/// How long to wait after submission before the first status poll, and between polls.
const POLL_INITIAL_DELAY: Duration = Duration::from_millis(500);
const POLL_INTERVAL: Duration = Duration::from_secs(1);

pub struct JitoClient {
    http: Client,
    endpoint: String,
    api_key: Option<String>,
}

impl JitoClient {
    pub fn new(endpoint: Option<String>, api_key: Option<String>) -> Self {
        Self {
            http: Client::new(),
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_BUNDLE_ENDPOINT.to_string()),
            api_key,
        }
    }

    /// The block-engine URL with the optional auth uuid appended (matches p0-app).
    fn url(&self) -> String {
        match &self.api_key {
            Some(key) => format!("{}?uuid={}", self.endpoint, key),
            None => self.endpoint.clone(),
        }
    }

    /// Submit a bundle of transactions. Returns the bundle id on success.
    pub fn send_bundle(&self, txs: &[VersionedTransaction]) -> Result<String> {
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

        let resp: Value = self.http.post(self.url()).json(&body).send()?.json()?;
        parse_send_bundle_result(&resp)
    }

    /// Returns the `confirmation_status` of the bundle, if the block engine reports one yet.
    pub fn get_bundle_status(&self, bundle_id: &str) -> Result<Option<String>> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBundleStatuses",
            "params": [[bundle_id]],
        });
        let resp: Value = self.http.post(self.url()).json(&body).send()?.json()?;
        parse_bundle_status(&resp)
    }

    /// Submit a bundle and poll until it is `confirmed`/`finalized` or `max_attempts` elapse.
    pub fn send_bundle_and_confirm(
        &self,
        txs: &[VersionedTransaction],
        max_attempts: usize,
    ) -> Result<String> {
        let bundle_id = self.send_bundle(txs)?;
        // "0x0" is the block engine's "already processed" sentinel — nothing to poll.
        if bundle_id == "0x0" {
            return Ok(bundle_id);
        }

        thread::sleep(POLL_INITIAL_DELAY);
        for attempt in 0..max_attempts {
            match self.get_bundle_status(&bundle_id) {
                Ok(Some(status)) if status == "confirmed" || status == "finalized" => {
                    debug!("Bundle {} {} after {} poll(s)", bundle_id, status, attempt + 1);
                    return Ok(bundle_id);
                }
                Ok(_) => {}
                Err(e) => warn!("Bundle {} status poll failed: {}", bundle_id, e),
            }
            thread::sleep(POLL_INTERVAL);
        }
        Err(anyhow!("Bundle {} did not confirm in time", bundle_id))
    }

    /// Fetch the current set of Jito tip accounts (one must receive the tip ix in a bundle).
    pub fn get_tip_accounts(&self) -> Result<Vec<Pubkey>> {
        let resp: Value = self.http.get(TIP_ACCOUNTS_URL).send()?.json()?;
        parse_tip_accounts(&resp)
    }

    /// Simulate a bundle atomically against a `simulateBundle`-capable RPC endpoint.
    ///
    /// Uses `skipSigVerify` + `replaceRecentBlockhash` so unsigned, blockhash-less txs can be
    /// simulated. This is how the executor does simulate-first crank detection: simulate
    /// `[buy?, liquidate]` without a crank, and only prepend a crank tx if the failure is a
    /// stale-oracle error. `rpc_url` must support `simulateBundle` (block-engine sim is on the
    /// RPC, not the bundle endpoint); `sim_api_key` is sent as a Bearer token when present.
    pub fn simulate_bundle(
        &self,
        rpc_url: &str,
        sim_api_key: Option<&str>,
        txs: &[VersionedTransaction],
        accounts_to_inspect: &[Pubkey],
    ) -> Result<BundleSimulation> {
        let encoded = encode_transactions(txs)?;
        let config = build_simulate_config(txs.len(), accounts_to_inspect);
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "simulateBundle",
            "params": [{ "encodedTransactions": encoded }, config],
        });

        let mut req = self.http.post(rpc_url).json(&body);
        if let Some(key) = sim_api_key {
            req = req.bearer_auth(key);
        }
        let resp: Value = req.send()?.json()?;
        parse_simulate_bundle(&resp)
    }
}

/// Outcome of a `simulateBundle` call.
#[derive(Debug, Clone)]
pub struct BundleSimulation {
    pub succeeded: bool,
    /// The on-chain error message of the first failing tx (e.g. "... custom program error: 0x17a1").
    pub error_message: Option<String>,
    /// Index of the first failing tx within the bundle, if reported.
    pub failed_tx_index: Option<usize>,
}

impl BundleSimulation {
    /// Whether the failure (if any) is the Switchboard stale-price error that warrants a crank.
    pub fn is_stale_price_failure(&self) -> bool {
        self.error_message
            .as_deref()
            .map(|m| m.contains(crate::utils::swb_cranker::SWB_STALE_PRICE_ERROR_CODE))
            .unwrap_or(false)
    }
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

/// Extract the first bundle's `confirmation_status` from a `getBundleStatuses` response.
fn parse_bundle_status(resp: &Value) -> Result<Option<String>> {
    if let Some(err) = resp.get("error") {
        return Err(anyhow!("Jito getBundleStatuses error: {}", err));
    }
    Ok(resp
        .pointer("/result/value/0/confirmation_status")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string()))
}

/// Parse the tip-accounts REST response (a plain JSON array of base58 pubkeys).
fn parse_tip_accounts(resp: &Value) -> Result<Vec<Pubkey>> {
    let arr = resp
        .as_array()
        .ok_or_else(|| anyhow!("Unexpected tip_accounts response: {}", resp))?;
    arr.iter()
        .filter_map(|v| v.as_str())
        .map(|s| Pubkey::from_str(s).map_err(|e| anyhow!("Invalid tip account {s}: {e}")))
        .collect()
}

/// Build the `simulateBundle` config, mirroring p0-app's `createBundleConfig`: skip sig
/// verification, replace the recent blockhash, and only inspect accounts after the last tx.
fn build_simulate_config(num_txs: usize, accounts_to_inspect: &[Pubkey]) -> Value {
    let pre: Vec<Value> = (0..num_txs).map(|_| json!({ "addresses": [] })).collect();
    let post: Vec<Value> = (0..num_txs)
        .map(|i| {
            if i + 1 == num_txs && !accounts_to_inspect.is_empty() {
                let addrs: Vec<String> =
                    accounts_to_inspect.iter().map(|a| a.to_string()).collect();
                json!({ "addresses": addrs })
            } else {
                json!({ "addresses": [] })
            }
        })
        .collect();
    json!({
        "skipSigVerify": true,
        "replaceRecentBlockhash": true,
        "preExecutionAccountsConfigs": pre,
        "postExecutionAccountsConfigs": post,
    })
}

/// Parse a `simulateBundle` response into a [`BundleSimulation`]. On failure, extract the first
/// failing tx index and its on-chain error message from `summary.failed.error.TransactionFailure`.
fn parse_simulate_bundle(resp: &Value) -> Result<BundleSimulation> {
    if let Some(err) = resp.get("error") {
        return Err(anyhow!("Jito simulateBundle error: {}", err));
    }
    let summary = resp
        .pointer("/result/value/summary")
        .ok_or_else(|| anyhow!("simulateBundle: missing result.value.summary: {}", resp))?;

    if summary.as_str() == Some("succeeded") {
        return Ok(BundleSimulation {
            succeeded: true,
            error_message: None,
            failed_tx_index: None,
        });
    }

    // Failure: `{ "failed": { "error": { "TransactionFailure": [[idx...], "msg"] }, .. } }`
    let tf = summary.pointer("/failed/error/TransactionFailure");
    let (failed_tx_index, error_message) = match tf {
        Some(tf) => {
            let idx = tf
                .get(0)
                .and_then(|a| a.as_array())
                .and_then(|a| a.first())
                .and_then(|n| n.as_u64())
                .map(|n| n as usize);
            let msg = tf.get(1).and_then(|m| m.as_str()).map(|s| s.to_string());
            (idx, msg)
        }
        None => (None, summary.get("failed").map(|f| f.to_string())),
    };

    Ok(BundleSimulation {
        succeeded: false,
        error_message,
        failed_tx_index,
    })
}

const TIP_REFRESH_INTERVAL: Duration = Duration::from_secs(60 * 60);
const TIP_FALLBACK_LAMPORTS: u64 = 10_000;
const LAMPORTS_PER_SOL_F64: f64 = 1_000_000_000.0;

/// Dynamic Jito tip estimator with a max cap and TTL cache.
///
/// Reads the EMA 50th-percentile landed tip from Jito's `tip_floor` endpoint, converts SOL ->
/// lamports, and clamps to `max_lamports`. The value is cached for `TIP_REFRESH_INTERVAL`, so it
/// is not refetched on every execution.
pub struct TipEstimator {
    http: Client,
    max_lamports: u64,
    cached: Mutex<Option<(u64, Instant)>>,
}

impl TipEstimator {
    pub fn new(max_lamports: u64) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            http,
            max_lamports,
            cached: Mutex::new(None),
        }
    }

    /// Current tip in lamports (EMA-50th landed tip, clamped to the max cap), cached on a TTL so
    /// repeated executions reuse the same value instead of refetching.
    pub fn current_tip(&self) -> u64 {
        let now = Instant::now();
        if let Ok(guard) = self.cached.lock() {
            if let Some((tip, at)) = *guard {
                if now.duration_since(at) < TIP_REFRESH_INTERVAL {
                    return tip;
                }
            }
        }

        let tip = match self.fetch_tip_lamports() {
            Ok(lamports) => lamports.min(self.max_lamports),
            Err(e) => {
                // Back off to the last good value (or the fallback), capped, and reuse it for the
                // refresh interval so a flaky endpoint doesn't add latency to every execution.
                let last = self
                    .cached
                    .lock()
                    .ok()
                    .and_then(|g| g.map(|(t, _)| t))
                    .unwrap_or(TIP_FALLBACK_LAMPORTS)
                    .min(self.max_lamports);
                warn!("Tip floor fetch failed ({e}); using {last} lamports");
                last
            }
        };

        if let Ok(mut guard) = self.cached.lock() {
            *guard = Some((tip, now));
        }
        tip
    }

    fn fetch_tip_lamports(&self) -> Result<u64> {
        let resp: Value = self.http.get(TIP_FLOOR_URL).send()?.json()?;
        parse_tip_floor_lamports(&resp)
    }
}

/// Parse `ema_landed_tips_50th_percentile` (SOL) from the tip_floor response into lamports.
fn parse_tip_floor_lamports(resp: &Value) -> Result<u64> {
    let sol = resp
        .get(0)
        .and_then(|d| d.get("ema_landed_tips_50th_percentile"))
        .and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow!("tip_floor missing ema_landed_tips_50th_percentile: {resp}"))?;
    if sol < 0.0 {
        return Err(anyhow!("negative tip floor: {sol}"));
    }
    Ok((sol * LAMPORTS_PER_SOL_F64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_parse_bundle_status_confirmed() {
        let resp = json!({
            "result": { "context": { "slot": 1 }, "value": [ { "confirmation_status": "confirmed" } ] }
        });
        assert_eq!(parse_bundle_status(&resp).unwrap().as_deref(), Some("confirmed"));
    }

    #[test]
    fn test_parse_bundle_status_none_yet() {
        let resp = json!({ "result": { "context": { "slot": 1 }, "value": [] } });
        assert_eq!(parse_bundle_status(&resp).unwrap(), None);
    }

    #[test]
    fn test_parse_bundle_status_error() {
        let resp = json!({ "error": { "code": -32000, "message": "boom" } });
        assert!(parse_bundle_status(&resp).is_err());
    }

    #[test]
    fn test_parse_tip_accounts() {
        let a = Pubkey::new_unique().to_string();
        let b = Pubkey::new_unique().to_string();
        let resp = json!([a, b]);
        let parsed = parse_tip_accounts(&resp).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].to_string(), a);
    }

    #[test]
    fn test_parse_tip_accounts_bad_shape() {
        let resp = json!({ "not": "an array" });
        assert!(parse_tip_accounts(&resp).is_err());
    }

    #[test]
    fn test_parse_simulate_bundle_succeeded() {
        let resp = json!({
            "result": { "context": { "slot": 1 }, "value": {
                "summary": "succeeded",
                "transactionResults": []
            } }
        });
        let sim = parse_simulate_bundle(&resp).unwrap();
        assert!(sim.succeeded);
        assert!(sim.error_message.is_none());
        assert!(!sim.is_stale_price_failure());
    }

    #[test]
    fn test_parse_simulate_bundle_stale_failure() {
        let resp = json!({
            "result": { "context": { "slot": 1 }, "value": {
                "summary": { "failed": { "error": { "TransactionFailure": [
                    [1],
                    "Error processing Instruction 3: custom program error: 0x17a1"
                ] }, "tx_signature": "sig" } },
                "transactionResults": []
            } }
        });
        let sim = parse_simulate_bundle(&resp).unwrap();
        assert!(!sim.succeeded);
        assert_eq!(sim.failed_tx_index, Some(1));
        assert!(sim.is_stale_price_failure());
    }

    #[test]
    fn test_parse_simulate_bundle_other_failure() {
        let resp = json!({
            "result": { "context": { "slot": 1 }, "value": {
                "summary": { "failed": { "error": { "TransactionFailure": [
                    [0],
                    "Error processing Instruction 1: custom program error: 0x1"
                ] } } },
                "transactionResults": []
            } }
        });
        let sim = parse_simulate_bundle(&resp).unwrap();
        assert!(!sim.succeeded);
        assert!(!sim.is_stale_price_failure());
    }

    #[test]
    fn test_parse_simulate_bundle_rpc_error() {
        let resp = json!({ "error": { "code": -32601, "message": "method not found" } });
        assert!(parse_simulate_bundle(&resp).is_err());
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

    #[test]
    fn test_build_simulate_config_inspect_last_only() {
        let acct = Pubkey::new_unique();
        let cfg = build_simulate_config(3, std::slice::from_ref(&acct));
        let post = cfg["postExecutionAccountsConfigs"].as_array().unwrap();
        assert_eq!(post.len(), 3);
        assert_eq!(post[0]["addresses"].as_array().unwrap().len(), 0);
        assert_eq!(post[2]["addresses"].as_array().unwrap().len(), 1);
        assert_eq!(post[2]["addresses"][0].as_str().unwrap(), acct.to_string());
        assert_eq!(cfg["skipSigVerify"], json!(true));
        assert_eq!(cfg["replaceRecentBlockhash"], json!(true));
    }
}
