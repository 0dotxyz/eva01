//! The few RPC reads the liquidation path makes while an account is in flight, each bounded by a
//! short timeout. The Solana RPC client retries a 429 five times with sleeps, which a slow or
//! rate-limited node turns into minutes; these calls fail fast instead, and the caller decides.

use std::{str::FromStr, time::Duration};

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use solana_account::Account;
use solana_message::Hash;
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;

/// Each call's whole budget.
const TIMEOUT: Duration = Duration::from_secs(2);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// A rate-limited call is asked once more after this.
const RATE_LIMIT_RETRY: Duration = Duration::from_millis(250);

pub struct QuickRpc {
    http: Client,
    url: String,
}

/// The node answered with a JSON-RPC error: the request reached it and was refused. Any other
/// error is the transport's (a timeout, a dropped connection, a gateway error), after which a
/// request may still have been acted on.
#[derive(Debug)]
pub struct RpcRefused(pub String);

/// The JSON-RPC error code for a transaction that failed its preflight simulation.
pub const PREFLIGHT_FAILURE: i64 = -32002;

impl RpcRefused {
    /// Whether the node refused a transaction because its preflight simulation failed, other
    /// than for having processed it already, or for not knowing its blockhash (a node behind, or
    /// the hash's last slots): those share the code and say nothing about whether it lands.
    pub fn is_preflight_failure(&self) -> bool {
        self.0.contains(&format!("\"code\":{PREFLIGHT_FAILURE}"))
            && !self.0.contains("AlreadyProcessed")
            && !self.0.contains("already been processed")
            && !self.0.contains("BlockhashNotFound")
            && !self.0.contains("Blockhash not found")
    }
}

impl std::fmt::Display for RpcRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RpcRefused {}

/// A transaction's status, when the node knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureStatus {
    pub slot: u64,
    /// Why it failed on chain, if it did.
    pub err: Option<String>,
    /// At confirmed commitment or beyond.
    pub confirmed: bool,
}

impl QuickRpc {
    /// The one client for `url`, so every caller shares its warm connections.
    pub fn shared(url: String) -> std::sync::Arc<Self> {
        static CLIENTS: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<QuickRpc>>>,
        > = std::sync::OnceLock::new();
        let mut clients = CLIENTS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::sync::Arc::clone(
            clients
                .entry(url.clone())
                .or_insert_with(|| std::sync::Arc::new(Self::new(url))),
        )
    }

    pub fn new(url: String) -> Self {
        Self {
            http: Client::builder()
                .timeout(TIMEOUT)
                .connect_timeout(CONNECT_TIMEOUT)
                .build()
                .expect("the RPC HTTP client builds"),
            url,
        }
    }

    fn call(&self, method: &str, params: Value) -> Result<Value> {
        let body = json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
        // Without the URL, which may carry a token.
        // A connection never made is a request never sent. A 429 is asked once more, shortly.
        let mut retried = false;
        let started = std::time::Instant::now();
        let resp = loop {
            // The retry gets what is left of the one budget.
            let left = TIMEOUT.saturating_sub(started.elapsed());
            if left.is_zero() {
                return Err(anyhow!("{method}: out of time"));
            }
            let resp = match self.http.post(&self.url).json(&body).timeout(left).send() {
                Ok(resp) => resp,
                Err(e) if e.is_connect() => {
                    return Err(RpcRefused(format!("{method}: {}", e.without_url())).into())
                }
                Err(e) => return Err(e.without_url().into()),
            };
            if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && !retried {
                retried = true;
                std::thread::sleep(RATE_LIMIT_RETRY);
                continue;
            }
            break resp;
        };
        // By status first: a 4xx is the node refusing, whatever its body, except a proxy's
        // timeout (408, 499), after which the request may have been acted on; a 5xx is a gateway
        // or node failing, after which it may also have been.
        let status = resp.status();
        if status.is_client_error() && !matches!(status.as_u16(), 408 | 499) {
            return Err(RpcRefused(format!("{method}: HTTP {status}")).into());
        }
        if !status.is_success() {
            return Err(anyhow!("{method}: HTTP {status}"));
        }
        let resp: Value = resp.json().map_err(|e| e.without_url())?;
        if let Some(err) = resp.get("error").filter(|err| !err.is_null()) {
            return Err(RpcRefused(format!("{method}: {err}")).into());
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| anyhow!("{method}: no result"))
    }

    /// The statuses of `sigs`, in order; `None` for one the node doesn't know.
    pub fn signature_statuses(&self, sigs: &[Signature]) -> Result<Vec<Option<SignatureStatus>>> {
        let sigs: Vec<String> = sigs.iter().map(ToString::to_string).collect();
        let result = self.call(
            "getSignatureStatuses",
            json!([sigs, { "searchTransactionHistory": false }]),
        )?;
        parse_signature_statuses(&result)
    }

    /// Whether `blockhash` can still land a transaction, at processed commitment.
    pub fn is_blockhash_valid(&self, blockhash: &Hash) -> Result<bool> {
        let result = self.call(
            "isBlockhashValid",
            json!([blockhash.to_string(), { "commitment": "processed" }]),
        )?;
        result
            .get("value")
            .and_then(Value::as_bool)
            .ok_or_else(|| anyhow!("isBlockhashValid: unexpected answer {result}"))
    }

    /// The newest blockhash, at confirmed commitment (what `getLatestBlockhash` defaults to).
    pub fn latest_blockhash(&self) -> Result<Hash> {
        let result = self.call("getLatestBlockhash", json!([{ "commitment": "confirmed" }]))?;
        let hash = result
            .pointer("/value/blockhash")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("getLatestBlockhash: unexpected answer {result}"))?;
        Hash::from_str(hash).map_err(|e| anyhow!("getLatestBlockhash: {e}"))
    }

    /// Sends `tx`, with the node's preflight simulation unless `skip_preflight`. Returns its
    /// signature; an error carries the node's reason (with preflight, the simulation's).
    pub fn send_transaction(
        &self,
        tx: &VersionedTransaction,
        skip_preflight: bool,
    ) -> Result<Signature> {
        let encoded = BASE64.encode(bincode::serialize(tx)?);
        let result = self.call(
            "sendTransaction",
            json!([encoded, {
                "encoding": "base64",
                "skipPreflight": skip_preflight,
                "preflightCommitment": "processed",
            }]),
        )?;
        let sig = result
            .as_str()
            .ok_or_else(|| anyhow!("sendTransaction: unexpected answer {result}"))?;
        Signature::from_str(sig).map_err(|e| anyhow!("sendTransaction: {e}"))
    }

    /// The node's slot at confirmed commitment.
    pub fn confirmed_slot(&self) -> Result<u64> {
        self.call("getSlot", json!([{ "commitment": "confirmed" }]))?
            .as_u64()
            .ok_or_else(|| anyhow!("getSlot: not a number"))
    }

    /// The node's slot at processed commitment.
    pub fn processed_slot(&self) -> Result<u64> {
        self.call("getSlot", json!([{ "commitment": "processed" }]))?
            .as_u64()
            .ok_or_else(|| anyhow!("getSlot: not a number"))
    }

    /// `addresses`' accounts at processed commitment, and the slot read at.
    pub fn multiple_accounts(&self, addresses: &[Pubkey]) -> Result<(u64, Vec<Option<Account>>)> {
        let keys: Vec<String> = addresses.iter().map(ToString::to_string).collect();
        let result = self.call(
            "getMultipleAccounts",
            json!([keys, { "commitment": "processed", "encoding": "base64" }]),
        )?;
        parse_multiple_accounts(&result)
    }

    /// Simulates `tx` against the current state: `None` when it passes, otherwise its error and
    /// logs.
    pub fn simulate(&self, tx: &VersionedTransaction) -> Result<Option<(String, Vec<String>)>> {
        let encoded = BASE64.encode(bincode::serialize(tx)?);
        let result = self.call(
            "simulateTransaction",
            json!([encoded, {
                "encoding": "base64",
                "sigVerify": false,
                "commitment": "processed",
            }]),
        )?;
        let value = result
            .get("value")
            .ok_or_else(|| anyhow!("simulateTransaction: no value"))?;
        match value.get("err") {
            None | Some(Value::Null) => Ok(None),
            Some(err) => {
                let logs = value
                    .get("logs")
                    .and_then(Value::as_array)
                    .map(|logs| {
                        logs.iter()
                            .filter_map(|log| log.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(Some((err.to_string(), logs)))
            }
        }
    }
}

fn parse_signature_statuses(result: &Value) -> Result<Vec<Option<SignatureStatus>>> {
    let values = result
        .get("value")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("getSignatureStatuses: unexpected answer {result}"))?;
    values
        .iter()
        .map(|status| {
            if status.is_null() {
                return Ok(None);
            }
            let slot = status
                .get("slot")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("getSignatureStatuses: no slot in {status}"))?;
            let err = match status.get("err") {
                None | Some(Value::Null) => None,
                Some(err) => Some(err.to_string()),
            };
            // Without `confirmationStatus` (a non-standard node), `confirmations`: null once
            // finalized, at least one once confirmed.
            let confirmed = match status.get("confirmationStatus").and_then(Value::as_str) {
                Some(level) => matches!(level, "confirmed" | "finalized"),
                None => match status.get("confirmations") {
                    None | Some(Value::Null) => true,
                    Some(count) => count.as_u64().is_some_and(|count| count > 1),
                },
            };
            Ok(Some(SignatureStatus {
                slot,
                err,
                confirmed,
            }))
        })
        .collect()
}

fn parse_multiple_accounts(result: &Value) -> Result<(u64, Vec<Option<Account>>)> {
    let slot = result
        .pointer("/context/slot")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("getMultipleAccounts: no slot"))?;
    let values = result
        .get("value")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("getMultipleAccounts: no value"))?;
    let accounts = values
        .iter()
        .map(|account| {
            if account.is_null() {
                return Ok(None);
            }
            let field = |name: &str| {
                account
                    .get(name)
                    .ok_or_else(|| anyhow!("getMultipleAccounts: no {name}"))
            };
            let data = field("data")?
                .get(0)
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("getMultipleAccounts: data is not base64"))?;
            Ok(Some(Account {
                lamports: field("lamports")?.as_u64().unwrap_or_default(),
                data: BASE64.decode(data)?,
                owner: Pubkey::from_str(field("owner")?.as_str().unwrap_or_default())?,
                executable: field("executable")?.as_bool().unwrap_or_default(),
                rent_epoch: account
                    .get("rentEpoch")
                    .and_then(Value::as_u64)
                    .unwrap_or(u64::MAX),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((slot, accounts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_tell_unknown_landed_confirmed_and_failed() {
        let statuses = parse_signature_statuses(&json!({
            "context": { "slot": 10 },
            "value": [
                null,
                { "slot": 5, "confirmations": 0, "err": null, "confirmationStatus": "processed" },
                { "slot": 6, "confirmations": null, "err": null, "confirmationStatus": "finalized" },
                { "slot": 7, "confirmations": 1, "err": { "InstructionError": [2, { "Custom": 6068 }] }, "confirmationStatus": "confirmed" },
            ]
        }))
        .unwrap();
        assert_eq!(statuses[0], None);
        assert!(!statuses[1].as_ref().unwrap().confirmed);
        assert!(statuses[2].as_ref().unwrap().confirmed);
        assert!(statuses[3]
            .as_ref()
            .unwrap()
            .err
            .as_ref()
            .unwrap()
            .contains("6068"));
        // A node that leaves `confirmationStatus` out.
        let statuses = parse_signature_statuses(&json!({ "value": [
            { "slot": 5, "confirmations": 0, "err": null },
            { "slot": 5, "confirmations": 1, "err": null },
            { "slot": 5, "confirmations": 3, "err": null },
            { "slot": 5, "confirmations": null, "err": null },
        ]}))
        .unwrap();
        let confirmed: Vec<bool> = statuses
            .iter()
            .map(|status| status.as_ref().unwrap().confirmed)
            .collect();
        assert_eq!(confirmed, [false, false, true, true]);
    }

    #[test]
    fn a_preflight_failure_is_told_from_already_processed() {
        let refused = |message: &str| RpcRefused(format!("sendTransaction: {message}"));
        assert!(refused(r#"{"code":-32002,"message":"Transaction simulation failed: Error processing Instruction 4: custom program error: 0x17b4"}"#).is_preflight_failure());
        assert!(!refused(r#"{"code":-32002,"message":"Transaction simulation failed: This transaction has already been processed"}"#).is_preflight_failure());
        assert!(
            !refused(r#"{"code":-32002,"data":{"err":"AlreadyProcessed"}}"#).is_preflight_failure()
        );
        assert!(!refused("HTTP 429 Too Many Requests").is_preflight_failure());
        assert!(!refused(r#"{"code":-32002,"message":"Transaction simulation failed: Blockhash not found","data":{"err":"BlockhashNotFound"}}"#).is_preflight_failure());
    }

    #[test]
    fn accounts_decode_from_base64() {
        let owner = Pubkey::new_unique();
        let (slot, accounts) = parse_multiple_accounts(&json!({
            "context": { "slot": 42 },
            "value": [
                null,
                { "data": [BASE64.encode([1u8, 2, 3]), "base64"], "executable": false,
                  "lamports": 7, "owner": owner.to_string(), "rentEpoch": 18446744073709551615u64,
                  "space": 3 },
            ]
        }))
        .unwrap();
        assert_eq!(slot, 42);
        assert!(accounts[0].is_none());
        let account = accounts[1].as_ref().unwrap();
        assert_eq!(account.data, vec![1, 2, 3]);
        assert_eq!(account.owner, owner);
        assert_eq!(account.lamports, 7);
    }

    fn rand_bytes() -> [u8; 32] {
        solana_signer::Signer::pubkey(&solana_keypair::Keypair::new()).to_bytes()
    }

    /// Reads mainnet with every call. `cargo test quick_rpc -- --ignored`.
    #[test]
    #[ignore = "reads mainnet"]
    fn every_call_reads_mainnet() {
        let rpc = QuickRpc::new("https://api.mainnet-beta.solana.com".to_string());
        let slot = rpc.processed_slot().unwrap();
        assert!(slot > 400_000_000);
        assert!(!rpc.is_blockhash_valid(&Hash::new_unique()).unwrap());
        let latest = rpc.latest_blockhash().unwrap();
        assert!(rpc.is_blockhash_valid(&latest).unwrap());
        let usdc = Pubkey::from_str("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v").unwrap();
        // A random address, unlike `new_unique`'s predictable ones, which may exist.
        let absent = Pubkey::new_from_array(rand_bytes());
        let (read_at, accounts) = rpc.multiple_accounts(&[usdc, absent]).unwrap();
        assert!(read_at >= slot - 100);
        assert_eq!(accounts[0].as_ref().unwrap().data.len(), 82, "a mint");
        assert!(accounts[1].is_none());
        // A landed eva push transaction.
        let landed = Signature::from_str(
            "5E9wE8imewjZB878X8rvkQbqLa7Fc75qSfRYte27Y1aRFBbXxj6vzd8di3BEbRTbYyEAGd3cDkJdD6cGaHfqqSHh",
        )
        .unwrap();
        // A recent transaction of a busy program: known to the node's status cache.
        let recent = rpc
            .call(
                "getSignaturesForAddress",
                json!(["rec2HHDDnjLfj4kE7VyEtFA1HPGQLK33259532cRyHp", { "limit": 1 }]),
            )
            .unwrap();
        let recent = Signature::from_str(recent[0]["signature"].as_str().unwrap()).unwrap();
        let statuses = rpc
            .signature_statuses(&[landed, Signature::default(), recent])
            .unwrap();
        println!("{statuses:?}");
        assert!(statuses[1].is_none());
        assert!(statuses[2].as_ref().is_some_and(|status| status.slot > 0));
    }
}
