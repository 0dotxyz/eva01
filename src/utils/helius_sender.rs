//! Minimal Helius Sender bundle client.
//!
//! Sender takes up to four signed legacy or v0 transactions as an atomic bundle (`sendBundle`; it
//! can't decode v1) and routes it across several block-building paths, Jito among them, paying
//! their tips itself out of the one tip the bundle carries to a Sender tip account. It has no bundle
//! status: whether a bundle landed is read from the chain, by its liquidation's signature.

use std::{
    collections::VecDeque,
    str::FromStr,
    sync::Mutex,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use log::{info, warn};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_keypair::Keypair;
use solana_message::{v0, Hash, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_system_interface::instruction::transfer;
use solana_transaction::versioned::VersionedTransaction;

use crate::transaction_limits::{bundle_cu_price, MAX_SENDER_BUNDLE_TXS};

const DEFAULT_ENDPOINT: &str = "https://sender.helius-rpc.com/fast";

/// Sender's tip accounts; a bundle must pay one of them at least [`MIN_TIP_LAMPORTS`].
const TIP_ACCOUNTS: [&str; 10] = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
    "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
];

/// The smallest tip Sender takes for a bundle: 0.001 SOL.
pub const MIN_TIP_LAMPORTS: u64 = 1_000_000;

/// Requests per second: 50 per key and region with an API key, 1 per IP and region without.
const KEYED_REQUESTS: usize = 50;
const KEYLESS_REQUESTS: usize = 1;
const RATE_WINDOW: Duration = Duration::from_secs(1);

const REQUEST_TIMEOUT: Duration = Duration::from_secs(3);

/// How often an idle connection is kept warm, so a bundle never waits for a TLS handshake.
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// The tip transaction's compute limit: a transfer and the budget instructions take well under it.
const TIP_TX_CU_LIMIT: u32 = 1_000;

pub struct HeliusSender {
    http: Client,
    url: String,
    capacity: usize,
    sent: Mutex<VecDeque<Instant>>,
}

/// Sender's answer to `sendBundle`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SenderSend {
    Accepted,
    /// Refused or never delivered: it will not land.
    Refused(String),
}

impl HeliusSender {
    /// `endpoint` defaults to Sender's global HTTPS endpoint.
    pub fn new(endpoint: Option<String>, api_key: Option<String>) -> Self {
        let endpoint = endpoint.unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        let capacity = if api_key.is_some() {
            KEYED_REQUESTS
        } else {
            KEYLESS_REQUESTS
        };
        info!(
            "Helius Sender: {capacity} bundle(s) per second ({})",
            if api_key.is_some() {
                "API key"
            } else {
                "no API key"
            }
        );
        let url = match &api_key {
            Some(key) => format!("{endpoint}?api-key={key}"),
            None => endpoint.clone(),
        };
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .pool_idle_timeout(None)
            .build()
            .unwrap_or_else(|_| Client::new());
        spawn_keepalive(http.clone(), ping_url(&endpoint));
        Self {
            http,
            url,
            capacity,
            sent: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    /// Takes a request out of this second's budget, if one is left. Never waits: a bundle with
    /// none goes through the other paths only.
    fn try_admit(&self) -> bool {
        let now = Instant::now();
        let mut sent = self.sent.lock().unwrap_or_else(|e| e.into_inner());
        while sent
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) >= RATE_WINDOW)
        {
            sent.pop_front();
        }
        if sent.len() >= self.capacity {
            return false;
        }
        sent.push_back(now);
        true
    }

    /// Submits `txs` as an atomic bundle.
    pub fn send_bundle(&self, txs: &[VersionedTransaction]) -> SenderSend {
        if txs.is_empty() || txs.len() > MAX_SENDER_BUNDLE_TXS {
            return SenderSend::Refused(format!("{} transactions", txs.len()));
        }
        if !self.try_admit() {
            return SenderSend::Refused("over the request rate".to_string());
        }
        let encoded: Result<Vec<String>> = txs
            .iter()
            .map(|tx| Ok(BASE64.encode(bincode::serialize(tx)?)))
            .collect();
        let encoded = match encoded {
            Ok(encoded) => encoded,
            Err(e) => return SenderSend::Refused(format!("can't encode: {e}")),
        };
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [encoded, { "encoding": "base64" }],
        });
        let resp = match self.http.post(&self.url).json(&body).send() {
            Ok(resp) => resp,
            Err(e) => return SenderSend::Refused(strip_key(&e.to_string())),
        };
        let status = resp.status();
        match resp.json::<Value>() {
            Ok(value) => parse_send_bundle(status, &value),
            Err(e) => SenderSend::Refused(format!("HTTP {status}, unreadable answer: {e}")),
        }
    }
}

/// `reqwest` errors name the URL, which carries the API key.
fn strip_key(message: &str) -> String {
    match message.find("api-key=") {
        Some(at) => {
            let end = message[at..]
                .find(|c: char| c == ')' || c == '&' || c.is_whitespace())
                .map_or(message.len(), |len| at + len);
            format!("{}api-key=…{}", &message[..at], &message[end..])
        }
        None => message.to_string(),
    }
}

fn parse_send_bundle(status: reqwest::StatusCode, resp: &Value) -> SenderSend {
    if let Some(err) = resp.get("error") {
        return SenderSend::Refused(format!("HTTP {status}: {err}"));
    }
    if !status.is_success() || resp.get("result").is_none() {
        return SenderSend::Refused(format!("HTTP {status}: {resp}"));
    }
    SenderSend::Accepted
}

fn ping_url(endpoint: &str) -> String {
    match endpoint.strip_suffix("/fast") {
        Some(base) => format!("{base}/ping"),
        None => format!("{}/ping", endpoint.trim_end_matches('/')),
    }
}

fn spawn_keepalive(http: Client, url: String) {
    thread::Builder::new()
        .name("sender-ping".to_string())
        .spawn(move || loop {
            if let Err(e) = http.get(&url).send() {
                warn!("Helius Sender ping failed: {}", strip_key(&e.to_string()));
            }
            thread::sleep(PING_INTERVAL);
        })
        .ok();
}

/// The bundle's tip to a Sender tip account, with the priority fee Sender asks of every
/// transaction.
pub fn build_tip_transaction(
    signer: &Keypair,
    blockhash: Hash,
    lamports: u64,
    cu_price_micro_lamports: u64,
) -> Result<VersionedTransaction> {
    let ixs = [
        ComputeBudgetInstruction::set_compute_unit_limit(TIP_TX_CU_LIMIT),
        ComputeBudgetInstruction::set_compute_unit_price(bundle_cu_price(
            TIP_TX_CU_LIMIT,
            cu_price_micro_lamports,
        )),
        tip_instruction(&signer.pubkey(), lamports)?,
    ];
    let msg = v0::Message::try_compile(&signer.pubkey(), &ixs, &[], blockhash)?;
    VersionedTransaction::try_new(VersionedMessage::V0(msg), &[signer]).map_err(Into::into)
}

/// A transfer of `lamports` from `payer` to one of Sender's tip accounts.
pub fn tip_instruction(payer: &Pubkey, lamports: u64) -> Result<solana_instruction::Instruction> {
    Ok(transfer(payer, &select_tip_account()?, lamports))
}

fn select_tip_account() -> Result<Pubkey> {
    let idx = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as usize)
        .unwrap_or(0)
        % TIP_ACCOUNTS.len();
    Pubkey::from_str(TIP_ACCOUNTS[idx]).map_err(|e| anyhow!("Sender tip account is invalid: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tip_accounts_are_valid_and_distinct() {
        let accounts: std::collections::HashSet<Pubkey> = TIP_ACCOUNTS
            .iter()
            .map(|account| Pubkey::from_str(account).unwrap())
            .collect();
        assert_eq!(accounts.len(), TIP_ACCOUNTS.len());
    }

    #[test]
    fn an_answer_with_a_result_is_accepted() {
        let ok = reqwest::StatusCode::OK;
        assert_eq!(
            parse_send_bundle(ok, &json!({ "jsonrpc": "2.0", "id": 1, "result": "abc" })),
            SenderSend::Accepted
        );
        assert!(matches!(
            parse_send_bundle(
                ok,
                &json!({ "error": { "code": -32602, "message": "no tip" } })
            ),
            SenderSend::Refused(_)
        ));
        assert!(matches!(
            parse_send_bundle(reqwest::StatusCode::TOO_MANY_REQUESTS, &json!({})),
            SenderSend::Refused(_)
        ));
    }

    #[test]
    fn the_rate_is_kept_without_waiting() {
        let sender = HeliusSender::new(Some("http://127.0.0.1:1/fast".to_string()), None);
        assert!(sender.try_admit());
        assert!(!sender.try_admit(), "one a second without a key");
    }

    #[test]
    fn errors_never_show_the_key() {
        let message = "error sending request for url (https://x/fast?api-key=secret)";
        assert!(!strip_key(message).contains("secret"));
        assert_eq!(
            ping_url("https://sender.helius-rpc.com/fast"),
            "https://sender.helius-rpc.com/ping"
        );
    }

    #[test]
    fn the_tip_transaction_pays_the_minimum_priority_fee() {
        let signer = Keypair::new();
        let tx = build_tip_transaction(&signer, Hash::new_unique(), MIN_TIP_LAMPORTS, 1).unwrap();
        assert_eq!(tx.message.instructions().len(), 3);
        assert!(
            u64::from(TIP_TX_CU_LIMIT) * bundle_cu_price(TIP_TX_CU_LIMIT, 1)
                >= crate::transaction_limits::MIN_BUNDLE_PRIORITY_FEE_LAMPORTS * 1_000_000
        );
    }
}
