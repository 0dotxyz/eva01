//! Inventory-funded liquidation strategy.
//!
//! Funds each liquidation just-in-time: it buys the *shortfall* of the liability token (from the
//! swap_mint reserve, e.g. USDC) via a separate Jupiter **ExactIn** swap tx, then liquidates. The
//! seized collateral — and any liability bought in excess of what the repay consumes — is sold
//! back to swap_mint by the rebalancer on a later pass. No pre-held per-liability inventory is
//! required beyond the swap_mint reserve.
//!
//! ExactIn (rather than ExactOut) because many routes don't support ExactOut: we size the
//! swap_mint input from the liability's oracle price (+ a buffer), then verify the Jupiter quote
//! actually yields at least the shortfall, bumping the input once if it falls short.
//!
//! Produces an ordered plan `[buy?] [liquidate]`; the executor handles simulate-first cranking,
//! the Jito tip, and submission.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use fixed::types::I80F48;
use log::{debug, info};
use reqwest::{blocking::Client, header::HeaderMap};
use serde_json::{json, Value};
use solana_program::pubkey::Pubkey;
use solana_sdk::{signer::Signer, transaction::VersionedTransaction};

use crate::clock_manager;
use crate::wrappers::{
    liquidator_account::{LiquidatorAccount, PROFIT_SHARE},
    oracle::OracleWrapper,
    token_account::TokenAccountWrapper,
};

use super::{ExecutionPlan, LiquidationIntent, LiquidationStrategy};

/// Extra margin over the oracle-derived input estimate, to absorb the oracle/market price gap and
/// rounding so the first ExactIn quote usually already covers the shortfall (the quote is still
/// verified, and the input bumped, if it doesn't).
const INPUT_BUFFER: f64 = 1.02;

pub struct InventoryStrategy {
    liquidator_account: Arc<LiquidatorAccount>,
    http: Client,
    jup_swap_api_url: String,
    swap_mint: Pubkey,
    slippage_bps: u16,
}

impl InventoryStrategy {
    pub fn new(
        liquidator_account: Arc<LiquidatorAccount>,
        swap_mint: Pubkey,
        jup_swap_api_url: String,
        jupiter_api_key: String,
        slippage_bps: u16,
    ) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            jupiter_api_key
                .parse()
                .map_err(|e| anyhow!("Invalid Jupiter API key header: {e}"))?,
        );
        let http = Client::builder().default_headers(headers).build()?;

        Ok(Self {
            liquidator_account,
            http,
            jup_swap_api_url,
            swap_mint,
            slippage_bps,
        })
    }

    /// A price-only token wrapper (balance 0) for `mint`, used solely for the oracle-price <-> token
    /// amount conversions when sizing the swap input. Avoids depending on a wallet token account for
    /// a liability we may hold none of.
    fn value_wrapper(&self, mint: &Pubkey) -> Result<TokenAccountWrapper<OracleWrapper>> {
        let cache = &self.liquidator_account.cache;
        let bank_address = cache.banks.try_get_account_for_mint(mint)?;
        let bank_wrapper = cache.banks.try_get_bank(&bank_address)?;
        let clock = clock_manager::get_clock(&cache.clock)?;
        let oracle_wrapper = OracleWrapper::build_lenient(cache, &clock, &bank_address)?;
        Ok(TokenAccountWrapper {
            balance: 0,
            bank_wrapper,
            oracle_wrapper,
        })
    }

    /// Build a signed `swap_mint -> output_mint` **ExactIn** buy tx that yields at least `min_out`
    /// of `output_mint`. Sizes the input from oracle prices, verifies against the Jupiter quote,
    /// and bumps the input once if the quote falls short. Any overshoot is sold back later.
    fn build_buy_tx(&self, output_mint: Pubkey, min_out: u64) -> Result<VersionedTransaction> {
        // Size the swap_mint input from oracle prices: value(min_out of liab) -> swap_mint amount.
        let needed_value = self
            .value_wrapper(&output_mint)?
            .get_value_for_amount(I80F48::from_num(min_out))?;
        let swap_wrapper = self.value_wrapper(&self.swap_mint)?;
        let buffered_value = needed_value
            .checked_mul(I80F48::from_num(INPUT_BUFFER))
            .ok_or_else(|| anyhow!("input sizing overflow"))?;
        let mut input_amount = swap_wrapper
            .get_amount_from_value(buffered_value)?
            .to_num::<u64>();
        if input_amount == 0 {
            return Err(anyhow!(
                "computed zero swap_mint input for buying {} {}",
                min_out,
                output_mint
            ));
        }

        let mut quote = self.quote_exact_in(self.swap_mint, output_mint, input_amount)?;
        let mut out = quote_out_amount(&quote)?;
        if out < min_out {
            // The oracle-sized input under-delivered; scale up by the deficit ratio (+ margin) and
            // re-quote once before giving up.
            let ratio = (min_out as f64 / out.max(1) as f64) * 1.02;
            input_amount = ((input_amount as f64) * ratio).ceil() as u64;
            quote = self.quote_exact_in(self.swap_mint, output_mint, input_amount)?;
            out = quote_out_amount(&quote)?;
            if out < min_out {
                return Err(anyhow!(
                    "Jupiter ExactIn quote still short after re-quote: out {out} < needed {min_out}"
                ));
            }
        }
        debug!(
            "InventoryStrategy: ExactIn spends {} {} for {} {} (need {})",
            input_amount, self.swap_mint, out, output_mint, min_out
        );

        let swap_tx_b64 = self.request_swap_tx(&quote)?;
        let bytes = BASE64
            .decode(swap_tx_b64.as_bytes())
            .map_err(|e| anyhow!("Failed to base64-decode Jupiter swap tx: {e}"))?;
        let unsigned: VersionedTransaction = bincode::deserialize(&bytes)
            .map_err(|e| anyhow!("Failed to deserialize Jupiter swap tx: {e}"))?;
        // Jupiter builds the tx for our wallet; sign it.
        let signed =
            VersionedTransaction::try_new(unsigned.message, &[&self.liquidator_account.signer])?;
        Ok(signed)
    }

    /// GET `{base}/quote` for an ExactIn swap; returns the raw quote JSON (fed back into `/swap`).
    fn quote_exact_in(
        &self,
        input_mint: Pubkey,
        output_mint: Pubkey,
        amount: u64,
    ) -> Result<Value> {
        let resp = self
            .http
            .get(format!("{}/quote", self.jup_swap_api_url))
            .query(&[
                ("inputMint", input_mint.to_string()),
                ("outputMint", output_mint.to_string()),
                ("amount", amount.to_string()),
                ("swapMode", "ExactIn".to_string()),
                ("slippageBps", self.slippage_bps.to_string()),
            ])
            .send()?;
        json_or_err(resp, "Jupiter quote")
    }

    /// POST `{base}/swap` with the quote; returns the base64 `swapTransaction`.
    fn request_swap_tx(&self, quote: &Value) -> Result<String> {
        let body = json!({
            "userPublicKey": self.liquidator_account.signer.pubkey().to_string(),
            "quoteResponse": quote,
            // Keep WSOL as a plain SPL token in the ATA; never auto-wrap/unwrap.
            "wrapAndUnwrapSol": false,
        });
        let resp = self
            .http
            .post(format!("{}/swap", self.jup_swap_api_url))
            .json(&body)
            .send()?;
        let v = json_or_err(resp, "Jupiter swap")?;
        v.get("swapTransaction")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Jupiter swap response missing swapTransaction: {v}"))
    }
}

/// Amount of the liability token to buy: the part of `repay_amount` not already in the wallet.
/// Uses the same `u64` truncation as the repay ix so the buy lands exactly what repay consumes.
fn buy_shortfall(repay_amount: I80F48, wallet_balance: u64) -> u64 {
    repay_amount.to_num::<u64>().saturating_sub(wallet_balance)
}

/// Parse the (stringified) `outAmount` from a Jupiter quote response.
fn quote_out_amount(quote: &Value) -> Result<u64> {
    quote
        .get("outAmount")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Jupiter quote missing outAmount: {quote}"))?
        .parse::<u64>()
        .map_err(|e| anyhow!("Jupiter quote outAmount not a u64: {e}"))
}

/// Read a reqwest response as JSON, surfacing the body on a non-2xx status (so route errors like
/// `NO_ROUTES_FOUND` propagate to the executor's quarantine).
fn json_or_err(resp: reqwest::blocking::Response, ctx: &str) -> Result<Value> {
    let status = resp.status();
    let text = resp.text()?;
    if !status.is_success() {
        return Err(anyhow!("{ctx} failed ({status}): {text}"));
    }
    serde_json::from_str(&text).map_err(|e| anyhow!("{ctx}: invalid JSON ({e}): {text}"))
}

impl LiquidationStrategy for InventoryStrategy {
    fn name(&self) -> &'static str {
        "inventory"
    }

    fn assemble(&self, intent: &LiquidationIntent) -> Result<Option<ExecutionPlan>> {
        let liab_bank = self
            .liquidator_account
            .cache
            .banks
            .try_get_bank(&intent.liab_bank)?;
        let liab_mint = liab_bank.bank.mint;

        // JIT-buy covers the liability, so we always liquidate the full amount (no proportional
        // reduction). The repay ix consumes `repay_amount` of the liab token.
        let asset_amount = intent.asset_amount;
        let repay_amount = intent
            .liab_amount
            .checked_mul(I80F48::from_num(1.0 - PROFIT_SHARE))
            .ok_or_else(|| anyhow!("repay amount overflow"))?;

        let wallet_liab = self
            .liquidator_account
            .get_token_balance_for_mint(&liab_mint)
            .unwrap_or(0);
        let shortfall = buy_shortfall(repay_amount, wallet_liab);

        let mut txs = Vec::new();
        let mut temp_luts = Vec::new();
        if shortfall > 0 {
            info!(
                "InventoryStrategy: buying shortfall of {} {} (have {}, need {})",
                shortfall,
                liab_mint,
                wallet_liab,
                repay_amount.to_num::<u64>()
            );
            txs.push(self.build_buy_tx(liab_mint, shortfall)?);
        } else {
            debug!(
                "InventoryStrategy: wallet already covers {} {}; no buy needed",
                repay_amount.to_num::<u64>(),
                liab_mint
            );
        }

        let (liquidate_tx, _ixs, temp_lut) =
            self.liquidator_account
                .build_liquidate_tx(intent, asset_amount, repay_amount)?;
        txs.push(liquidate_tx);
        if let Some(lut_key) = temp_lut {
            temp_luts.push(lut_key);
        }

        Ok(Some(ExecutionPlan { txs, temp_luts }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixed_macro::types::I80F48;

    #[test]
    fn test_buy_shortfall_partial() {
        assert_eq!(buy_shortfall(I80F48!(1000), 600), 400);
    }

    #[test]
    fn test_buy_shortfall_covered() {
        assert_eq!(buy_shortfall(I80F48!(1000), 1000), 0);
        assert_eq!(buy_shortfall(I80F48!(1000), 1500), 0);
    }

    #[test]
    fn test_buy_shortfall_empty_wallet() {
        assert_eq!(buy_shortfall(I80F48!(1000), 0), 1000);
    }

    #[test]
    fn test_buy_shortfall_truncates_like_repay() {
        // repay ix uses to_num::<u64>() (truncation); the shortfall must match it.
        assert_eq!(buy_shortfall(I80F48!(1000.9), 0), 1000);
    }

    #[test]
    fn test_quote_out_amount_parses_string() {
        let q = json!({ "outAmount": "1908764533" });
        assert_eq!(quote_out_amount(&q).unwrap(), 1908764533);
    }

    #[test]
    fn test_quote_out_amount_missing() {
        let q = json!({ "inAmount": "100" });
        assert!(quote_out_amount(&q).is_err());
    }
}
