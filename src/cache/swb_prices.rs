use anyhow::{anyhow, Result};
use fixed::types::I80F48;
use solana_sdk::pubkey::Pubkey;
use std::{collections::HashMap, sync::RwLock};

#[derive(Clone, Debug)]
pub struct SwbPrice {
    pub price_realtime: I80F48,
    pub conf_realtime: I80F48,
    pub price_weighted: I80F48,
    pub conf_weighted: I80F48,
}

#[derive(Default)]
pub struct SwbPricesCache {
    prices: RwLock<HashMap<Pubkey, SwbPrice>>,
}

impl SwbPricesCache {
    pub fn upsert(&self, bank_address: Pubkey, price: SwbPrice) -> Result<()> {
        self.prices
            .write()
            .map_err(|e| anyhow!("Failed to lock SwbPricesCache for write: {}", e))?
            .insert(bank_address, price);
        Ok(())
    }

    pub fn get(&self, bank_address: &Pubkey) -> Option<SwbPrice> {
        self.prices.read().ok()?.get(bank_address).cloned()
    }
}
