//! Pyth push-feed prices: what the chain holds, as opposed to what eva prices with.
//!
//! eva writes the newest Pyth price it knows into the oracle cache, usually Hermes's, ahead of
//! the sponsored feed marginfi reads. A liquidation sized on such a price only passes once the feed
//! holds it, so the publish time each feed has on chain is kept here: a cached price newer than it
//! must be pushed before the liquidation.

use std::{collections::HashMap, sync::RwLock};

use anchor_lang::AccountDeserialize;
use pyth_solana_receiver_sdk::price_update::PriceUpdateV2;
use solana_account::Account;
use solana_pubkey::Pubkey;

/// The Pyth price update an account holds, if it is a receiver-owned `PriceUpdateV2`.
pub fn price_update(account: &Account) -> Option<PriceUpdateV2> {
    if account.owner != pyth_solana_receiver_sdk::ID {
        return None;
    }
    PriceUpdateV2::try_deserialize(&mut account.data.as_slice()).ok()
}

#[derive(Default)]
pub struct PythPrices {
    onchain: RwLock<HashMap<Pubkey, i64>>,
}

impl PythPrices {
    /// Records that the chain holds `oracle` at `publish_time`, unless a newer one is known.
    pub fn record_onchain(&self, oracle: Pubkey, publish_time: i64) {
        let mut onchain = self.onchain.write().unwrap_or_else(|e| e.into_inner());
        let known = onchain.entry(oracle).or_insert(publish_time);
        *known = (*known).max(publish_time);
    }

    /// The publish time the chain holds for `oracle`, if known.
    pub fn onchain(&self, oracle: &Pubkey) -> Option<i64> {
        self.onchain
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(oracle)
            .copied()
    }
}

/// A receiver-owned `PriceUpdateV2` account holding `price` for `feed_id`, published at
/// `publish_time`.
#[cfg(test)]
pub fn test_price_update_account(feed_id: [u8; 32], publish_time: i64, price: i64) -> Account {
    use anchor_lang::AccountSerialize;
    use pyth_solana_receiver_sdk::price_update::{PriceFeedMessage, VerificationLevel};
    let update = PriceUpdateV2 {
        write_authority: Default::default(),
        verification_level: VerificationLevel::Full,
        price_message: PriceFeedMessage {
            feed_id,
            price,
            conf: 1,
            exponent: -8,
            publish_time,
            prev_publish_time: publish_time - 1,
            ema_price: price,
            ema_conf: 1,
        },
        posted_slot: 0,
    };
    let mut data = Vec::new();
    update.try_serialize(&mut data).unwrap();
    Account {
        lamports: 1,
        data,
        owner: pyth_solana_receiver_sdk::ID,
        executable: false,
        rent_epoch: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_onchain_publish_time_never_goes_back() {
        let prices = PythPrices::default();
        let oracle = Pubkey::new_unique();
        assert_eq!(prices.onchain(&oracle), None);
        prices.record_onchain(oracle, 100);
        prices.record_onchain(oracle, 90);
        assert_eq!(prices.onchain(&oracle), Some(100));
        prices.record_onchain(oracle, 110);
        assert_eq!(prices.onchain(&oracle), Some(110));
    }
}
