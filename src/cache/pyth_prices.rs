//! Pyth push-feed prices: what the chain holds, as opposed to what eva prices with.
//!
//! eva writes the newest Pyth price it knows into the oracle cache, usually Hermes's, ahead of
//! the sponsored feed marginfi reads. A liquidation sized on such a price only passes once the feed
//! holds it, so the price account each feed has on chain is kept here: eva evaluates an account on
//! both, to know before building anything whether a liquidation needs a push. The last few signed
//! updates behind each streamed price are kept too, so a push is built from them, with no request
//! to Hermes.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
};

use anchor_lang::{AccountDeserialize, AccountSerialize};
use pyth_solana_receiver_sdk::price_update::{PriceFeedMessage, PriceUpdateV2};
use pythnet_sdk::wire::v1::MerklePriceUpdate;
use solana_account::Account;
use solana_pubkey::Pubkey;

/// The Pyth price update an account holds, if it is a receiver-owned `PriceUpdateV2`.
pub fn price_update(account: &Account) -> Option<PriceUpdateV2> {
    if account.owner != pyth_solana_receiver_sdk::ID {
        return None;
    }
    PriceUpdateV2::try_deserialize(&mut account.data.as_slice()).ok()
}

/// A feed's price as Hermes signed it: the VAA carrying the merkle root, the feed's update proven
/// against it, and the price message it carries. Feeds published in the same Pythnet slot share
/// one VAA.
#[derive(Clone)]
pub struct SignedUpdate {
    pub message: PriceFeedMessage,
    pub vaa: Arc<Vec<u8>>,
    pub update: MerklePriceUpdate,
}

impl SignedUpdate {
    pub fn publish_time(&self) -> i64 {
        self.message.publish_time
    }
}

/// What one push posts: one VAA and, per oracle, the feed's update under it. A liquidation behind
/// the push is sized on exactly these prices.
#[derive(Clone)]
pub struct PushedPrices {
    pub vaa: Arc<Vec<u8>>,
    /// The oracles pushed and their updates, in push order.
    pub feeds: Vec<(Pubkey, SignedUpdate)>,
}

impl PushedPrices {
    /// The account the program reads at `oracle` once the push is posted: `cached` carrying the
    /// pushed price message. `None` when `oracle` isn't pushed.
    pub fn account(&self, oracle: &Pubkey, cached: &Account) -> Option<Account> {
        let (_, signed) = self.feeds.iter().find(|(pushed, _)| pushed == oracle)?;
        let mut update = price_update(cached)?;
        update.price_message = signed.message;
        let mut data = Vec::with_capacity(cached.data.len());
        update.try_serialize(&mut data).ok()?;
        data.resize(cached.data.len(), 0);
        Some(Account {
            data,
            ..cached.clone()
        })
    }
}

/// The prices an account is valued on.
#[derive(Clone, Copy)]
pub enum Pricing<'a> {
    /// The newest eva knows: Hermes's, for the feeds it streams.
    Newest,
    /// The program's as the chain stands: every Pyth feed eva prices ahead of it at its price.
    Chain,
    /// The program's once this push is posted; every other feed at the chain's price.
    Pushed(&'a PushedPrices),
}

impl Pricing<'_> {
    /// `cached`, eva's account for `oracle`, as this pricing reads it.
    pub fn account(&self, prices: &PythPrices, oracle: &Pubkey, cached: Account) -> Account {
        match self {
            Self::Newest => cached,
            Self::Pushed(push) => match push.account(oracle, &cached) {
                Some(pushed) => pushed,
                None => prices.onchain_if_behind(oracle, &cached).unwrap_or(cached),
            },
            Self::Chain => prices.onchain_if_behind(oracle, &cached).unwrap_or(cached),
        }
    }
}

/// Signed updates kept per feed: about 10 s, at one Hermes event a second. An event carries only
/// the feeds that published in it, so pushing a quiet feed with an active one needs an older VAA
/// both are under.
const SIGNED_HISTORY: usize = 10;

#[derive(Default)]
pub struct PythPrices {
    /// The price account each feed holds on chain, with its publish time.
    onchain: RwLock<HashMap<Pubkey, (i64, Account)>>,
    /// The last signed updates streamed for each feed id, oldest first.
    signed: RwLock<HashMap<[u8; 32], VecDeque<SignedUpdate>>>,
}

impl PythPrices {
    /// Records that the chain holds `account` for `oracle`, unless a newer one is known. Accounts
    /// that are not Pyth price updates are ignored.
    pub fn record_onchain(&self, oracle: Pubkey, account: &Account) {
        let Some(update) = price_update(account) else {
            return;
        };
        let publish_time = update.price_message.publish_time;
        let mut onchain = self.onchain.write().unwrap_or_else(|e| e.into_inner());
        if onchain
            .get(&oracle)
            .is_none_or(|(known, _)| publish_time >= *known)
        {
            onchain.insert(oracle, (publish_time, account.clone()));
        }
    }

    /// The publish time the chain holds for `oracle`, if known.
    pub fn onchain(&self, oracle: &Pubkey) -> Option<i64> {
        self.onchain
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(oracle)
            .map(|(publish_time, _)| *publish_time)
    }

    /// The account the chain holds for `oracle`, when it is older than `cached`, the price eva
    /// prices with; `None` when the chain is level or the feed is unknown.
    pub fn onchain_if_behind(&self, oracle: &Pubkey, cached: &Account) -> Option<Account> {
        let cached = price_update(cached)?.price_message.publish_time;
        let onchain = self.onchain.read().unwrap_or_else(|e| e.into_inner());
        match onchain.get(oracle) {
            Some((publish_time, account)) if *publish_time < cached => Some(account.clone()),
            _ => None,
        }
    }

    /// Records a signed update streamed for `feed_id`, ignoring one already known for its publish
    /// time (each stream delivers every event).
    pub fn record_signed(&self, feed_id: [u8; 32], signed: SignedUpdate) {
        let mut all = self.signed.write().unwrap_or_else(|e| e.into_inner());
        let history = all.entry(feed_id).or_default();
        let at = history.partition_point(|known| known.publish_time() < signed.publish_time());
        if history
            .get(at)
            .is_some_and(|known| known.publish_time() == signed.publish_time())
        {
            return;
        }
        history.insert(at, signed);
        if history.len() > SIGNED_HISTORY {
            history.pop_front();
        }
    }

    /// The push of `feeds` (each an oracle and its feed id) at the newest prices one VAA carries for
    /// all of them, each still ahead of the chain. `None` when no VAA covers them all.
    pub fn push_of(&self, feeds: &[(Pubkey, [u8; 32])]) -> Option<PushedPrices> {
        let ids: Vec<[u8; 32]> = feeds.iter().map(|(_, feed_id)| *feed_id).collect();
        let (vaa, updates) = self.signed_together(&ids, |feed_id, publish_time| {
            feeds
                .iter()
                .find(|(_, id)| id == feed_id)
                .is_some_and(|(oracle, _)| {
                    self.onchain(oracle)
                        .is_none_or(|chain| publish_time > chain)
                })
        })?;
        Some(PushedPrices {
            vaa,
            feeds: feeds
                .iter()
                .map(|(oracle, _)| *oracle)
                .zip(updates)
                .collect(),
        })
    }

    /// The newest VAA under which every feed in `feed_ids` has an update `ahead` accepts (given the
    /// feed and the update's publish time), with those updates in order. A push carries one VAA.
    /// `None` when no VAA covers them all.
    pub fn signed_together(
        &self,
        feed_ids: &[[u8; 32]],
        ahead: impl Fn(&[u8; 32], i64) -> bool,
    ) -> Option<(Arc<Vec<u8>>, Vec<SignedUpdate>)> {
        let all = self.signed.read().unwrap_or_else(|e| e.into_inner());
        let (first, rest) = feed_ids.split_first()?;
        let candidates = all
            .get(first)?
            .iter()
            .rev()
            .filter(|signed| ahead(first, signed.publish_time()));
        'vaa: for candidate in candidates {
            let mut updates = Vec::with_capacity(feed_ids.len());
            updates.push(candidate.clone());
            for feed_id in rest {
                let Some(signed) = all.get(feed_id).and_then(|history| {
                    history.iter().rev().find(|signed| {
                        same_vaa(&signed.vaa, &candidate.vaa)
                            && ahead(feed_id, signed.publish_time())
                    })
                }) else {
                    continue 'vaa;
                };
                updates.push(signed.clone());
            }
            return Some((Arc::clone(&candidate.vaa), updates));
        }
        None
    }
}

/// Each stream decodes its own copy of an event, so equal VAAs need not share an allocation.
fn same_vaa(a: &Arc<Vec<u8>>, b: &Arc<Vec<u8>>) -> bool {
    Arc::ptr_eq(a, b) || a.as_slice() == b.as_slice()
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
    fn the_onchain_price_never_goes_back() {
        let prices = PythPrices::default();
        let oracle = Pubkey::new_unique();
        let feed = [1u8; 32];
        assert_eq!(prices.onchain(&oracle), None);
        prices.record_onchain(oracle, &test_price_update_account(feed, 100, 5));
        prices.record_onchain(oracle, &test_price_update_account(feed, 90, 6));
        assert_eq!(prices.onchain(&oracle), Some(100));
        prices.record_onchain(oracle, &test_price_update_account(feed, 110, 7));
        assert_eq!(prices.onchain(&oracle), Some(110));
    }

    #[test]
    fn the_chain_price_is_returned_only_when_eva_prices_ahead_of_it() {
        let prices = PythPrices::default();
        let oracle = Pubkey::new_unique();
        let feed = [1u8; 32];
        let chain = test_price_update_account(feed, 100, 5);
        prices.record_onchain(oracle, &chain);

        let ahead = test_price_update_account(feed, 101, 9);
        let behind = prices.onchain_if_behind(&oracle, &ahead).unwrap();
        assert_eq!(price_update(&behind).unwrap().price_message.price, 5);
        assert!(prices.onchain_if_behind(&oracle, &chain).is_none());
        assert!(prices
            .onchain_if_behind(&Pubkey::new_unique(), &ahead)
            .is_none());
    }

    fn message(publish_time: i64, price: i64) -> PriceFeedMessage {
        PriceFeedMessage {
            feed_id: [0u8; 32],
            price,
            conf: 1,
            exponent: -8,
            publish_time,
            prev_publish_time: publish_time - 1,
            ema_price: price,
            ema_conf: 1,
        }
    }

    fn signed(publish_time: i64, vaa: &Arc<Vec<u8>>, tag: u8) -> SignedUpdate {
        SignedUpdate {
            message: message(publish_time, i64::from(tag)),
            vaa: Arc::clone(vaa),
            update: MerklePriceUpdate {
                message: vec![tag].into(),
                proof: pythnet_sdk::accumulators::merkle::MerklePath::new(vec![]),
            },
        }
    }

    fn any(_: &[u8; 32], _: i64) -> bool {
        true
    }

    fn tags(updates: &[SignedUpdate]) -> Vec<u8> {
        updates
            .iter()
            .map(|u| u.update.message.as_ref()[0])
            .collect()
    }

    #[test]
    fn feeds_push_from_the_stream_under_the_newest_vaa_they_share() {
        let prices = PythPrices::default();
        let (sol, thin, other) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let (k, k1, k2) = (
            Arc::new(vec![1u8; 3]),
            Arc::new(vec![2u8; 3]),
            Arc::new(vec![3u8; 3]),
        );
        // SOL publishes every second; the thin feed last did at k.
        prices.record_signed(sol, signed(100, &k, 10));
        prices.record_signed(thin, signed(100, &k, 20));
        prices.record_signed(sol, signed(101, &k1, 11));
        prices.record_signed(sol, signed(102, &k2, 12));

        let (vaa, updates) = prices.signed_together(&[thin, sol], any).unwrap();
        assert_eq!(
            vaa.as_slice(),
            k.as_slice(),
            "the newest VAA both are under"
        );
        assert_eq!(tags(&updates), vec![20, 10], "in the order asked");
        let (vaa, updates) = prices.signed_together(&[sol], any).unwrap();
        assert_eq!((vaa.as_slice(), tags(&updates)), (k2.as_slice(), vec![12]));

        // Only updates still ahead of the chain count: with SOL's chain price at 100, k is out.
        let after_100 = |feed: &[u8; 32], t: i64| feed != &sol || t > 100;
        assert!(prices.signed_together(&[sol, thin], after_100).is_none());

        // The same event from the second stream changes nothing, nor does a late older one.
        prices.record_signed(sol, signed(102, &Arc::new(vec![3u8; 3]), 99));
        prices.record_signed(sol, signed(99, &k, 9));
        assert_eq!(
            tags(&prices.signed_together(&[sol], any).unwrap().1),
            vec![12]
        );
        assert!(prices.signed_together(&[other], any).is_none());
        assert!(prices.signed_together(&[], any).is_none());
    }

    #[test]
    fn a_push_carries_and_prices_exactly_the_updates_under_one_vaa() {
        let prices = PythPrices::default();
        let (sol, thin) = (Pubkey::new_unique(), Pubkey::new_unique());
        let (sol_id, thin_id) = ([1u8; 32], [2u8; 32]);
        let (k, k1) = (Arc::new(vec![1u8; 3]), Arc::new(vec![2u8; 3]));
        prices.record_onchain(sol, &test_price_update_account(sol_id, 99, 1));
        prices.record_onchain(thin, &test_price_update_account(thin_id, 99, 1));
        prices.record_signed(sol_id, signed(100, &k, 10));
        prices.record_signed(thin_id, signed(100, &k, 20));
        prices.record_signed(sol_id, signed(101, &k1, 11));

        let push = prices.push_of(&[(sol, sol_id), (thin, thin_id)]).unwrap();
        assert_eq!(push.vaa.as_slice(), k.as_slice());
        let oracles: Vec<Pubkey> = push.feeds.iter().map(|(oracle, _)| *oracle).collect();
        assert_eq!(oracles, vec![sol, thin]);
        // SOL is read at its price under that VAA, not at its newest.
        let cached = test_price_update_account(sol_id, 101, 11);
        let read = |pricing: Pricing| {
            price_update(&pricing.account(&prices, &sol, cached.clone()))
                .unwrap()
                .price_message
        };
        let pushed = read(Pricing::Pushed(&push));
        assert_eq!((pushed.price, pushed.publish_time), (10, 100));
        assert_eq!(read(Pricing::Newest).price, 11);
        assert_eq!(read(Pricing::Chain).price, 1);

        // An update the chain already holds isn't pushed.
        prices.record_onchain(thin, &test_price_update_account(thin_id, 100, 20));
        assert!(prices.push_of(&[(sol, sol_id), (thin, thin_id)]).is_none());
        let alone = prices.push_of(&[(sol, sol_id)]).unwrap();
        assert_eq!(alone.vaa.as_slice(), k1.as_slice());
    }

    #[test]
    fn only_the_last_signed_updates_are_kept() {
        let prices = PythPrices::default();
        let feed = [1u8; 32];
        for t in 0..(SIGNED_HISTORY as i64 + 3) {
            prices.record_signed(feed, signed(t, &Arc::new(vec![t as u8]), t as u8));
        }
        let history = &prices.signed.read().unwrap()[&feed];
        assert_eq!(history.len(), SIGNED_HISTORY);
        assert_eq!(history.front().unwrap().publish_time(), 3);
    }
}
