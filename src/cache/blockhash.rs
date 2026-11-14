//! The chain's newest blockhash, refreshed in the background, so a transaction built on a
//! liquidation's path never waits on the RPC for one.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock, RwLock,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use log::{info, warn};
use solana_client::rpc_client::RpcClient;
use solana_message::Hash;

use crate::utils::quick_rpc::QuickRpc;

/// How often the blockhash is refreshed. It stays valid for 150 blocks (~30 s), so a second's age
/// only slightly shortens how long a transaction can be re-sent.
pub const REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// A blockhash older than this isn't served (the refresh has stalled); the caller asks the RPC.
/// A blockhash lives 150 slots (~32 s at ~215 ms), so one this old still leaves ~12 s to land:
/// better than waiting on an RPC that isn't answering.
const MAX_AGE: Duration = Duration::from_secs(20);

/// How many recent blockhashes remember when they were fetched (a minute's worth at one a second).
const REMEMBERED: usize = 64;

/// How long a blockhash lands transactions: 150 blocks, ~32 s of slots at ~215 ms, a little more
/// with skipped slots.
pub const LIFETIME: Duration = Duration::from_secs(34);

#[derive(Default)]
pub struct RecentBlockhash {
    latest: RwLock<Option<(Hash, Instant)>>,
    /// The latest blockhashes and when each was fetched, newest last.
    fetched: std::sync::Mutex<std::collections::VecDeque<(Hash, Instant)>>,
    /// Fetches, the refresh's and a caller's: fails fast instead of the RPC client's retries.
    quick: OnceLock<std::sync::Arc<QuickRpc>>,
    /// When a caller's fetch last failed: callers don't ask again within `FAILED_FETCH_BACKOFF`.
    failed_at: std::sync::Mutex<Option<Instant>>,
}

/// A caller's fetch that failed isn't repeated by other callers for this long: an attempt makes
/// several, and each would wait the full timeout.
const FAILED_FETCH_BACKOFF: Duration = Duration::from_secs(1);

impl RecentBlockhash {
    pub fn record(&self, hash: Hash) {
        let now = Instant::now();
        *self.latest.write().unwrap_or_else(|e| e.into_inner()) = Some((hash, now));
        let mut fetched = self.fetched.lock().unwrap_or_else(|e| e.into_inner());
        if !fetched.iter().any(|(known, _)| *known == hash) {
            fetched.push_back((hash, now));
            if fetched.len() > REMEMBERED {
                fetched.pop_front();
            }
        }
    }

    /// Whether `hash` is past its lifetime by the time it was fetched, for when the RPC can't
    /// say. A hash not remembered is taken as `MAX_AGE` old at `fallback_from`.
    pub fn assumed_expired(&self, hash: &Hash, fallback_from: Instant) -> bool {
        let fetched = self.fetched.lock().unwrap_or_else(|e| e.into_inner());
        match fetched.iter().rev().find(|(known, _)| known == hash) {
            Some((_, at)) => at.elapsed() >= LIFETIME,
            None => fallback_from.elapsed() + MAX_AGE >= LIFETIME,
        }
    }

    /// The newest blockhash, if it was fetched within `MAX_AGE` of `now`.
    fn fresh(&self, now: Instant) -> Option<Hash> {
        let latest = *self.latest.read().unwrap_or_else(|e| e.into_inner());
        latest
            .filter(|(_, at)| now.saturating_duration_since(*at) <= MAX_AGE)
            .map(|(hash, _)| hash)
    }

    fn quick(&self, rpc_client: &RpcClient) -> &QuickRpc {
        self.quick
            .get_or_init(|| QuickRpc::shared(rpc_client.url()))
    }

    /// The newest blockhash, from the RPC only when the background refresh has none fresh.
    pub fn get(&self, rpc_client: &RpcClient) -> Result<Hash> {
        if let Some(hash) = self.fresh(Instant::now()) {
            return Ok(hash);
        }
        {
            let failed_at = self.failed_at.lock().unwrap_or_else(|e| e.into_inner());
            if failed_at.is_some_and(|at| at.elapsed() < FAILED_FETCH_BACKOFF) {
                return Err(anyhow::anyhow!(
                    "no recent blockhash: the RPC failed just now"
                ));
            }
        }
        match self.quick(rpc_client).latest_blockhash() {
            Ok(hash) => {
                self.record(hash);
                Ok(hash)
            }
            Err(e) => {
                *self.failed_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
                Err(e)
            }
        }
    }

    /// Keeps the blockhash fresh until `stop` is set.
    pub fn refresh(&self, rpc_client: &RpcClient, stop: &AtomicBool) {
        let mut failing = false;
        while !stop.load(Ordering::Relaxed) {
            match self.quick(rpc_client).latest_blockhash() {
                Ok(hash) => {
                    self.record(hash);
                    if failing {
                        info!("RecentBlockhash: refreshing again");
                        failing = false;
                    }
                }
                Err(e) => {
                    // Reported once per outage: transactions fetch their own meanwhile.
                    if !failing {
                        warn!("RecentBlockhash: refresh failed ({e}); fetching per transaction");
                        failing = true;
                    }
                }
            }
            thread::sleep(REFRESH_INTERVAL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blockhash_is_served_only_while_fresh() {
        let blockhash = RecentBlockhash::default();
        assert_eq!(blockhash.fresh(Instant::now()), None);

        let hash = Hash::new_unique();
        blockhash.record(hash);
        assert_eq!(blockhash.fresh(Instant::now()), Some(hash));
        assert_eq!(
            blockhash.fresh(Instant::now() + MAX_AGE + Duration::from_millis(1)),
            None
        );
    }

    #[test]
    fn expiry_is_judged_from_when_the_hash_was_fetched() {
        let blockhash = RecentBlockhash::default();
        let hash = Hash::new_unique();
        blockhash.record(hash);
        assert!(
            !blockhash.assumed_expired(&hash, Instant::now()),
            "just fetched"
        );
        // Recorded long ago.
        blockhash.fetched.lock().unwrap().back_mut().unwrap().1 = Instant::now() - LIFETIME;
        assert!(blockhash.assumed_expired(&hash, Instant::now()));
        // Unknown: taken as `MAX_AGE` old at the fallback time.
        let unknown = Hash::new_unique();
        assert!(!blockhash.assumed_expired(&unknown, Instant::now()));
        assert!(blockhash.assumed_expired(&unknown, Instant::now() - (LIFETIME - MAX_AGE)));
    }
}
