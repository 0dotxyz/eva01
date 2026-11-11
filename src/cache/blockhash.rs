//! The chain's newest blockhash, refreshed in the background, so a transaction built on a
//! liquidation's path never waits on the RPC for one.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use log::{info, warn};
use solana_client::rpc_client::RpcClient;
use solana_message::Hash;

/// How often the blockhash is refreshed. It stays valid for 150 blocks (~30 s), so a second's age
/// only slightly shortens how long a transaction can be re-sent.
pub const REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// A blockhash older than this isn't served (the refresh has stalled); the caller asks the RPC.
const MAX_AGE: Duration = Duration::from_secs(5);

#[derive(Default)]
pub struct RecentBlockhash {
    latest: RwLock<Option<(Hash, Instant)>>,
}

impl RecentBlockhash {
    pub fn record(&self, hash: Hash) {
        *self.latest.write().unwrap_or_else(|e| e.into_inner()) = Some((hash, Instant::now()));
    }

    /// The newest blockhash, if it was fetched within `MAX_AGE` of `now`.
    fn fresh(&self, now: Instant) -> Option<Hash> {
        let latest = *self.latest.read().unwrap_or_else(|e| e.into_inner());
        latest
            .filter(|(_, at)| now.saturating_duration_since(*at) <= MAX_AGE)
            .map(|(hash, _)| hash)
    }

    /// The newest blockhash, from the RPC only when the background refresh has none fresh.
    pub fn get(&self, rpc_client: &RpcClient) -> Result<Hash> {
        if let Some(hash) = self.fresh(Instant::now()) {
            return Ok(hash);
        }
        let hash = rpc_client.get_latest_blockhash()?;
        self.record(hash);
        Ok(hash)
    }

    /// Keeps the blockhash fresh until `stop` is set.
    pub fn refresh(&self, rpc_client: &RpcClient, stop: &AtomicBool) {
        let mut failing = false;
        while !stop.load(Ordering::Relaxed) {
            match rpc_client.get_latest_blockhash() {
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
}
