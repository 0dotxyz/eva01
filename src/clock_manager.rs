use log::{debug, error, info};
use solana_client::rpc_client::RpcClient;
use solana_clock::Clock;
use solana_sdk_ids::sysvar;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const REFRESH_INTERVAL_SEC: u64 = 10;
const TICK_INTERVAL: Duration = Duration::from_secs(1);

pub struct ClockManager {
    rpc_client: RpcClient,
    clock: Arc<Mutex<Clock>>,
    refresh_interval: Duration,
}

impl ClockManager {
    pub fn new(clock: Arc<Mutex<Clock>>, rpc_url: String) -> anyhow::Result<Self> {
        info!("Initializing ClockManager");

        let rpc_client = RpcClient::new(rpc_url);
        let refresh_interval = Duration::from_secs(REFRESH_INTERVAL_SEC);

        Ok(Self {
            rpc_client,
            clock,
            refresh_interval,
        })
    }

    pub fn start(&mut self, stop_liquidator: Arc<AtomicBool>) {
        info!("Starting the ClockManager loop.");
        let mut last_refresh = Instant::now();

        while !stop_liquidator.load(Ordering::Relaxed) {
            std::thread::sleep(TICK_INTERVAL);

            if last_refresh.elapsed() < self.refresh_interval {
                if let Ok(mut clock_guard) = self.clock.lock() {
                    clock_guard.unix_timestamp += TICK_INTERVAL.as_secs() as i64;
                }
                continue;
            }

            debug!("Updating the Solana Clock...");
            match fetch_clock(&self.rpc_client) {
                Ok(clock) => {
                    match self.clock.lock() {
                        Ok(mut clock_guard) => {
                            // Oracle updates may have moved the timestamp past the fetched one.
                            let newest = clock_guard.unix_timestamp.max(clock.unix_timestamp);
                            *clock_guard = clock;
                            clock_guard.unix_timestamp = newest;
                        }
                        Err(err) => {
                            error!("Failed to lock the clock mutex for update: {:?}", err);
                        }
                    }
                    last_refresh = Instant::now();
                    debug!("Updated the Solana Clock: {:?}", self.clock);
                }
                Err(e) => {
                    error!("Failed to update the Solana Clock! {:?}", e);
                }
            }
        }
    }
}

pub fn fetch_clock(rpc_client: &RpcClient) -> anyhow::Result<Clock> {
    Ok(rpc_client
        .get_account(&sysvar::clock::id())?
        .deserialize_data()?)
}

/// Moves the clock forward to `timestamp` (e.g. an oracle's publish time) if it is newer.
pub fn advance_to(clock: &Arc<Mutex<Clock>>, timestamp: i64) -> anyhow::Result<()> {
    let mut clock_guard = clock
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to obtain the Clock lock: {:?}", e))?;
    clock_guard.unix_timestamp = clock_guard.unix_timestamp.max(timestamp);
    Ok(())
}

pub fn get_clock(clock: &Arc<Mutex<Clock>>) -> anyhow::Result<Clock> {
    let clock_guard = clock
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to obtain the Clock lock: {:?}", e))?;
    Ok(clock_guard.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_to_only_moves_forward() {
        let clock = Arc::new(Mutex::new(Clock {
            unix_timestamp: 1_000,
            ..Clock::default()
        }));

        advance_to(&clock, 1_010).unwrap();
        assert_eq!(get_clock(&clock).unwrap().unix_timestamp, 1_010);

        advance_to(&clock, 990).unwrap();
        assert_eq!(get_clock(&clock).unwrap().unix_timestamp, 1_010);
    }
}
