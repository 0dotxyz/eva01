use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use anyhow::Result;
use log::{info, warn};
use solana_account::Account;
use solana_client::rpc_client::RpcClient;
use solana_pubkey::Pubkey;

use crate::{
    cache::Cache,
    geyser::{AccountType, GeyserUpdate},
};
use crossbeam::channel::Sender;

const FETCH_INTERVAL: Duration = Duration::from_secs(1800);

// Kamino MinimalReserve: 8-byte discriminator + 8-byte version, then slot:u64 at [16..24]
const KAMINO_SLOT_OFFSET: usize = 16;
// Juplend Lending: 8-byte discriminator + packed fields; last_update_timestamp:u64 at [123..131]
const JUPLEND_TIMESTAMP_OFFSET: usize = 123;

pub struct IntegrationAccountFetcher {
    rpc_client: RpcClient,
    cache: Arc<Cache>,
    geyser_tx: Sender<GeyserUpdate>,
    stop: Arc<AtomicBool>,
}

impl IntegrationAccountFetcher {
    pub fn new(
        rpc_url: String,
        cache: Arc<Cache>,
        geyser_tx: Sender<GeyserUpdate>,
        stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            rpc_client: RpcClient::new(rpc_url),
            cache,
            geyser_tx,
            stop,
        }
    }

    /// These accounts are excluded from the Geyser subscription, so the liquidator never sees them
    /// change. Publish every write into the same channel, otherwise a reserve / lending state
    /// refresh queues no accounts for the liquidatability check.
    fn publish(&self, address: Pubkey, account: Account) {
        if let Err(e) = self.geyser_tx.send(GeyserUpdate {
            account_type: AccountType::Oracle,
            address,
            account,
        }) {
            warn!("IntegrationAccountFetcher: failed to publish the {address} update: {e}");
        }
    }

    pub fn start(&self) {
        info!("IntegrationAccountFetcher starting.");
        loop {
            if let Err(e) = self.fetch_and_update() {
                warn!("IntegrationAccountFetcher: fetch cycle failed: {e}");
            }
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(FETCH_INTERVAL);
        }
        info!("IntegrationAccountFetcher stopped.");
    }

    pub fn fetch_and_update(&self) -> Result<()> {
        // Publish only once the whole cycle is written: a scan triggered halfway through would
        // still price against the not-yet-patched half and fail on its staleness check.
        let mut updated: Vec<(Pubkey, Account)> = vec![];
        let kamino = self.refresh_kamino(&mut updated);
        let juplend = self.refresh_juplend(&mut updated);

        for (address, account) in updated {
            self.publish(address, account);
        }

        kamino.and(juplend)
    }

    fn refresh_kamino(&self, updated: &mut Vec<(Pubkey, Account)>) -> Result<()> {
        let addresses: Vec<Pubkey> = self.cache.banks.get_kamino_reserves().into_iter().collect();
        if addresses.is_empty() {
            return Ok(());
        }
        let accounts = self.rpc_client.get_multiple_accounts(&addresses)?;
        let mut count = 0usize;
        for (addr, maybe_acct) in addresses.iter().zip(accounts) {
            let Some(mut acct) = maybe_acct else { continue };
            if acct.data.len() >= KAMINO_SLOT_OFFSET + 8 {
                // Write u64::MAX so reserve.slot < current_slot is always false
                acct.data[KAMINO_SLOT_OFFSET..KAMINO_SLOT_OFFSET + 8]
                    .copy_from_slice(&u64::MAX.to_le_bytes());
            }
            if let Err(e) = self.cache.oracles.try_update(addr, acct.clone()) {
                warn!("IntegrationAccountFetcher: kamino update failed for {addr}: {e}");
            } else {
                updated.push((*addr, acct));
                count += 1;
            }
        }
        info!("IntegrationAccountFetcher: refreshed {count} Kamino reserves.");
        Ok(())
    }

    fn refresh_juplend(&self, updated: &mut Vec<(Pubkey, Account)>) -> Result<()> {
        let addresses: Vec<Pubkey> = self
            .cache
            .banks
            .get_juplend_lending_states()
            .into_iter()
            .collect();
        if addresses.is_empty() {
            return Ok(());
        }
        let accounts = self.rpc_client.get_multiple_accounts(&addresses)?;
        let mut count = 0usize;
        for (addr, maybe_acct) in addresses.iter().zip(accounts) {
            let Some(mut acct) = maybe_acct else { continue };
            if acct.data.len() >= JUPLEND_TIMESTAMP_OFFSET + 8 {
                // Write i64::MAX as u64 so (last_update_timestamp as i64) < current_timestamp is always false
                acct.data[JUPLEND_TIMESTAMP_OFFSET..JUPLEND_TIMESTAMP_OFFSET + 8]
                    .copy_from_slice(&(i64::MAX as u64).to_le_bytes());
            }
            if let Err(e) = self.cache.oracles.try_update(addr, acct.clone()) {
                warn!("IntegrationAccountFetcher: juplend update failed for {addr}: {e}");
            } else {
                updated.push((*addr, acct));
                count += 1;
            }
        }
        info!("IntegrationAccountFetcher: refreshed {count} Juplend lending states.");
        Ok(())
    }
}
