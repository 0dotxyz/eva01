use crate::{
    config::Eva01Config,
    utils::{account_update_to_account, decode_bank},
    ward,
};
use anchor_lang::AccountDeserialize;
use anyhow::Result;
use crossbeam::channel::Sender;
use futures::StreamExt;
use log::{error, info, warn};
use marginfi_type_crate::types::MarginfiAccount;
use solana_account::Account;
use solana_pubkey::Pubkey;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::runtime::{Builder, Runtime};
use yellowstone_grpc_client::{ClientTlsConfig, GeyserGrpcClient};
use yellowstone_grpc_proto::prelude::*;

const RATE_LIMIT_LOG_INTERVAL_SECS: u64 = 60;
const INITIAL_RECONNECT_BACKOFF: Duration = Duration::from_secs(1);
const MAX_RECONNECT_BACKOFF: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct GeyserUpdate {
    pub account_type: AccountType,
    pub address: Pubkey,
    pub account: Account,
}

#[derive(Clone, Copy, Debug)]
pub enum AccountType {
    Oracle,
    Marginfi,
    Bank,
    Token,
}

/// Rate-limited logger that logs messages at most once per interval.
struct RateLimitedLogger {
    last_logged_at: Mutex<Option<Instant>>,
}

impl RateLimitedLogger {
    fn new() -> Self {
        Self {
            last_logged_at: Mutex::new(None),
        }
    }

    fn warn(&self, message: &str) {
        let now = Instant::now();
        let mut last_logged_at = self.last_logged_at.lock().unwrap();
        let should_log = match *last_logged_at {
            Some(last_logged_at) => {
                now.duration_since(last_logged_at)
                    >= Duration::from_secs(RATE_LIMIT_LOG_INTERVAL_SECS)
            }
            None => true,
        };

        if should_log {
            warn!("{}", message);
            *last_logged_at = Some(now);
        }
    }
}

/// Geyser service is responsible for receiving and distributing the
/// messages to the other services.
pub struct GeyserService {
    endpoint: String,
    x_token: Option<String>,
    tracked_accounts: HashMap<Pubkey, AccountType>,
    marginfi_group_pk: Pubkey,
    geyser_tx: Sender<GeyserUpdate>,
    tokio_rt: Runtime,
    stop: Arc<AtomicBool>,
    error_logger: RateLimitedLogger,
}

impl GeyserService {
    pub fn new(
        config: Eva01Config,
        tracked_accounts: HashMap<Pubkey, AccountType>,
        geyser_tx: Sender<GeyserUpdate>,
        stop: Arc<AtomicBool>,
    ) -> Result<Self> {
        let tokio_rt = Builder::new_multi_thread()
            .thread_name("GeyserService")
            .worker_threads(2)
            .enable_all()
            .build()?;

        Ok(Self {
            endpoint: config.yellowstone_endpoint,
            x_token: config.yellowstone_x_token,
            tracked_accounts,
            marginfi_group_pk: config.marginfi_group_key,
            geyser_tx,
            tokio_rt,
            stop,
            error_logger: RateLimitedLogger::new(),
        })
    }

    pub fn start(&self) -> Result<()> {
        info!("Staring GeyserService");

        let tracked_accounts_vec: Vec<Pubkey> = self.tracked_accounts.keys().copied().collect();
        let tls_config = ClientTlsConfig::new().with_native_roots();
        let mut from_slot: Option<u64> = None;
        let mut backoff = INITIAL_RECONNECT_BACKOFF;

        while !self.stop.load(Ordering::Relaxed) {
            info!("Connecting to Geyser...");
            let sub_req = Self::build_geyser_subscribe_request(&tracked_accounts_vec, from_slot);

            // TODO: replace from_slot with auto-reconnect once we migrate to the up-to-date client (requires updating Solana deps):
            // https://docs.triton.one/project-yellowstone/dragons-mouth-grpc-subscriptions#auto-reconnect-rust-client
            //
            // Establish the connection and subscription inside the reconnect loop so that a
            // transient Geyser outage (e.g. connection refused) triggers a retry with backoff
            // instead of propagating out of `start()` and panicking the whole process.
            let connect_result = self.tokio_rt.block_on(async {
                let mut client = GeyserGrpcClient::build_from_shared(self.endpoint.clone())?
                    .x_token(self.x_token.clone())?
                    .tls_config(tls_config.clone())?
                    .connect()
                    .await?;
                let (_, stream) = client.subscribe_with_request(Some(sub_req.clone())).await?;
                Ok::<_, anyhow::Error>(stream)
            });

            let mut stream = match connect_result {
                // Don't reset the backoff yet: a server can accept the connection and then
                // immediately reset the stream (REFUSED_STREAM). Only treat the connection as
                // healthy once it actually delivers a message (see below).
                Ok(stream) => stream,
                Err(e) => {
                    self.error_logger.warn(&format!(
                        "Failed to connect to Geyser, retrying in {:?}: {:?}",
                        backoff, e
                    ));
                    self.sleep_interruptible(backoff);
                    backoff = (backoff * 2).min(MAX_RECONNECT_BACKOFF);
                    continue;
                }
            };

            // TODO: use IndexerFlags
            info!("Entering the GeyserService loop");
            while let Some(msg) = self.tokio_rt.block_on(stream.next()) {
                match msg {
                    Ok(msg) => {
                        // A delivered message proves the connection is healthy: reset the backoff.
                        backoff = INITIAL_RECONNECT_BACKOFF;
                        let update_oneof = ward!(msg.update_oneof, continue);
                        if let subscribe_update::UpdateOneof::Account(account) = update_oneof {
                            from_slot = Some(account.slot);

                            let account_update = ward!(&account.account, continue);
                            let account =
                                ward!(account_update_to_account(account_update).ok(), continue);
                            let address = ward!(
                                Pubkey::try_from(account_update.pubkey.clone()).ok(),
                                continue
                            );

                            if account.owner == marginfi_type_crate::ID {
                                if let Ok(marginfi_account) = MarginfiAccount::try_deserialize(
                                    &mut account.data.clone().as_slice(),
                                ) {
                                    if marginfi_account.group != self.marginfi_group_pk {
                                        continue;
                                    }
                                    self.send_update(AccountType::Marginfi, address, &account);
                                }

                                if let Ok(bank) = decode_bank(&account.data) {
                                    if bank.group != self.marginfi_group_pk {
                                        continue;
                                    }
                                    self.send_update(AccountType::Bank, address, &account);
                                }
                            } else if let Some(account_type) = self.tracked_accounts.get(&address) {
                                self.send_update(*account_type, address, &account);
                            }
                        }
                    }
                    Err(error) => {
                        self.error_logger.warn(&format!(
                            "Received error message from Geyser, reconnecting in {:?}: {:?}",
                            backoff, error
                        ));

                        // Back off before reconnecting so a server that keeps resetting the
                        // stream isn't hammered in a tight loop.
                        self.sleep_interruptible(backoff);
                        backoff = (backoff * 2).min(MAX_RECONNECT_BACKOFF);

                        // Break the inner loop so the outer loop reconnects.
                        break;
                    }
                }

                // Breaking the loop on stop request
                if self.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }
        info!("The GeyserService loop is stopped.");

        Ok(())
    }

    /// Sleeps up to `duration`, waking early if a stop is requested so the reconnect
    /// backoff never delays a clean shutdown.
    fn sleep_interruptible(&self, duration: Duration) {
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline && !self.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200).min(deadline - Instant::now()));
        }
    }

    fn send_update(&self, account_type: AccountType, address: Pubkey, account: &Account) {
        let update = GeyserUpdate {
            account_type,
            address,
            account: account.clone(),
        };
        if let Err(e) = self.geyser_tx.send(update) {
            error!("Error channeling update to the Geyser processor! {:?}", e);
        }
    }

    /// Builds a geyser subscription request payload
    fn build_geyser_subscribe_request(
        tracked_accounts: &[Pubkey],
        from_slot: Option<u64>,
    ) -> SubscribeRequest {
        let mut request = SubscribeRequest {
            ..Default::default()
        };

        let subscribe_to_static_account_updates = SubscribeRequestFilterAccounts {
            account: tracked_accounts.iter().map(|a| a.to_string()).collect(),
            ..Default::default()
        };

        let marginfi_account_subscription = SubscribeRequestFilterAccounts {
            owner: vec![marginfi_type_crate::ID.to_string()],
            ..Default::default()
        };

        let mut req = HashMap::new();
        req.insert(
            "static_accounts".to_string(),
            subscribe_to_static_account_updates,
        );
        req.insert(
            "marginfi_accounts".to_string(),
            marginfi_account_subscription,
        );

        request.accounts = req;
        request.from_slot = from_slot;

        request
    }
}
