use crate::{
    cache::Cache,
    cache_loader::{get_accounts_to_track, CacheLoader},
    clock_manager::{self, ClockManager},
    config::Eva01Config,
    geyser::{GeyserService, GeyserUpdate},
    liquidator::Liquidator,
    utils::{
        integration_account_fetcher::IntegrationAccountFetcher, swb_price_fetcher::SwbPriceFetcher,
    },
    wrappers::liquidator_account::LiquidatorAccount,
};
use log::{error, info};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{signature::Keypair, signer::Signer};
use std::{
    sync::{atomic::AtomicBool, Arc, Mutex},
    thread,
};

pub fn run_liquidator(config: Eva01Config, stop_liquidator: Arc<AtomicBool>) -> anyhow::Result<()> {
    info!(
        "Starting liquidator for group: {:?}",
        config.marginfi_group_key
    );

    let wallet_pubkey = Keypair::try_from(config.wallet_keypair.as_slice())?.pubkey();
    info!("Liquidator public key: {}", wallet_pubkey);

    let clock = {
        let rpc_client = RpcClient::new(config.rpc_url.clone());
        Arc::new(Mutex::new(clock_manager::fetch_clock(&rpc_client)?))
    };
    let mut clock_manager = ClockManager::new(clock.clone(), config.rpc_url.clone())?;

    info!("Loading Cache...");
    let mut cache = Cache::new(wallet_pubkey, config.marginfi_group_key, clock.clone());

    let cache_loader = CacheLoader::new(
        &config.wallet_keypair,
        config.rpc_url.clone(),
        config.luts_group1.clone(),
        config.luts_group2.clone(),
        config.luts_group3.clone(),
    )?;
    cache_loader.load_cache(&mut cache)?;

    let accounts_to_track = get_accounts_to_track(&cache)?;
    let swb_fetcher_api_url = config.project0_api_url.clone();
    let swb_fetcher_crossbar_url = config.crossbar_api_url.clone();
    let integration_fetcher_rpc_url = config.rpc_url.clone();

    info!("Initializing services...");

    let (geyser_tx, geyser_rx) = crossbeam::channel::unbounded::<GeyserUpdate>();
    let cache = Arc::new(cache);

    let liquidator_account = Arc::new(LiquidatorAccount::new(
        &config.clone(),
        config.marginfi_group_key,
        config.swap_mint,
        cache.clone(),
    )?);

    let mut liquidator = Liquidator::new(
        config.clone(),
        liquidator_account.clone(),
        geyser_rx,
        stop_liquidator.clone(),
        cache.clone(),
    )?;

    let geyser_service = GeyserService::new(
        config,
        accounts_to_track,
        geyser_tx,
        stop_liquidator.clone(),
    )?;

    let swb_fetcher_cache = cache.clone();
    let swb_fetcher_stop = stop_liquidator.clone();
    let integration_fetcher_cache = cache.clone();
    let integration_fetcher_stop = stop_liquidator.clone();

    info!("Starting services...");

    thread::spawn(move || {
        SwbPriceFetcher::new(
            swb_fetcher_api_url,
            swb_fetcher_crossbar_url,
            swb_fetcher_cache,
            swb_fetcher_stop,
        )
        .start();
    });

    thread::spawn(move || {
        IntegrationAccountFetcher::new(
            integration_fetcher_rpc_url,
            integration_fetcher_cache,
            integration_fetcher_stop,
        )
        .start();
    });

    let cloned_stop = stop_liquidator.clone();
    thread::spawn(move || clock_manager.start(cloned_stop));

    thread::spawn(move || {
        if let Err(e) = liquidator.start() {
            error!("The Liquidator service failed! {:?}", e);
            panic!("Fatal error in the Liquidator service!");
        }
    });

    thread::spawn(move || {
        if let Err(e) = geyser_service.start() {
            error!("GeyserService failed! {:?}", e);
            panic!("Fatal error in GeyserService!");
        }
    });

    info!("Entering the Main loop.");
    while !stop_liquidator.load(std::sync::atomic::Ordering::SeqCst) {
        thread::sleep(std::time::Duration::from_secs(30));
    }
    info!("The Main loop stopped.");

    Ok(())
}
