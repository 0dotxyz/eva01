use crate::{
    cache::Cache,
    cache_loader::{get_accounts_to_track, CacheLoader},
    clock_manager::{self, ClockManager},
    config::Eva01Config,
    geyser::{GeyserService, GeyserUpdate},
    geyser_processor::GeyserProcessor,
    liquidator::Liquidator,
    metrics::{FAILED_LIQUIDATIONS, LIQUIDATION_ATTEMPTS},
    utils::{
        integration_account_fetcher::IntegrationAccountFetcher, swb_price_fetcher::SwbPriceFetcher,
    },
    wrappers::liquidator_account::LiquidatorAccount,
};
use anchor_lang::AccountDeserialize;
use log::{error, info};
use marginfi_type_crate::types::Bank;
use solana_account_decoder::UiAccountEncoding;
use solana_client::{
    rpc_client::RpcClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig},
    rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType},
};
use solana_sdk::{
    commitment_config::CommitmentConfig, pubkey::Pubkey, signature::Keypair, signer::Signer,
};
use std::{
    collections::HashSet,
    str::FromStr,
    sync::{atomic::AtomicBool, Arc, Mutex},
    thread,
};

pub fn create_lut_entry() -> anyhow::Result<()> {
    let rpc_url = std::env::var("RPC_URL").unwrap_or_else(|_| {
        let endpoint = std::env::var("YELLOWSTONE_ENDPOINT")
            .expect("Either RPC_URL or YELLOWSTONE_ENDPOINT must be set");
        let token = std::env::var("YELLOWSTONE_X_TOKEN").unwrap_or_default();
        let endpoint = endpoint.trim_end_matches('/');
        if token.trim().is_empty() {
            endpoint.to_string()
        } else {
            format!("{endpoint}/{}", token.trim())
        }
    });
    let wallet_keypair_env =
        std::env::var("WALLET_KEYPAIR").expect("WALLET_KEYPAIR environment variable is not set");
    let wallet_keypair_bytes: Vec<u8> =
        serde_json::from_str(&wallet_keypair_env).expect("Invalid WALLET_KEYPAIR JSON format");
    let signer =
        Keypair::from_bytes(&wallet_keypair_bytes).expect("Failed to parse WALLET_KEYPAIR");
    let marginfi_group_key = Pubkey::from_str(
        &std::env::var("MARGINFI_GROUP_KEY")
            .expect("MARGINFI_GROUP_KEY environment variable is not set"),
    )
    .expect("Invalid MARGINFI_GROUP_KEY");

    println!("Signer: {}", signer.pubkey());
    println!("Group:  {}", marginfi_group_key);

    let rpc_client = RpcClient::new_with_commitment(&rpc_url, CommitmentConfig::confirmed());

    // --- Step 1: Collect all bank + oracle addresses ---
    const BANK_GROUP_PK_OFFSET: usize = 32 + 1 + 8;

    println!("Fetching all banks from the marginfi program...");
    let bank_accounts = rpc_client.get_program_accounts_with_config(
        &marginfi_type_crate::ID,
        RpcProgramAccountsConfig {
            account_config: RpcAccountInfoConfig {
                encoding: Some(UiAccountEncoding::Base64),
                ..Default::default()
            },
            filters: Some(vec![
                RpcFilterType::Memcmp(Memcmp::new(
                    0,
                    MemcmpEncodedBytes::Base58(
                        solana_sdk::bs58::encode(Bank::DISCRIMINATOR).into_string(),
                    ),
                )),
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(
                    BANK_GROUP_PK_OFFSET,
                    marginfi_group_key.as_ref(),
                )),
            ]),
            with_context: Some(false),
            sort_results: None,
        },
    )?;

    let mut addresses: HashSet<Pubkey> = HashSet::new();
    for (bank_address, bank_account) in &bank_accounts {
        addresses.insert(*bank_address);
        let mut data = bank_account.data.as_slice();
        if let Ok(bank) = Bank::try_deserialize(&mut data) {
            for key in bank
                .config
                .oracle_keys
                .iter()
                .filter(|k| k != &&Pubkey::default())
            {
                addresses.insert(*key);
            }
        }
    }

    let all_addresses: Vec<Pubkey> = addresses.into_iter().collect();
    println!(
        "Found {} banks. Total addresses to store: {}",
        bank_accounts.len(),
        all_addresses.len()
    );

    // A single LUT holds at most 256 entries. Split into buckets of 256.
    const LUT_MAX: usize = 256;
    const EXTEND_CHUNK: usize = 20; // max addresses per extend tx

    let lut_buckets: Vec<&[Pubkey]> = all_addresses.chunks(LUT_MAX).collect();
    let num_luts = lut_buckets.len();
    println!(
        "Need {} LUT(s) to hold {} addresses.",
        num_luts,
        all_addresses.len()
    );

    // --- Step 2: Create + extend each LUT via the solana CLI ---
    // The SDK's address_lookup_table::instruction has a version mismatch with the
    // on-chain program in Solana 2.x, so we shell out to the CLI binary instead.
    let wallet_path = write_wallet_to_tempfile(&wallet_keypair_bytes)?;
    let mut lut_addresses: Vec<Pubkey> = Vec::new();

    for (lut_idx, bucket) in lut_buckets.iter().enumerate() {
        println!("\nCreating LUT {}/{}...", lut_idx + 1, num_luts);
        let cli_output = std::process::Command::new("solana")
            .args([
                "address-lookup-table",
                "create",
                "--url",
                &rpc_url,
                "--keypair",
                &wallet_path,
            ])
            .output()
            .map_err(|e| {
                anyhow::anyhow!("Failed to run `solana` CLI: {e}. Is solana-cli installed?")
            })?;

        if !cli_output.status.success() {
            let stderr = String::from_utf8_lossy(&cli_output.stderr);
            anyhow::bail!("solana address-lookup-table create failed:\n{stderr}");
        }

        let stdout = String::from_utf8_lossy(&cli_output.stdout);
        let lut_address_str = stdout
            .lines()
            .find_map(|line| {
                let line = line.trim();
                if line.starts_with("Lookup Table Address:") {
                    line.split_whitespace().last().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Could not parse LUT address from CLI output:\n{stdout}")
            })?;

        let lut_address: Pubkey = lut_address_str
            .parse()
            .expect("Invalid LUT pubkey from CLI");
        println!("  Created empty LUT: {}", lut_address);

        let total_chunks = bucket.chunks(EXTEND_CHUNK).count();
        for (i, chunk) in bucket.chunks(EXTEND_CHUNK).enumerate() {
            println!("  Extending ({}/{})...", i + 1, total_chunks);
            let addr_list: Vec<String> = chunk.iter().map(|k| k.to_string()).collect();
            let output = std::process::Command::new("solana")
                .args(["address-lookup-table", "extend"])
                .arg(lut_address.to_string())
                .args(["--url", &rpc_url, "--keypair", &wallet_path])
                .args(["--addresses", &addr_list.join(",")])
                .output()
                .map_err(|e| anyhow::anyhow!("Failed to run `solana` CLI: {e}"))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!(
                    "solana address-lookup-table extend failed on chunk {}: {stderr}",
                    i + 1
                );
            }
        }

        lut_addresses.push(lut_address);
        println!("  LUT {}/{} done: {}", lut_idx + 1, num_luts, lut_address);
    }

    std::fs::remove_file(&wallet_path).ok();

    let lut_list = lut_addresses
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");

    println!("\nAll LUTs populated successfully!");
    println!("\nAdd this to your .env:");
    println!("ADDRESS_LOOKUP_TABLES={}", lut_list);

    Ok(())
}

fn write_wallet_to_tempfile(keypair_bytes: &[u8]) -> anyhow::Result<String> {
    use std::io::Write;
    let path = std::env::temp_dir().join("eva01_wallet_tmp.json");
    let mut f = std::fs::File::create(&path)?;
    // solana CLI expects a JSON array of u8
    let json = serde_json::to_string(keypair_bytes)?;
    f.write_all(json.as_bytes())?;
    Ok(path.to_string_lossy().into_owned())
}

pub fn run_liquidator(config: Eva01Config, stop_liquidator: Arc<AtomicBool>) -> anyhow::Result<()> {
    info!(
        "Starting liquidator for group: {:?}",
        config.marginfi_group_key
    );

    let wallet_pubkey = Keypair::from_bytes(&config.wallet_keypair)?.pubkey();
    info!("Liquidator public key: {}", wallet_pubkey);

    // Solana Clock
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
        config.clone().address_lookup_tables,
    )?;

    cache_loader.load_cache(&mut cache)?;

    let accounts_to_track = get_accounts_to_track(&cache)?;

    let swb_fetcher_api_url = config.project0_api_url.clone();
    let swb_fetcher_crossbar_url = config.crossbar_api_url.clone();
    let integration_fetcher_rpc_url = config.rpc_url.clone();

    info!("Initializing services...");

    // GeyserService -> GeyserProcessor
    // GeyserProcessor -> Liquidator/Rebalancer
    // Liquidator/Rebalancer -> TransactionManager
    let (geyser_tx, geyser_rx) = crossbeam::channel::unbounded::<GeyserUpdate>();
    let run_liquidation = Arc::new(AtomicBool::new(false));

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
        run_liquidation.clone(),
        stop_liquidator.clone(),
        cache.clone(),
    )?;

    let geyser_service = GeyserService::new(
        config,
        accounts_to_track,
        geyser_tx,
        stop_liquidator.clone(),
        clock.clone(),
    )?;

    let swb_fetcher_cache = cache.clone();
    let swb_fetcher_stop = stop_liquidator.clone();
    let integration_fetcher_cache = cache.clone();
    let integration_fetcher_stop = stop_liquidator.clone();

    let geyser_processor = GeyserProcessor::new(
        geyser_rx.clone(),
        run_liquidation.clone(),
        stop_liquidator.clone(),
        cache,
    )?;

    info!("Starting services...");

    thread::spawn(move || {
        let fetcher = SwbPriceFetcher::new(
            swb_fetcher_api_url,
            swb_fetcher_crossbar_url,
            swb_fetcher_cache,
            swb_fetcher_stop,
        );
        fetcher.start();
    });

    thread::spawn(move || {
        let fetcher = IntegrationAccountFetcher::new(
            integration_fetcher_rpc_url,
            integration_fetcher_cache,
            integration_fetcher_stop,
        );
        fetcher.start();
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
        if let Err(e) = geyser_processor.start() {
            error!("GeyserProcessor failed! {:?}", e);
            panic!("Fatal error in GeyserProcessor!");
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
        info!(
            "Stats: Liqudations [attempts, failed] -> [{},{}]",
            LIQUIDATION_ATTEMPTS.get(),
            FAILED_LIQUIDATIONS.get()
        );
        thread::sleep(std::time::Duration::from_secs(30));
    }
    info!("The Main loop stopped.");

    Ok(())
}
