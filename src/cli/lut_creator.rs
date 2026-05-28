use crate::utils::{find_bank_liquidity_vault_authority, marginfi_account_by_authority};
use anchor_lang::AccountDeserialize;
use anchor_spl::associated_token::get_associated_token_address_with_program_id;
use log::warn;
use marginfi_type_crate::{
    constants::{ASSET_TAG_DEFAULT, ASSET_TAG_KAMINO, ASSET_TAG_SOL, ASSET_TAG_STAKED},
    types::{Bank, MarginfiGroup},
};
use solana_account_decoder::UiAccountEncoding;
use solana_client::{
    rpc_client::RpcClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig},
    rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType},
};
use solana_sdk::{
    address_lookup_table::state::AddressLookupTable, commitment_config::CommitmentConfig,
    pubkey::Pubkey, signature::Keypair, signer::Signer, system_program, sysvar,
};
use std::{collections::HashSet, str::FromStr};

// ── Shared context ────────────────────────────────────────────────────────────

struct LutContext {
    rpc_url: String,
    wallet_keypair_bytes: Vec<u8>,
    rpc_client: RpcClient,
    g1: HashSet<Pubkey>,
    g2: HashSet<Pubkey>,
    g3: HashSet<Pubkey>,
}

fn build_lut_context() -> anyhow::Result<LutContext> {
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
    let wallet_keypair_bytes: Vec<u8> = serde_json::from_str(
        &std::env::var("WALLET_KEYPAIR").expect("WALLET_KEYPAIR environment variable is not set"),
    )
    .expect("Invalid WALLET_KEYPAIR JSON format");
    let signer =
        Keypair::from_bytes(&wallet_keypair_bytes).expect("Failed to parse WALLET_KEYPAIR");
    let marginfi_group_key = Pubkey::from_str(
        &std::env::var("MARGINFI_GROUP_KEY")
            .expect("MARGINFI_GROUP_KEY environment variable is not set"),
    )
    .expect("Invalid MARGINFI_GROUP_KEY");

    println!("Signer:  {}", signer.pubkey());
    println!("Group:   {}", marginfi_group_key);

    let rpc_client = RpcClient::new_with_commitment(&rpc_url, CommitmentConfig::confirmed());

    let (global_fee_state_key, _) = Pubkey::find_program_address(
        &[marginfi_type_crate::constants::FEE_STATE_SEED.as_bytes()],
        &marginfi_type_crate::ID,
    );
    let marginfi_group_account = rpc_client.get_account(&marginfi_group_key)?;
    let marginfi_group = bytemuck::from_bytes::<MarginfiGroup>(&marginfi_group_account.data[8..]);
    let global_fee_wallet = marginfi_group.fee_state_cache.global_fee_wallet;

    let liquidator_marginfi_account =
        marginfi_account_by_authority(signer.pubkey(), &rpc_client, marginfi_group_key)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                anyhow::anyhow!("No marginfi account found for signer {}", signer.pubkey())
            })?;

    let common: Vec<Pubkey> = vec![
        signer.pubkey(),
        liquidator_marginfi_account,
        marginfi_group_key,
        marginfi_type_crate::ID,
        global_fee_state_key,
        global_fee_wallet,
        spl_token::id(),
        anchor_spl::token_2022::ID,
        system_program::id(),
        sysvar::instructions::id(),
    ];

    println!(
        "Liquidator marginfi account: {}",
        liquidator_marginfi_account
    );
    println!("Global fee state:            {}", global_fee_state_key);
    println!("Global fee wallet:           {}", global_fee_wallet);

    // Fetch and partition banks
    const BANK_GROUP_PK_OFFSET: usize = 32 + 1 + 8;
    println!("\nFetching all banks...");
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
    println!("Found {} banks.", bank_accounts.len());

    let mut g1: HashSet<Pubkey> = common.iter().cloned().collect();
    let mut g2: HashSet<Pubkey> = common.iter().cloned().collect();
    let mut g3: HashSet<Pubkey> = common.iter().cloned().collect();
    let mut g1_mints: HashSet<Pubkey> = HashSet::new();
    let mut g2_mints: HashSet<Pubkey> = HashSet::new();
    let mut g3_mints: HashSet<Pubkey> = HashSet::new();
    let mut counts = [0u32; 3];

    for (bank_address, bank_account) in &bank_accounts {
        let mut data = bank_account.data.as_slice();
        let bank = match Bank::try_deserialize(&mut data) {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed to deserialize bank {}: {}", bank_address, e);
                continue;
            }
        };

        let tag = bank.config.asset_tag;
        let vault_authority =
            find_bank_liquidity_vault_authority(bank_address, &marginfi_type_crate::ID);

        let mut per_bank = vec![
            *bank_address,
            bank.liquidity_vault,
            bank.insurance_vault,
            bank.fee_vault,
            bank.mint,
            vault_authority,
        ];
        per_bank.extend(
            bank.config
                .oracle_keys
                .iter()
                .filter(|k| **k != Pubkey::default())
                .copied(),
        );

        let in_g1 = tag == ASSET_TAG_DEFAULT || tag == ASSET_TAG_SOL;
        let in_g2 = tag == ASSET_TAG_SOL || tag == ASSET_TAG_STAKED;
        let in_g3 = tag >= ASSET_TAG_KAMINO;

        if in_g3 {
            per_bank.extend(
                [
                    bank.integration_acc_1,
                    bank.integration_acc_2,
                    bank.integration_acc_3,
                ]
                .iter()
                .filter(|k| **k != Pubkey::default())
                .copied(),
            );
        }

        if in_g1 {
            counts[0] += 1;
            g1.extend(per_bank.iter().copied());
            g1_mints.insert(bank.mint);
        }
        if in_g2 {
            counts[1] += 1;
            g2.extend(per_bank.iter().copied());
            g2_mints.insert(bank.mint);
        }
        if in_g3 {
            counts[2] += 1;
            g3.extend(per_bank.iter().copied());
            g3_mints.insert(bank.mint);
        }
    }

    println!(
        "Groups: g1={} banks, g2={} banks, g3={} banks",
        counts[0], counts[1], counts[2]
    );

    // Derive liquidator ATAs
    let all_mints: Vec<Pubkey> = g1_mints
        .union(&g2_mints)
        .chain(&g3_mints)
        .copied()
        .collect();
    println!(
        "Fetching {} mint accounts to derive ATAs...",
        all_mints.len()
    );
    let mint_accounts: Vec<_> = all_mints
        .chunks(100)
        .map(|chunk| rpc_client.get_multiple_accounts(chunk))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    for (mint, mint_acct) in all_mints.iter().zip(mint_accounts.iter()) {
        if let Some(acct) = mint_acct {
            let ata =
                get_associated_token_address_with_program_id(&signer.pubkey(), mint, &acct.owner);
            if g1_mints.contains(mint) {
                g1.insert(ata);
            }
            if g2_mints.contains(mint) {
                g2.insert(ata);
            }
            if g3_mints.contains(mint) {
                g3.insert(ata);
            }
        } else {
            warn!("Mint {} not found, skipping ATA derivation", mint);
        }
    }

    println!(
        "Address counts: g1={}, g2={}, g3={}",
        g1.len(),
        g2.len(),
        g3.len()
    );

    Ok(LutContext {
        rpc_url,
        wallet_keypair_bytes,
        rpc_client,
        g1,
        g2,
        g3,
    })
}

// ── create-lut ────────────────────────────────────────────────────────────────

pub fn create_lut_entry() -> anyhow::Result<()> {
    let ctx = build_lut_context()?;

    let existing_g1 = read_existing_luts("ADDRESS_LOOKUP_TABLES_GROUP1");
    let existing_g2 = read_existing_luts("ADDRESS_LOOKUP_TABLES_GROUP2");
    let existing_g3 = read_existing_luts("ADDRESS_LOOKUP_TABLES_GROUP3");

    let wallet_path = write_wallet_to_tempfile(&ctx.wallet_keypair_bytes)?;

    let g1_luts = populate_group(
        &ctx.rpc_url,
        &wallet_path,
        &ctx.rpc_client,
        1,
        ctx.g1.into_iter().collect(),
        existing_g1,
    )?;
    let g2_luts = populate_group(
        &ctx.rpc_url,
        &wallet_path,
        &ctx.rpc_client,
        2,
        ctx.g2.into_iter().collect(),
        existing_g2,
    )?;
    let g3_luts = populate_group(
        &ctx.rpc_url,
        &wallet_path,
        &ctx.rpc_client,
        3,
        ctx.g3.into_iter().collect(),
        existing_g3,
    )?;

    std::fs::remove_file(&wallet_path).ok();

    print_env_output(&g1_luts, &g2_luts, &g3_luts);
    Ok(())
}

// ── sync-lut ──────────────────────────────────────────────────────────────────

/// Checks existing LUTs against the current on-chain bank state and extends them
/// with any missing addresses (new banks, changed oracle/integration keys, etc.).
/// LUTs are append-only so changed keys are added alongside old ones.
pub fn sync_lut_entry() -> anyhow::Result<()> {
    let existing_g1 = read_existing_luts("ADDRESS_LOOKUP_TABLES_GROUP1");
    let existing_g2 = read_existing_luts("ADDRESS_LOOKUP_TABLES_GROUP2");
    let existing_g3 = read_existing_luts("ADDRESS_LOOKUP_TABLES_GROUP3");

    if existing_g1.is_empty() && existing_g2.is_empty() && existing_g3.is_empty() {
        anyhow::bail!(
            "No existing LUT addresses found. Set ADDRESS_LOOKUP_TABLES_GROUP1/2/3 in your env, or run create-lut first."
        );
    }

    let ctx = build_lut_context()?;
    let wallet_path = write_wallet_to_tempfile(&ctx.wallet_keypair_bytes)?;

    let g1_luts = populate_group(
        &ctx.rpc_url,
        &wallet_path,
        &ctx.rpc_client,
        1,
        ctx.g1.into_iter().collect(),
        existing_g1,
    )?;
    let g2_luts = populate_group(
        &ctx.rpc_url,
        &wallet_path,
        &ctx.rpc_client,
        2,
        ctx.g2.into_iter().collect(),
        existing_g2,
    )?;
    let g3_luts = populate_group(
        &ctx.rpc_url,
        &wallet_path,
        &ctx.rpc_client,
        3,
        ctx.g3.into_iter().collect(),
        existing_g3,
    )?;

    std::fs::remove_file(&wallet_path).ok();

    println!("\nSync complete!");
    print_env_output(&g1_luts, &g2_luts, &g3_luts);
    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn print_env_output(g1: &[Pubkey], g2: &[Pubkey], g3: &[Pubkey]) {
    let fmt = |luts: &[Pubkey]| {
        luts.iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    println!("\nAdd these to your .env:");
    println!("ADDRESS_LOOKUP_TABLES_GROUP1={}", fmt(g1));
    println!("ADDRESS_LOOKUP_TABLES_GROUP2={}", fmt(g2));
    println!("ADDRESS_LOOKUP_TABLES_GROUP3={}", fmt(g3));
}

fn read_existing_luts(env_var: &str) -> Vec<Pubkey> {
    std::env::var(env_var)
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// Create or resume a group of LUTs.
///
/// Fetches existing on-chain LUTs to determine which addresses are already stored,
/// then only creates/extends LUTs for the remainder. Returns the complete set of LUT keys.
fn populate_group(
    rpc_url: &str,
    wallet_path: &str,
    rpc_client: &RpcClient,
    group_num: u8,
    all_addresses: Vec<Pubkey>,
    existing_keys: Vec<Pubkey>,
) -> anyhow::Result<Vec<Pubkey>> {
    const LUT_MAX: usize = 256;

    let mut already_stored: HashSet<Pubkey> = HashSet::new();
    let mut last_lut_space: usize = 0;

    if !existing_keys.is_empty() {
        println!(
            "\n[Group {}] Found {} existing LUT(s), fetching on-chain state...",
            group_num,
            existing_keys.len()
        );
        let accounts = rpc_client.get_multiple_accounts(&existing_keys)?;
        for (key, acct_opt) in existing_keys.iter().zip(accounts.iter()) {
            if let Some(acct) = acct_opt {
                let lut = AddressLookupTable::deserialize(&acct.data)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize LUT {}: {}", key, e))?;
                println!(
                    "[Group {}]   {} — {} addresses",
                    group_num,
                    key,
                    lut.addresses.len()
                );
                already_stored.extend(lut.addresses.iter().copied());
                last_lut_space = LUT_MAX.saturating_sub(lut.addresses.len());
            } else {
                anyhow::bail!("Existing LUT {} not found on-chain", key);
            }
        }
    }

    let remaining: Vec<Pubkey> = all_addresses
        .into_iter()
        .filter(|a| !already_stored.contains(a))
        .collect();

    if remaining.is_empty() {
        println!(
            "\n[Group {}] Up to date ({} addresses). No changes needed.",
            group_num,
            already_stored.len()
        );
        return Ok(existing_keys);
    }

    println!(
        "\n[Group {}] {} addresses already stored, {} new to add.",
        group_num,
        already_stored.len(),
        remaining.len()
    );

    let mut result_keys = existing_keys.clone();

    let (extend_into_last, fresh) = if !existing_keys.is_empty() && last_lut_space > 0 {
        let take = remaining.len().min(last_lut_space);
        (remaining[..take].to_vec(), remaining[take..].to_vec())
    } else {
        (vec![], remaining)
    };

    if !extend_into_last.is_empty() {
        let last_key = *existing_keys.last().unwrap();
        println!(
            "[Group {}] Extending last LUT {} with {} addresses...",
            group_num,
            last_key,
            extend_into_last.len()
        );
        extend_lut_cli(rpc_url, wallet_path, group_num, last_key, &extend_into_last)?;
    }

    if !fresh.is_empty() {
        let new_keys = create_lut_group(rpc_url, wallet_path, group_num, fresh)?;
        result_keys.extend(new_keys);
    }

    Ok(result_keys)
}

fn create_lut_group(
    rpc_url: &str,
    wallet_path: &str,
    group_num: u8,
    addresses: Vec<Pubkey>,
) -> anyhow::Result<Vec<Pubkey>> {
    const LUT_MAX: usize = 256;

    let buckets: Vec<&[Pubkey]> = addresses.chunks(LUT_MAX).collect();
    println!(
        "\n[Group {}] Creating {} new LUT(s) for {} addresses...",
        group_num,
        buckets.len(),
        addresses.len()
    );

    let mut lut_keys: Vec<Pubkey> = Vec::new();

    for (idx, bucket) in buckets.iter().enumerate() {
        println!("[Group {}] LUT {}/{}...", group_num, idx + 1, buckets.len());

        let out = std::process::Command::new("solana")
            .args([
                "address-lookup-table",
                "create",
                "--url",
                rpc_url,
                "--keypair",
                wallet_path,
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run `solana` CLI: {e}"))?;

        if !out.status.success() {
            anyhow::bail!(
                "solana address-lookup-table create failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let lut_address: Pubkey = stdout
            .lines()
            .find_map(|line| {
                let line = line.trim();
                line.starts_with("Lookup Table Address:")
                    .then(|| line.split_whitespace().last()?.parse().ok())
                    .flatten()
            })
            .ok_or_else(|| anyhow::anyhow!("Could not parse LUT address from:\n{stdout}"))?;

        println!("[Group {}] Created LUT: {}", group_num, lut_address);
        extend_lut_cli(rpc_url, wallet_path, group_num, lut_address, bucket)?;
        println!(
            "[Group {}] LUT {}/{} done: {}",
            group_num,
            idx + 1,
            buckets.len(),
            lut_address
        );
        lut_keys.push(lut_address);
    }

    Ok(lut_keys)
}

fn extend_lut_cli(
    rpc_url: &str,
    wallet_path: &str,
    group_num: u8,
    lut_address: Pubkey,
    addresses: &[Pubkey],
) -> anyhow::Result<()> {
    const EXTEND_CHUNK: usize = 20;

    let total = addresses.chunks(EXTEND_CHUNK).count();
    for (i, chunk) in addresses.chunks(EXTEND_CHUNK).enumerate() {
        println!(
            "[Group {}] Extending {}/{} ({} addresses)...",
            group_num,
            i + 1,
            total,
            chunk.len()
        );
        let out = std::process::Command::new("solana")
            .args(["address-lookup-table", "extend"])
            .arg(lut_address.to_string())
            .args(["--url", rpc_url, "--keypair", wallet_path])
            .args([
                "--addresses",
                &chunk
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run `solana` CLI: {e}"))?;

        if !out.status.success() {
            anyhow::bail!(
                "solana address-lookup-table extend failed on chunk {}: {}",
                i + 1,
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    Ok(())
}

fn write_wallet_to_tempfile(keypair_bytes: &[u8]) -> anyhow::Result<String> {
    use std::io::Write;
    let path = std::env::temp_dir().join("eva01_wallet_tmp.json");
    let mut f = std::fs::File::create(&path)?;
    f.write_all(serde_json::to_string(keypair_bytes)?.as_bytes())?;
    Ok(path.to_string_lossy().into_owned())
}
