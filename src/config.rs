use solana_pubkey::Pubkey;
use std::str::FromStr;

/// The debt tokens the wallet keeps a working balance of by default, so a liquidation whose swap
/// does not fit can repay from the wallet: the largest liabilities in the main group besides the
/// swap mint (USDC), which is the reserve the balances are bought from.
pub const DEFAULT_INVENTORY_MINTS: [&str; 7] = [
    "So11111111111111111111111111111111111111112", // SOL (wrapped)
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", // USDT
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn", // JitoSOL
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So", // mSOL
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1", // bSOL
    "USDSwr9ApdHk5bvJKMjzff41FfuX8bSxdKcR81vTwcA", // USDS
    "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo", // PYUSD
];

#[derive(Clone, Debug)]
pub struct Eva01Config {
    pub rpc_url: String,
    pub yellowstone_endpoint: String,
    pub yellowstone_x_token: Option<String>,
    pub wallet_keypair: Vec<u8>,
    pub compute_unit_price_micro_lamports: u64,
    pub marginfi_group_key: Pubkey,
    pub luts_group1: Vec<Pubkey>,
    pub luts_group2: Vec<Pubkey>,
    pub luts_group3: Vec<Pubkey>,
    pub excluded_liquidation_mints: Vec<Pubkey>,
    pub min_profit: f64,
    /// Accounts whose liquidation would earn more than this are left to another instance: set on
    /// a dust instance so it only takes what the main instance's `min_profit` passes over.
    pub max_profit: Option<f64>,
    /// Debt tokens the rebalancer keeps a working balance of (`INVENTORY_MINTS`).
    pub inventory_mints: Vec<Pubkey>,
    /// The working balance per inventory token, in USD (`INVENTORY_TARGET_USD`).
    pub inventory_target_usd: f64,
    pub healthcheck_port: u16,
    pub pyth_hermes_url: String,
    pub pyth_api_key: Option<String>,
    pub pyth_cranker_enabled: bool,
    pub jup_swap_api_url: String,
    pub swap_mint: Pubkey,
    pub slippage_bps: u16,
    pub titan_ws_endpoint: String,
    pub titan_api_key: String,
    pub dflow_api_key: Option<String>,
    pub jupiter_api_key: String,
    /// Land liquidations as Jito bundles, and price them on Hermes prices pushed on chain in the
    /// same bundle when the chain's aren't enough (`PRICE_PUSH_ENABLED`, on unless "false").
    pub price_push_enabled: bool,
    /// Jito block-engine `sendBundle` endpoint; `None` uses the client's built-in default.
    pub jito_block_engine_url: Option<String>,
    /// Hard cap on the Jito tip per bundle (lamports).
    pub jito_tip_max_lamports: u64,
    /// The share of a bundle's expected profit tipped (`BUNDLE_TIP_PROFIT_SHARE`).
    pub bundle_tip_profit_share: f64,
    /// The share of a liquidation's full profit a wallet-funded plan must earn to go first, ahead
    /// of the plans that quote a swap (`LIQUIDATION_WALLET_MIN_PROFIT_SHARE`).
    pub wallet_min_profit_share: f64,
    /// API key/uuid for `sendBundle`; `None` for the public (unauthenticated) path.
    pub bundle_api_key: Option<String>,
}

impl Eva01Config {
    pub fn new() -> anyhow::Result<Self> {
        //General configuration
        let yellowstone_endpoint = std::env::var("YELLOWSTONE_ENDPOINT")
            .expect("YELLOWSTONE_ENDPOINT environment variable is not set");
        let yellowstone_x_token = std::env::var("YELLOWSTONE_X_TOKEN").ok();
        let derived_rpc_url = derive_rpc_url(&yellowstone_endpoint, yellowstone_x_token.as_deref());
        let rpc_url = std::env::var("RPC_URL").unwrap_or(derived_rpc_url);

        let wallet_keypair_env = std::env::var("WALLET_KEYPAIR")
            .expect("WALLET_KEYPAIR environment variable is not set");
        let wallet_keypair: Vec<u8> =
            serde_json::from_str(&wallet_keypair_env).expect("Invalid WALLET_KEYPAIR JSON format");

        let compute_unit_price_micro_lamports: u64 =
            std::env::var("COMPUTE_UNIT_PRICE_MICRO_LAMPORTS")
                .expect("COMPUTE_UNIT_PRICE_MICRO_LAMPORTS environment variable is not set")
                .parse()
                .expect("Invalid COMPUTE_UNIT_PRICE_MICRO_LAMPORTS number");

        let marginfi_group_key = Pubkey::from_str(
            &std::env::var("MARGINFI_GROUP_KEY")
                .expect("MARGINFI_GROUP_KEY environment variable is not set"),
        )
        .expect("Invalid MARGINFI_GROUP_KEY Pubkey");

        let luts_group1 = parse_pubkey_list("ADDRESS_LOOKUP_TABLES_GROUP1").unwrap_or_default();
        let luts_group2 = parse_pubkey_list("ADDRESS_LOOKUP_TABLES_GROUP2").unwrap_or_default();
        let luts_group3 = parse_pubkey_list("ADDRESS_LOOKUP_TABLES_GROUP3").unwrap_or_default();

        let excluded_liquidation_mints = parse_pubkey_list("EXCLUDED_LIQUIDATION_MINTS")?;

        let min_profit: f64 = std::env::var("MIN_PROFIT")
            .expect("MIN_PROFIT environment variable is not set")
            .parse()
            .expect("Invalid MIN_PROFIT number");
        let max_profit: Option<f64> = std::env::var("MAX_PROFIT")
            .ok()
            .map(|value| value.parse().expect("Invalid MAX_PROFIT number"));

        let inventory_mints = match std::env::var_os("INVENTORY_MINTS") {
            Some(_) => parse_pubkey_list("INVENTORY_MINTS")?,
            None => DEFAULT_INVENTORY_MINTS
                .iter()
                .map(|mint| Pubkey::from_str(mint).expect("valid default inventory mint"))
                .collect(),
        };
        let inventory_target_usd: f64 = std::env::var("INVENTORY_TARGET_USD")
            .unwrap_or_else(|_| "50".to_string())
            .parse()
            .expect("Invalid INVENTORY_TARGET_USD number");

        let healthcheck_port: u16 = std::env::var("PORT")
            .unwrap_or("3000".to_string())
            .parse()
            .expect("Invalid PORT number");

        let jup_swap_api_url = std::env::var("JUP_SWAP_API_URL")
            .expect("JUP_SWAP_API_URL environment variable is not set");

        let slippage_bps: u16 = std::env::var("SLIPPAGE_BPS")
            .expect("SLIPPAGE_BPS environment variable is not set")
            .parse()
            .expect("Invalid SLIPPAGE_BPS number: {:#?}");

        let swap_mint = Pubkey::from_str(
            &std::env::var("SWAP_MINT").expect("SWAP_MINT environment variable is not set"),
        )
        .expect("Invalid SWAP_MINT Pubkey");

        let titan_ws_endpoint = std::env::var("TITAN_WS_ENDPOINT")
            .expect("TITAN_WS_ENDPOINT environment variable is not set");
        let titan_api_key =
            std::env::var("TITAN_API_KEY").expect("TITAN_API_KEY environment variable is not set");
        let dflow_api_key = std::env::var("DFLOW_API_KEY")
            .ok()
            .filter(|key| !key.is_empty());

        let jupiter_api_key = std::env::var("JUP_SWAP_API_KEY")
            .expect("JUP_SWAP_API_KEY environment variable is not set");

        let pyth_hermes_url = std::env::var("PYTH_HERMES_URL")
            .unwrap_or_else(|_| crate::utils::pyth_cranker::DEFAULT_HERMES_URL.to_string());

        let pyth_api_key = std::env::var("PYTH_API_KEY").ok();
        let pyth_cranker_enabled =
            std::env::var("PYTH_CRANKER_ENABLED").is_ok_and(|value| value == "true");

        let price_push_enabled = std::env::var("PRICE_PUSH_ENABLED").map_or(true, |v| v != "false");
        let jito_block_engine_url = std::env::var("JITO_BLOCK_ENGINE_URL").ok();
        // Default cap 0.005 SOL: competitors' largest tips on marginfi liquidations (Nov 2026).
        let jito_tip_max_lamports: u64 = std::env::var("JITO_TIP_MAX_LAMPORTS")
            .unwrap_or_else(|_| "5000000".to_string())
            .parse()
            .expect("Invalid JITO_TIP_MAX_LAMPORTS number");
        // Competitors tipped 6–9% of their profit (Nov 2026).
        let bundle_tip_profit_share = share_from_env("BUNDLE_TIP_PROFIT_SHARE", 0.08);
        let wallet_min_profit_share = share_from_env("LIQUIDATION_WALLET_MIN_PROFIT_SHARE", 0.1);
        let bundle_api_key = std::env::var("JITO_API_KEY").ok();

        Ok(Eva01Config {
            rpc_url,
            yellowstone_endpoint,
            yellowstone_x_token,
            wallet_keypair,
            compute_unit_price_micro_lamports,
            marginfi_group_key,
            luts_group1,
            luts_group2,
            luts_group3,
            excluded_liquidation_mints,
            min_profit,
            max_profit,
            inventory_mints,
            inventory_target_usd,
            healthcheck_port,
            pyth_hermes_url,
            pyth_api_key,
            pyth_cranker_enabled,
            jup_swap_api_url,
            swap_mint,
            slippage_bps,
            titan_ws_endpoint,
            titan_api_key,
            dflow_api_key,
            jupiter_api_key,
            price_push_enabled,
            jito_block_engine_url,
            jito_tip_max_lamports,
            bundle_tip_profit_share,
            wallet_min_profit_share,
            bundle_api_key,
        })
    }
}

/// A share between 0 and 1 from `env_var`, or `default` when it is unset.
fn share_from_env(env_var: &str, default: f64) -> f64 {
    let share: f64 = std::env::var(env_var)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("Invalid {env_var} number"))
        })
        .unwrap_or(default);
    assert!(
        (0.0..=1.0).contains(&share),
        "{env_var} must be between 0 and 1, got {share}"
    );
    share
}

fn derive_rpc_url(yellowstone_endpoint: &str, yellowstone_x_token: Option<&str>) -> String {
    let endpoint = yellowstone_endpoint.trim_end_matches('/');
    match yellowstone_x_token
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        Some(token) => format!("{endpoint}/{token}"),
        None => endpoint.to_string(),
    }
}

fn parse_pubkey_list(env_var: &str) -> anyhow::Result<Vec<Pubkey>> {
    match std::env::var_os(env_var) {
        Some(val) => {
            let value = val.to_string_lossy();
            if value.trim().is_empty() {
                Ok(vec![])
            } else {
                value
                    .split(',')
                    .map(|s| {
                        Pubkey::from_str(s.trim()).map_err(|e| {
                            anyhow::anyhow!("Invalid pubkey in the {} list: {:#?}", env_var, e)
                        })
                    })
                    .collect()
            }
        }
        None => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use figment::Jail;
    use serial_test::serial;
    use solana_keypair::Keypair;

    use super::*;

    fn setup_general_env(jail: &mut Jail) {
        let keypair = serde_json::to_string(&Keypair::new().to_bytes().to_vec()).unwrap();

        let yellowstone_endpoint = "https://dummy";
        let yellowstone_x_token = "token";
        let compute_unit_price_micro_lamports = "1000";
        let marginfi_group_key = Pubkey::new_unique().to_string();
        let lut_group1 = Pubkey::new_unique().to_string();
        let lut_group2 = Pubkey::new_unique().to_string();
        let lut_group3 = Pubkey::new_unique().to_string();
        let min_profit = "0.01";
        let healthcheck_port = "3000";

        jail.set_env("YELLOWSTONE_ENDPOINT", yellowstone_endpoint);
        jail.set_env("YELLOWSTONE_X_TOKEN", yellowstone_x_token);
        jail.set_env("WALLET_KEYPAIR", &keypair);
        jail.set_env(
            "COMPUTE_UNIT_PRICE_MICRO_LAMPORTS",
            compute_unit_price_micro_lamports,
        );
        jail.set_env("MARGINFI_GROUP_KEY", &marginfi_group_key);
        jail.set_env("ADDRESS_LOOKUP_TABLES_GROUP1", &lut_group1);
        jail.set_env("ADDRESS_LOOKUP_TABLES_GROUP2", &lut_group2);
        jail.set_env("ADDRESS_LOOKUP_TABLES_GROUP3", &lut_group3);
        jail.set_env("MIN_PROFIT", min_profit);
        jail.set_env("PORT", healthcheck_port);
    }

    fn setup_rebalancer_env(jail: &mut Jail) {
        jail.set_env("TOKEN_ACCOUNT_DUST_THRESHOLD", "0.0001");
        jail.set_env("SWAP_MINT", &Pubkey::new_unique().to_string());
        jail.set_env("JUP_SWAP_API_URL", "https://dummy/swap");
        jail.set_env("JUP_SWAP_API_KEY", "dummy_jupiter_api_key");
        jail.set_env("TITAN_WS_ENDPOINT", "dummy.titan.exchange");
        jail.set_env("TITAN_API_KEY", "dummy_titan_api_key");
        jail.set_env("SLIPPAGE_BPS", "50");
    }

    #[test]
    #[serial]
    fn test_parse_pubkey_list_empty() {
        Jail::expect_with(|_jail| {
            // TEST_PUBKEY_LIST is not set in the jail, so it should return empty
            let result = parse_pubkey_list("TEST_PUBKEY_LIST").unwrap();
            assert_eq!(result.len(), 0);
            Ok(())
        });
    }

    #[test]
    #[serial]
    fn test_parse_pubkey_list_valid() {
        Jail::expect_with(|jail| {
            jail.set_env(
                "TEST_PUBKEY_LIST",
                format!(
                    "{},{}",
                    &Pubkey::new_unique().to_string(),
                    &Pubkey::new_unique().to_string()
                )
                .as_str(),
            );
            let result = parse_pubkey_list("TEST_PUBKEY_LIST").unwrap();
            assert_eq!(result.len(), 2);
            Ok(())
        });
    }

    #[test]
    #[serial]
    fn test_parse_pubkey_list_invalid() {
        Jail::expect_with(|jail| {
            jail.set_env("TEST_PUBKEY_LIST", "invalid_pubkey");
            let result = parse_pubkey_list("TEST_PUBKEY_LIST");
            assert!(result.is_err());
            Ok(())
        });
    }

    #[test]
    #[serial]
    fn test_eva01_config_new_success() {
        Jail::expect_with(|mut jail| {
            setup_general_env(&mut jail);
            setup_rebalancer_env(&mut jail);
            let config = Eva01Config::new();
            assert!(config.is_ok());
            Ok(())
        });
    }

    #[test]
    #[serial]
    fn test_eva01_config_new_derives_rpc_url_from_yellowstone() {
        Jail::expect_with(|mut jail| {
            setup_general_env(&mut jail);
            setup_rebalancer_env(&mut jail);
            let config = Eva01Config::new().unwrap();
            assert_eq!(config.rpc_url, "https://dummy/token");
            Ok(())
        });
    }

    #[test]
    #[serial]
    fn test_eva01_config_new_prefers_explicit_rpc_url_override() {
        Jail::expect_with(|mut jail| {
            setup_general_env(&mut jail);
            setup_rebalancer_env(&mut jail);
            jail.set_env("RPC_URL", "https://explicit-rpc.example");
            let config = Eva01Config::new().unwrap();
            assert_eq!(config.rpc_url, "https://explicit-rpc.example");
            Ok(())
        });
    }

    #[test]
    #[serial]
    #[should_panic(expected = "YELLOWSTONE_ENDPOINT environment variable is not set")]
    fn test_eva01_config_new_missing_env() {
        Jail::expect_with(|_jail| {
            // YELLOWSTONE_ENDPOINT is not set in the jail, so it should panic
            let _result = Eva01Config::new();
            Ok(())
        });
    }

    #[test]
    #[serial]
    #[should_panic(expected = "Invalid COMPUTE_UNIT_PRICE_MICRO_LAMPORTS number")]
    fn test_eva01_config_new_invalid_compute_unit_price() {
        Jail::expect_with(|mut jail| {
            setup_general_env(&mut jail);
            setup_rebalancer_env(&mut jail);
            jail.set_env("COMPUTE_UNIT_PRICE_MICRO_LAMPORTS", "not_a_number");
            Eva01Config::new().unwrap();
            Ok(())
        });
    }
}
