use anyhow::Context;
use lazy_static::lazy_static;
use prometheus::{
    CounterVec, Encoder, Histogram, HistogramOpts, IntCounter as Counter, IntGauge, Opts, Registry,
    TextEncoder,
};

// TODO: revisit metrics, some of them are not that useful anymore
lazy_static! {
    pub static ref REGISTRY: Registry =
        Registry::new_custom(Some("eva01".to_string()), None).expect("init registry");
    pub static ref LIQUIDATION_ATTEMPTS: Counter = register_counter(Opts::new(
        "eva01_liquidation_attempts_total",
        "Total number of liquidation attempts"
    ));
    pub static ref FAILED_LIQUIDATIONS: Counter = register_counter(Opts::new(
        "eva01_failed_liquidations_total",
        "Total number of failed liquidation attempts"
    ));
    pub static ref ACCOUNTS_SCANNED_TOTAL: Counter = register_counter(Opts::new(
        "eva01_accounts_scanned_total",
        "Total number of accounts evaluated for liquidation"
    ));
    pub static ref LIQUIDATABLE_ACCOUNTS_FOUND_TOTAL: Counter = register_counter(Opts::new(
        "eva01_liquidatable_accounts_found_total",
        "Total number of liquidatable accounts detected during evaluation"
    ));
    pub static ref ERROR_COUNT: Counter = register_counter(Opts::new(
        "eva01_errors_total",
        "Total number of errors encountered"
    ));
    pub static ref GEYSER_UPDATES_TOTAL: Counter = register_counter(Opts::new(
        "eva01_geyser_updates_total",
        "Total number of Yellowstone Geyser updates processed"
    ));
    pub static ref LIQUIDATION_SUCCESSES: Counter = register_counter(Opts::new(
        "eva01_liquidations_succeeded_total",
        "Total number of successful liquidations"
    ));
    pub static ref LIQUIDATABLE_ACCOUNTS_FOUND: IntGauge = register_gauge(Opts::new(
        "eva01_liquidatable_accounts",
        "Current count of detected liquidatable accounts"
    ));
    pub static ref LIQUIDATION_SCAN_IN_PROGRESS: IntGauge = register_gauge(Opts::new(
        "eva01_liquidation_scan_in_progress",
        "Flag indicating whether the liquidator is scanning accounts"
    ));
    pub static ref LIQUIDATION_FAILURES: CounterVec = register_counter_vec(
        Opts::new(
            "eva01_liquidation_failures_total",
            "Total number of liquidation failures grouped by reason, liability mint, and oracle"
        ),
        &["reason", "liab_mint", "oracle"]
    );
    pub static ref ACCOUNT_SCAN_DURATION_SECONDS: Histogram = register_histogram(
        HistogramOpts::new(
            "eva01_account_scan_duration_seconds",
            "Time spent scanning all accounts for liquidation"
        )
        .buckets(vec![0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0])
    );
    pub static ref LIQUIDATION_LATENCY_SECONDS: Histogram = register_histogram(
        HistogramOpts::new(
            "eva01_liquidation_latency_seconds",
            "Time spent executing a liquidation transaction"
        )
        .buckets(vec![
            0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0
        ])
    );
}

pub const FAILURE_REASON_INTERNAL: &str = "internal";
pub const FAILURE_REASON_RPC_ERROR: &str = "rpc_error";
pub const FAILURE_REASON_STALE_ORACLES: &str = "stale_oracles";
pub const FAILURE_REASON_NOT_ENOUGH_FUNDS: &str = "not_enough_funds";

pub fn record_liquidation_failure(
    reason: &str,
    liab_mint: Option<solana_program::pubkey::Pubkey>,
    oracle: Option<solana_program::pubkey::Pubkey>,
) {
    FAILED_LIQUIDATIONS.inc();
    let mint_str = liab_mint
        .map(|m| m.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let oracle_str = oracle
        .map(|o| o.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    LIQUIDATION_FAILURES
        .with_label_values(&[reason, &mint_str, &oracle_str])
        .inc();
}

fn register_counter(opts: Opts) -> Counter {
    let counter = Counter::with_opts(opts).expect("create counter");
    REGISTRY
        .register(Box::new(counter.clone()))
        .expect("register counter");
    counter
}

fn register_counter_vec(opts: Opts, labels: &[&str]) -> CounterVec {
    let counter = CounterVec::new(opts, labels).expect("create counter vec");
    REGISTRY
        .register(Box::new(counter.clone()))
        .expect("register counter vec");
    counter
}

fn register_gauge(opts: Opts) -> IntGauge {
    let gauge = IntGauge::with_opts(opts).expect("create gauge");
    REGISTRY
        .register(Box::new(gauge.clone()))
        .expect("register gauge");
    gauge
}

fn register_histogram(opts: HistogramOpts) -> Histogram {
    let histogram = Histogram::with_opts(opts).expect("create histogram");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("register histogram");
    histogram
}

fn encode_metrics() -> anyhow::Result<Vec<u8>> {
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    TextEncoder::new()
        .encode(&metric_families, &mut buffer)
        .context("failed to encode prometheus metrics")?;
    Ok(buffer)
}

pub mod server {
    use super::encode_metrics;
    use log::{debug, error, info};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use tiny_http::{Header, Response, Server};

    const HTTP_STATUS_OK: u16 = 200;
    const HTTP_STATUS_NOT_FOUND: u16 = 404;
    const HTTP_STATUS_INTERNAL_SERVER_ERROR: u16 = 500;

    pub struct MetricsServer {
        bind_addr: String,
        port: u16,
        stop: Arc<AtomicBool>,
    }

    impl MetricsServer {
        pub fn new(bind_addr: String, port: u16, stop: Arc<AtomicBool>) -> Self {
            Self {
                bind_addr,
                port,
                stop,
            }
        }

        pub fn start(&self) -> anyhow::Result<()> {
            let bind_target = format!("{}:{}", self.bind_addr, self.port);
            info!("Running the Prometheus metrics server on {}", bind_target);

            let server = Server::http(&bind_target).map_err(|err| {
                anyhow::anyhow!("Failed to bind metrics server to {}: {}", bind_target, err)
            })?;

            while !self.stop.load(Ordering::Relaxed) {
                for request in server.incoming_requests() {
                    if self.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let response = match request.url() {
                        "/metrics" => match encode_metrics() {
                            Ok(body) => {
                                debug!("Serving /metrics");
                                let header = Header::from_bytes(
                                    b"Content-Type",
                                    b"text/plain; version=0.0.4",
                                )
                                .expect("create header");
                                Response::from_data(body)
                                    .with_header(header)
                                    .with_status_code(HTTP_STATUS_OK)
                            }
                            Err(error) => {
                                error!("Failed to encode metrics: {:?}", error);
                                Response::from_string("failed to encode metrics")
                                    .with_status_code(HTTP_STATUS_INTERNAL_SERVER_ERROR)
                            }
                        },
                        _ => Response::from_string("Not Found")
                            .with_status_code(HTTP_STATUS_NOT_FOUND),
                    };

                    if let Err(err) = request.respond(response) {
                        error!("Failed to respond to metrics request: {}", err);
                    }
                }
            }

            info!("Metrics server stopped");
            Ok(())
        }
    }
}
