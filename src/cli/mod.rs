use std::{
    sync::{atomic::AtomicBool, Arc, Mutex},
    thread,
};

use crate::{
    config::Eva01Config,
    metrics::server::MetricsServer,
    utils::healthcheck::{HealthCheckServer, HealthState},
};
use log::error;

pub mod entrypoints;
pub use entrypoints::create_lut_entry;

/// Main entrypoint for Eva
pub fn main_entry(stop: Arc<AtomicBool>) -> anyhow::Result<()> {
    // Load environment variables from .env file if it exists
    // This allows local development without needing to export all env vars
    dotenv::dotenv().ok();

    let config = Eva01Config::new()?;

    let health_state = Arc::new(Mutex::new(HealthState::Initializing));
    let healthcheck_server =
        HealthCheckServer::new(config.healthcheck_port, health_state.clone(), stop.clone());
    thread::spawn(move || healthcheck_server.start());

    let metrics_server = MetricsServer::new(
        config.metrics_bind_addr.clone(),
        config.metrics_port,
        stop.clone(),
    );
    thread::spawn(move || {
        if let Err(err) = metrics_server.start() {
            error!("Metrics server stopped unexpectedly: {:?}", err);
        }
    });

    *health_state
        .lock()
        .expect("Failed to acquire the health_state lock") = HealthState::Running;

    entrypoints::run_liquidator(config, stop.clone())
}
