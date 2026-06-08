#!/bin/bash
set -e

# Load env vars manually
set -o allexport
source docker.prod.env
set +o allexport

# Add hardcoded vars
export RUST_LOG="debug,hyper=warn,h2::codec=warn"
export RUST_BACKTRACE=0

# Run all steps
cargo fmt
cargo clippy -- -D warnings
# cargo build --bin eva01 --package eva01
# cargo run --bin eva01 --features pretty_logs

# Staging network build/run example:
cargo build --bin eva01 --package eva01 --no-default-features --features network-staging
cargo run --bin eva01 --no-default-features --features "network-staging,pretty_logs"
