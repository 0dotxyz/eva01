FROM lukemathwalker/cargo-chef:latest-rust-1.94 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json --no-default-features --features network-mainnet

COPY . .

RUN cargo build --release --bin eva01 --no-default-features --features network-mainnet

FROM debian:bookworm-slim AS runner

RUN apt-get update && apt-get install -y ca-certificates

COPY --from=builder /app/target/release/eva01 /usr/local/bin/

ENTRYPOINT ["/usr/local/bin/eva01"]
