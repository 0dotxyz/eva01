//! The Solana runtime limits every transaction eva builds must respect, in one place. Each is the
//! Solana crates' own constant where one is public, so a protocol change reaches eva with the
//! dependency.

use solana_compute_budget::compute_budget_limits::MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES;

/// Max serialized size of a legacy or v0 transaction (and of a Jito bundle's transactions).
pub use solana_packet::PACKET_DATA_SIZE as MAX_V0_TX_SIZE;

/// Max serialized size of a v1 transaction.
pub use solana_message::v1::MAX_TRANSACTION_SIZE as MAX_V1_TX_SIZE;

/// The most compute units a transaction may request. Liquidations and price pushes ask for all of
/// it: a heavy route, VAA verification or a Kamino refresh can each take a large share.
pub use solana_compute_budget::compute_budget_limits::MAX_COMPUTE_UNIT_LIMIT;

/// The most account data a transaction may load. A v1 header must state it: unset reads as 0,
/// which fails every transaction.
pub const MAX_LOADED_ACCOUNTS_DATA: u32 = MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES.get();

/// Accounts one transaction may lock, lookup-table-resolved or not. `solana-transaction` says 128,
/// the limit SIMD-0268 raises it to, but that feature is not active on mainnet: the runtime still
/// enforces 64.
pub const MAX_TX_ACCOUNT_LOCKS: usize = 64;

/// A compute-unit price is in micro-lamports. (The compute-budget crate's own constant is private.)
pub const MICRO_LAMPORTS_PER_LAMPORT: u64 = 1_000_000;

/// A Jito bundle holds at most this many transactions.
pub const MAX_BUNDLE_TXS: usize = 5;

/// A Helius Sender bundle holds at most this many transactions.
pub const MAX_SENDER_BUNDLE_TXS: usize = 4;

/// The least priority fee a bundled transaction pays, in lamports: Helius Sender refuses one under
/// 5,000 and recommends 10,000.
pub const MIN_BUNDLE_PRIORITY_FEE_LAMPORTS: u64 = 10_000;

/// The compute-unit price of a bundled transaction limited to `cu_limit`: `micro_lamports_per_cu`,
/// or more when that would pay under [`MIN_BUNDLE_PRIORITY_FEE_LAMPORTS`].
pub fn bundle_cu_price(cu_limit: u32, micro_lamports_per_cu: u64) -> u64 {
    let floor = (MIN_BUNDLE_PRIORITY_FEE_LAMPORTS * MICRO_LAMPORTS_PER_LAMPORT)
        .div_ceil(u64::from(cu_limit.max(1)));
    micro_lamports_per_cu.max(floor)
}

/// The priority fee, in lamports, of a transaction that requests [`MAX_COMPUTE_UNIT_LIMIT`] at
/// `micro_lamports_per_cu`: what a v1 header states, since it carries a total, not a price.
pub fn priority_fee_lamports(micro_lamports_per_cu: u64) -> u64 {
    micro_lamports_per_cu.saturating_mul(u64::from(MAX_COMPUTE_UNIT_LIMIT))
        / MICRO_LAMPORTS_PER_LAMPORT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_v1_priority_fee_matches_the_price_per_compute_unit() {
        // 10,000 micro-lamports per CU over 1.4M CU.
        assert_eq!(priority_fee_lamports(10_000), 14_000);
        assert_eq!(priority_fee_lamports(0), 0);
        assert_eq!(
            priority_fee_lamports(u64::MAX),
            u64::MAX / MICRO_LAMPORTS_PER_LAMPORT
        );
    }

    #[test]
    fn a_bundled_transaction_pays_at_least_the_minimum_priority_fee() {
        for limit in [1_000, 120_000, MAX_COMPUTE_UNIT_LIMIT] {
            let fee = bundle_cu_price(limit, 1) * u64::from(limit) / MICRO_LAMPORTS_PER_LAMPORT;
            assert!(fee >= MIN_BUNDLE_PRIORITY_FEE_LAMPORTS, "{limit}: {fee}");
        }
        // A higher configured price is kept.
        assert_eq!(bundle_cu_price(MAX_COMPUTE_UNIT_LIMIT, 10_000), 10_000);
    }

    #[test]
    fn the_limits_are_the_mainnet_values() {
        assert_eq!(MAX_V0_TX_SIZE, 1232);
        assert_eq!(MAX_V1_TX_SIZE, 4096);
        assert_eq!(MAX_COMPUTE_UNIT_LIMIT, 1_400_000);
        assert_eq!(MAX_LOADED_ACCOUNTS_DATA, 64 * 1024 * 1024);
    }
}
