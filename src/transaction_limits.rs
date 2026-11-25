//! The Solana runtime limits every transaction eva builds must respect, and the one way it builds
//! them: v1, every address in full and the compute budget in the header. Each limit is the Solana
//! crates' own constant where one is public, so a protocol change reaches eva with the dependency.

use anyhow::{anyhow, Result};
use solana_compute_budget::compute_budget_limits::MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::{v1, Hash, VersionedMessage};
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;

/// Max serialized size of a v1 transaction, in a bundle or not.
pub use solana_message::v1::MAX_TRANSACTION_SIZE as MAX_TX_SIZE;

/// The most compute units a transaction may request. Liquidations ask for all of it: a heavy
/// route or a Kamino refresh can each take a large share.
pub use solana_compute_budget::compute_budget_limits::MAX_COMPUTE_UNIT_LIMIT;

/// The most account data a transaction may load. A v1 header must state it: unset reads as 0,
/// which fails every transaction.
pub const MAX_LOADED_ACCOUNTS_DATA: u32 = MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES.get();

/// Accounts one transaction may lock. `solana-transaction` says 128, the limit SIMD-0268 raises it
/// to, but that feature is not active on mainnet: the runtime still enforces 64.
pub const MAX_TX_ACCOUNT_LOCKS: usize = 64;

/// A compute-unit price is in micro-lamports. (The compute-budget crate's own constant is private.)
pub const MICRO_LAMPORTS_PER_LAMPORT: u64 = 1_000_000;

/// A Jito bundle holds at most this many transactions.
pub const MAX_BUNDLE_TXS: usize = 5;

/// A Helius Sender bundle holds at most this many transactions.
pub const MAX_SENDER_BUNDLE_TXS: usize = 4;

/// The least priority fee eva pays on a transaction (all but Jito's tip, which only the block
/// engine sees), in lamports: Helius Sender refuses one under 5,000 and recommends 10,000.
pub const MIN_PRIORITY_FEE_LAMPORTS: u64 = 10_000;

/// The compute limit of a tip transaction: its one transfer takes 150.
pub const TIP_TX_CU_LIMIT: u32 = 1_000;

/// `tx` as the network reads it, for sending and for measuring against the size limit. A v1
/// transaction is its message followed by its signatures; bincode writes the legacy layout
/// (signatures first) for every version, which no node or relay can decode as v1.
pub fn wire_bytes(tx: &VersionedTransaction) -> Result<Vec<u8>> {
    wincode::serialize(tx).map_err(|e| anyhow!("can't serialize the transaction: {e}"))
}

/// The priority fee, in lamports, of a transaction limited to `cu_limit` at `micro_lamports_per_cu`,
/// and at least [`MIN_PRIORITY_FEE_LAMPORTS`]. A v1 header states this total, not a price.
pub fn priority_fee(cu_limit: u32, micro_lamports_per_cu: u64) -> u64 {
    (micro_lamports_per_cu.saturating_mul(u64::from(cu_limit)) / MICRO_LAMPORTS_PER_LAMPORT)
        .max(MIN_PRIORITY_FEE_LAMPORTS)
}

/// `ixs` as a v1 message paid by `payer`, limited to `cu_limit` and paying `priority_fee` lamports,
/// checked against the v1 message rules (signers, instructions, addresses; the size is checked
/// once signed, by [`v1_transaction`] or [`signed_size`]).
pub fn v1_message(
    payer: &solana_pubkey::Pubkey,
    ixs: &[Instruction],
    blockhash: Hash,
    cu_limit: u32,
    priority_fee: u64,
) -> Result<v1::Message> {
    let config = v1::TransactionConfig {
        priority_fee: Some(priority_fee),
        compute_unit_limit: Some(cu_limit),
        loaded_accounts_data_size_limit: Some(MAX_LOADED_ACCOUNTS_DATA),
        ..v1::TransactionConfig::empty()
    };
    let message = v1::Message::try_compile_with_config(payer, ixs, blockhash, config).map_err(
        |e| match e {
            // More accounts than an index can address.
            solana_message::CompileError::AccountIndexOverflow => {
                DoesNotFit(format!("can't compile the transaction: {e}")).into()
            }
            e => anyhow!("can't compile the transaction: {e}"),
        },
    )?;
    // Every account a v1 transaction locks is in its keys: it has no lookup tables.
    let locks = message.account_keys.len();
    if locks > MAX_TX_ACCOUNT_LOCKS {
        return Err(DoesNotFit(format!(
            "transaction loads {locks} accounts (max {MAX_TX_ACCOUNT_LOCKS})"
        ))
        .into());
    }
    message.validate().map_err(|e| {
        use v1::MessageError::*;
        let message = format!("not a valid v1 transaction: {e:?}");
        match e {
            TooManyAddresses
            | TooManyInstructions
            | TooManySignatures
            | InstructionAccountsTooLarge
            | InstructionDataTooLarge
            | TransactionTooLarge => DoesNotFit(message).into(),
            // Instructions built wrong, not too many of them.
            _ => anyhow!(message),
        }
    })?;
    Ok(message)
}

/// Instructions that no transaction can carry: too many accounts, instructions or bytes.
#[derive(Debug)]
pub struct DoesNotFit(pub String);

impl std::fmt::Display for DoesNotFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DoesNotFit {}

/// The wire size of `message` once signed: the signatures' values don't change it, so nothing is
/// signed.
pub fn signed_size(message: &v1::Message) -> Result<usize> {
    let tx = VersionedTransaction {
        signatures: vec![
            solana_signature::Signature::default();
            usize::from(message.header.num_required_signatures)
        ],
        message: VersionedMessage::V1(message.clone()),
    };
    Ok(wire_bytes(&tx)?.len())
}

/// `ixs` as a signed v1 transaction, `signers[0]` paying: see [`v1_message`]. Fails with
/// [`DoesNotFit`] when it locks more accounts than a transaction may, or is larger than one may be.
pub fn v1_transaction(
    signers: &[&Keypair],
    ixs: &[Instruction],
    blockhash: Hash,
    cu_limit: u32,
    priority_fee: u64,
) -> Result<VersionedTransaction> {
    let payer = signers
        .first()
        .ok_or_else(|| anyhow!("a transaction needs a payer"))?
        .pubkey();
    let message = v1_message(&payer, ixs, blockhash, cu_limit, priority_fee)?;
    let tx = VersionedTransaction::try_new(VersionedMessage::V1(message), signers)?;
    let size = wire_bytes(&tx)?.len();
    if size > MAX_TX_SIZE {
        return Err(DoesNotFit(format!("transaction is {size} bytes (max {MAX_TX_SIZE})")).into());
    }
    Ok(tx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_pubkey::Pubkey;

    #[test]
    fn the_priority_fee_is_the_price_times_the_limit_and_at_least_the_minimum() {
        // 10,000 micro-lamports per CU over 1.4M CU.
        assert_eq!(priority_fee(MAX_COMPUTE_UNIT_LIMIT, 10_000), 14_000);
        for limit in [1, TIP_TX_CU_LIMIT, 120_000, MAX_COMPUTE_UNIT_LIMIT] {
            assert_eq!(priority_fee(limit, 0), MIN_PRIORITY_FEE_LAMPORTS);
        }
        assert_eq!(
            priority_fee(MAX_COMPUTE_UNIT_LIMIT, u64::MAX),
            u64::MAX / MICRO_LAMPORTS_PER_LAMPORT
        );
    }

    #[test]
    fn a_v1_transaction_is_written_message_first_and_signatures_last() {
        let payer = Keypair::new();
        let ix = solana_system_interface::instruction::transfer(
            &payer.pubkey(),
            &Pubkey::new_unique(),
            1,
        );
        let tx = v1_transaction(
            &[&payer],
            &[ix],
            Hash::new_unique(),
            TIP_TX_CU_LIMIT,
            MIN_PRIORITY_FEE_LAMPORTS,
        )
        .unwrap();
        let bytes = wire_bytes(&tx).unwrap();
        assert_eq!(bytes[0], v1::V1_PREFIX, "starts with the version byte");
        assert_eq!(&bytes[bytes.len() - 64..], tx.signatures[0].as_ref());
        let decoded: VersionedTransaction = wincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, tx);
        // The legacy layout bincode writes is not the v1 wire format.
        assert_ne!(bincode::serialize(&tx).unwrap(), bytes);
        let VersionedMessage::V1(message) = &tx.message else {
            panic!("not v1");
        };
        assert_eq!(signed_size(message).unwrap(), bytes.len());
        assert_eq!(message.config.priority_fee, Some(MIN_PRIORITY_FEE_LAMPORTS));
        assert_eq!(message.config.compute_unit_limit, Some(TIP_TX_CU_LIMIT));
        assert_eq!(
            message.config.loaded_accounts_data_size_limit,
            Some(MAX_LOADED_ACCOUNTS_DATA)
        );
    }

    #[test]
    fn a_transaction_over_the_limits_is_refused() {
        let payer = Keypair::new();
        let program = Pubkey::new_unique();
        let accounts = |n: usize| {
            (0..n)
                .map(|_| solana_instruction::AccountMeta::new(Pubkey::new_unique(), false))
                .collect()
        };
        let ix = |n, data: usize| Instruction {
            program_id: program,
            accounts: accounts(n),
            data: vec![7; data],
        };
        let build = |ix: Instruction| {
            v1_transaction(&[&payer], &[ix], Hash::new_unique(), 1_000, 0).map(|tx| {
                let VersionedMessage::V1(message) = tx.message else {
                    unreachable!()
                };
                message.account_keys.len()
            })
        };
        // The payer and the program are locked too.
        assert_eq!(build(ix(MAX_TX_ACCOUNT_LOCKS - 2, 0)).unwrap(), 64);
        let too_many = build(ix(MAX_TX_ACCOUNT_LOCKS - 1, 0)).unwrap_err();
        assert!(
            too_many.to_string().contains("loads 65 accounts"),
            "{too_many}"
        );
        assert!(too_many.is::<DoesNotFit>());
        // 64 addresses take 2 KiB, which leaves room for 1.5 KiB of instruction data.
        assert!(build(ix(MAX_TX_ACCOUNT_LOCKS - 2, 1_500)).is_ok());
        assert!(build(ix(1, MAX_TX_SIZE)).unwrap_err().is::<DoesNotFit>());
        // The same limit as the v1 format's own.
        assert_eq!(MAX_TX_ACCOUNT_LOCKS, usize::from(v1::MAX_ADDRESSES));
    }

    #[test]
    fn the_limits_are_the_mainnet_values() {
        assert_eq!(MAX_TX_SIZE, 4096);
        assert_eq!(MAX_COMPUTE_UNIT_LIMIT, 1_400_000);
        assert_eq!(MAX_LOADED_ACCOUNTS_DATA, 64 * 1024 * 1024);
    }
}
