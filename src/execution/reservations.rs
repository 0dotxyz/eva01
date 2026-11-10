//! Wallet token reservations.
//!
//! Plans that spend wallet tokens run concurrently. Instead of serializing them, each plan claims
//! the tokens it will spend before it builds its transactions, and other plans and the rebalancer
//! only count the unclaimed part of a balance. A claim is held until the plan has landed or failed
//! and the wallet balances have been re-read, so a released claim never leaves tokens that were
//! just spent looking free.
//!
//! The ledger lock is held only for the arithmetic of a claim, never across I/O.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use solana_pubkey::Pubkey;

#[derive(Default)]
pub struct TokenReservations {
    reserved: Mutex<HashMap<Pubkey, u64>>,
}

impl TokenReservations {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<Pubkey, u64>> {
        // The map holds only counters, each update a single statement, so a poisoned lock still
        // holds consistent data.
        self.reserved.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// How much of `balance` of `mint` no claim holds.
    pub fn unclaimed(&self, mint: &Pubkey, balance: u64) -> u64 {
        balance.saturating_sub(self.lock().get(mint).copied().unwrap_or(0))
    }

    /// Claims up to `wanted` of the unclaimed part of `balance` of `mint`, and returns how much
    /// was granted. Reading the free amount and claiming it is one step, so two plans can never be
    /// granted the same tokens.
    pub fn claim_available(
        &self,
        claim: &mut Claim,
        mint: Pubkey,
        wanted: u64,
        balance: u64,
    ) -> u64 {
        let mut reserved = self.lock();
        let held = reserved.entry(mint).or_insert(0);
        let granted = wanted.min(balance.saturating_sub(*held));
        *held += granted;
        drop(reserved);
        claim.record(mint, granted);
        granted
    }

    /// Claims `amount` of `mint` the wallet does not hold yet but a plan is about to receive (a
    /// buy's output), so nothing else counts it as free once it arrives.
    pub fn claim_incoming(&self, claim: &mut Claim, mint: Pubkey, amount: u64) {
        *self.lock().entry(mint).or_insert(0) += amount;
        claim.record(mint, amount);
    }

    fn release(&self, amounts: &HashMap<Pubkey, u64>) {
        let mut reserved = self.lock();
        for (mint, amount) in amounts {
            if let Some(held) = reserved.get_mut(mint) {
                *held = held.saturating_sub(*amount);
                if *held == 0 {
                    reserved.remove(mint);
                }
            }
        }
    }

    pub fn new_claim(self: &Arc<Self>) -> Claim {
        Claim {
            ledger: Arc::clone(self),
            amounts: HashMap::new(),
        }
    }
}

/// The tokens one plan (or one rebalancer sweep) has claimed. Dropping it releases them.
pub struct Claim {
    ledger: Arc<TokenReservations>,
    amounts: HashMap<Pubkey, u64>,
}

impl Claim {
    fn record(&mut self, mint: Pubkey, amount: u64) {
        if amount > 0 {
            *self.amounts.entry(mint).or_insert(0) += amount;
        }
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.ledger.release(&self.amounts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_claims_never_share_tokens() {
        let ledger = TokenReservations::new();
        let mint = Pubkey::new_unique();
        let mut a = ledger.new_claim();
        let mut b = ledger.new_claim();
        assert_eq!(ledger.claim_available(&mut a, mint, 70, 100), 70);
        assert_eq!(ledger.claim_available(&mut b, mint, 70, 100), 30);
        assert_eq!(ledger.unclaimed(&mint, 100), 0);
        drop(a);
        assert_eq!(ledger.unclaimed(&mint, 100), 70);
        drop(b);
        assert_eq!(ledger.unclaimed(&mint, 100), 100);
    }

    #[test]
    fn incoming_tokens_are_not_free_when_they_arrive() {
        let ledger = TokenReservations::new();
        let mint = Pubkey::new_unique();
        let mut buyer = ledger.new_claim();
        ledger.claim_incoming(&mut buyer, mint, 50);
        // Before the buy lands the wallet holds nothing, and after it lands the 50 are the buyer's.
        assert_eq!(ledger.unclaimed(&mint, 0), 0);
        assert_eq!(ledger.unclaimed(&mint, 50), 0);
        let mut other = ledger.new_claim();
        assert_eq!(ledger.claim_available(&mut other, mint, 10, 50), 0);
        drop(buyer);
        assert_eq!(ledger.claim_available(&mut other, mint, 10, 50), 10);
    }
}
