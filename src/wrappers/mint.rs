use solana_account::Account;
use solana_pubkey::Pubkey;

#[derive(Clone)]
pub struct MintWrapper {
    pub token: Pubkey,
    pub account: Account,
}

impl MintWrapper {
    pub fn new(token: Pubkey, account: Account) -> Self {
        Self { token, account }
    }
}
