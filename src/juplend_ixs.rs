use anchor_lang::{Id, InstructionData, ToAccountMetas};
use fixed::types::I80F48;

use solana_instruction::Instruction;
use solana_pubkey::Pubkey;

use crate::juplend_earn::{accounts::Lending, client as juplend, program};

pub fn make_update_lending_rate_ix(
    lending_state_address: Pubkey,
    lending_state: &Lending,
) -> Instruction {
    let accounts = juplend::accounts::UpdateRate {
        lending: lending_state_address,
        mint: lending_state.mint,
        f_token_mint: lending_state.f_token_mint,
        supply_token_reserves_liquidity: lending_state.token_reserves_liquidity,
        rewards_rate_model: lending_state.rewards_rate_model,
    }
    .to_account_metas(None);

    Instruction {
        program_id: program::Lending::id(),
        accounts,
        data: juplend::args::UpdateRate.data(),
    }
}

/// JupLend's `token_exchange_price` is scaled by this: underlying = fTokens * price / 1e12.
const JUPLEND_EXCHANGE_PRICE_PRECISION: u128 = 1_000_000_000_000;

/// Underlying tokens that `f_tokens` of a JupLend position redeem to, rounded down so the program's
/// `ceil(underlying * 1e12 / price)` burns no more fTokens than the position holds.
pub fn juplend_underlying_amount(f_tokens: I80F48, token_exchange_price: u64) -> u64 {
    let f_tokens: u128 = f_tokens.to_num::<u64>().into();
    u64::try_from(f_tokens * u128::from(token_exchange_price) / JUPLEND_EXCHANGE_PRICE_PRECISION)
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn juplend_underlying_never_burns_more_shares_than_held() {
        let precision = JUPLEND_EXCHANGE_PRICE_PRECISION;
        for (shares, price) in [
            (535_916u64, 1_054_321_987_654u64),
            (1u64, 1_999_999_999_999u64),
            (170_782u64, 1_000_000_000_001u64),
            (123_456_789u64, 999_999_999_999u64),
        ] {
            let underlying = juplend_underlying_amount(I80F48::from_num(shares), price);
            // The program burns ceil(underlying * 1e12 / price) fTokens.
            let burned = (u128::from(underlying) * precision).div_ceil(u128::from(price));
            assert!(
                burned <= u128::from(shares),
                "{shares} @ {price}: burns {burned}"
            );
            // And never leaves more than one share's worth on the table.
            assert!(u128::from(shares) - burned <= 1);
        }
    }
}
