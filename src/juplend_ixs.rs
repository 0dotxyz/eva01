use anchor_lang::{Id, InstructionData, ToAccountMetas};

use solana_sdk::{instruction::Instruction, pubkey::Pubkey};

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
