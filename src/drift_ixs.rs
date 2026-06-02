use anchor_lang::{Id, InstructionData, ToAccountMetas};

use marginfi_type_crate::pdas::derive_drift_state;
use solana_sdk::{instruction::Instruction, pubkey::Pubkey};

use crate::drift::{client as drift, program::Drift};

pub fn make_refresh_spot_market_ix(
    spot_market: Pubkey,
    spot_market_vault: Pubkey,
    oracle: Pubkey,
) -> Instruction {
    let accounts = drift::accounts::UpdateSpotMarketCumulativeInterest {
        state: derive_drift_state().0,
        spot_market,
        oracle,
        spot_market_vault,
    }
    .to_account_metas(None);

    Instruction {
        program_id: Drift::id(),
        accounts,
        data: drift::args::UpdateSpotMarketCumulativeInterest.data(),
    }
}
