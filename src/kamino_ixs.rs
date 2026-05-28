use anchor_lang::{prelude::AccountMeta, Id, InstructionData, ToAccountMetas};

use solana_sdk::{instruction::Instruction, pubkey::Pubkey};

use crate::{
    cache::KaminoReserve,
    kamino_lending::{client as kamino, program::KaminoLending},
};

pub fn make_refresh_reserve_ix(
    kamino_reserve_address: Pubkey,
    kamino_reserve: &KaminoReserve,
) -> Instruction {
    let (pyth_oracle, switchboard_price_oracle, switchboard_twap_oracle, scope_prices) =
        get_oracle_setup(kamino_reserve);
    let accounts = kamino::accounts::RefreshReserve {
        reserve: kamino_reserve_address,
        lending_market: kamino_reserve.reserve.lending_market,
        pyth_oracle,
        switchboard_price_oracle,
        switchboard_twap_oracle,
        scope_prices,
    }
    .to_account_metas(None);

    Instruction {
        program_id: KaminoLending::id(),
        accounts,
        data: kamino::args::RefreshReserve.data(),
    }
}

pub fn make_refresh_obligation_ix(
    obligation: Pubkey,
    lending_market: Pubkey,
    remaining: &[Pubkey],
) -> Instruction {
    let mut accounts = kamino::accounts::RefreshObligation {
        lending_market,
        obligation,
    }
    .to_account_metas(None);

    accounts.extend(
        remaining
            .iter()
            .map(|a| AccountMeta::new_readonly(*a, false)),
    );

    Instruction {
        program_id: KaminoLending::id(),
        accounts,
        data: kamino::args::RefreshObligation.data(),
    }
}

fn get_oracle_setup(
    reserve: &KaminoReserve,
) -> (
    Option<Pubkey>,
    Option<Pubkey>,
    Option<Pubkey>,
    Option<Pubkey>,
) {
    (
        none_if_default(reserve.reserve.config.token_info.pyth_configuration.price),
        none_if_default(
            reserve
                .reserve
                .config
                .token_info
                .switchboard_configuration
                .price_aggregator,
        ),
        none_if_default(
            reserve
                .reserve
                .config
                .token_info
                .switchboard_configuration
                .twap_aggregator,
        ),
        none_if_default(
            reserve
                .reserve
                .config
                .token_info
                .scope_configuration
                .price_feed,
        ),
    )
}

fn none_if_default(pk: Pubkey) -> Option<Pubkey> {
    (pk != Pubkey::default()).then_some(pk)
}
