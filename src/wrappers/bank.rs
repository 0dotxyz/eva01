use crate::wrappers::oracle::OracleWrapperTrait;
use fixed::types::I80F48;
use marginfi::state::marginfi_account::{calc_amount, calc_value};
use marginfi_type_crate::types::{BalanceSide, Bank, OraclePriceType, PriceBias, RequirementType};
use solana_account::Account;
use solana_pubkey::Pubkey;

#[derive(Clone)]
pub struct BankWrapper {
    pub address: Pubkey,
    pub bank: Bank,
    pub account: Account,
}

impl BankWrapper {
    pub fn new(address: Pubkey, bank: Bank, account: Account) -> Self {
        Self {
            address,
            bank,
            account,
        }
    }

    fn get_pricing_params(
        side: BalanceSide,
        requirement_type: RequirementType,
    ) -> (Option<PriceBias>, OraclePriceType) {
        match (side, requirement_type) {
            (BalanceSide::Assets, RequirementType::Initial) => {
                (Some(PriceBias::Low), OraclePriceType::TimeWeighted)
            }
            (BalanceSide::Assets, RequirementType::Maintenance) => {
                (Some(PriceBias::Low), OraclePriceType::RealTime)
            }
            (BalanceSide::Liabilities, RequirementType::Initial) => {
                (Some(PriceBias::High), OraclePriceType::TimeWeighted)
            }
            (BalanceSide::Liabilities, RequirementType::Maintenance) => {
                (Some(PriceBias::High), OraclePriceType::RealTime)
            }
            (BalanceSide::Assets, RequirementType::Equity) => {
                (Some(PriceBias::Low), OraclePriceType::TimeWeighted)
            }
            (BalanceSide::Liabilities, RequirementType::Equity) => {
                (Some(PriceBias::High), OraclePriceType::TimeWeighted)
            }
        }
    }

    /// The price the program uses for `side` under `requirement_type`.
    pub fn price(
        &self,
        oracle_wrapper: &impl OracleWrapperTrait,
        side: BalanceSide,
        requirement_type: RequirementType,
    ) -> anyhow::Result<I80F48> {
        let (price_bias, oracle_type) = Self::get_pricing_params(side, requirement_type);
        oracle_wrapper.get_price_of_type(
            oracle_type,
            price_bias,
            self.bank.config.oracle_max_confidence,
        )
    }

    pub fn calc_amount(
        &self,
        oracle_wrapper: &impl OracleWrapperTrait,
        value: I80F48,
        side: BalanceSide,
        requirement_type: RequirementType,
    ) -> anyhow::Result<I80F48> {
        let (price_bias, oracle_type) = Self::get_pricing_params(side, requirement_type);

        let price = oracle_wrapper.get_price_of_type(
            oracle_type,
            price_bias,
            self.bank.config.oracle_max_confidence,
        )?;

        Ok(calc_amount(value, price, self.bank.mint_decimals)?)
    }

    pub fn calc_value(
        &self,
        oracle_wrapper: &impl OracleWrapperTrait,
        amount: I80F48,
        side: BalanceSide,
        requirement_type: RequirementType,
    ) -> anyhow::Result<I80F48> {
        let (price_bias, oracle_type) = Self::get_pricing_params(side, requirement_type);

        let price = oracle_wrapper.get_price_of_type(
            oracle_type,
            price_bias,
            self.bank.config.oracle_max_confidence,
        )?;

        Ok(calc_value(
            amount,
            price,
            self.bank.get_balance_decimals(),
            None,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equity_prices_match_end_liquidation() {
        // `end_liquidation` values the seizure and the repayment at time-weighted prices, collateral
        // biased low and debt biased high; sizing at any other price breaks its premium cap.
        assert!(matches!(
            BankWrapper::get_pricing_params(BalanceSide::Assets, RequirementType::Equity),
            (Some(PriceBias::Low), OraclePriceType::TimeWeighted)
        ));
        assert!(matches!(
            BankWrapper::get_pricing_params(BalanceSide::Liabilities, RequirementType::Equity),
            (Some(PriceBias::High), OraclePriceType::TimeWeighted)
        ));
    }
}
