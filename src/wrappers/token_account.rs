use crate::wrappers::oracle::OracleWrapperTrait;

use super::bank::BankWrapper;
use fixed::types::I80F48;
use marginfi_type_crate::{constants::EXP_10_I80F48, types::OraclePriceType};

#[derive(Clone)]
pub struct TokenAccountWrapper<T: OracleWrapperTrait> {
    pub balance: u64,
    pub bank_wrapper: BankWrapper,
    pub oracle_wrapper: T,
}

impl<T: OracleWrapperTrait> TokenAccountWrapper<T> {
    pub fn get_value_for_amount(&self, amount: I80F48) -> anyhow::Result<I80F48> {
        let ui_amount = {
            let decimal_scale = EXP_10_I80F48[self.bank_wrapper.bank.mint_decimals as usize];

            amount
                .checked_div(decimal_scale)
                .ok_or(anyhow::anyhow!("Division failed"))?
        };

        let price = self.oracle_wrapper.get_price_of_type(
            OraclePriceType::RealTime,
            None,
            self.bank_wrapper.bank.config.oracle_max_confidence,
        )?;

        Ok(ui_amount * price)
    }

    pub fn get_amount_from_value(&self, value: I80F48) -> anyhow::Result<I80F48> {
        let price = self.oracle_wrapper.get_price_of_type(
            OraclePriceType::RealTime,
            None,
            self.bank_wrapper.bank.config.oracle_max_confidence,
        )?;

        let amount = value
            .checked_div(price)
            .ok_or(anyhow::anyhow!("Division failed"))?;

        let decimal_scale = EXP_10_I80F48[self.bank_wrapper.bank.mint_decimals as usize];

        Ok(amount * decimal_scale)
    }
}
