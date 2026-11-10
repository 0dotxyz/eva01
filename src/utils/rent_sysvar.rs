//! Serves the Rent sysvar to the marginfi program code eva runs off chain.
//!
//! The program prices on-ramp staked collateral with `Rent::get()`. On chain that is a syscall;
//! off chain the Solana crates answer it through a process-wide stub table whose default answers
//! `UnsupportedSysvar`, which made every account holding staked collateral fail evaluation. At
//! startup eva reads the Rent sysvar from the chain and installs it in that table.
//!
//! The stub has to be installed in the `solana-sysvar` version the program is built against
//! (pinned in Cargo.toml); the test below calls `Rent::get()` through the same `anchor_lang` the
//! program uses, so a version drift fails it.

use anyhow::{anyhow, Result};
use log::info;
use solana_client::rpc_client::RpcClient;
use solana_sysvar::{
    program_stubs::{set_syscall_stubs, SyscallStubs},
    rent::Rent,
};

/// `lamports_per_byte_year: u64`, `exemption_threshold: f64`, `burn_percent: u8`.
const RENT_SYSVAR_LEN: usize = 17;

/// Answers `Rent::get()` with the chain's rent. Every other syscall keeps its default stub.
struct RentSysvarStub(Rent);

impl SyscallStubs for RentSysvarStub {
    fn sol_get_rent_sysvar(&self, var_addr: *mut u8) -> u64 {
        // SAFETY: `Rent::get` passes a pointer to its own `Rent` value.
        unsafe { *(var_addr as *mut Rent) = self.0.clone() };
        0
    }
}

// The fields are deprecated for building new values, but they are the sysvar's on-chain layout.
#[allow(deprecated)]
fn parse_rent(data: &[u8]) -> Result<Rent> {
    if data.len() < RENT_SYSVAR_LEN {
        return Err(anyhow!("the Rent sysvar is {} bytes", data.len()));
    }
    Ok(Rent {
        lamports_per_byte_year: u64::from_le_bytes(data[0..8].try_into()?),
        exemption_threshold: f64::from_le_bytes(data[8..16].try_into()?),
        burn_percent: data[16],
    })
}

fn install(rent: Rent) {
    set_syscall_stubs(Box::new(RentSysvarStub(rent)));
}

/// Reads the Rent sysvar and serves it to `Rent::get()` for the rest of the process. Mainnet rent
/// is not `Rent::default()`, so the chain's value is used. It changes only by feature activation,
/// so reading it once at startup is enough.
pub fn install_rent_sysvar(rpc_client: &RpcClient) -> Result<()> {
    let data = rpc_client.get_account_data(&solana_sysvar::rent::ID)?;
    let rent = parse_rent(&data)?;
    info!(
        "Rent sysvar: a 0-byte account needs {} lamports to be rent exempt",
        rent.minimum_balance(0)
    );
    install(rent);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anchor_lang::prelude::{Rent as ProgramRent, SolanaSysvar};

    #[test]
    #[allow(deprecated)]
    fn parses_the_sysvar_layout() {
        let mut data = 5080u64.to_le_bytes().to_vec();
        data.extend(1.0f64.to_le_bytes());
        data.push(50);
        let rent = parse_rent(&data).unwrap();
        assert_eq!(rent.lamports_per_byte_year, 5080);
        assert_eq!(rent.exemption_threshold, 1.0);
        assert_eq!(rent.burn_percent, 50);
        assert!(parse_rent(&data[..16]).is_err());
    }

    #[test]
    fn the_program_reads_the_installed_rent() {
        let mut data = 5080u64.to_le_bytes().to_vec();
        data.extend(1.0f64.to_le_bytes());
        data.push(50);
        install(parse_rent(&data).unwrap());
        // The same `Rent::get()` the program's staked pricing calls.
        let rent = ProgramRent::get().unwrap();
        assert_eq!(rent.minimum_balance(520), 3_291_840);
    }
}
