use anchor_lang::{self, declare_program};
use env_logger::Builder;
use env_logger::Target;
use signal_hook::consts::{SIGINT, SIGTERM};
use std::{
    backtrace::Backtrace,
    error::Error,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
#[cfg(not(feature = "pretty_logs"))]
use {log::Level, std::io::Write as _};

#[cfg(all(feature = "network-mainnet", feature = "network-staging"))]
compile_error!("Enable only one network feature: network-mainnet or network-staging");

#[cfg(not(any(feature = "network-mainnet", feature = "network-staging")))]
compile_error!("Missing network feature. Enable network-mainnet or network-staging");

mod cache;
mod cache_loader;
mod cli;
mod clock_manager;
mod config;
mod execution;
mod geyser;
mod health_watch;
mod juplend_ixs;
mod kamino_ixs;
mod liquidator;
mod marginfi_ixs;
mod rebalancer;
mod tag_policy;
mod transaction_limits;
mod utils;
#[warn(clippy::type_complexity)]
mod wrappers;

declare_program!(kamino_lending);
declare_program!(kamino_farms);
declare_program!(juplend_earn);
declare_program!(liquidity);

fn main() -> Result<(), Box<dyn Error>> {
    init_logging();

    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:#?}", panic_info);

        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("Payload: {}", s);
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            eprintln!("Payload: {}", s);
        } else if let Some(err) = panic_info.payload().downcast_ref::<anyhow::Error>() {
            eprintln!("Payload: {:?}", err);
        } else {
            eprintln!("Payload: (unknown type)");
        }

        eprintln!("Backtrace: {}", Backtrace::capture());

        std::process::exit(1);
    }));

    // Register signal handlers for graceful shutdown
    let stop = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(SIGINT, stop.clone()).unwrap();
    signal_hook::flag::register(SIGTERM, stop.clone()).unwrap();

    let stop_hook = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        stop_hook.store(true, Ordering::SeqCst);
        println!("Received stop signal");
    })
    .expect("Error setting Ctrl-C handler");

    // Main entrypoint
    crate::cli::main_entry(stop)?;

    Ok(())
}

/// `message` with every URL cut to its scheme and host: RPC and API URLs carry their access token
/// in the path or the query, and errors quote the URL they failed on.
fn redact_urls(message: &str) -> std::borrow::Cow<'_, str> {
    const SCHEMES: [&str; 4] = ["https://", "http://", "wss://", "ws://"];
    if !SCHEMES.iter().any(|scheme| message.contains(scheme)) {
        return message.into();
    }
    let ends_url =
        |c: char| c.is_whitespace() || matches!(c, ')' | '(' | '"' | '\'' | '<' | '>' | ',');
    let mut out = String::with_capacity(message.len());
    let mut rest = message;
    while let Some((at, scheme)) = SCHEMES
        .iter()
        .filter_map(|scheme| rest.find(scheme).map(|at| (at, *scheme)))
        .min_by_key(|(at, _)| *at)
    {
        out.push_str(&rest[..at + scheme.len()]);
        rest = &rest[at + scheme.len()..];
        let host_end = rest
            .find(|c: char| ends_url(c) || matches!(c, '/' | '?' | '#'))
            .unwrap_or(rest.len());
        // Credentials before the host (`user:token@host`) are dropped too.
        let host = &rest[..host_end];
        out.push_str(host.rsplit_once('@').map_or(host, |(_, host)| host));
        rest = &rest[host_end..];
        let url_end = rest.find(ends_url).unwrap_or(rest.len());
        if url_end > 0 {
            out.push_str("/…");
        }
        rest = &rest[url_end..];
    }
    out.push_str(rest);
    out.into()
}

fn init_logging() {
    let mut b = Builder::from_env(env_logger::Env::default().default_filter_or("info"));

    // LOG_FILE=<path>: also append every log line to that file (for post-mortems of a run).
    let file = std::env::var("LOG_FILE")
        .ok()
        .filter(|p| !p.is_empty())
        .map(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| panic!("Failed to open LOG_FILE {path}: {e}"))
        });

    #[cfg(feature = "pretty_logs")]
    {
        if let Some(file) = file {
            b.target(Target::Pipe(Box::new(Tee {
                console: std::io::stderr(),
                file,
            })));
        }
        b.init();
    }
    #[cfg(not(feature = "pretty_logs"))]
    {
        match file {
            Some(file) => b.target(Target::Pipe(Box::new(Tee {
                console: std::io::stdout(),
                file,
            }))),
            None => b.target(Target::Stdout),
        };
        b.format(|buf, record| {
            use serde_json::json;

            let level = match record.level() {
                Level::Error => "error",
                Level::Warn => "warn",
                Level::Info => "info",
                Level::Debug | Level::Trace => "debug",
            };

            let line = json!({
                "level": level,
                "message": redact_urls(&record.args().to_string()),
                "target": record.target(),
            });

            writeln!(buf, "{}", line)
        })
        .init();
    }
}

/// Writes every log line to the console and to the log file.
struct Tee<C: std::io::Write> {
    console: C,
    file: std::fs::File,
}

impl<C: std::io::Write> std::io::Write for Tee<C> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.console.write_all(buf)?;
        self.file.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.console.flush()?;
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::Tee;
    use std::io::Write;

    #[test]
    fn tee_writes_console_and_file() {
        let path = std::env::temp_dir().join(format!("eva01-tee-{}.log", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let mut tee = Tee {
            console: Vec::new(),
            file,
        };
        tee.write_all(b"line one\n").unwrap();
        tee.write_all(b"line two\n").unwrap();
        tee.flush().unwrap();

        assert_eq!(tee.console, b"line one\nline two\n");
        assert_eq!(std::fs::read(&path).unwrap(), b"line one\nline two\n");
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod log_tests {
    use super::redact_urls;

    #[test]
    fn a_logged_url_keeps_only_its_host() {
        assert_eq!(
            redact_urls("error sending request for url (https://rpc.example.com/0212-secret)"),
            "error sending request for url (https://rpc.example.com/…)"
        );
        assert_eq!(
            redact_urls("GET https://api.example.com/v1/x?api-key=secret failed, http://h:8899 ok"),
            "GET https://api.example.com/… failed, http://h:8899 ok"
        );
        assert_eq!(
            redact_urls("connecting to https://mainnet.block-engine.jito.wtf as X"),
            "connecting to https://mainnet.block-engine.jito.wtf as X"
        );
        assert_eq!(redact_urls("no url here"), "no url here");
        assert_eq!(redact_urls("https://user:token@h/x é"), "https://h/… é");
        assert_eq!(redact_urls("wss://h/a/b"), "wss://h/…");
    }
}
