use std::{ffi::c_void, path::PathBuf, process::ExitCode};

use log::LevelFilter;

use clap::Parser;
use object::{Object, ObjectSection};

#[derive(Parser, Debug)]
#[command(name = "check-stack-usage", version, about)]
struct Cli {
    /// Path to the ELF file to analyze.
    elf: PathBuf,

    /// Maximum permitted stack size (in bytes) for any single function.
    #[arg(
        short = 'w',
        long = "warn",
        value_name = "BYTES",
        default_value_t = 1024
    )]
    warn: u64,

    /// Increase log verbosity. (May be repeated: -v, -vv, -vvv)
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    verbose: u8,
}

fn read_uleb128(buf: &[u8]) -> Result<(u64, usize), &'static str> {
    let mut result: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in buf.iter().enumerate() {
        // Low 7 bits contribute to the value.
        result |= ((byte & 0x7F) as u64) << shift;

        // High bit clear means this was the last byte.
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }

        shift += 7;
        if shift >= 64 {
            return Err("ULEB128 value too large (more than 64 bits)");
        }
    }

    return Err("truncated ULEB128 sequence at end of section");
}

fn main() -> ExitCode {
    // Parse command-line arguments
    let mut cli = Cli::parse();

    // Set log level
    let level = match cli.verbose {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    env_logger::Builder::new()
        .filter_level(level)
        .format_timestamp(None)
        .format_target(false)
        .init();

    // Check warn size
    if cli.warn == 0 {
        log::warn!("Invalid threshold stack size: 0. Assuming warn=1 instead");
        cli.warn = 1;
    }

    // Try to open ELF
    log::info!("Read ELF file");
    let file = match std::fs::read(&cli.elf) {
        Ok(file) => file,
        Err(error) => {
            log::error!("Unable to open {:?}: {}", cli.elf, error);
            return ExitCode::FAILURE;
        }
    };

    // Start parsing ELF file
    let elf = match object::File::parse(&*file) {
        Ok(elf) => elf,
        Err(error) => {
            log::error!("Unable to parse ELF: {}", error);
            return ExitCode::FAILURE;
        }
    };

    // Check architecture
    log::info!("Check target architecture");
    match elf.architecture() {
        object::Architecture::Aarch64 => { /* Nothing to do here */ }
        object::Architecture::X86_64 => { /* Nothing to do here */ }
        arch => {
            log::error!("Unsupported architecture: {:?}", arch);
            return ExitCode::FAILURE;
        }
    };

    // Get list of symbols
    log::info!("Trying to get symbol table map");
    let symbols = elf.symbol_map();

    // Try to parse `.stack_sizes` section
    log::info!("Find `.stack_size` section");
    let stack_sizes = match elf.section_by_name(".stack_sizes") {
        Some(stack_sizes) => stack_sizes,
        None => {
            log::error!("Unable to find `.stack_sizes` section");
            return ExitCode::FAILURE;
        }
    };

    // Try to read `.stack_sizes` section
    log::info!("Try to read `.stack_size` section");
    let data = match stack_sizes.data() {
        Ok(data) => data,
        Err(error) => {
            log::error!("Unable to read `.stack_sizes` section: {}", error);
            return ExitCode::FAILURE;
        }
    };

    let mut offset = 0;

    log::info!("Check stack size of all functions");
    let mut exit_code = ExitCode::SUCCESS;
    while offset < data.len() {
        // Function address, little-endian, target pointer width.
        let address = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += std::mem::size_of::<u64>();

        // ULEB128-encoded stack size immediately follows.
        let (stack_size, consumed) = match read_uleb128(&data[offset..]) {
            Ok(result) => result,
            Err(error) => {
                log::error!("Unable to parse ULEB128-encoded stack size: {}", error);
                return ExitCode::FAILURE;
            }
        };
        offset += consumed;

        let symbol = match symbols.containing(address) {
            Some(symbol) => symbol.name().to_string(),
            None => {
                let ptr = format!("{:p}", address as *const c_void);
                log::warn!("Unable to find symbol for address {}", ptr);
                ptr
            }
        };

        log::debug!("{}: {} byte(s)", symbol, stack_size);

        if stack_size > cli.warn {
            if exit_code == ExitCode::SUCCESS {
                eprintln!("Detected exceeded stack size:");
            }
            exit_code = ExitCode::FAILURE;

            println!(
                "{}: {} ({:.2}% over)",
                symbol,
                stack_size,
                (((stack_size - cli.warn) as f32) / (cli.warn as f32)) * 100.0
            );
        }
    }

    exit_code
}
