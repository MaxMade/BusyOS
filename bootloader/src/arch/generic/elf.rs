//! Architecture-specific ELF handling.

use elf::{ElfBytes, endian::AnyEndian};

/// Architecture-specific ELF handling.
pub trait ELF {
    /// Checks if the target kernel ELF file is valid for target architecture.
    ///
    /// # Panics
    ///
    /// If any check fails, `panic` will be triggered.
    fn check_header(elf: &ElfBytes<AnyEndian>);
}
