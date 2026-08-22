//! Architecture-specific ELF handling.

use elf::{ElfBytes, endian::AnyEndian};

/// Architecture-specific ELF handling.
pub trait ELF {
    /// Relocation type that asks for the load bias to be added to a
    /// link-time address.
    ///
    /// This is the only relocation type a statically linked kernel image is
    /// expected to carry: it has no dynamic symbols to resolve, only its own
    /// absolute addresses to adjust. Every other type is rejected by the
    /// loader.
    const R_RELATIVE: u32;

    /// Checks if the target kernel ELF file is valid for target architecture.
    ///
    /// # Panics
    ///
    /// If any check fails, `panic` will be triggered.
    fn check_header(elf: &ElfBytes<AnyEndian>);
}
