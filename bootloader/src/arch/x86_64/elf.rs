//! Handle `x86_64`-specific internals of ELFs.

use elf::{ElfBytes, endian::AnyEndian};

/// Checks if the target kernel ELF file is valid for `x86_64`.
///
/// # Panics
///
/// If any check fails, `panic` will be triggered.
pub fn check_elf_header(elf: &ElfBytes<AnyEndian>) {
    if elf.ehdr.class != elf::file::Class::ELF64 {
        panic!("Unexpected ELF class: {:?}", elf.ehdr.class);
    }

    if elf.ehdr.e_type != elf::abi::ET_DYN {
        panic!("Unexpected ELF type: {:?}", elf.ehdr.e_type);
    }

    if elf.ehdr.e_machine != elf::abi::EM_X86_64 {
        panic!("Unexpected ELF machine: {:?}", elf.ehdr.e_machine);
    }
}
