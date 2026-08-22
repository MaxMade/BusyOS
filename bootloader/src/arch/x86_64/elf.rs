//! x86_64-specific ELF handling.

use elf::{ElfBytes, endian::AnyEndian};

/// x86_64-specific ELF handling.
#[derive(Debug)]
pub struct ELF;

impl crate::arch::generic::elf::ELF for ELF {
    const R_RELATIVE: u32 = elf::abi::R_X86_64_RELATIVE;

    fn check_header(elf: &ElfBytes<AnyEndian>) {
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
}
