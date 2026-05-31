#![no_std]
#![no_main]

pub mod elf_loader;

extern crate alloc;

use uefi::prelude::*;
use uefi::println;

/// Path to the kernel ELF on the ESP, relative to the volume root.
pub const KERNEL_PATH: &str = r"\EFI\BOOT\busyos.elf";

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();

    // Starting the bootloader
    println!("Booting BUSYOS!");

    // Try to load BUSYOS kernel ELF
    let handle = uefi::boot::image_handle();
    let kernel_elf = elf_loader::ELF::load(KERNEL_PATH, handle);

    // Exit boot service.
    //
    // # Safety
    //
    // From then on, only UEFI configuration tables and runtime service can be used.
    let memory_map = unsafe { uefi::boot::exit_boot_services(None) };

    // Perform shutdown
    uefi::runtime::reset(runtime::ResetType::SHUTDOWN, Status::SUCCESS, None)
}
