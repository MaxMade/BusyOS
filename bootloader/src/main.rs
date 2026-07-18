#![no_std]
#![no_main]

pub mod arch;
pub mod elf_loader;
pub mod paging;

extern crate alloc;

use busyos::kernel::locking::RootToken;
use busyos::kernel::locking::SyscallLevel;
use uefi::prelude::*;
use uefi::println;

use busyos::kernel::bootinfo::Bootinfo;

use crate::arch::generic::features::Features;
use crate::arch::generic::handover::HandOver;

/// Path to the kernel ELF on the ESP, relative to the volume root.
pub const KERNEL_PATH: &str = r"\EFI\BOOT\busyos.elf";

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();

    // Starting the bootloader
    println!("Welcome to BUSYOS!");
    println!("Starting BUSYOS UEFI Bootloader...");

    // Simulate system-call entry to get top-level token
    let root_token = unsafe { RootToken::forge() };
    let (syscall_level, token) = SyscallLevel::enter(root_token);

    // Try to load BUSYOS kernel ELF
    let mut bootinfo = Bootinfo::default();
    let handle = uefi::boot::image_handle();
    let kernel_elf = elf_loader::ELF::read(KERNEL_PATH, handle);
    kernel_elf.load(&mut bootinfo);

    // Check and active extensions
    arch::Features::activate();

    // Create page tables for hand-over
    let (mut handover, token) = arch::HandOver::prepare(&mut bootinfo, token);

    // Begin handover
    println!("Beginning handover to BUSYOS Kernel...");

    // Exit boot service.
    //
    // # Safety
    //
    // From then on, only UEFI configuration tables and runtime service can be used.
    let memory_map = unsafe { uefi::boot::exit_boot_services(None) };

    // Simulate system-call exit
    syscall_level.leave(token);

    // Perform handover
    let poweroff = unsafe { handover.handover() };

    // Perform shutdown
    if poweroff {
        uefi::runtime::reset(runtime::ResetType::SHUTDOWN, Status::SUCCESS, None);
    } else {
        uefi::runtime::reset(runtime::ResetType::WARM, Status::SUCCESS, None);
    }
}
