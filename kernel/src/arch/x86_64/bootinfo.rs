//! x86_64-specific boot information

use core::ffi::c_void;
use core::ptr;

use crate::{
    arch::{CR3, generic::paging::PhysicalAddress},
    driver::acpi::rsdp::RSDP,
};

/// x86_64-specific boot information
#[repr(C)]
#[derive(Debug)]
pub struct Bootinfo {
    /// x86_64's `cr3` register used by UEFI.
    pub uefi_cr3: CR3,

    /// Physical address of x86_64's descriptor of the GDT used by UEFI.
    pub uefi_gdt: PhysicalAddress<c_void>, // TODO(@MaxMade): replace by actual GDT type

    /// Physical address of x86_64's descriptor of the IDT used by UEFI.
    pub uefi_idt: PhysicalAddress<c_void>, // TODO(@MaxMade): replace by actual IDT type

    /// Base address of x86_64's `gs` segment used by UEFI.
    pub gs: usize,

    /// Physical address of the ACPI Root System Description Pointer.
    ///
    /// Located by the bootloader and validated by the ACPI driver, see
    /// [`RSDP::verify`].
    pub rsdp: PhysicalAddress<RSDP>,
}

impl Default for Bootinfo {
    fn default() -> Self {
        Self {
            uefi_cr3: Default::default(),
            uefi_gdt: PhysicalAddress::new(ptr::null_mut()),
            uefi_idt: PhysicalAddress::new(ptr::null_mut()),
            gs: 0,
            rsdp: PhysicalAddress::new(ptr::null_mut()),
        }
    }
}
