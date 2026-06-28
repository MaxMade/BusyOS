//! AMD64-specific boot information

use core::ffi::c_void;

use crate::arch::{CR3, generic::paging::PhysicalAddress};

/// AMD64-specific boot information
#[repr(C)]
#[derive(Debug)]
pub struct Bootinfo {
    /// AMD64's `cr3` register used by UEFI.
    pub uefi_cr3: CR3,

    /// Physical address of x86_64's descriptor of the GDT used by UEFI.
    pub uefi_gdt: PhysicalAddress<c_void>, // TODO(@MaxMade): replace by actual GDT type

    /// Physical address of x86_64's descriptor of the IDT used by UEFI.
    pub uefi_idt: PhysicalAddress<c_void>, // TODO(@MaxMade): replace by actual IDT type
}

/// Stores `cr3` register in [`Bootinfo`].
#[unsafe(no_mangle)]
pub extern "C" fn __arch_amd64_bootinfo_store_cr3(bootinfo: *mut Bootinfo, cr3: u64) {
    let bootinfo = unsafe { bootinfo.as_mut().unwrap() };
    bootinfo.uefi_cr3 = CR3::try_from(cr3).unwrap();
}

/// Stores address of `gdt` (*G*lobal *D*escriptor *T*able) descriptor in [`Bootinfo`].
#[unsafe(no_mangle)]
pub extern "C" fn __arch_amd64_bootinfo_store_gdt(bootinfo: *mut Bootinfo, gdt: u64) {
    let bootinfo = unsafe { bootinfo.as_mut().unwrap() };
    bootinfo.uefi_gdt = PhysicalAddress::new(gdt as _);
}

/// Stores address of `idt` (*I*nterrupt *D*escriptor *T*able)  descriptor in [`Bootinfo`].
#[unsafe(no_mangle)]
pub extern "C" fn __arch_amd64_bootinfo_store_idt(bootinfo: *mut Bootinfo, idt: u64) {
    let bootinfo = unsafe { bootinfo.as_mut().unwrap() };
    bootinfo.uefi_idt = PhysicalAddress::new(idt as _);
}
