use core::ffi::c_void;
use core::ptr;

use crate::arch::Bootinfo as ArchBootinfo;
use crate::arch::generic::paging::PhysicalAddress;

#[repr(C)]
#[derive(Debug)]
pub struct Bootinfo {
    /// Offset between every virtual and physical address of the kernel
    pub kernel_virt_phys_offset: usize,

    /// Physical address of the kernel ELF file.
    pub kernel_elf_start: PhysicalAddress<c_void>,

    /// Size of the kernel ELF file in physical memory.
    pub kernel_elf_size: usize,

    /// Architecture-specific boot information
    pub arch_bootinfo: ArchBootinfo,
}

impl Default for Bootinfo {
    fn default() -> Self {
        Self {
            kernel_virt_phys_offset: 0,
            kernel_elf_start: PhysicalAddress::new(ptr::null_mut()),
            kernel_elf_size: 0,
            arch_bootinfo: Default::default(),
        }
    }
}

/// Gets address of [`ArchBootinfo`] for [`Bootinfo`].
#[unsafe(no_mangle)]
pub extern "C" fn __kernel_bootinfo_get_arch_bootinfo(
    bootinfo: *mut Bootinfo,
) -> *mut ArchBootinfo {
    let bootinfo = unsafe { bootinfo.as_mut().unwrap() };
    core::ptr::addr_of_mut!(bootinfo.arch_bootinfo)
}
