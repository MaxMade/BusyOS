use core::ffi::c_void;

use crate::arch::generic::paging::PhysicalAddress;
use crate::arch::Bootinfo as ArchBootinfo;

#[repr(C)]
#[derive(Debug)]
pub struct Bootinfo {
    /// Offset between every virtual and physical address of the kernel
    pub kernel_virt_phys_offset: usize,

    /// Physical address of the kernel ELF file.
    pub kernel_elf_start: PhysicalAddress<c_void>,

    /// Size of the kernel ELF file in physical memory.
    pub kernel_elf_size: usize,

    /// Physical address of kernel start.
    pub kernel_start_phys: PhysicalAddress<c_void>,
    /// Physical address of kernel end (exclusive).
    pub kernel_end_phys: PhysicalAddress<c_void>,

    /// Physical start address of kernel `.text` segment.
    pub text_start_phys: PhysicalAddress<c_void>,
    /// Physical end address of kernel `.text` segment (exclusive).
    pub text_end_phys: PhysicalAddress<c_void>,

    /// Physical start address of kernel `.rodata` segment.
    pub rodata_start_phys: PhysicalAddress<c_void>,
    /// Physical end address of kernel `.rodata` segment (exclusive).
    pub rodata_end_phys: PhysicalAddress<c_void>,

    /// Physical start address of kernel `.data` segment.
    pub data_start_phys: PhysicalAddress<c_void>,
    /// Physical end address of kernel `.data` segment (exclusive).
    pub data_end_phys: PhysicalAddress<c_void>,

    /// Physical start address of kernel `.bss` segment.
    pub bss_start_phys: PhysicalAddress<c_void>,
    /// Physical end address of kernel `.bss` segment (exclusive).
    pub bss_end_phys: PhysicalAddress<c_void>,

    /// Architecture-specific boot information
    pub arch_bootinfo: ArchBootinfo,
}


/// Gets address of [`ArchBootinfo`] for [`Bootinfo`].
#[unsafe(no_mangle)]
pub extern "C" fn __kernel_bootinfo_get_arch_bootinfo(bootinfo: *mut Bootinfo) -> *mut ArchBootinfo {   
    let bootinfo = unsafe { bootinfo.as_mut().unwrap() };
    core::ptr::addr_of_mut!(bootinfo.arch_bootinfo)
}
