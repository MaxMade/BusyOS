use core::array;
use core::ffi::c_void;
use core::mem::MaybeUninit;

use crate::arch::Bootinfo as ArchBootinfo;
use crate::arch::generic::paging::{PhysicalAddress, VirtualAddress};
use crate::utils::range_tree::Range;

#[repr(C)]
#[derive(Debug)]
pub struct Bootinfo {
    /// Number of available CPUs.
    pub num_cpus: usize,

    /// Offset between every virtual and physical address of the kernel
    pub kernel_virt_phys_offset: usize,

    /// Virtual address of the `_start` symbol.
    pub kernel_start_symbol: VirtualAddress<c_void>,

    /// Physical address of the kernel ELF file.
    pub kernel_elf_start: PhysicalAddress<c_void>,

    /// Size of the kernel ELF file in physical memory.
    pub kernel_elf_size: usize,

    /// Physical address of the kernel `.text` segment.
    pub kernel_text_start: PhysicalAddress<c_void>,

    /// Size of the kernel `.text` segment.
    pub kernel_text_size: usize,

    /// Physical address of the kernel `.rodata` segment.
    pub kernel_rodata_start: PhysicalAddress<c_void>,

    /// Size of the kernel `.rodata` segment.
    pub kernel_rodata_size: usize,

    /// Physical address of the kernel `.data` segment.
    pub kernel_data_start: PhysicalAddress<c_void>,

    /// Size of the kernel `.data` segment.
    pub kernel_data_size: usize,

    /// Physical address of the kernel `.bss` segment.
    pub kernel_bss_start: PhysicalAddress<c_void>,

    /// Size of the kernel `.bss` segment.
    pub kernel_bss_size: usize,

    /// Physical address of the kernel `.percpu` segment.
    ///
    /// This is core 0's block: the bootloader replicates the template in
    /// place, so the first block is the template itself.
    pub kernel_percpu_start: PhysicalAddress<c_void>,

    /// Size of the kernel `.percpu` segment, spanning the blocks of *all*
    /// [`num_cpus`](Self::num_cpus) cores rather than a single one.
    ///
    /// The blocks sit one stride apart in one contiguous range, so mapping
    /// this range maps every core's block.
    pub kernel_percpu_size: usize,

    pub memory_ranges: [Range<PhysicalAddress<c_void>, usize>; 64],

    /// Architecture-specific boot information
    pub arch_bootinfo: ArchBootinfo,
}

impl Default for Bootinfo {
    fn default() -> Self {
        Self {
            kernel_virt_phys_offset: 0,
            kernel_start_symbol: VirtualAddress::null(),
            kernel_elf_start: PhysicalAddress::null(),
            kernel_elf_size: 0,
            arch_bootinfo: Default::default(),
            kernel_text_start: PhysicalAddress::null(),
            kernel_text_size: 0,
            kernel_rodata_start: PhysicalAddress::null(),
            kernel_rodata_size: 0,
            kernel_data_start: PhysicalAddress::null(),
            kernel_data_size: 0,
            kernel_bss_start: PhysicalAddress::null(),
            kernel_bss_size: 0,
            kernel_percpu_start: PhysicalAddress::null(),
            kernel_percpu_size: 0,
            num_cpus: 0,
            memory_ranges: array::from_fn(|_| Range::new(PhysicalAddress::null(), 0)),
        }
    }
}

#[unsafe(no_mangle)]
pub static BOOTINFO: MaybeUninit<Bootinfo> = MaybeUninit::zeroed();
