use core::ffi::c_void;

use busyos::{
    arch::generic::paging::PhysicalAddress, kernel::bootinfo::Bootinfo, utils::range_tree::Range,
};
use uefi::{
    boot::{MemoryType, PAGE_SIZE},
    mem::memory_map::{MemoryMap, MemoryMapMut, MemoryMapOwned},
};

/// Collects the system's usable physical memory into `bootinfo`.
///
/// `bootinfo.memory_ranges` ends up holding the conventional memory as
/// coalesced ranges in ascending order, followed by zero-length entries for
/// the slots that stay unused.
///
/// # Panics
///
/// If the coalesced ranges outnumber the slots in `bootinfo.memory_ranges`.
pub fn parse(mut memory_map: MemoryMapOwned, bootinfo: &mut Bootinfo) {
    if memory_map.len() > 1 {
        memory_map.sort();
    }

    let mut ranges = 0;

    for mem_desc in memory_map.entries() {
        if mem_desc.ty != MemoryType::CONVENTIONAL {
            continue;
        }

        let range = Range::new(
            PhysicalAddress::new(mem_desc.phys_start as *mut c_void),
            mem_desc.page_count as usize * PAGE_SIZE,
        );

        if ranges > 0 {
            let last = &mut bootinfo.memory_ranges[ranges - 1];

            if let Ok(merged) = last.try_merge(range) {
                *last = merged;
                continue;
            }
        }

        // TODO(@MaxMade): Handle these situations accordingly
        if ranges >= bootinfo.memory_ranges.len() {
            panic!(
                "more disjoint memory ranges than the {} slots of `Bootinfo`",
                bootinfo.memory_ranges.len()
            );
        }

        bootinfo.memory_ranges[ranges] = range;
        ranges += 1;
    }
}
