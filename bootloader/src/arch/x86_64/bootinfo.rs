//! Handle `x86_64` specifics of the boot information.

use busyos::kernel::bootinfo::Bootinfo;

pub fn update_bootinfo(bootinfo: &mut Bootinfo) {
    bootinfo.arch_bootinfo.uefi_cr3 = busyos::arch::x86_64::paging::CR3::read();

    // TODO(@MaxMade): Save address of GDT

    // TODO(@MaxMade): Save address of IDT
}
