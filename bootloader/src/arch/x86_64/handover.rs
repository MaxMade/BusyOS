use core::ffi::c_void;

use busyos::{
    arch::{
        CPU,
        generic::paging::{Paging as _, PhysicalAddress, VirtualAddress},
    },
    kernel::{
        bootinfo::Bootinfo,
        locking::{CanAcquire, PreviousToken, level::Epilogue},
    },
};
use uefi::table::cfg::ConfigTableEntry;

use crate::paging::Paging;

unsafe extern "C" {
    fn _start() -> i32;
}

pub struct HandOver {
    /// Temporary page tables used to jump into the higher half.
    paging: Paging,

    /// Virtual address of the kernel entry symbol.
    entry: VirtualAddress<c_void>,
}

impl crate::arch::generic::handover::HandOver for HandOver {
    fn prepare<Token>(bootinfo: &mut Bootinfo, token: Token) -> (Self, Token)
    where
        Token: CanAcquire<Epilogue> + PreviousToken,
    {
        // Save UEFI `cr3` register
        bootinfo.arch_bootinfo.uefi_cr3 = busyos::arch::x86_64::paging::CR3::read();

        // TODO(@MaxMade): Save address of UEFI's GDT

        // TODO(@MaxMade): Save address of UEFI's IDT

        // Save value of UEFI's `gs` register
        bootinfo.arch_bootinfo.gs = unsafe { CPU::gs_base() };

        // Save address of RSDP
        let rsdp = uefi::system::with_config_table(|entries| {
            entries
                .iter()
                .find(|entry| entry.guid == ConfigTableEntry::ACPI2_GUID)
                .or_else(|| {
                    entries
                        .iter()
                        .find(|entry| entry.guid == ConfigTableEntry::ACPI_GUID)
                })
                .map(|entry| PhysicalAddress::new(entry.address as _))
        });

        bootinfo.arch_bootinfo.rsdp = match rsdp {
            Some(rsdp) => rsdp,
            None => panic!("Unable to find RSDP"),
        };

        // Save the address of `_start` symbol
        let entry = bootinfo.kernel_start_symbol;

        // Prepare temporary mapping for jumping to higher half kernel
        let (paging, token) = crate::paging::prepare_handover(token);

        (Self { paging, entry }, token)
    }

    unsafe fn handover(&mut self, cpu_id: usize) -> bool {
        // Activate temporary mapping
        unsafe { self.paging.activate() };

        // Jump to BUSYOS kernel
        let ptr = self.entry.as_ptr() as *const c_void;

        let entry: extern "C" fn(usize) -> i32 = unsafe { core::mem::transmute(ptr) };

        // TODO(@MaxMade): Currently only one CPU supported...
        let success = entry(cpu_id) == 0;

        success
    }
}
