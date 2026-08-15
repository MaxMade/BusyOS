use core::ffi::c_void;

use busyos::{
    arch::generic::paging::{Paging as _, VirtualAddress},
    kernel::{
        bootinfo::Bootinfo,
        locking::{CanAcquire, PreviousToken, level::Epilogue},
    },
};

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

        // Save the address of `_start` symbol
        //
        // XXX: the temporary mapping creates an identity mapping for the first
        // 512 GiB of the address space. The first 512 GiB of the upper half are
        // also mapped to the 512 GiB of the physical address space. Therefore,
        // we make use of BUSYOS being position-independent: we interpret the
        // physical address oft the kernel image as an offset and add it to the
        // start address.
        let entry = unsafe {
            bootinfo.kernel_start_symbol.byte_add(bootinfo.kernel_text_start.addr())
        };

        // Prepare temporary mapping for jumping to higher half kernel
        let (paging, token) = crate::paging::prepare_handover(token);

        (Self { paging, entry }, token)
    }

    unsafe fn handover(&mut self) -> bool {
        // Activate temporary mapping
        unsafe { self.paging.active() };

        // Jump to BUSYOS kernel
        let ptr = self.entry.as_ptr() as *const c_void;
        let entry: extern "C" fn() -> i32 = unsafe { core::mem::transmute(ptr) };
        let success = entry() == 0;

        success
    }
}
