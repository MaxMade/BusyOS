use busyos::{arch::generic::paging::Paging as _, kernel::{
    bootinfo::Bootinfo,
    locking::{CanAcquire, PreviousToken, level::Epilogue},
}};

use crate::paging::Paging;

pub struct HandOver {
    /// Temporary page tables used to jump into the higher half.
    paging: Paging,
}

impl crate::arch::generic::handover::HandOver for HandOver {
    fn prepare<Token>(bootinfo: &mut Bootinfo, token: Token) -> (Self, Token)
    where
        Token: CanAcquire<Epilogue> + PreviousToken {
        // Save UEFI `cr3` register
        bootinfo.arch_bootinfo.uefi_cr3 = busyos::arch::x86_64::paging::CR3::read();

        // TODO(@MaxMade): Save address of UEFI's GDT

        // TODO(@MaxMade): Save address of UEFI's IDT

        let (paging, token) = crate::paging::prepare_handover(token);

        (Self { paging }, token)
    }

    unsafe fn handover(&mut self) -> bool {
        // Activate temporary mapping
        unsafe { self.paging.active() };

        todo!();
    }
}
