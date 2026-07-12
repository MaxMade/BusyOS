use busyos::kernel::{bootinfo::Bootinfo, locking::{CanAcquire, PreviousToken, level::Epilogue}};

#[derive(Debug)]
pub struct HandOver;

impl crate::arch::generic::handover::HandOver for HandOver {
    fn prepare<Token>(bootinfo: &mut Bootinfo, token: Token) -> (Self, Token)
    where
        Token: CanAcquire<Epilogue> + PreviousToken {
        // Save UEFI `cr3` register
        bootinfo.arch_bootinfo.uefi_cr3 = busyos::arch::x86_64::paging::CR3::read();

        // TODO(@MaxMade): Save address of UEFI's GDT

        // TODO(@MaxMade): Save address of UEFI's IDT

        todo!()
    }

    unsafe fn handover(&mut self) -> bool {
        todo!()
    }
}
