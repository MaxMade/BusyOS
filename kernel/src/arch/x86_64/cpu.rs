//! x86_64 implementation of the generic [`CPU`](crate::arch::generic::cpu::CPU)
//! interface.

use crate::arch::generic::cpu::InterruptFlag;
use crate::arch::x86_64::rflags::RFLAGS;

#[derive(Debug)]
pub struct CPU;

impl crate::arch::generic::cpu::CPU for CPU {
    /// Returns whether maskable interrupts are currently enabled.
    ///
    /// Determined by the `IF` bit of the [`RFLAGS`] register.
    #[inline]
    fn interrupt_flag() -> InterruptFlag {
        if RFLAGS::read().interrupt() {
            InterruptFlag::Enabled
        } else {
            InterruptFlag::Disabled
        }
    }

    /// Enables maskable interrupts by executing `STI`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the interrupt handling infrastructure (IDT,
    /// per-CPU state, ...) is fully initialized and that enabling interrupts
    /// does not break the invariants of an enclosing critical section.
    #[inline]
    unsafe fn raw_enable_interrupts() {
        unsafe {
            core::arch::asm!(
                "sti",
                options(nostack, preserves_flags)
            );
        }
    }

    /// Disables maskable interrupts by executing `CLI`.
    ///
    /// # Safety
    ///
    /// The caller is responsible for restoring the previous interrupt state,
    /// otherwise interrupts stay masked indefinitely.
    #[inline]
    unsafe fn raw_disable_interrupts() {
        unsafe {
            core::arch::asm!(
                "cli",
                options(nostack, preserves_flags)
            );
        }
    }
}
