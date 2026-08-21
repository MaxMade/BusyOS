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

    /// Required minimum stack alignment.
    const STACK_ALIGNMENT: usize = 16;

    /// Kernel stack size.
    const KERNEL_STACK_SIZE: usize = 16 * 1024;
}

impl CPU {
    /// Reads the current `GS` base address using `RDGSBASE`.
    ///
    /// # Safety
    ///
    /// `RDGSBASE` raises `#UD` unless `CR4.FSGSBASE` is set, which in turn
    /// requires the processor to support
    /// [`fsgsbase`](crate::arch::x86_64::cpuid::StructuredExtendedFeatureEBX::fsgsbase).
    #[inline]
    pub unsafe fn gs_base() -> usize {
        let base: usize;

        unsafe {
            core::arch::asm!(
                "rdgsbase {}",
                out(reg) base,
                options(nomem, nostack, preserves_flags)
            );
        }

        base
    }

    /// Sets the `GS` base address to `base` using `WRGSBASE`.
    ///
    /// # Safety
    ///
    /// As for [`gs_base`](CPU::gs_base), `WRGSBASE` requires `CR4.FSGSBASE`
    /// to be set.
    ///
    /// Additionally, `GS` is the anchor of this core's core-local storage: a
    /// base that does not point at this core's block makes every subsequent
    /// [`PerCPU`](crate::kernel::core_local::PerCPU) access read or write
    /// unrelated memory. `base` must be canonical, otherwise the write
    /// raises `#GP`.
    #[inline]
    pub unsafe fn set_gs_base(base: usize) {
        unsafe {
            core::arch::asm!(
                "wrgsbase {}",
                in(reg) base,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
}
