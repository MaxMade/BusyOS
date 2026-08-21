//! Abstraction for the x86_64 `cr4` control register.

use bitfield_struct::bitfield;

/// x86_64 `cr4` register.
///
/// Holds the enable bits of a number of processor extensions. Only the bits
/// used by the kernel are named; every other bit is left untouched by a
/// read-modify-write cycle, since [`read`](CR4::read) preserves the raw value.
#[bitfield(u64)]
pub struct CR4 {
    /// Unused by the kernel (bits [6:0]).
    #[bits(7)]
    __: u8,

    /// Page Global Enable (`PGE`, bit 7).
    ///
    /// When set, page table entries marked global survive a `cr3` reload.
    #[bits(1)]
    pub pge: bool,

    /// Unused by the kernel (bits [15:8]).
    #[bits(8)]
    __: u8,

    /// `FSGSBASE` instructions (bit 16).
    ///
    /// Must be set before executing `RDFSBASE`, `RDGSBASE`, `WRFSBASE` or
    /// `WRGSBASE`; otherwise those instructions raise `#UD`. Availability
    /// must be checked via
    /// [`StructuredExtendedFeatureEBX::fsgsbase`](crate::arch::x86_64::cpuid::StructuredExtendedFeatureEBX::fsgsbase)
    /// beforehand.
    #[bits(1)]
    pub fsgsbase: bool,

    /// Unused by the kernel (bits [21:17]).
    #[bits(5)]
    __: u8,

    /// Protection Key Enable (`PKE`, bit 22).
    ///
    /// When set, the protection-key field of a page table entry selects one of
    /// 16 protection-key domains.
    #[bits(1)]
    pub pke: bool,

    /// Unused by the kernel (bits [63:23]).
    #[bits(41)]
    __: u64,
}

impl CR4 {
    /// Reads the current value of the `cr4` register.
    pub fn read() -> Self {
        let val: u64;
        unsafe {
            core::arch::asm!(
                "mov {}, cr4",
                out(reg) val,
                options(nomem, nostack, preserves_flags)
            );
        }
        Self(val)
    }

    /// Writes `self` to the `cr4` register.
    ///
    /// # Safety
    ///
    /// Every bit of this register changes how the processor interprets state
    /// that is already in place: clearing an enable bit can turn instructions
    /// in use into `#UD`, and changing a paging-related bit reinterprets the
    /// active page tables. The caller must ensure that the written value is
    /// consistent with the state the kernel is running on, and that the
    /// corresponding feature is supported by the processor.
    pub unsafe fn write(self) {
        unsafe {
            core::arch::asm!(
                "mov cr4, {}",
                in(reg) self.0,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
}
