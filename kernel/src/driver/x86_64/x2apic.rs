//! The *Local Advanced Programmable Interrupt Controller* in x2APIC mode.
//!
//! Every logical processor owns one local APIC, which is the core's interface
//! to the interrupt subsystem. x2APIC is its extended mode: the register file
//! is reached through MSRs rather than a memory-mapped page, and the
//! identifier of a local APIC widens from 8 to 32 bits.
//!
//! The controller itself is not programmed here yet. For now the module only
//! provides [`LapicID`], so that the ACPI driver can record the identifiers
//! the firmware reports in its
//! [`MADT`](crate::driver::acpi::madt::MADT) while walking the tables.

/// The 32-bit identifier of a local APIC in x2APIC mode.
///
/// The identifier is assigned by the hardware and is unique across the system.
/// It names the core a local APIC belongs to and is therefore what an
/// interrupt is addressed to.
///
/// The value is not an index: the firmware is free to leave gaps, so the
/// identifiers of `n` cores are not necessarily `0..n`.
pub struct LapicID(u32);

impl LapicID {
    /// Wraps a raw 32-bit identifier.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` is an identifier the platform
    /// actually reports, either through a firmware table such as the
    /// [`MADT`](crate::driver::acpi::madt::MADT) or through the local APIC's
    /// own ID register. Constructing an identifier no local APIC answers to
    /// is safe in the memory-safety sense, but interrupts addressed to it are
    /// never delivered.
    pub const unsafe fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}
