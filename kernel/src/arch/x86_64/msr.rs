//! Abstractions for x86_64 *Model Specific Registers* (MSRs).
//!
//! MSRs are processor-specific 64-bit registers accessed using the `RDMSR`
//! and `WRMSR` instructions. They provide control over processor features
//! such as long mode, system calls, memory management, virtualization, and
//! performance monitoring.
//!
//! This module provides a trait for implementing typed wrappers around
//! individual MSRs, allowing register contents to be represented as
//! strongly-typed Rust structures instead of raw `u64` values.

use core::fmt::Debug;

use bitfield_struct::bitfield;

/// A typed wrapper around an x86_64 *Model Specific Register* (MSR).
///
/// Each implementation corresponds to a single MSR identified by its
/// compile-time address (`ADDR`).
///
/// Implementations are typically thin wrappers around a `u64` or a bitfield
/// representation that expose the individual register fields.
///
/// # Safety
///
/// Reading from or writing to an MSR is inherently unsafe because:
///
/// - the addressed MSR may not exist on the current processor, accessing an
/// - unsupported MSR raises a `#GP` exception,
///
/// - writing invalid values may leave the processor in an undefined or
///   unrecoverable state.
///
/// Callers must ensure that the targeted MSR is implemented and that any
/// written values satisfy the architectural requirements described in the
/// processor's software developer manual.
pub trait MSR<const ADDR: u32>
where
    Self: Clone + Copy + Debug,
{
    /// Returns the raw 64-bit value of this register.
    fn raw(&self) -> u64;

    /// Constructs a typed register from its raw 64-bit representation.
    ///
    /// # Safety
    ///
    /// The supplied value must represent a valid encoding of this register.
    /// Reserved bits or architecturally invalid field combinations may violate
    /// the invariants expected by the implementation.
    unsafe fn from_raw(value: u64) -> Self;

    /// Reads the MSR from the current processor.
    ///
    /// Executes the `RDMSR` instruction using the MSR address specified by
    /// `ADDR`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the processor implements this MSR.
    /// Otherwise, a `#GP` exception is generated.
    #[inline]
    unsafe fn read() -> Self {
        let low: u32;
        let high: u32;

        unsafe {
            core::arch::asm!(
                "rdmsr",
                in("ecx") ADDR,
                out("eax") low,
                out("edx") high,
                options(nomem, nostack)
            );
        }

        let value = ((high as u64) << 32) | low as u64;

        unsafe { Self::from_raw(value) }
    }

    /// Writes this value to the MSR.
    ///
    /// Executes the `WRMSR` instruction using the MSR address specified by
    /// `ADDR`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - the processor implements this MSR,
    /// - the value satisfies all architectural constraints for this register.
    ///
    /// Writing reserved or invalid values may result in a `#GP` exception or
    /// processor misconfiguration.
    #[inline]
    unsafe fn write(&self) {
        let value = self.raw();

        unsafe {
            core::arch::asm!(
                "wrmsr",
                in("ecx") ADDR,
                in("eax") value as u32,
                in("edx") (value >> 32) as u32,
                options(nomem, nostack)
            )
        };
    }
}

/// Extended Feature Enable Register (`IA32_EFER`, MSR `0xC000_0080`).
///
/// The Extended Feature Enable Register controls several processor features,
/// including
///
/// # Safety
/// Before modifying [`EFER::sce`] and [`EFER::nxe`],
/// [`ExtentedFunction::edx().syscall()`](crate::arch::x86_64::cpuid::ExtendedFunctionEDX::syscall) and
/// [`ExtentedFunction::edx().nx()`](crate::arch::x86_64::cpuid::ExtendedFunctionEDX::nx)
/// must be respectively checked.
#[bitfield(u64)]
pub struct EFER {
    /// Enables the `SYSCALL`/`SYSRET` instruction pair.
    #[bits(1)]
    pub sce: bool,

    #[bits(7)]
    __: u8,

    /// Enables IA-32e (64-bit) mode.
    #[bits(1)]
    pub lme: bool,

    #[bits(1)]
    __: u8,

    /// Indicates that IA-32e mode is currently active (read-only).
    #[bits(1, access = RO)]
    pub lma: bool,

    /// Enables the Execute Disable (NX) page-table bit.
    #[bits(1)]
    pub nxe: bool,

    #[bits(52)]
    __: u64,
}

impl MSR<0xC0000080> for EFER {
    fn raw(&self) -> u64 {
        self.0
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value)
    }
}
