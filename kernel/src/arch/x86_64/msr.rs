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

use core::fmt::{Debug, Display};

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

/// A memory type, as encoded in an entry of the [`PAT`] register.
///
/// The numeric value of each variant is the encoding the hardware expects;
/// the two encodings the architecture leaves undefined, `0x02` and `0x03`,
/// have no variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MemoryType {
    /// Uncacheable (`UC`) — every access reaches the bus, in program order.
    Uncacheable = 0x00,

    /// Write Combining (`WC`) — writes are gathered in a buffer and released
    /// as bursts; reads are uncached.
    WriteCombining = 0x01,

    /// Write Through (`WT`) — reads are cached, writes update cache and
    /// memory both.
    WriteThrough = 0x04,

    /// Write Protected (`WP`) — reads are cached, writes go to memory and
    /// invalidate the line, in every cache holding it.
    WriteProtected = 0x05,

    /// Write Back (`WB`) — reads and writes are cached, writes reach memory
    /// on eviction.
    WriteBack = 0x06,

    /// Uncached (`UC-`) — as [`Uncacheable`](MemoryType::Uncacheable), except
    /// that an MTRR asking for write-combining wins over it.
    UncachedMinus = 0x07,
}

impl Display for MemoryType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MemoryType::Uncacheable => write!(f, "UC"),
            MemoryType::WriteCombining => write!(f, "WC"),
            MemoryType::WriteThrough => write!(f, "WT"),
            MemoryType::WriteProtected => write!(f, "WP"),
            MemoryType::WriteBack => write!(f, "WB"),
            MemoryType::UncachedMinus => write!(f, "UC-"),
        }
    }
}

impl MemoryType {
    /// The hardware encoding of this memory type.
    pub const fn into_bits(self) -> u8 {
        self as u8
    }

    /// The memory type `bits` encodes, or `None` for the two encodings the
    /// architecture reserves.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0x00 => Some(MemoryType::Uncacheable),
            0x01 => Some(MemoryType::WriteCombining),
            0x04 => Some(MemoryType::WriteThrough),
            0x05 => Some(MemoryType::WriteProtected),
            0x06 => Some(MemoryType::WriteBack),
            0x07 => Some(MemoryType::UncachedMinus),
            _ => None,
        }
    }
}

/// Page Attribute Table (`IA32_PAT`, MSR `0x277`).
///
/// Eight [`MemoryType`] slots, `PA0` through `PA7`, one per byte. A leaf page
/// table entry does not name its memory type directly: its `PAT`, `PCD` and
/// `PWT` bits form the three-bit index `PAT << 2 | PCD << 1 | PWT` of the slot
/// that names it. Which mapping ends up with which memory type is therefore a
/// property of this register, and the layout the kernel installs is
/// [`PAT_LAYOUT`](crate::arch::x86_64::paging::PAT_LAYOUT).
///
/// The register is per core, and comes out of reset holding
/// `WB`, `WT`, `UC-`, `UC`, `WB`, `WT`, `UC-`, `UC`.
///
/// # Safety
///
/// Availability must be checked via
/// [`FeatureInformationEDX::pat`](crate::arch::x86_64::cpuid::FeatureInformationEDX::pat)
/// before this MSR is touched.
#[derive(Clone, Copy)]
pub struct PAT(u64);

impl Debug for PAT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PAT(0x{:016x})", self.0)
    }
}

impl Display for PAT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x{:016x} (", self.0)?;

        for index in 0..PAT::ENTRIES {
            if index != 0 {
                write!(f, ", ")?;
            }

            match self.entry(index) {
                Some(memory_type) => write!(f, "PA{index}: {memory_type}")?,
                None => write!(f, "PA{index}: reserved")?,
            }
        }

        write!(f, ")")
    }
}

impl PAT {
    /// Number of memory type slots in the register.
    pub const ENTRIES: usize = 8;

    /// Builds a register value from the memory type of each of its eight
    /// slots, `PA0` first.
    pub const fn from_entries(entries: [MemoryType; PAT::ENTRIES]) -> Self {
        let mut value = 0u64;

        let mut index = 0;
        while index < PAT::ENTRIES {
            value |= (entries[index].into_bits() as u64) << (index * 8);
            index += 1;
        }

        Self(value)
    }

    /// The memory type held by slot `index`, or `None` if the slot holds one
    /// of the reserved encodings.
    ///
    /// # Panics
    ///
    /// If `index` is not below [`ENTRIES`](PAT::ENTRIES).
    pub const fn entry(&self, index: usize) -> Option<MemoryType> {
        assert!(index < PAT::ENTRIES, "PAT slot index out of range");

        MemoryType::from_bits((self.0 >> (index * 8)) as u8 & 0b111)
    }
}

impl MSR<0x277> for PAT {
    fn raw(&self) -> u64 {
        self.0
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value)
    }
}
