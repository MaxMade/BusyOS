//! Core ACPI driver module and shared table abstractions.
//!
//! This module hosts the [`Acpi`] driver, which is registered with the
//! kernel's module framework and performs the initial walk over the firmware
//! tables during boot. It also defines the pieces that every ACPI table has in
//! common:
//!
//! - [`SDTHeader`] is the *System Description Table* header that prefixes
//!   every table.
//! - [`Signature`] enumerates the four-character signatures of the tables the
//!   kernel knows about.
//! - [`Table`] ties a concrete table type to its signature and provides the
//!   shared validation logic.

use core::{fmt::Display, slice};

use driver_macro::module;

use crate::arch::generic::paging::ReversePaging;
use crate::driver::acpi::madt::MADT;
use crate::driver::acpi::rsdp::RSDP;
use crate::driver::acpi::xsdt::XSDT;
use crate::driver::module::Modules;
use crate::kernel::arc::Arc;
use crate::{
    arch::Paging,
    driver::module::Module,
    kernel::{
        bootinfo::BOOTINFO,
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
    },
    mem::page_frames::PageFrames,
    user::errno::Errno,
};

#[cfg(target_arch = "x86_64")]
use crate::driver::x86_64::x2apic::LapicID;

#[cfg(target_arch = "x86_64")]
use crate::kernel::linked_list::LinkedList;

module! {
    name: "acpi",
    priority: 0,
    driver: crate::driver::acpi::acpi::Acpi,
}

/// The ACPI driver module.
///
/// Registered through the [`module!`] macro and initialized by the kernel's
/// module framework. See the [`Module`] implementation for what happens during
/// initialization.
pub struct Acpi {
    /// The local APIC identifier of every core the firmware reported through
    /// the [`MADT`], collected during [`init`](Module::init).
    ///
    /// This is what tells the kernel how many cores exist and how to address
    /// each of them, so it is the starting point for bringing up the
    /// secondary cores later on.
    #[cfg(target_arch = "x86_64")]
    x2apic_lapic_ids: LinkedList<LapicID>,
}

/// The four-character signature identifying an ACPI table.
///
/// Each variant stores the signature as it appears in memory, so the
/// discriminant can be compared directly against the raw bytes of an
/// [`SDTHeader`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signature {
    /// *Extended System Description Table*, see [`XSDT`].
    XSDT = u32::from_ne_bytes([b'X', b'S', b'D', b'T']),

    /// *Multiple APIC Description Table*, see [`MADT`]. Signed `APIC` rather
    /// than `MADT`, since the table predates the name it is known by.
    MADT = u32::from_ne_bytes([b'A', b'P', b'I', b'C']),
}

impl Signature {
    /// Returns the signature as the four raw bytes found in an [`SDTHeader`].
    pub const fn as_bytes(self) -> [u8; 4] {
        u32::to_ne_bytes(self as _)
    }
}

impl Display for Signature {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        let str = str::from_utf8(bytes.as_slice()).unwrap();
        write!(f, "{}", str)
    }
}

/// The *System Description Table* header shared by all ACPI tables.
///
/// Every table starts with this header, immediately followed by the
/// table-specific payload. The [`length`](Self::length) field covers the
/// header *and* the payload, which is what makes a table self-describing and
/// allows [`verify`](Self::verify) to checksum it as a whole.
#[repr(packed)]
pub struct SDTHeader {
    /// Four-character ASCII signature identifying the table, see
    /// [`Signature`].
    pub signature: [u8; 4],

    /// Total size of the table in bytes, including this header.
    pub length: u32,

    /// Revision of the table's layout.
    pub revision: u8,

    /// Checksum byte chosen such that all bytes of the table sum to zero.
    pub checksum: u8,

    /// ASCII identifier of the OEM that supplied the table.
    pub oem_id: [u8; 6],

    /// OEM-specific identifier for this particular table.
    pub oem_table_id: [u8; 8],

    /// OEM-supplied revision of [`oem_table_id`](Self::oem_table_id).
    pub oem_revision: u32,

    /// Identifier of the utility that created the table.
    pub creator_id: u32,

    /// Revision of the utility that created the table.
    pub creator_revision: u32,
}

impl SDTHeader {
    /// Returns the raw signature as a string slice.
    ///
    /// Useful for reporting tables the kernel does not know about, since those
    /// have no matching [`Signature`] variant.
    ///
    /// The bytes are *not* validated. ACPI mandates printable ASCII here, so
    /// firmware-provided headers always satisfy this, but a header from any
    /// other source may produce a slice that is not valid UTF-8.
    pub fn raw_signature(&self) -> &str {
        unsafe { str::from_utf8_unchecked(&self.signature[..]) }
    }

    /// Checks that this table carries the expected `signature` and that its
    /// checksum is intact.
    ///
    /// The checksum spans [`length`](Self::length) bytes starting at the
    /// header. A well-formed table sums to zero across all of those bytes.
    ///
    /// The whole table, not just the header, must be mapped and readable for
    /// this to be sound. That holds for firmware tables reached through the
    /// physical memory mapping, since the firmware reserves them contiguously.
    pub fn verify(&self, signature: Signature) -> bool {
        if self.signature != signature.as_bytes() {
            return false;
        }

        let length = self.length as usize;
        let ptr = (self as *const SDTHeader) as *const u8;

        unsafe {
            let slice = slice::from_raw_parts(ptr, length);

            let mut checksum: u8 = 0;
            for byte in slice.iter() {
                checksum = checksum.wrapping_add(*byte);
            }

            checksum == 0
        }
    }
}

/// An ACPI table that is prefixed by an [`SDTHeader`].
///
/// Implementors only need to name their [`SIGNATURE`](Self::SIGNATURE); the
/// shared validation is provided by [`verify_header`](Self::verify_header).
/// The implementing type must be `#[repr(packed)]` and start with an
/// [`SDTHeader`], so that a validated header pointer can be reinterpreted as a
/// pointer to the concrete table.
pub trait Table: Sized {
    /// The signature that identifies this table in an [`SDTHeader`].
    const SIGNATURE: Signature;

    /// Validates the header at `ptr` and reinterprets it as this table.
    ///
    /// Returns [`None`] if `ptr` is null, if the signature does not match
    /// [`SIGNATURE`](Self::SIGNATURE), or if the checksum is invalid.
    ///
    /// The returned lifetime `'a` is unconstrained and therefore chosen by the
    /// caller. Firmware tables live for the lifetime of the system, so
    /// `'static` is the usual choice.
    fn verify_header<'a>(ptr: *const SDTHeader) -> Option<&'a Self> {
        let header = unsafe { ptr.as_ref()? };

        if header.verify(Self::SIGNATURE) {
            let ptr = ptr as *const Self;
            return unsafe { Some(ptr.as_ref_unchecked()) };
        }

        None
    }
}

impl Module for Acpi {
    /// Walks the ACPI tables provided by the firmware.
    ///
    /// The RSDP is taken from the boot information, validated, and used to
    /// locate the XSDT. The XSDT is validated in turn and its entries are then
    /// iterated to discover the remaining tables.
    ///
    /// Each entry is offered to every table type the kernel knows about; a
    /// table whose signature or checksum does not match is simply not
    /// recognized by that type, so an unknown table is skipped rather than
    /// rejected. On `x86_64` the [`MADT`] is the only table interpreted so
    /// far, and only for the local APIC identifiers it reports.
    ///
    /// # Panics
    ///
    /// Panics if the boot information does not contain a valid RSDP, or if the
    /// XSDT it refers to fails validation. Both indicate firmware that the
    /// kernel cannot make sense of, so there is nothing to fall back to.
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        let mut acpi = Acpi {
            x2apic_lapic_ids: LinkedList::new(),
        };

        let boot_info = unsafe { BOOTINFO.assume_init_ref() };

        // Parse Root System Descriptor Pointer (RSDP)
        let rsdp_ptr =
            unsafe { Paging::<PageFrames>::phys_to_virt(boot_info.arch_bootinfo.rsdp).as_ref() };
        let rsdp = match RSDP::verify(rsdp_ptr) {
            Some(rsdp) => rsdp,
            None => panic!("Unable to find valid RSDP"),
        };

        // Parse Extended System Descriptor Table (XSDT: successor of RSDT)
        let xsdt_ptr = unsafe { Paging::<PageFrames>::phys_to_virt(rsdp.xsdt()) };
        let xsdt = match XSDT::verify(xsdt_ptr.as_ptr()) {
            Some(xsdt) => xsdt,
            None => panic!("Unable to find valid XSDT"),
        };

        for table in xsdt.tables() {
            let table_ptr = unsafe { Paging::<PageFrames>::phys_to_virt(table) };

            // Try to interpret as Multiple APIC Description Table (MADT)
            if let Some(madt) = MADT::verify_header(table_ptr.as_ptr()) {
                #[cfg(target_arch = "x86_64")]
                {
                    // Collect x2apic LAPIC IDs
                    token = madt.x2apic_lapic_ids(&mut acpi, token);
                }
            }
        }

        // Register driver
        let driver;
        (driver, token) = match Arc::try_new(acpi, token) {
            Ok(success) => success,
            Err((error, _)) => {
                panic!("Unable to create sharable ACPI driver instance: {}", error);
            }
        };
        token = match Modules::register(driver, token) {
            Ok(token) => token,
            Err((error, _)) => {
                panic!("Unable to register ACPI driver instance: {}", error);
            }
        };

        Ok(token)
    }
}

impl Acpi {
    /// Records the local APIC identifier of one core.
    ///
    /// Called once per *Processor Local x2APIC* record while the [`MADT`] is
    /// walked, see [`MADT::x2apic_lapic_ids`]. The identifiers are kept in the
    /// order the firmware listed them.
    ///
    /// # Panics
    ///
    /// Panics if the identifier cannot be stored. This only happens if the
    /// allocation fails, and a kernel that cannot afford one node per core
    /// this early during boot has no way to continue.
    #[cfg(target_arch = "x86_64")]
    pub fn register_x2apic_lapic_id<Token>(&mut self, lapic_id: LapicID, token: Token) -> Token
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        token = match self.x2apic_lapic_ids.try_push_back(lapic_id, token) {
            Ok(token) => token,
            Err((error, _)) => {
                panic!("Unable to register x2apic LAPIC ID: {}", error);
            }
        };

        token
    }
}
