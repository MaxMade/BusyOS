//! The *Multiple APIC Description Table* (MADT).
//!
//! The MADT describes the platform's interrupt topology: which interrupt
//! controllers exist, which core each of them belongs to, and how legacy
//! interrupt sources are routed onto them. It is the table the kernel uses to
//! learn how many cores the system has before any of them is started.
//!
//! Its payload is a flat list of variable-length records rather than an array,
//! so each entry has to be walked in turn. Every record starts with a common
//! header giving its type and its total length, which is what makes the list
//! self-describing and skippable even for the record types the kernel does not
//! know about.
//!
//! Only the *Processor Local x2APIC* record is interpreted so far, see
//! [`MADT::x2apic_lapic_ids`].

#[cfg(target_arch = "x86_64")]
use crate::driver::acpi::acpi::Acpi;

use crate::{
    driver::acpi::acpi::{SDTHeader, Signature, Table},
    kernel::locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
};

/// The *Multiple APIC Description Table*.
///
/// The record list that makes up the table's payload directly follows the
/// fields below and is therefore not representable as a struct field. Use
/// [`x2apic_lapic_ids`](Self::x2apic_lapic_ids) to walk it.
#[repr(packed)]
pub struct MADT {
    /// The common table header, see [`SDTHeader`].
    header: SDTHeader,

    /// Physical address of the local interrupt controller as seen by every
    /// core. Superseded by the per-core addresses the records may override,
    /// and unused in x2APIC mode, where the register file is reached through
    /// MSRs rather than a mapping.
    #[allow(unused)]
    lapic_address: u32,

    /// Bit 0 indicates that the system also has a dual 8259 PIC wired up,
    /// which must be masked off before the APICs are used.
    #[allow(unused)]
    flags: u32,
    // Interrupt controller records follow
}

/// The type of a record in the [`MADT`] payload.
///
/// Only the record types the kernel acts on are named; all others are skipped
/// by their [`Header::record_len`] without being interpreted.
#[repr(u8)]
#[derive(Debug)]
enum EntryType {
    /// *Processor Local APIC*, see [`LocalAPICEntry`].
    LocalAPIC = 0,
    /// *Processor Local x2APIC*, see [`Localx2APICEntry`].
    Localx2APIC = 9,
}

/// The header shared by all records in the [`MADT`] payload.
///
/// Every record starts with this header, immediately followed by the
/// type-specific payload. As with [`SDTHeader`], the length field covers the
/// header *and* the payload, so it is the amount by which the cursor advances
/// to reach the next record.
#[repr(packed)]
struct Header {
    /// Identifies the record's type, see [`EntryType`].
    entry_type: u8,

    /// Total size of the record in bytes, including this header.
    record_len: u8,
}

/// A *Processor Local x2APIC* record ([`EntryType::Localx2APIC`]).
///
/// The firmware provides one such record per logical processor that the
/// platform exposes through x2APIC. Note that a record being present does not
/// mean the core is usable: a core is only enabled, or capable of being
/// brought online later, if `flags` says so.
#[repr(packed)]
struct Localx2APICEntry {
    /// The common record header, see [`Header`].
    #[allow(unused)]
    header: Header,

    /// Reserved, must be zero.
    #[allow(unused)]
    reserved: u16,

    /// The 32-bit identifier of this core's local APIC.
    lapic_id: u32,

    /// Bit 0 marks the core as enabled, bit 1 as online-capable. A core with
    /// neither bit set must not be started.
    #[allow(unused)]
    flags: u32,

    /// The processor's UID, matching the one its object carries in the
    /// namespace described by the *Differentiated System Description Table*.
    #[allow(unused)]
    acpi_id: u32,
}

/// A *Processor Local APIC* record ([`EntryType::LocalAPIC`]).
///
/// The firmware provides one such record per logical processor that the
/// platform exposes through APIC or x2APIC. Note that a record being present
/// does not mean the core is usable: a core is only enabled, or capable of
/// being brought online later, if `flags` says so.
///
/// Only used instead of [`EntryType::Localx2APIC`], if [`EntryType::LocalAPIC`]
/// is able to hold the required values.
#[repr(packed)]
struct LocalAPICEntry {
    /// The common record header, see [`Header`].
    #[allow(unused)]
    header: Header,

    /// The 8-bit identifier of this core's local APIC.
    lapic_id: u8,

    /// The processor's UID, matching the one its object carries in the
    /// namespace described by the *Differentiated System Description Table*.
    #[allow(unused)]
    acpi_id: u8,

    /// Bit 0 marks the core as enabled, bit 1 as online-capable. A core with
    /// neither bit set must not be started.
    #[allow(unused)]
    flags: u32,
}

impl MADT {
    /// Walks the record list and registers the local APIC identifier of every
    /// *Processor Local x2APIC* record with `acpi`.
    ///
    /// The list is walked record by record, each one skipped by the length its
    /// header declares, since the records are of variable size and the kernel
    /// only interprets one of the types. The walk stops early, rather than
    /// panicking, as soon as a record would reach past the end of the table
    /// that [`SDTHeader::length`] declares, so that a truncated trailing
    /// record is ignored instead of read out of bounds. A record shorter than
    /// its own header ends the walk for the same reason, and because it would
    /// otherwise not advance the cursor.
    ///
    /// The `token` is threaded through because registering an identifier
    /// allocates, see [`Acpi::register_x2apic_lapic_id`].
    ///
    /// # Panics
    ///
    /// Panics if a record claims to be a *Processor Local x2APIC* but is
    /// shorter than one. Unlike a truncated trailing record, this is a
    /// well-formed record that contradicts itself, which means the table
    /// cannot be trusted at all.
    #[cfg(target_arch = "x86_64")]
    pub fn x2apic_lapic_ids<Token>(&self, acpi: &mut Acpi, token: Token) -> Token
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        use core::ffi::c_void;
        use core::mem;

        use crate::driver::x86_64::x2apic::LapicID;

        let mut token = token;

        let ptr = self as *const MADT as *const c_void;
        let start = unsafe { ptr.byte_add(mem::size_of::<MADT>()) };
        let end = unsafe { ptr.byte_add(self.header.length as _) };

        let mut entry_ptr = start;
        loop {
            // Sanity check #1: Header must be valid
            if entry_ptr >= end {
                break;
            }
            let mut entry_end_ptr = unsafe { entry_ptr.byte_add(mem::size_of::<Header>()) };
            if entry_end_ptr > end {
                break;
            }

            // Ok, interpret pointer as header
            let header_ptr = entry_ptr as *const Header;
            let header = unsafe { header_ptr.as_ref_unchecked() };

            // Sanity check #2: Entry must be valid and must advance the cursor
            if (header.record_len as usize) < mem::size_of::<Header>() {
                break;
            }
            entry_end_ptr = unsafe { entry_ptr.byte_add(header.record_len as _) };
            if entry_end_ptr > end {
                break;
            }

            if header.entry_type == EntryType::LocalAPIC as u8 {
                // Sanity check #3: Local apic entry must be valid
                if mem::size_of::<LocalAPICEntry>() > header.record_len as usize {
                    panic!("Truncated Local APIC entry in MADT");
                }

                let entry_ptr = header_ptr as *const LocalAPICEntry;
                let entry = unsafe { entry_ptr.as_ref_unchecked() };

                let lapic_id = unsafe { LapicID::from_raw(entry.lapic_id as _) };

                token = acpi.register_x2apic_lapic_id(lapic_id, token);
            }

            if header.entry_type == EntryType::Localx2APIC as u8 {
                // Sanity check #3: Local x2apic entry must be valid
                if mem::size_of::<Localx2APICEntry>() > header.record_len as usize {
                    panic!("Truncated Local x2APIC entry in MADT");
                }

                let entry_ptr = header_ptr as *const Localx2APICEntry;
                let entry = unsafe { entry_ptr.as_ref_unchecked() };

                let lapic_id = unsafe { LapicID::from_raw(entry.lapic_id) };

                token = acpi.register_x2apic_lapic_id(lapic_id, token);
            }

            // Continue with next entry
            entry_ptr = entry_end_ptr;
        }
        token
    }
}

impl Table for MADT {
    const SIGNATURE: Signature = Signature::MADT;
}
