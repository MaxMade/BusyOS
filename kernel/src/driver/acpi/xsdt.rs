//! The *Extended System Description Table* (XSDT).
//!
//! The XSDT is the root of the ACPI table tree and the 64-bit successor of the
//! legacy RSDT. It consists of nothing but an [`SDTHeader`] followed by an
//! array of physical addresses, one per table the firmware provides. Its
//! address is obtained from the [`RSDP`](crate::driver::acpi::rsdp::RSDP).

use core::{ffi::c_void, marker::PhantomData, mem};

use crate::{
    arch::generic::paging::PhysicalAddress,
    driver::acpi::acpi::{SDTHeader, Signature, Table},
};

/// The *Extended System Description Table*.
///
/// The array of 64-bit physical addresses that makes up the table's payload
/// directly follows the header and is therefore not representable as a struct
/// field. Use [`tables`](Self::tables) to walk it.
#[repr(packed)]
pub struct XSDT {
    /// The common table header, see [`SDTHeader`].
    header: SDTHeader,
    // Pointers to other headers follow
}

/// Iterator over the table addresses listed in an [`XSDT`].
///
/// Created by [`XSDT::tables`]. Yields the raw physical addresses as stored by
/// the firmware; neither the mapping nor the validation of the referenced
/// tables is performed here.
pub struct Tables<'a> {
    /// Cursor into the address array following the [`XSDT`] header.
    ptr: *const PhysicalAddress<SDTHeader>,

    /// Number of entries left to yield.
    remaining_entires: usize,

    /// Ties the iterator to the lifetime of the [`XSDT`] it came from.
    phantom: PhantomData<&'a ()>,
}

impl<'a> Iterator for Tables<'a> {
    type Item = PhysicalAddress<SDTHeader>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_entires == 0 {
            return None;
        }
        // The entries are only 4 byte aligned within the packed table
        let addr = unsafe { self.ptr.read_unaligned() };

        self.remaining_entires -= 1;
        self.ptr = unsafe { self.ptr.add(1) };

        Some(addr)
    }
}

impl XSDT {
    /// Validates the XSDT at `ptr`.
    ///
    /// Returns [`None`] if `ptr` is null, if the signature is not
    /// [`Signature::XSDT`], or if the checksum is invalid. See
    /// [`Table::verify_header`] for the details.
    pub fn verify<'a>(ptr: *const Self) -> Option<&'a Self> {
        let header = ptr as *const SDTHeader;
        Self::verify_header(header)
    }

    /// Returns an iterator over the physical addresses of all tables listed in
    /// this XSDT.
    ///
    /// The number of entries is derived from the header's `length` field: the
    /// bytes remaining after the header, divided by the size of one address.
    /// A truncated trailing entry is therefore ignored rather than yielded.
    pub fn tables(&self) -> Tables<'_> {
        let length = self.header.length as usize;
        let remaining = length.saturating_sub(mem::size_of::<SDTHeader>());
        let entries = remaining / mem::size_of::<PhysicalAddress<SDTHeader>>();

        let mut ptr = self as *const Self as *const c_void;
        ptr = unsafe { ptr.byte_add(mem::size_of::<SDTHeader>()) };

        Tables {
            ptr: ptr.cast(),
            remaining_entires: entries,
            phantom: PhantomData,
        }
    }
}

impl Table for XSDT {
    const SIGNATURE: Signature = Signature::XSDT;
}
