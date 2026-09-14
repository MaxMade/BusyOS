//! The *Root System Description Pointer* (RSDP).
//!
//! The RSDP is the entry point into the ACPI table tree. It is not itself an
//! [`SDTHeader`]-prefixed table, but a small standalone structure that the
//! firmware places in memory and that the bootloader locates and records in
//! the boot information.
//!
//! Only revision 2 (ACPI 2.0 and later) is supported, so the 64-bit
//! [`XSDT`] pointer is always available and the legacy 32-bit RSDT is ignored.

use crate::{
    arch::generic::paging::PhysicalAddress,
    driver::acpi::{acpi::SDTHeader, xsdt::XSDT},
};

/// The *Root System Description Pointer*.
///
/// The first 20 bytes up to and including `legacy_rsdt`
/// form the original ACPI 1.0 structure, covered by
/// `checksum`. Revision 2 appends the remaining fields,
/// which are covered separately by `ext_checksum`.
#[repr(packed)]
pub struct RSDP {
    /// Must be the ASCII string `"RSD PTR "`, see `RSDP::SIGNATURE`.
    signature: [u8; 8],

    /// Checksum byte over the first 20 bytes of the structure.
    checksum: u8,

    /// ASCII identifier of the OEM that supplied the structure.
    oemid: [u8; 6],

    /// Revision of the structure. Must be 2 for ACPI 2.0 and later.
    revision: u8,

    /// Physical address of the legacy 32-bit RSDT. Superseded by
    /// `xsdt` and therefore unused.
    legacy_rsdt: u32,

    /// Total size of the structure in bytes.
    length: u32,

    /// Physical address of the [`XSDT`].
    xsdt: PhysicalAddress<XSDT>,

    /// Checksum byte over the whole structure, including the legacy fields.
    ext_checksum: u8,

    /// Reserved, must be zero.
    reserved: [u8; 3],
}

impl RSDP {
    /// The signature every valid RSDP starts with.
    const SIGNATURE: &'static str = "RSD PTR ";

    /// Validates the RSDP at `ptr`.
    ///
    /// Returns [`None`] if `ptr` is null, if the signature does not match
    /// `SIGNATURE`, if the revision is not 2, or if either
    /// checksum is invalid.
    ///
    /// Both checksums are verified. The legacy checksum covers the first 20
    /// bytes; the extended one covers the whole structure. Since the legacy
    /// checksum is already known to be zero at that point, the running sum is
    /// simply carried over into the second check instead of restarted.
    ///
    /// The returned lifetime `'a` is unconstrained and therefore chosen by the
    /// caller. The firmware keeps the RSDP alive for the lifetime of the
    /// system, so `'static` is the usual choice.
    pub fn verify<'a>(ptr: *const Self) -> Option<&'a Self> {
        let tmp = unsafe { ptr.as_ref()? };

        if tmp.signature != Self::SIGNATURE.as_bytes() {
            return None;
        }

        if tmp.revision != 2 {
            return None;
        }

        // Check checksum
        let mut checksum: u8 = 0;
        for byte in tmp.signature.iter() {
            checksum = checksum.wrapping_add(*byte);
        }
        checksum = checksum.wrapping_add(tmp.checksum);
        for byte in tmp.oemid.iter() {
            checksum = checksum.wrapping_add(*byte);
        }
        checksum = checksum.wrapping_add(tmp.revision);
        for byte in tmp.legacy_rsdt.to_ne_bytes() {
            checksum = checksum.wrapping_add(byte);
        }
        if checksum != 0 {
            return None;
        }

        // Check extended checksum
        for byte in tmp.length.to_ne_bytes() {
            checksum = checksum.wrapping_add(byte);
        }
        for byte in tmp.xsdt.addr().to_ne_bytes() {
            checksum = checksum.wrapping_add(byte);
        }
        checksum = checksum.wrapping_add(tmp.ext_checksum);
        for byte in tmp.reserved.iter() {
            checksum = checksum.wrapping_add(*byte);
        }

        if checksum != 0 {
            return None;
        }

        Some(tmp)
    }
}

impl RSDP {
    /// Returns the physical address of the [`XSDT`].
    ///
    /// The address is taken verbatim from the structure. It still has to be
    /// validated through [`XSDT::verify`] once it has been mapped.
    pub fn xsdt(&self) -> PhysicalAddress<XSDT> {
        self.xsdt
    }
}
