//! Drivers for the *Advanced Configuration and Power Interface* (ACPI).
//!
//! ACPI describes the platform's hardware topology and its power management
//! capabilities through a tree of tables provided by the firmware. Discovery
//! always starts at the [`RSDP`](rsdp::RSDP), which is located by the
//! bootloader and handed to the kernel through the boot information. The RSDP
//! points at the [`XSDT`](xsdt::XSDT), which in turn lists the physical
//! addresses of all remaining tables.
//!
//! # Overview
//!
//! - [`acpi`] contains the driver module itself together with the shared
//!   [`SDTHeader`](acpi::SDTHeader), [`Signature`](acpi::Signature) and
//!   [`Table`](acpi::Table) abstractions.
//! - [`madt`] implements the *Multiple APIC Description Table*.
//! - [`rsdp`] implements the *Root System Description Pointer*.
//! - [`xsdt`] implements the *Extended System Description Table*.

pub mod acpi;
pub mod madt;
pub mod rsdp;
pub mod xsdt;
