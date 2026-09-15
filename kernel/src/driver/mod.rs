//! Device drivers and the framework that brings them up.
//!
//! # Overview
//!
//! - [`module`] provides the driver module framework: registration,
//!   priorities and the initialization the kernel performs during boot.
//! - [`acpi`] discovers the platform through the firmware's ACPI tables.
//! - [`x86_64`] holds the drivers that only exist on `x86_64`.

pub mod acpi;
pub mod module;
pub mod x86_64;
