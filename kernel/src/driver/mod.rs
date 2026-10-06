//! Device drivers and the framework that brings them up.
//!
//! # Overview
//!
//! - [`module`] provides the driver module framework: registration,
//!   priorities and the initialization the kernel performs during boot.
//! - [`acpi`] discovers the platform through the firmware's ACPI tables.
//! - [`irq`] routes an interrupt vector to the driver in charge of it.
//! - [`x86_64`] holds the drivers that only exist on `x86_64`.

pub mod acpi;
pub mod console;
pub mod framebuffer;
pub mod ipi;
pub mod irq;
pub mod ksymbols;
pub mod module;
pub mod timer;
pub mod x86_64;
