//! Drivers for devices that only exist on `x86_64`.
//!
//! Unlike the architecture-independent drivers next to it, everything below
//! this module is tied to the platform and is therefore only referenced from
//! code guarded by `#[cfg(target_arch = "x86_64")]`.
//!
//! # Overview
//!
//! - [`x2apic`] implements the *Local Advanced Programmable Interrupt
//!   Controller* in x2APIC mode.

pub mod x2apic;
