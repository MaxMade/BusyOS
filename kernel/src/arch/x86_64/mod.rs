//! x86_64-specific abstractions.
pub mod bootinfo;
pub mod cpu;
pub mod cpuid;
pub mod cr4;
pub mod gdt;
pub mod msr;
pub mod paging;
pub mod rflags;

// The bootstrap stub belongs to the kernel image alone.
pub mod idt;
#[cfg(all(not(test), not(feature = "library")))]
pub mod init;
pub mod io;
pub mod pit;
