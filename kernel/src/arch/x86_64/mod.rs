//! x86_64-specific abstractions.
pub mod bootinfo;
pub mod cpuid;
pub mod cr4;
pub mod msr;
pub mod paging;
pub mod cpu;
pub mod rflags;

// The bootstrap stub belongs to the kernel image alone.
#[cfg(all(not(test), not(feature = "library")))]
pub mod init;
