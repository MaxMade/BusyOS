//! Architecture-specific abstractions.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
pub mod generic;

// Export all x86_64-specific implementations
#[cfg(target_arch = "x86_64")]
mod imp {
    pub use crate::arch::x86_64::elf::*;
    pub use crate::arch::x86_64::features::*;
    pub use crate::arch::x86_64::paging::*;
    pub use crate::arch::x86_64::handover::*;
}

// Re-export without "::imp::*"
pub use imp::*;
