//! Architecture-specific abstractions.
pub mod generic;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

// Export all x86_64-specific implementations
#[cfg(target_arch = "x86_64")]
mod imp {
    pub use crate::arch::x86_64::bootinfo::*;
    pub use crate::arch::x86_64::paging::*;
}

// Re-export without "::imp::*"
pub use imp::*;
