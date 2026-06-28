//! Architecture-specific abstractions.
pub mod generic;

#[cfg(target_arch = "x86_64")]
pub mod amd64;

// Export all amd64-specific implementations
#[cfg(target_arch = "x86_64")]
mod imp {
    pub use crate::arch::amd64::paging::*;
    pub use crate::arch::amd64::bootinfo::*;
}

// Re-export without "::imp::*"
pub use imp::*;
