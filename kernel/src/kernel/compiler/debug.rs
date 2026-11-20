//! Build-profile queries.
//!
//! [`debug_build`] makes debug-only checks plain `if` conditions rather than
//! `#[cfg]`s, so the checked code is compiled — and kept compiling — in every
//! profile, and optimised away where the condition is `false`.
//!
//! ```ignore
//! if debug_build() {
//!     assert!(queue.iter().all(|t| t.id() != thread.id()));
//! }
//! ```

/// Returns `true` if the kernel is built with debug assertions.
#[inline(always)]
pub const fn debug_build() -> bool {
    cfg!(debug_assertions)
}
