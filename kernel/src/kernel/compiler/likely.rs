//! Branch prediction hints.
//!
//! [`likely`] and [`unlikely`] wrap a condition and return it unchanged, but
//! mark the branch that is not expected to be taken as cold through
//! [`cold_path`]. The compiler may then lay out the expected branch as the
//! fall-through path and move the other one out of line.
//!
//! ```ignore
//! if unlikely(ptr.is_null()) {
//!     return Err(Error::NullPointer);
//! }
//! ```
//!
//! The hints are only useful for branches that go one way almost always, such
//! as error checks on hot paths. A wrong hint makes the common case slower.

use core::hint::cold_path;

/// Returns `b`, hinting that it is expected to be `true`.
#[inline(always)]
pub const fn likely(b: bool) -> bool {
    if !b {
        cold_path();
    }
    b
}

/// Returns `b`, hinting that it is expected to be `false`.
#[inline(always)]
pub const fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
    }
    b
}
