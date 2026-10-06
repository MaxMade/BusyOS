//! Fallible memory allocation behind the kernel's lock-level token system.
//!
//! # Overview
//!
//! This module provides the [`Allocator`] trait, the single interface through
//! which the kernel obtains and releases heap memory.
//!
//! ## Deadlock prevention via the token API
//!
//! The kernel uses a compile-time
//! lock-level system ([`crate::kernel::locking`]) to guarantee that locks are
//! always acquired in a fixed order, ruling out deadlock by construction.
//! Allocators participate in this system: because an allocator implementation
//! typically acquires an internal lock (e.g. on the heap metadata), it is
//! assigned a [`LockId`] that encodes which lock level it sits at. Callers
//! must pass a [`Token`](CanAcquire) proving they do not already hold any lock
//! at or above that level, and they receive it back after each call, allowing
//! the borrow checker to enforce the ordering statically.

use core::alloc::Layout;
use core::fmt::Display;
use core::ptr::NonNull;

use crate::{
    kernel::locking::{CanAcquire, LockId, PreviousToken},
    user::errno::{Errno, ToErrno},
};

/// Error type returned by allocation operations.
#[derive(Debug)]
pub enum Error {
    /// The allocator has no memory available to satisfy the request.
    OutOfMemory,
}

impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn core::error::Error> {
        self.source()
    }
}

impl ToErrno for Error {
    fn to_errno(&self) -> Errno {
        match self {
            Error::OutOfMemory => Errno::ENOMEM,
        }
    }
}

/// A minimal, fallible, token-aware allocator interface.
///
/// This trait is the kernel's single abstraction over heap allocators. It is
/// intentionally narrow — only [`allocate`](Allocator::allocate) and
/// [`deallocate`](Allocator::deallocate) — to keep implementations simple and
/// auditable.
///
/// Implementations must uphold these invariants:
///
/// - **Validity.** A successful [`allocate`](Allocator::allocate) returns a
///   `NonNull<u8>` pointing to a block of at least `layout.size()` bytes,
///   aligned to at least `layout.align()`, that is not aliased by any other
///   live allocation.
/// - **Exact layout on dealloc.** [`deallocate`](Allocator::deallocate) must
///   be called with the *same* `Layout` that was passed to the corresponding
///   `allocate` call. Passing a different layout is undefined behaviour.
/// - **Exactly once.** Each successfully allocated block must be deallocated
///   exactly once. Double-free and use-after-free are undefined behaviour.
/// - **No use after dealloc.** After `deallocate` returns, the memory must not
///   be read or written.
pub unsafe trait Allocator<ID: LockId> {
    /// Allocates a block of memory described by `layout`.
    ///
    /// On success returns `(ptr, token)` where `ptr` is the start of the
    /// allocated block. On failure returns `(Error, token)` so the caller
    /// can continue using the token regardless of outcome.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] if the allocator cannot satisfy the
    /// request at this time.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both the `Ok` and `Err` arms.
    /// The caller must thread the returned token through subsequent operations.
    fn allocate<Token>(
        &self,
        layout: Layout,
        token: Token,
    ) -> Result<(NonNull<u8>, Token), (Error, Token)>
    where
        Token: CanAcquire<ID::Level> + PreviousToken;

    /// Deallocates the block at `ptr`, previously obtained from this allocator.
    ///
    /// Consumes and returns the `token` so callers can continue holding lock
    /// state across the call.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a pointer returned by a prior call to
    ///   [`allocate`](Allocator::allocate) on **this** allocator instance.
    /// - `layout` must be identical to the `Layout` passed to that `allocate`
    ///   call.
    /// - `ptr` must not have been deallocated already.
    /// - `ptr` must not be used (read or written) after this call returns.
    unsafe fn deallocate<Token>(&self, ptr: NonNull<u8>, layout: Layout, token: Token) -> Token
    where
        Token: CanAcquire<ID::Level> + PreviousToken;
}

/// Blanket implementation forwarding through a shared reference.
///
/// This allows passing `&A` wherever `A: Allocator<ID>` is required, which is
/// useful when the allocator needs to be shared across multiple owners without
/// requiring `Arc` or similar wrappers.  Because `&A` is `Copy`, it can be
/// stored cheaply in stack-allocated data structures.
// Safety: we simply delegate to the underlying implementation; all invariants
// are preserved by the inner A.
unsafe impl<ID: LockId, A: Allocator<ID> + ?Sized> Allocator<ID> for &A {
    #[inline]
    fn allocate<Token>(
        &self,
        layout: Layout,
        token: Token,
    ) -> Result<(NonNull<u8>, Token), (Error, Token)>
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        (**self).allocate(layout, token)
    }

    #[inline]
    unsafe fn deallocate<Token>(&self, ptr: NonNull<u8>, layout: Layout, token: Token) -> Token
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        // Safety: the caller upholds all deallocate preconditions; we forward
        // them unchanged to the inner implementation.
        unsafe { (**self).deallocate(ptr, layout, token) }
    }
}
