//! Interrupt-safe locks for the lowest level of the locking hierarchy.
//!
//! A [`Lock`](crate::kernel::locking::Lock) at [`level::Prologue`] protects
//! data that is also touched by interrupt handlers. Taking such a lock with
//! interrupts enabled is a self-deadlock: an interrupt arriving on the same
//! CPU while the lock is held runs a prologue that spins on the very lock the
//! interrupted code already owns.
//!
//! [`PrologueLock`] closes that window by disabling interrupts *before*
//! acquiring the raw lock and restoring the previous interrupt state *after*
//! releasing it.
//!
//! # Difference to [`Lock`](crate::kernel::locking::Lock)
//!
//! [`Lock::acquire`](crate::kernel::locking::Lock::acquire) hands back a
//! `(guard, token)` pair, because the token is needed to descend further down
//! the hierarchy. Prologue is the *lowest* level, so there is nothing left to
//! descend to and the token would only be a burden. [`PrologueLock::acquire`]
//! therefore returns the guard alone and keeps the caller's token inside it;
//! [`PrologueWriteGuard::release`] gives that token back.
//!
//! Storing the token inside the guard preserves linearity just as well as
//! handing it out: while the guard lives, the caller has no token and can
//! acquire no other lock.
//!
//! Because no token is exposed, nested shared holds
//! ([`Lock::acquire_shared_nested`](crate::kernel::locking::Lock::acquire_shared_nested))
//! have no counterpart here — they need the primary token as their input.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};

use crate::arch::CPU;
use crate::arch::generic::cpu::{CPU as _, InterruptState};
use crate::kernel::locking::{CanAcquire, HierarchicalLock, LockId, PreviousToken, level};

/// A data-carrying hierarchical lock that masks interrupts while held.
///
/// The raw lock `L` must live at [`level::Prologue`]; locks at higher levels
/// are never contended against an interrupt handler and should use
/// [`Lock`](crate::kernel::locking::Lock) instead.
pub struct PrologueLock<T, L: HierarchicalLock> {
    raw: L,
    data: UnsafeCell<T>,
}

// SAFETY: sending the PrologueLock sends the T inside it.
unsafe impl<T: Send, L: HierarchicalLock + Send> Send for PrologueLock<T, L> {}

// SAFETY: read guards hand out `&T` on multiple threads (T: Sync); write
// guards allow moving values out via `&mut T` (T: Send). Identical bounds to
// those of `Lock`.
unsafe impl<T: Send + Sync, L: HierarchicalLock + Sync> Sync for PrologueLock<T, L> {}

impl<T, L> PrologueLock<T, L>
where
    L: HierarchicalLock,
    L::Id: LockId<Level = level::Prologue>,
{
    /// Creates a new lock around `value`.
    pub const fn new(raw: L, value: T) -> Self {
        Self {
            raw,
            data: UnsafeCell::new(value),
        }
    }

    /// Consumes the lock, returning the inner value.
    ///
    /// Safe without a token: `self` by value proves no guards exist.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the lock mutably, no actual locking needs to
    /// take place - the mutable borrow statically guarantees no new locks can
    /// be acquired while this reference exists.
    pub const fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Exclusive access with interrupts disabled.
    ///
    /// Disables interrupts first, then acquires the lock. The caller's token
    /// is stored inside the guard and recovered by
    /// [`PrologueWriteGuard::release`].
    pub fn acquire<From>(&self, token: From) -> PrologueWriteGuard<'_, T, L, From>
    where
        From: CanAcquire<level::Prologue> + PreviousToken,
    {
        // Interrupts must be off *before* the lock is taken, otherwise a
        // prologue running on this CPU can spin on a lock this code owns.
        let state = CPU::disable_interrupts(token);

        // SAFETY: the hierarchy is proven by the token consumed above, which
        // now lives inside `state` and is unreachable until `release`. The
        // matching `raw_unlock` happens there.
        unsafe { self.raw.raw_lock() };

        PrologueWriteGuard { lock: self, state }
    }

    /// Shared access with interrupts disabled.
    ///
    /// Disables interrupts first, then acquires the lock. The caller's token
    /// is stored inside the guard and recovered by
    /// [`PrologueReadGuard::release`].
    pub fn acquire_shared<From>(&self, token: From) -> PrologueReadGuard<'_, T, L, From>
    where
        From: CanAcquire<level::Prologue> + PreviousToken,
    {
        // See `acquire` for why the order matters.
        let state = CPU::disable_interrupts(token);

        // SAFETY: as in `acquire`, shared mode. Since no token is handed out,
        // no nested shared hold can be taken and the count stays at one.
        unsafe { self.raw.raw_lock_shared() };

        PrologueReadGuard { lock: self, state }
    }
}

/// Exclusive data guard. Dereferences to `T` (mutably).
///
/// Holds the interrupt state and the caller's token for as long as the lock
/// is held. Release explicitly via [`PrologueWriteGuard::release`].
pub struct PrologueWriteGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    lock: &'a PrologueLock<T, L>,
    state: InterruptState<From>,
}

impl<'a, T, L, From> PrologueWriteGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    /// Unlocks, restores the previous interrupt state and returns the token
    /// that was consumed by [`PrologueLock::acquire`].
    pub fn release(self) -> From {
        // SAFETY: this guard exists only because `acquire` took the lock
        // exclusively, and it is consumed here, so the unlock happens once.
        unsafe { self.lock.raw.raw_unlock() };

        // Interrupts stay off until the lock is released, and are only
        // re-enabled if they were enabled on entry.
        CPU::restore_interrupts(self.state)
    }
}

impl<'a, T, L, From> Deref for PrologueWriteGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: exclusive lock held for the guard's lifetime.
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T, L, From> DerefMut for PrologueWriteGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: exclusive lock held; `&mut self` ensures uniqueness.
        unsafe { &mut *self.lock.data.get() }
    }
}

/// Shared data guard. Dereferences to `T` (immutably).
///
/// Holds the interrupt state and the caller's token for as long as the lock
/// is held. Release explicitly via [`PrologueReadGuard::release`].
pub struct PrologueReadGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    lock: &'a PrologueLock<T, L>,
    state: InterruptState<From>,
}

impl<'a, T, L, From> PrologueReadGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    /// Unlocks, restores the previous interrupt state and returns the token
    /// that was consumed by [`PrologueLock::acquire_shared`].
    pub fn release(self) -> From {
        // SAFETY: this guard exists only because `acquire_shared` took the
        // lock, and it is consumed here, so the unlock happens once. No
        // nested holds can be outstanding, since no token was ever exposed.
        unsafe { self.lock.raw.raw_unlock_shared() };

        // See `PrologueWriteGuard::release`.
        CPU::restore_interrupts(self.state)
    }
}

impl<'a, T, L, From> Deref for PrologueReadGuard<'a, T, L, From>
where
    L: HierarchicalLock,
    From: CanAcquire<level::Prologue> + PreviousToken,
{
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: shared lock held for the guard's lifetime.
        unsafe { &*self.lock.data.get() }
    }
}

