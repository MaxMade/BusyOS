//! A single value on the heap, owned by one handle.
//!
//! # Overview
//!
//! [`Box`] moves a value out of the frame it was built in and into one
//! allocation from an [`Allocator`] at the `MemoryManagement` level — the
//! kernel [`Heap`] unless another one is named, which is what `Box<T>`
//! resolves to. The value keeps its address for as long as the box lives, and
//! there is exactly one owner: no counting,
//! [`DerefMut`] straight to the value, and
//! [`into_inner`](Box::into_inner) to take it back off the heap.
//!
//! # Tokens and dropping
//!
//! The allocation and the freeing go through the kernel's lock-level token
//! system (see [`crate::kernel::locking`]), so
//! [`try_new`](Box::try_new) takes a token and so do the two ways to give a box
//! up: [`release`](Box::release), which drops the value, and
//! [`into_inner`](Box::into_inner), which hands it back.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot free the allocation.

use core::alloc::Layout;
use core::borrow::{Borrow, BorrowMut};
use core::fmt;
use core::marker::{PhantomData, Unsize};
use core::mem::ManuallyDrop;
use core::ops::{CoerceUnsized, Deref, DerefMut};
use core::ptr::{self, NonNull};

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// The sole owner of one heap-allocated `T`.
///
/// # Type parameters
///
/// - `T` — the value on the heap.
/// - `A` — allocator for the one block holding it. Defaults to the kernel
///   [`Heap`], which carries no state, so `Box<T>` is a single pointer.
///
/// # Dropping
///
/// A box must be given up with [`release`](Box::release) or
/// [`into_inner`](Box::into_inner); dropping one implicitly panics. See the
/// [module documentation](self).
pub struct Box<T: ?Sized, A: Allocator<MemoryManagementLevelID> = Heap> {
    ptr: NonNull<T>,
    /// The allocator the block came from, and will go back to. Kept next to
    /// the pointer rather than inside the block, so that the value on the heap
    /// is nothing but the value.
    alloc: A,
    /// Marks the value as owned, so that drop checking sees `T` as a type this
    /// box may drop.
    phantom: PhantomData<T>,
}

// SAFETY: the box is the only owner of its value and hands out borrows of it
// only through borrows of itself, so sending and sharing follow `T` and the
// allocator, exactly as they do for a `T` held directly.
unsafe impl<T, A> Send for Box<T, A>
where
    T: ?Sized + Send,
    A: Allocator<MemoryManagementLevelID> + Send,
{
}

// SAFETY: as above.
unsafe impl<T, A> Sync for Box<T, A>
where
    T: ?Sized + Sync,
    A: Allocator<MemoryManagementLevelID> + Sync,
{
}

impl<T, U, A> CoerceUnsized<Box<U, A>> for Box<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
    A: Allocator<MemoryManagementLevelID>,
{
}

impl<T> Box<T, Heap> {
    /// Moves `value` onto the kernel [`Heap`].
    ///
    /// Shorthand for [`try_new_in`](Box::try_new_in) with the default
    /// allocator.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the heap cannot serve the request; `value` is
    /// dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_new<Token>(value: T, token: Token) -> Result<(Self, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        Self::try_new_in(value, Heap, token)
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Box<T, A> {
    /// Moves `value` into a new allocation from `alloc`.
    ///
    /// The allocator is kept in the box, so it is also the one that will free
    /// the block. Pass a shared reference (`&A`, which [`Allocator`] is
    /// implemented for) or a handle like [`Heap`] to share one allocator
    /// between several boxes.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `alloc` cannot serve the request; `value` and
    /// `alloc` are dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_new_in<Token>(
        value: T,
        alloc: A,
        token: Token,
    ) -> Result<(Self, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let layout = Layout::new::<T>();

        let (ptr, token) = match alloc.allocate(layout, token) {
            Ok((ptr, token)) => (ptr.cast::<T>(), token),
            Err(error) => return Err(error),
        };

        // SAFETY: `ptr` is freshly allocated, sized and aligned for a `T`, and
        // owned by this call alone, so writing initialises it exactly once.
        unsafe { ptr.as_ptr().write(value) };

        Ok((
            Self {
                ptr,
                alloc,
                phantom: PhantomData,
            },
            token,
        ))
    }

    /// Takes the value back off the heap and returns the block to the
    /// allocator.
    ///
    /// The counterpart to [`try_new`](Box::try_new); use
    /// [`release`](Box::release) where the value is not wanted.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn into_inner<Token>(self, token: Token) -> (T, Token)
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // The block is freed right here, so `Drop` must not run — it would
        // panic, and there would be nothing left for it to free anyway.
        let this = ManuallyDrop::new(self);

        let ptr = this.ptr;
        let layout = Layout::new::<T>();

        // SAFETY: this box was the only owner, so both moves take their value
        // for good: the value goes to the caller, the allocator lives until the
        // end of this call, and the block is gone by then.
        let (value, alloc) = unsafe { (ptr.as_ptr().read(), ptr::read(&this.alloc)) };

        // SAFETY: the block came from `alloc` with exactly this layout, has not
        // been freed before, and nothing touches it after this call — the value
        // in it was moved out above.
        let token = unsafe { alloc.deallocate(ptr.cast(), layout, token) };

        (value, token)
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Box<T, A> {
    /// Borrows the allocator the block came from.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Drops the value and returns the block to the allocator.
    ///
    /// This is how a box is given up; `Drop` cannot do it, because it cannot
    /// be handed a token.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn release<Token>(self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // As in `into_inner`: the block is freed here, so `Drop` must not run.
        let this = ManuallyDrop::new(self);

        let ptr = this.ptr;

        // SAFETY: the box owns the value, so it is live. The layout has to be
        // taken while it is: for an unsized `T` its size is only known through
        // the value.
        let layout = Layout::for_value(unsafe { ptr.as_ref() });

        // SAFETY: this box was the only owner, so the value has not been
        // dropped before and nothing reads it afterwards.
        unsafe { ptr::drop_in_place(ptr.as_ptr()) };

        // SAFETY: as above — the allocator is moved out of a box that is never
        // dropped, and the block is its own with this very layout.
        let alloc = unsafe { ptr::read(&this.alloc) };

        // SAFETY: the block has not been freed before and nothing touches it
        // after this call.
        unsafe { alloc.deallocate(ptr.cast(), layout, token) }
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Drop for Box<T, A> {
    /// Always panics.
    ///
    /// A box is the only owner of its value, so an implicit drop could only
    /// leak both the value and the block: freeing them needs a token, which
    /// `Drop` cannot be given. Use [`Box::release`] or [`Box::into_inner`].
    fn drop(&mut self) {
        panic!("A Box must never be dropped. Use Box::release(...) instead!");
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Deref for Box<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the box owns the value, so it is live, and it is borrowed for
        // no longer than the box itself.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> DerefMut for Box<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`, and the box is the only owner and is itself
        // borrowed exclusively for as long as the returned borrow lives.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> AsRef<T> for Box<T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> AsMut<T> for Box<T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Borrow<T> for Box<T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> BorrowMut<T> for Box<T, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: ?Sized + fmt::Debug, A: Allocator<MemoryManagementLevelID>> fmt::Debug for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display, A: Allocator<MemoryManagementLevelID>> fmt::Display for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    /// `Box<T>` has to resolve to the kernel heap without naming it. Never
    /// called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(
        token: Token,
    ) -> Result<(Box<u32>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        Box::try_new(0u32, token)
    }

    /// Creates a box on the test allocator, panicking if the allocation fails.
    /// A macro rather than a function because the error arm cannot be
    /// unwrapped: a token is not [`Debug`], so `expect` is unavailable.
    macro_rules! new_box {
        ($value:expr, $token:expr) => {
            match Box::try_new_in($value, HeapAllocator, $token) {
                Ok(result) => result,
                Err(_) => panic!("try_new_in() returned an allocation error"),
            }
        };
    }

    /// A fresh box reads the value back and gives it up again.
    #[test]
    fn new_deref_and_release() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(42u32, token);

        assert_eq!(*boxed, 42);
        assert_eq!(boxed.as_ref(), &42);

        let token = boxed.release(token);
        level.leave(token);
    }

    /// The value can be changed through the box, which is its only owner.
    #[test]
    fn deref_mut_writes_through() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut boxed, token) = new_box!(1u32, token);

        *boxed += 1;
        *boxed.as_mut() *= 10;

        assert_eq!(*boxed, 20);

        let token = boxed.release(token);
        level.leave(token);
    }

    /// The value keeps its address while the box lives, so borrows of it stay
    /// valid across a move of the box itself.
    #[test]
    fn the_value_keeps_its_address() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(7u64, token);
        let address = &*boxed as *const u64;

        let moved = boxed;
        assert_eq!(&*moved as *const u64, address);

        let token = moved.release(token);
        level.leave(token);
    }

    /// The block is aligned for the value, not just for a pointer.
    #[test]
    fn alignment_is_honoured() {
        #[repr(align(64))]
        struct Aligned(u64);

        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(Aligned(1), token);

        assert_eq!((&*boxed as *const Aligned).addr() % 64, 0);
        assert_eq!(boxed.0, 1);

        let token = boxed.release(token);
        level.leave(token);
    }

    /// `into_inner` hands the value back off the heap without dropping it.
    #[test]
    fn into_inner_returns_the_value() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountingDrop(u32);

        impl Drop for CountingDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(CountingDrop(5), token);
        let (value, token) = boxed.into_inner(token);

        assert_eq!(value.0, 5);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);

        // The value belongs to this frame now, so it is dropped here.
        drop(value);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 1);

        level.leave(token);
    }

    /// `release` drops the value exactly once.
    #[test]
    fn release_drops_the_value() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountingDrop;

        impl Drop for CountingDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(CountingDrop, token);

        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);

        let token = boxed.release(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 1);

        level.leave(token);
    }

    /// Formatting goes through to the value.
    #[test]
    fn formatting_delegates_to_the_value() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(7u32, token);

        assert_eq!(std::format!("{boxed}"), "7");
        assert_eq!(std::format!("{boxed:?}"), "7");

        let token = boxed.release(token);
        level.leave(token);
    }

    /// A box has no owner but itself, so any implicit drop would leak both the
    /// value and its block: it panics instead.
    #[test]
    #[should_panic(expected = "A Box must never be dropped")]
    fn implicit_drop_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (boxed, token) = new_box!(1u32, token);

        // The level guard panics when it is dropped as well, which would turn
        // the panic below into a double panic and abort the test process.
        core::mem::forget(level);
        core::mem::forget(token);

        drop(boxed);
    }
}
