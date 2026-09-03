//! Atomically reference-counted shared ownership.
//!
//! # Overview
//!
//! [`Arc`] hands the same value to several owners and keeps it alive until the
//! last of them is gone. The value and the reference count live in one
//! allocation taken from an [`Allocator`] at the `MemoryManagement` level —
//! the kernel [`Heap`] unless another one is named, which is what
//! `Arc<T>` resolves to.
//!
//! # Tokens and dropping
//!
//! Allocating and freeing go through the kernel's lock-level token system (see
//! [`crate::kernel::locking`]), so both ends of an `Arc`'s life need a token:
//! [`try_new`](Arc::try_new) takes one, and so does
//! [`release`](Arc::release), which is how a handle is given up.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot free anything. A
//! handle that is *not* the last one only has to decrement the count, which
//! needs no token — dropping such a handle implicitly is therefore fine. The
//! last handle, though, would have to free the allocation, so dropping it
//! implicitly panics instead of leaking, in the same way a non-empty
//! [`RbTree`](crate::utils::rbtree::RbTree) does. Give up every handle with
//! [`release`](Arc::release) and this never comes up.
//!
//! # What is not here
//!
//! - **Weak references.** Nothing in the kernel needs them yet, and they would
//!   double the state each allocation carries.
//! - **Unsized values.** `Arc<T>` is written for `T: ?Sized` throughout, but
//!   only a sized `T` can be created: turning an `Arc<T>` into an
//!   `Arc<dyn Trait>` needs `CoerceUnsized`, which is still unstable, and the
//!   kernel enables no unstable features.

use core::alloc::Layout;
use core::borrow::Borrow;
use core::fmt;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering, fence};

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// Upper bound on the strong count.
///
/// A count beyond this many handles cannot exist — there is not enough memory
/// for that many — so reaching it means a count was incremented without a
/// handle behind it, and continuing would eventually wrap the count and free
/// the value while it is still in use.
const MAX_REFCOUNT: usize = isize::MAX as usize;

/// The shared allocation: the count, the allocator that owns the block, and
/// the value itself.
///
/// The allocator is kept here rather than in each handle, so that a handle
/// stays a single pointer, cloning one needs nothing from the allocator, and
/// the block is always freed by the very allocator it came from. `data` comes
/// last, which is what allows `T` to be unsized.
struct ArcInner<T: ?Sized, A: Allocator<MemoryManagementLevelID>> {
    /// Number of live handles on this allocation.
    strong: AtomicUsize,
    /// The allocator the block came from, and will go back to.
    alloc: A,
    /// The shared value.
    data: T,
}

/// A shared, reference-counted handle on a value of type `T`.
///
/// Cloning a handle is an atomic increment and hands out another owner of the
/// same value; [`release`](Arc::release) gives one up. The value is dropped
/// and its memory returned to `A` when the last handle is released.
///
/// # Type parameters
///
/// - `T` — the shared value.
/// - `A` — allocator for the one block holding count, allocator and value.
///   Defaults to the kernel [`Heap`].
///
/// # Dropping
///
/// Dropping the last handle implicitly panics; see the
/// [module documentation](self).
pub struct Arc<T: ?Sized, A: Allocator<MemoryManagementLevelID> = Heap> {
    ptr: NonNull<ArcInner<T, A>>,
    /// Marks the inner as owned, so that drop checking sees `T` and `A` as
    /// values this handle may drop.
    phantom: PhantomData<ArcInner<T, A>>,
}

// SAFETY: the handle shares `T` and `A` across threads, so both have to be
// `Sync` to be reachable from several of them and `Send` because the last
// thread to release is the one that drops them. The count itself is atomic.
unsafe impl<T, A> Send for Arc<T, A>
where
    T: ?Sized + Send + Sync,
    A: Allocator<MemoryManagementLevelID> + Send + Sync,
{
}

// SAFETY: as above.
unsafe impl<T, A> Sync for Arc<T, A>
where
    T: ?Sized + Send + Sync,
    A: Allocator<MemoryManagementLevelID> + Send + Sync,
{
}

impl<T, A: Allocator<MemoryManagementLevelID>> Arc<T, A> {
    /// Puts `value` into a new allocation from `alloc` and returns the first
    /// handle on it.
    ///
    /// The allocator is moved into the allocation, so it is also the one that
    /// will free it. Pass a shared reference (`&A`, which [`Allocator`] is
    /// implemented for) or a handle like [`Heap`] to share one allocator
    /// between several `Arc`s.
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
        let layout = Layout::new::<ArcInner<T, A>>();

        let (ptr, token) = match alloc.allocate(layout, token) {
            Ok((ptr, token)) => (ptr.cast::<ArcInner<T, A>>(), token),
            Err(error) => return Err(error),
        };

        // SAFETY: `ptr` is freshly allocated with the layout of an
        // `ArcInner<T, A>` and owned by this call alone, so writing the whole
        // inner into it initialises every field exactly once.
        unsafe {
            ptr.as_ptr().write(ArcInner {
                strong: AtomicUsize::new(1),
                alloc,
                data: value,
            })
        };

        Ok((
            Self {
                ptr,
                phantom: PhantomData,
            },
            token,
        ))
    }

    /// Takes the value back out if this is the only handle on it.
    ///
    /// On success the allocation is freed and the value returned. If other
    /// handles exist, this handle comes back untouched as
    /// `Err((self, token))`.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_unwrap<Token>(self, token: Token) -> Result<(T, Token), (Self, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // The handle must not be dropped along the success path: its `Drop`
        // would decrement a count this call has already taken to zero.
        let this = ManuallyDrop::new(self);

        // Claiming the count means going straight from one to zero: any other
        // handle would either be counted here, or be a clone of this one,
        // which cannot happen while this call owns it. `Acquire` on success
        // pairs with the `Release` of the handles released before, so their
        // writes to the value are visible here.
        if this
            .inner()
            .strong
            .compare_exchange(1, 0, AtomicOrdering::Acquire, AtomicOrdering::Relaxed)
            .is_err()
        {
            return Err((ManuallyDrop::into_inner(this), token));
        }

        let inner = this.ptr;
        let layout = Layout::new::<ArcInner<T, A>>();

        // SAFETY: the count is zero and this handle is the only one left, so
        // both moves take their field for good: the value is handed to the
        // caller, the allocator lives until the end of this call, and the
        // block itself is gone by then.
        let (value, alloc) = unsafe {
            (
                ptr::read(&inner.as_ref().data),
                ptr::read(&inner.as_ref().alloc),
            )
        };

        // SAFETY: the block came from `alloc` with exactly this layout, has
        // not been freed before, and nothing touches it after this call — the
        // two fields still in use were moved out above.
        let token = unsafe { alloc.deallocate(inner.cast::<u8>(), layout, token) };

        Ok((value, token))
    }
}

impl<T> Arc<T, Heap> {
    /// Puts `value` on the kernel [`Heap`] and returns the first handle on it.
    ///
    /// Shorthand for [`try_new_in`](Arc::try_new_in) with the default
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

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Arc<T, A> {
    /// Borrows the shared allocation.
    #[inline]
    fn inner(&self) -> &ArcInner<T, A> {
        // SAFETY: this handle is live, so it holds one strong count and the
        // allocation cannot have been freed.
        unsafe { self.ptr.as_ref() }
    }

    /// Gives up this handle, freeing the value once it was the last one.
    ///
    /// This is the counterpart to [`try_new`](Arc::try_new) and the only way
    /// to hand a handle back: `Drop` cannot free anything, because it cannot
    /// be given a token.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn release<Token>(self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // The count is decremented right here, so `Drop` must not run and do
        // it a second time.
        let mut this = ManuallyDrop::new(self);

        // `Release`, so that everything this owner wrote to the value is
        // visible to whoever ends up freeing it.
        if this.inner().strong.fetch_sub(1, AtomicOrdering::Release) != 1 {
            return token;
        }

        // This handle was the last one. The fence pairs with the `Release`
        // above of every other handle, making their writes visible before the
        // value is dropped.
        fence(AtomicOrdering::Acquire);

        // SAFETY: the count reached zero, so no other handle is left and the
        // allocation has not been freed yet.
        unsafe { this.drop_slow(token) }
    }

    /// Returns the number of live handles on the shared value.
    ///
    /// A snapshot: another core may add or drop handles right after the read,
    /// so this is only reliable where the caller knows no one else can hold
    /// one — for the rest, [`get_mut`](Arc::get_mut) and
    /// [`try_unwrap`](Arc::try_unwrap) do the check without the race.
    #[inline]
    pub fn strong_count(&self) -> usize {
        self.inner().strong.load(AtomicOrdering::Acquire)
    }

    /// Returns `true` if both handles refer to the same allocation.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Borrows the allocator the shared allocation came from.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.inner().alloc
    }

    /// Returns a mutable borrow of the value if this is the only handle on it,
    /// and `None` otherwise.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        // A count of one means this handle is the only one: no other handle
        // exists to be cloned, and this one cannot be cloned while it is
        // borrowed exclusively. `Acquire` pairs with the `Release` of the
        // handles released before, so their writes are visible through the
        // borrow.
        if self.inner().strong.load(AtomicOrdering::Acquire) != 1 {
            return None;
        }

        // SAFETY: the allocation is live and, per the check above, reachable
        // through this handle alone, which is borrowed exclusively for as long
        // as the returned borrow lives.
        Some(unsafe { &mut (*self.ptr.as_ptr()).data })
    }

    /// Drops the value and returns the allocation to its allocator.
    ///
    /// # Safety
    ///
    /// The strong count must have reached zero and the allocation must not
    /// have been freed yet, i.e. this must be called exactly once, from the
    /// handle that took the count to zero. `self` is dangling afterwards.
    unsafe fn drop_slow<Token>(&mut self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let inner = self.ptr;

        // SAFETY: the allocation is still there per the caller's promise. The
        // layout has to be taken while `data` is live: for an unsized `T` its
        // size is only known through the value.
        let layout = Layout::for_value(unsafe { inner.as_ref() });

        // SAFETY: as above. This moves the allocator out of the block before
        // the block is handed back to it; nothing reads the field again.
        let alloc = unsafe { ptr::read(&inner.as_ref().alloc) };

        // SAFETY: the count is zero, so this is the only reference to the
        // value and it has not been dropped before.
        unsafe { ptr::drop_in_place(&raw mut (*inner.as_ptr()).data) };

        // SAFETY: the block came from `alloc` with this very layout, has not
        // been freed before, and nothing touches it after this call.
        unsafe { alloc.deallocate(inner.cast::<u8>(), layout, token) }
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Clone for Arc<T, A> {
    /// Hands out another handle on the same value.
    ///
    /// # Panics
    ///
    /// If the strong count has run away past [`MAX_REFCOUNT`], which no
    /// legitimate number of handles can reach.
    fn clone(&self) -> Self {
        // `Relaxed` is enough: this handle already proves the allocation is
        // alive, and the count is only ever *read* for ordering when it drops
        // to zero, which the release path orders.
        let strong = self.inner().strong.fetch_add(1, AtomicOrdering::Relaxed);

        if strong > MAX_REFCOUNT {
            panic!("Arc strong count overflow");
        }

        Self {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Drop for Arc<T, A> {
    /// Gives up a handle that is not the last one.
    ///
    /// # Panics
    ///
    /// If this was the last handle: freeing the allocation needs a token,
    /// which `Drop` cannot be given, so the alternative would be to leak the
    /// value silently. Use [`Arc::release`] instead.
    fn drop(&mut self) {
        if self.inner().strong.fetch_sub(1, AtomicOrdering::Release) == 1 {
            panic!("The last Arc must never be dropped. Use Arc::release(...) instead!");
        }
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Deref for Arc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> AsRef<T> for Arc<T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized, A: Allocator<MemoryManagementLevelID>> Borrow<T> for Arc<T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized + fmt::Debug, A: Allocator<MemoryManagementLevelID>> fmt::Debug for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display, A: Allocator<MemoryManagementLevelID>> fmt::Display for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::AtomicUsize;

    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    /// `Arc<T>` has to resolve to the kernel heap without naming it. Never
    /// called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(
        token: Token,
    ) -> Result<(Arc<u32>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        Arc::try_new(0u32, token)
    }

    /// Creates an [`Arc`] on the test allocator, panicking if the allocation
    /// fails. A macro rather than a function because the error arm cannot be
    /// unwrapped: a token is not [`Debug`], so `expect` is unavailable.
    macro_rules! new_arc {
        ($value:expr, $token:expr) => {
            match Arc::try_new_in($value, HeapAllocator, $token) {
                Ok(result) => result,
                Err(_) => panic!("try_new_in() returned an allocation error"),
            }
        };
    }

    /// A handle on a fresh allocation reads the value back and counts once.
    #[test]
    fn new_deref_and_release() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (arc, token) = new_arc!(42u32, token);

        assert_eq!(*arc, 42);
        assert_eq!(arc.strong_count(), 1);

        let token = arc.release(token);
        level.leave(token);
    }

    /// A clone is another handle on the very same allocation.
    #[test]
    fn clone_shares_the_value() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (first, token) = new_arc!(7u64, token);
        let second = first.clone();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(*second, 7);
        assert_eq!(first.strong_count(), 2);
        assert_eq!(second.strong_count(), 2);

        let token = second.release(token);
        assert_eq!(first.strong_count(), 1);

        let token = first.release(token);
        level.leave(token);
    }

    /// Two allocations holding equal values are still distinct.
    #[test]
    fn separate_allocations_are_not_ptr_eq() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (first, token) = new_arc!(1u8, token);
        let (second, token) = new_arc!(1u8, token);

        assert!(!Arc::ptr_eq(&first, &second));

        let token = first.release(token);
        let token = second.release(token);
        level.leave(token);
    }

    /// Dropping a handle that is not the last one needs no token, so it may
    /// happen implicitly.
    #[test]
    fn implicit_drop_of_a_clone_is_fine() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (arc, token) = new_arc!(3u32, token);

        drop(arc.clone());

        assert_eq!(arc.strong_count(), 1);

        let token = arc.release(token);
        level.leave(token);
    }

    /// The value is mutable through the only handle, and through none while it
    /// is shared.
    #[test]
    fn get_mut_only_when_unique() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut arc, token) = new_arc!(1u32, token);

        *arc.get_mut().expect("unique") = 2;
        assert_eq!(*arc, 2);

        let mut second = arc.clone();
        assert!(arc.get_mut().is_none());
        assert!(second.get_mut().is_none());

        let token = second.release(token);

        // Unique again, so the value may be changed once more.
        *arc.get_mut().expect("unique again") = 3;
        assert_eq!(*arc, 3);

        let token = arc.release(token);
        level.leave(token);
    }

    /// The only handle hands the value back and frees the allocation.
    #[test]
    fn try_unwrap_returns_the_value_when_unique() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (arc, token) = new_arc!(99u32, token);

        let (value, token) = match arc.try_unwrap(token) {
            Ok(result) => result,
            Err(_) => panic!("try_unwrap() refused the only handle"),
        };

        assert_eq!(value, 99);
        level.leave(token);
    }

    /// A shared handle comes back untouched instead.
    #[test]
    fn try_unwrap_gives_the_handle_back_when_shared() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (first, token) = new_arc!(5u32, token);
        let second = first.clone();

        let (first, token) = match first.try_unwrap(token) {
            Ok(_) => panic!("try_unwrap() unwrapped a shared handle"),
            Err(result) => result,
        };

        assert_eq!(*first, 5);
        assert_eq!(first.strong_count(), 2);
        assert!(Arc::ptr_eq(&first, &second));

        let token = second.release(token);
        let token = first.release(token);
        level.leave(token);
    }

    /// The value is dropped exactly once, when the last handle goes.
    #[test]
    fn value_is_dropped_once_on_last_release() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountingDrop;

        impl Drop for CountingDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (arc, token) = new_arc!(CountingDrop, token);
        let second = arc.clone();

        let token = second.release(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);

        let token = arc.release(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 1);

        level.leave(token);
    }

    /// Dropping the last handle cannot free anything, so it panics rather than
    /// leaking the value.
    #[test]
    #[should_panic(expected = "The last Arc must never be dropped")]
    fn implicit_drop_of_the_last_handle_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (arc, token) = new_arc!(1u32, token);

        // The level guard panics when it is dropped as well, which would turn
        // the panic below into a double panic and abort the test process.
        core::mem::forget(level);
        core::mem::forget(token);

        drop(arc);
    }
}
