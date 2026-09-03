//! A growable array of contiguously stored elements.
//!
//! # Overview
//!
//! [`Vec`] owns one block of memory holding its elements back to back, taken
//! from an [`Allocator`] at the `MemoryManagement` level — the kernel [`Heap`]
//! unless another one is named, which is what `Vec<T>` resolves to.
//!
//! Pushing and popping at the end are `O(1)` amortised, the elements are
//! reachable as a slice, and an element costs nothing beyond itself. What it
//! costs instead is the growth: a push that runs out of capacity allocates a
//! larger block and moves every element over, which also invalidates every
//! pointer into the old one. Where elements have to keep their address, or be
//! removed from the middle in constant time, a
//! [`LinkedList`](crate::kernel::linked_list::LinkedList) is the better fit.
//!
//! # Tokens and dropping
//!
//! Only the operations that may allocate or free take a token (see
//! [`crate::kernel::locking`]): [`try_push`](Vec::try_push),
//! [`try_reserve`](Vec::try_reserve), [`try_insert`](Vec::try_insert) and
//! [`clear`](Vec::clear). Everything that stays inside the block it already
//! has — [`pop`](Vec::pop), [`remove`](Vec::remove),
//! [`truncate`](Vec::truncate), [`retain`](Vec::retain) and the slice
//! accessors — needs none, because dropping an element in place frees no
//! memory.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot return the block: a
//! vector that still holds elements *or* a block has to be given to
//! [`clear`](Vec::clear) before it goes out of scope, and dropping one that was
//! not panics rather than leaking the block. Note the difference to
//! [`RbTree`](crate::utils::rbtree::RbTree) and
//! [`LinkedList`](crate::kernel::linked_list::LinkedList), which only hold
//! memory while they hold elements: an *empty* `Vec` with capacity left over —
//! after [`truncate`](Vec::truncate), [`pop`](Vec::pop) or
//! [`try_with_capacity`](Vec::try_with_capacity) — still owns its block and
//! still has to be cleared.
//!
//! # Zero-sized elements
//!
//! A `T` of size zero needs no memory at all: such a vector never allocates,
//! reports a [`capacity`](Vec::capacity) of `usize::MAX` and only counts its
//! elements, so that dropping them still runs their [`Drop`].

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::slice;

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// Number of elements the first allocation holds.
///
/// Growing means copying everything over, so the smallest useful block is
/// worth a few elements rather than one; anything asked for explicitly through
/// [`Vec::try_reserve`] is allocated as asked.
const MIN_CAPACITY: usize = 4;

/// A contiguous, growable array of `T`.
///
/// The elements are reachable as a slice — `Vec` derefs to `[T]`, so indexing,
/// [`iter`](slice::iter), [`first`](slice::first), sorting and everything else
/// a slice offers work directly on it.
///
/// # Type parameters
///
/// - `T` — element type.
/// - `A` — allocator for the one block holding the elements. Defaults to the
///   kernel [`Heap`].
///
/// # Dropping
///
/// A vector holding elements or a block must be given to
/// [`clear`](Vec::clear) before it goes out of scope; dropping one that was not
/// panics. See the [module documentation](self).
pub struct Vec<T, A: Allocator<MemoryManagementLevelID> = Heap> {
    /// Start of the block, dangling while nothing is allocated.
    ptr: NonNull<T>,
    /// Elements the block can hold; `usize::MAX` for a zero-sized `T`.
    cap: usize,
    /// Elements currently in it, all at the front.
    len: usize,
    alloc: A,
    /// Marks the elements as owned, so that drop checking sees `T` as a type
    /// this vector may drop.
    phantom: PhantomData<T>,
}

// SAFETY: the vector owns its elements and hands out borrows of them only
// through borrows of itself, so sending and sharing follow `T` and the
// allocator, exactly as they do for a `T` held directly.
unsafe impl<T: Send, A: Allocator<MemoryManagementLevelID> + Send> Send for Vec<T, A> {}

// SAFETY: as above.
unsafe impl<T: Sync, A: Allocator<MemoryManagementLevelID> + Sync> Sync for Vec<T, A> {}

impl<T> Vec<T, Heap> {
    /// Creates an empty vector on the kernel [`Heap`].
    ///
    /// Allocates nothing: the first push pays for the first block.
    pub const fn new() -> Self {
        Self::new_in(Heap)
    }

    /// Creates an empty vector on the kernel [`Heap`] with room for
    /// `capacity` elements.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the heap cannot serve the request.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_with_capacity<Token>(
        capacity: usize,
        token: Token,
    ) -> Result<(Self, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        Self::try_with_capacity_in(capacity, Heap, token)
    }
}

impl<T> Default for Vec<T, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Vec<T, A> {
    /// Creates an empty vector backed by `alloc`.
    ///
    /// Allocates nothing: the first push pays for the first block.
    pub const fn new_in(alloc: A) -> Self {
        Self {
            ptr: NonNull::dangling(),
            // A zero-sized element needs no memory, so the capacity is not a
            // promise about a block and stays at the maximum for good.
            cap: if size_of::<T>() == 0 { usize::MAX } else { 0 },
            len: 0,
            alloc,
            phantom: PhantomData,
        }
    }

    /// Creates an empty vector backed by `alloc`, with room for `capacity`
    /// elements.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `alloc` cannot serve the request; `alloc` is
    /// dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_with_capacity_in<Token>(
        capacity: usize,
        alloc: A,
        token: Token,
    ) -> Result<(Self, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut vec = Self::new_in(alloc);

        // Nothing was allocated on failure, and the vector is empty, so
        // dropping it here is allowed.
        match vec.try_reserve(capacity, token) {
            Ok(token) => Ok((vec, token)),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns how many elements fit before the next allocation.
    ///
    /// `usize::MAX` for a zero-sized element type, which never allocates.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns the elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the first `len` slots of the block are initialised and stay
        // put while the vector is borrowed. A dangling `ptr` only ever comes
        // with `len == 0`, which is a valid empty slice.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the elements as an exclusive slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and the vector is borrowed exclusively for
        // as long as the returned slice lives.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    // --- capacity ------------------------------------------------------------

    /// Returns the layout of the block currently held, or `None` if there is
    /// none — an untouched vector, a cleared one, or any vector of a
    /// zero-sized element type.
    fn block_layout(&self) -> Option<Layout> {
        if self.cap == 0 || size_of::<T>() == 0 {
            return None;
        }

        // The block was allocated with this very layout, so it cannot have
        // overflowed.
        Some(Layout::array::<T>(self.cap).expect("layout of the block in hand"))
    }

    /// Makes sure `additional` more elements fit without allocating again.
    ///
    /// Allocates a larger block and moves the elements over if they do not,
    /// which invalidates every pointer into the old one.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if a large enough block cannot be had, including
    /// the case of the required capacity not fitting into a [`Layout`] at all.
    /// The vector is unchanged in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_reserve<Token>(
        &mut self,
        additional: usize,
        token: Token,
    ) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // A length past `usize::MAX` cannot be reached, let alone allocated.
        let Some(required) = self.len.checked_add(additional) else {
            return Err((Error::OutOfMemory, token));
        };

        // Also the way out for a zero-sized element type, whose capacity is
        // `usize::MAX` and needs no block behind it.
        if required <= self.cap {
            return Ok(token);
        }

        // Doubling keeps a series of pushes at amortised `O(1)`; `required`
        // wins where it asks for more than that.
        let new_cap = required.max(self.cap.saturating_mul(2)).max(MIN_CAPACITY);

        let Ok(new_layout) = Layout::array::<T>(new_cap) else {
            return Err((Error::OutOfMemory, token));
        };

        let (new_ptr, token) = match self.alloc.allocate(new_layout, token) {
            Ok((ptr, token)) => (ptr.cast::<T>(), token),
            Err(error) => return Err(error),
        };

        // SAFETY: the new block holds at least `len` elements and the two
        // blocks are distinct, so this moves the elements into it; the old
        // slots are not read again.
        unsafe { ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len) };

        let token = match self.block_layout() {
            // SAFETY: the old block came from this allocator with exactly this
            // layout, its elements have been moved out, and `self.ptr` is
            // replaced right below, so nothing reaches it again.
            Some(layout) => unsafe { self.alloc.deallocate(self.ptr.cast(), layout, token) },
            None => token,
        };

        self.ptr = new_ptr;
        self.cap = new_cap;

        Ok(token)
    }

    // --- adding and removing -------------------------------------------------

    /// Appends `value` to the end.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the vector had to grow and could not; it is
    /// unchanged and `value` is dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_push<Token>(&mut self, value: T, token: Token) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let token = match self.try_reserve(1, token) {
            Ok(token) => token,
            Err(error) => return Err(error),
        };

        // SAFETY: the slot at `len` is inside the block per the reservation
        // above, and uninitialised, so writing to it initialises it without
        // dropping anything.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };

        self.len += 1;

        Ok(token)
    }

    /// Takes the last element out, or returns `None` if the vector is empty.
    ///
    /// The block is kept, so this needs no token; the capacity stays as it was.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;

        // SAFETY: the element at the new `len` is initialised and no longer
        // counted, so this moves it out for good.
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }

    /// Inserts `value` at `index`, shifting everything from there on one slot
    /// to the right.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the vector had to grow and could not; it is
    /// unchanged and `value` is dropped in that case.
    ///
    /// # Panics
    ///
    /// If `index > len`.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_insert<Token>(
        &mut self,
        index: usize,
        value: T,
        token: Token,
    ) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        assert!(index <= self.len, "Vec::insert(...) index out of bounds");

        let token = match self.try_reserve(1, token) {
            Ok(token) => token,
            Err(error) => return Err(error),
        };

        // SAFETY: `index <= len` and the block holds one more element than
        // `len` per the reservation, so source and destination of the shift are
        // both inside it. The slot at `index` is left uninitialised by the move
        // and initialised by the write.
        unsafe {
            let slot = self.ptr.as_ptr().add(index);
            ptr::copy(slot, slot.add(1), self.len - index);
            slot.write(value);
        }

        self.len += 1;

        Ok(token)
    }

    /// Takes the element at `index` out, shifting everything after it one slot
    /// to the left.
    ///
    /// The block is kept, so this needs no token.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "Vec::remove(...) index out of bounds");

        self.len -= 1;

        // SAFETY: `index` was inside the initialised range, so the element is
        // moved out for good; the shift closes the hole it leaves, and every
        // slot it touches is inside the block. The length was cut first, so the
        // slot freed at the end is no longer counted.
        unsafe {
            let slot = self.ptr.as_ptr().add(index);
            let value = slot.read();
            ptr::copy(slot.add(1), slot, self.len - index);
            value
        }
    }

    /// Takes the element at `index` out and moves the last element into its
    /// place, which is `O(1)` but does not keep the order.
    ///
    /// The block is kept, so this needs no token.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "Vec::swap_remove(...) index out of bounds"
        );

        self.len -= 1;

        // SAFETY: both slots are inside the initialised range, and the last one
        // is no longer counted after the length was cut, so the element at
        // `index` is moved out and replaced by it. Where the two coincide the
        // copy is a no-op on a slot nothing counts any more.
        unsafe {
            let slot = self.ptr.as_ptr().add(index);
            let value = slot.read();
            ptr::copy(self.ptr.as_ptr().add(self.len), slot, 1);
            value
        }
    }

    /// Drops everything past the first `len` elements, keeping the block.
    ///
    /// Does nothing if the vector is already that short. The capacity is not
    /// touched, so the vector still owns its block afterwards and still has to
    /// be given to [`clear`](Vec::clear) — which is also what `truncate(0)`
    /// does *not* do.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }

        let dropped = self.len - len;

        // Cut the length first: from here on the tail is nobody's, so a drop
        // that panics cannot lead to it being dropped twice.
        self.len = len;

        // SAFETY: the tail holds `dropped` initialised elements inside the
        // block, and nothing counts them any more.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr().add(len),
                dropped,
            ))
        };
    }

    /// Drops every element `keep` returns `false` for, keeping the order of the
    /// rest and the block.
    ///
    /// The predicate sees each element exactly once, front to back.
    pub fn retain<Keep>(&mut self, mut keep: Keep)
    where
        Keep: FnMut(&T) -> bool,
    {
        let len = self.len;

        // The elements belong to this call while it shuffles them: a length of
        // zero means a `keep` that panics leaks them rather than leaving the
        // vector with slots it would drop a second time.
        self.len = 0;

        let mut kept = 0;

        for index in 0..len {
            // SAFETY: `index < len`, so the element is initialised, and the
            // borrow ends before anything is moved or dropped.
            let element = unsafe { self.ptr.as_ptr().add(index) };

            // SAFETY: as above.
            if !keep(unsafe { &*element }) {
                // SAFETY: the element is initialised and, with the length at
                // zero and `index` skipped below, unreachable from here on.
                unsafe { ptr::drop_in_place(element) };
                continue;
            }

            if kept != index {
                // SAFETY: `kept < index`, both inside the block; the slot at
                // `kept` was moved out of earlier in this walk, and the one at
                // `index` is left uninitialised, which is what the length
                // below accounts for.
                unsafe { ptr::copy_nonoverlapping(element, self.ptr.as_ptr().add(kept), 1) };
            }

            kept += 1;
        }

        self.len = kept;
    }

    /// Drops every element and returns the block to the allocator, leaving an
    /// empty vector that may be dropped or filled again.
    ///
    /// This is what a vector has to end with: see the
    /// [module documentation](self).
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn clear<Token>(&mut self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        self.truncate(0);

        let Some(layout) = self.block_layout() else {
            return token;
        };

        // SAFETY: the block came from this allocator with exactly this layout,
        // its elements have just been dropped, and `ptr` and `cap` are reset
        // below, so nothing reaches it again.
        let token = unsafe { self.alloc.deallocate(self.ptr.cast(), layout, token) };

        self.ptr = NonNull::dangling();
        self.cap = 0;

        token
    }
}

impl<T: Copy, A: Allocator<MemoryManagementLevelID>> Vec<T, A> {
    /// Appends every element of `values`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the vector had to grow and could not; it is
    /// unchanged in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_extend_from_slice<Token>(
        &mut self,
        values: &[T],
        token: Token,
    ) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let token = match self.try_reserve(values.len(), token) {
            Ok(token) => token,
            Err(error) => return Err(error),
        };

        // SAFETY: the reservation made room for `values.len()` more elements
        // past `len`, and `values` is borrowed from elsewhere, so the two
        // ranges cannot overlap. `T` is `Copy`, so copying the bytes is a
        // complete copy of each element.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.ptr.as_ptr().add(self.len),
                values.len(),
            )
        };

        self.len += values.len();

        Ok(token)
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Drop for Vec<T, A> {
    /// Panics if the vector still holds elements or a block.
    ///
    /// Neither can be released here because `Drop::drop` cannot accept the lock
    /// token the allocator needs. Call [`Vec::clear`] before the vector goes
    /// out of scope — an empty vector with capacity left over owns its block
    /// just the same.
    fn drop(&mut self) {
        if !self.is_empty() || self.block_layout().is_some() {
            panic!(
                "A Vec holding elements or a block must never be dropped. Use Vec::clear(...) instead!"
            );
        }
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Deref for Vec<T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> DerefMut for Vec<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> AsRef<[T]> for Vec<T, A> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> AsMut<[T]> for Vec<T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: fmt::Debug, A: Allocator<MemoryManagementLevelID>> fmt::Debug for Vec<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<'a, T, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a Vec<T, A> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a mut Vec<T, A> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
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

    type TestVec<T> = Vec<T, HeapAllocator>;

    /// `Vec<T>` has to resolve to the kernel heap without naming it. Never
    /// called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut vec: Vec<u32> = Vec::new();

        let token = match vec.try_push(0, token) {
            Ok(token) => token,
            Err((_, token)) => token,
        };

        vec.clear(token)
    }

    fn new_vec<T>() -> TestVec<T> {
        Vec::new_in(HeapAllocator)
    }

    /// The elements front to back.
    fn values<T: Copy>(vec: &TestVec<T>) -> std::vec::Vec<T> {
        vec.iter().copied().collect()
    }

    /// Pushes, panicking if the allocation fails. A macro rather than a
    /// function because the error arm cannot be unwrapped: a token is not
    /// [`Debug`], so `expect` is unavailable.
    macro_rules! push {
        ($vec:expr, $value:expr, $token:expr) => {
            match $vec.try_push($value, $token) {
                Ok(token) => token,
                Err(_) => panic!("try_push() returned an allocation error"),
            }
        };
    }

    /// Inserts at an index; see [`push`].
    macro_rules! insert {
        ($vec:expr, $index:expr, $value:expr, $token:expr) => {
            match $vec.try_insert($index, $value, $token) {
                Ok(token) => token,
                Err(_) => panic!("try_insert() returned an allocation error"),
            }
        };
    }

    /// Appends a slice; see [`push`].
    macro_rules! extend {
        ($vec:expr, $values:expr, $token:expr) => {
            match $vec.try_extend_from_slice($values, $token) {
                Ok(token) => token,
                Err(_) => panic!("try_extend_from_slice() returned an allocation error"),
            }
        };
    }

    /// Creates a vector with a capacity; see [`push`].
    macro_rules! with_capacity {
        ($capacity:expr, $token:expr) => {
            match Vec::try_with_capacity_in($capacity, HeapAllocator, $token) {
                Ok(result) => result,
                Err(_) => panic!("try_with_capacity_in() returned an allocation error"),
            }
        };
    }

    /// A fresh vector holds nothing and owns nothing, so it may be dropped as
    /// it is.
    #[test]
    fn empty_vec() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert_eq!(vec.as_slice(), &[] as &[u32]);
        assert!(vec.pop().is_none());

        // Nothing to free, so clearing is a no-op and dropping is allowed.
        let token = vec.clear(token);
        drop(vec);

        level.leave(token);
    }

    /// Pushing appends and popping takes the newest.
    #[test]
    fn push_and_pop() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        for value in 1..=3 {
            token = push!(vec, value, token);
        }

        assert_eq!(values(&vec), [1, 2, 3]);
        assert_eq!(vec.len(), 3);
        assert!(vec.capacity() >= 3);

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(values(&vec), [1, 2]);

        // Popping keeps the block, so the vector still has to be cleared.
        let token = vec.clear(token);
        level.leave(token);
    }

    /// Growing keeps every element, in order.
    #[test]
    fn growth_keeps_the_elements() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        // The first block holds `MIN_CAPACITY` elements and doubles from there.
        token = push!(vec, 0, token);
        assert_eq!(vec.capacity(), MIN_CAPACITY);

        for value in 1..MIN_CAPACITY as u32 + 1 {
            token = push!(vec, value, token);
        }

        assert_eq!(vec.capacity(), MIN_CAPACITY * 2);

        for value in MIN_CAPACITY as u32 + 1..100 {
            token = push!(vec, value, token);
        }

        assert_eq!(vec.len(), 100);
        assert!(vec.capacity() >= 100);
        assert_eq!(values(&vec), (0..100).collect::<std::vec::Vec<u32>>());

        let token = vec.clear(token);
        level.leave(token);
    }

    /// The elements are reachable as a slice, for reading and for writing.
    #[test]
    fn slice_access() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        for value in 1..=3 {
            token = push!(vec, value, token);
        }

        assert_eq!(vec[0], 1);
        assert_eq!(vec.first().copied(), Some(1));
        assert_eq!(vec.last().copied(), Some(3));
        assert_eq!(vec.iter().sum::<u32>(), 6);

        vec[0] = 10;
        for value in &mut vec {
            *value *= 2;
        }

        assert_eq!(vec.as_slice(), &[20u32, 4, 6]);

        let token = vec.clear(token);
        level.leave(token);
    }

    /// Inserting and removing shift the rest along and keep the order.
    #[test]
    fn insert_and_remove() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        token = insert!(vec, 0, 2, token); // [2]
        token = insert!(vec, 0, 1, token); // [1, 2]
        token = insert!(vec, 2, 4, token); // [1, 2, 4]
        token = insert!(vec, 2, 3, token); // [1, 2, 3, 4]

        assert_eq!(values(&vec), [1, 2, 3, 4]);

        assert_eq!(vec.remove(0), 1);
        assert_eq!(values(&vec), [2, 3, 4]);

        assert_eq!(vec.remove(1), 3);
        assert_eq!(values(&vec), [2, 4]);

        let token = vec.clear(token);
        level.leave(token);
    }

    /// `swap_remove` takes an element out in `O(1)`, at the cost of the order.
    #[test]
    fn swap_remove_moves_the_last_element() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        for value in 1..=4 {
            token = push!(vec, value, token);
        }

        assert_eq!(vec.swap_remove(0), 1);
        assert_eq!(values(&vec), [4, 2, 3]);

        // Removing the last element is the case where the two slots coincide.
        assert_eq!(vec.swap_remove(2), 3);
        assert_eq!(values(&vec), [4, 2]);

        let token = vec.clear(token);
        level.leave(token);
    }

    /// `truncate` drops the tail but keeps the block, so the vector still has
    /// to be cleared afterwards.
    #[test]
    fn truncate_keeps_the_capacity() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        for value in 1..=4 {
            token = push!(vec, value, token);
        }

        let capacity = vec.capacity();

        // Longer than the vector: nothing happens.
        vec.truncate(10);
        assert_eq!(vec.len(), 4);

        vec.truncate(2);
        assert_eq!(values(&vec), [1, 2]);
        assert_eq!(vec.capacity(), capacity);

        vec.truncate(0);
        assert!(vec.is_empty());
        assert_eq!(vec.capacity(), capacity);

        let token = vec.clear(token);
        assert_eq!(vec.capacity(), 0);

        level.leave(token);
    }

    /// `retain` drops the rejected elements and keeps the order of the rest.
    #[test]
    fn retain_removes_and_keeps_order() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        for value in 1..=6 {
            token = push!(vec, value, token);
        }

        token = push!(vec, 7, token);

        // Drops 1 (the front), 3 and 5 (the middle) and 7 (the back).
        vec.retain(|value| value % 2 == 0);

        assert_eq!(values(&vec), [2, 4, 6]);
        assert_eq!(vec.len(), 3);

        vec.retain(|_| false);
        assert!(vec.is_empty());

        let token = vec.clear(token);
        level.leave(token);
    }

    /// A capacity asked for up front is allocated as asked and used up before
    /// the vector grows again.
    #[test]
    fn with_capacity_does_not_grow_early() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut vec, mut token): (TestVec<u32>, _) = with_capacity!(16, token);

        assert_eq!(vec.capacity(), 16);
        assert!(vec.is_empty());

        for value in 0..16 {
            token = push!(vec, value, token);
        }

        assert_eq!(vec.capacity(), 16);

        token = push!(vec, 16, token);
        assert_eq!(vec.capacity(), 32);
        assert_eq!(vec.len(), 17);

        let token = vec.clear(token);
        level.leave(token);
    }

    /// A slice is appended element for element.
    #[test]
    fn extend_from_slice() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        token = push!(vec, 1, token);
        token = extend!(vec, &[2, 3, 4], token);
        token = extend!(vec, &[], token);

        assert_eq!(values(&vec), [1, 2, 3, 4]);

        let token = vec.clear(token);
        level.leave(token);
    }

    /// Every element is dropped exactly once, whichever way it leaves the
    /// vector — including the ones a growth moved to another block.
    #[test]
    fn elements_are_dropped_once() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountingDrop;

        impl Drop for CountingDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<CountingDrop>();

        // Past `MIN_CAPACITY`, so the elements are moved to a second block.
        for _ in 0..6 {
            token = push!(vec, CountingDrop, token);
        }

        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);

        // A popped element belongs to the caller, so nothing is dropped yet.
        let popped = vec.pop();
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);
        drop(popped);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 1);

        // The same goes for a removed one.
        drop(vec.remove(0));
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 2);

        // `retain`, `truncate` and `clear` drop what they remove.
        vec.retain(|_| false);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 6);

        token = push!(vec, CountingDrop, token);
        vec.truncate(0);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 7);

        token = push!(vec, CountingDrop, token);
        let token = vec.clear(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 8);

        level.leave(token);
    }

    /// A zero-sized element type needs no memory: nothing is ever allocated,
    /// the capacity is the maximum, and the elements are only counted.
    #[test]
    fn zero_sized_elements() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<()>();

        assert_eq!(vec.capacity(), usize::MAX);

        for _ in 0..3 {
            token = push!(vec, (), token);
        }

        assert_eq!(vec.len(), 3);
        assert_eq!(values(&vec), [(), (), ()]);
        assert_eq!(vec.pop(), Some(()));
        assert_eq!(vec.len(), 2);

        // Nothing was allocated, so clearing only drops the elements and the
        // capacity stays where it was.
        let token = vec.clear(token);
        assert_eq!(vec.capacity(), usize::MAX);
        assert!(vec.is_empty());

        level.leave(token);
    }

    /// Dropping a vector that still holds elements cannot free them, so it
    /// panics rather than leaking the block.
    #[test]
    #[should_panic(expected = "A Vec holding elements or a block must never be dropped")]
    fn implicit_drop_of_a_non_empty_vec_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        let token = push!(vec, 1, token);

        // The level guard panics when it is dropped as well, which would turn
        // the panic below into a double panic and abort the test process.
        core::mem::forget(level);
        core::mem::forget(token);

        drop(vec);
    }

    /// An empty vector that still owns its block panics just the same: the
    /// block is what `Drop` cannot give back.
    #[test]
    #[should_panic(expected = "A Vec holding elements or a block must never be dropped")]
    fn implicit_drop_of_an_emptied_vec_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut vec = new_vec::<u32>();

        let token = push!(vec, 1, token);
        vec.truncate(0);

        assert!(vec.is_empty());
        assert!(vec.capacity() > 0);

        core::mem::forget(level);
        core::mem::forget(token);

        drop(vec);
    }
}
