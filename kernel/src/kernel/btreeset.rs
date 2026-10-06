//! An ordered set of unique values.
//!
//! # Overview
//!
//! [`BTreeSet`] keeps its elements sorted and tests membership, inserts and
//! removes in `O(log n)`, with one allocation per element from an
//! [`Allocator`] at the `MemoryManagement` level — the kernel [`Heap`] unless
//! another one is named, which is what `BTreeSet<T>` resolves to.
//!
//! The set itself is [`RbTreeSet`], which the kernel already has, over an
//! [`RbTree`](crate::utils::rbtree::RbTree) whose values are all `()`: this
//! module reuses it and only pins down the lock level and the allocator, adds
//! the [`new`](BTreeSet::new) that needs no allocator argument, and gives the
//! type the name a reader looks for. As with
//! [`BTreeMap`](crate::kernel::btreemap::BTreeMap), that name follows the
//! interface rather than the red-black tree behind it.
//!
//! # Tokens and dropping
//!
//! Only the operations that allocate or free take a token (see
//! [`crate::kernel::locking`]): [`try_insert`](BTreeSet::try_insert),
//! [`remove`](BTreeSet::remove) and [`clear`](BTreeSet::clear). Membership
//! tests, iteration and the set relations need none.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot free the elements: a
//! non-empty set has to be emptied with [`clear`](BTreeSet::clear) before it
//! goes out of scope, and dropping one that is not panics — the [`RbTreeSet`]
//! inside it does, so the panic names that rather than this set.

use core::borrow::Borrow;
use core::cmp::Ordering;

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::{
        allocator::{Allocator, Error},
        rbtree::set::RbTreeSet,
    },
};

pub use crate::utils::rbtree::set::SetIter as Iter;

/// An ordered set of `T`, backed by an [`RbTreeSet`].
///
/// # Type parameters
///
/// - `T` — element type; [`Ord`] is required for everything but the accessors
///   that do not look at an element.
/// - `A` — allocator; one node-sized allocation is made per element. Defaults
///   to the kernel [`Heap`].
///
/// # Dropping
///
/// A non-empty set must be emptied with [`clear`](BTreeSet::clear) before it
/// goes out of scope; dropping one that still holds elements panics. See the
/// [module documentation](self).
pub struct BTreeSet<T, A: Allocator<MemoryManagementLevelID> = Heap> {
    inner: RbTreeSet<T, MemoryManagementLevelID, A>,
}

impl<T> BTreeSet<T, Heap> {
    /// Creates an empty set on the kernel [`Heap`].
    ///
    /// Allocates nothing: the first element pays for the first node.
    pub const fn new() -> Self {
        Self::new_in(Heap)
    }
}

impl<T> Default for BTreeSet<T, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> BTreeSet<T, A> {
    /// Creates an empty set backed by `alloc`.
    ///
    /// Allocates nothing: the first element pays for the first node.
    pub const fn new_in(alloc: A) -> Self {
        Self {
            inner: RbTreeSet::new_in(alloc),
        }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the set holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Borrows the set the elements live in.
    ///
    /// For the parts of [`RbTreeSet`] this set does not forward.
    #[inline]
    pub fn as_rbtree_set(&self) -> &RbTreeSet<T, MemoryManagementLevelID, A> {
        &self.inner
    }
}

impl<T: Ord, A: Allocator<MemoryManagementLevelID>> BTreeSet<T, A> {
    // --- lookup --------------------------------------------------------------

    /// Returns `true` if `value` is in the set.
    ///
    /// Takes any borrowed form of the element, so a `&str` finds an element of
    /// a set of owned strings.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contains(value)
    }

    /// Returns a shared borrow of the element equal to `value`, or `None` if
    /// there is none.
    ///
    /// Useful where the stored element carries more than what it is compared
    /// by.
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get(value)
    }

    /// Walks the tree with a comparator of the caller's own, which says for
    /// each element it is shown whether what is looked for sorts before it,
    /// after it, or is it.
    pub fn find<Q, CB>(&self, cb: CB) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
        CB: FnMut(&Q) -> Ordering,
    {
        self.inner.find(cb)
    }

    // --- insert and remove ---------------------------------------------------

    /// Puts `value` into the set.
    ///
    /// Returns whether the element was already there, in which case the stored
    /// one is kept and nothing is allocated.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if a node for a new element could not be
    /// allocated; the set is unchanged and `value` is dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_insert<Token>(
        &mut self,
        value: T,
        token: Token,
    ) -> Result<(bool, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        self.inner.try_insert(value, token)
    }

    /// Takes `value` out of the set, returning whether it was there.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned, whether an element was there or
    /// not.
    pub fn remove<Q, Token>(&mut self, value: &Q, token: Token) -> (bool, Token)
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        self.inner.remove(value, token)
    }

    /// Drops every element and returns all node memory to the allocator.
    ///
    /// This is what a set has to end with: see the
    /// [module documentation](self).
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn clear<Token>(&mut self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        self.inner.clear(token)
    }

    // --- iteration and relations ---------------------------------------------

    /// Returns an iterator over the elements, ascending.
    pub fn iter(&self) -> Iter<'_, T> {
        self.inner.iter()
    }

    /// Returns `true` if every element of this set is also in `other`.
    ///
    /// An empty set is a subset of every set. Walks both sets once, so this is
    /// `O(n + m)` rather than a lookup per element. `other` may be backed by a
    /// different allocator.
    pub fn is_subset<A2>(&self, other: &BTreeSet<T, A2>) -> bool
    where
        A2: Allocator<MemoryManagementLevelID>,
    {
        self.inner.is_subset(&other.inner)
    }

    /// Returns `true` if every element of `other` is also in this set.
    pub fn is_superset<A2>(&self, other: &BTreeSet<T, A2>) -> bool
    where
        A2: Allocator<MemoryManagementLevelID>,
    {
        self.inner.is_superset(&other.inner)
    }

    /// Returns `true` if the two sets share no element.
    pub fn is_disjoint<A2>(&self, other: &BTreeSet<T, A2>) -> bool
    where
        A2: Allocator<MemoryManagementLevelID>,
    {
        self.inner.is_disjoint(&other.inner)
    }
}

impl<'a, T: Ord, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a BTreeSet<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T, A> core::fmt::Debug for BTreeSet<T, A>
where
    T: Ord + core::fmt::Debug,
    A: Allocator<MemoryManagementLevelID>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type TestSet<T> = BTreeSet<T, HeapAllocator>;

    /// `BTreeSet<T>` has to resolve to the kernel heap without naming it.
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut set: BTreeSet<u32> = BTreeSet::new();

        let token = match set.try_insert(0, token) {
            Ok((_, token)) => token,
            Err((_, token)) => token,
        };

        set.clear(token)
    }

    fn new_set<T>() -> TestSet<T> {
        BTreeSet::new_in(HeapAllocator)
    }

    /// Inserts, panicking if the allocation fails. A macro rather than a
    /// function because the error arm cannot be unwrapped: a token is not
    /// [`Debug`], so `expect` is unavailable.
    macro_rules! insert {
        ($set:expr, $value:expr, $token:expr) => {
            match $set.try_insert($value, $token) {
                Ok(result) => result,
                Err(_) => panic!("try_insert() returned an allocation error"),
            }
        };
    }

    /// The elements, ascending.
    fn values<T: Copy + Ord>(set: &TestSet<T>) -> std::vec::Vec<T> {
        set.iter().copied().collect()
    }

    /// Builds a set from a list of elements.
    macro_rules! set_of {
        ([$($value:expr),*], $token:expr) => {{
            let mut set = new_set();
            let mut token = $token;
            $(token = insert!(set, $value, token).1;)*
            (set, token)
        }};
    }

    /// A fresh set holds nothing and owns nothing, so it may be dropped as it
    /// is.
    #[test]
    fn empty_set() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut set = new_set::<u32>();

        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(&1));
        assert!(set.get(&1).is_none());

        let (removed, token) = set.remove(&1, token);
        assert!(!removed);

        let token = set.clear(token);
        drop(set);

        level.leave(token);
    }

    /// An element goes in once; putting it in again says so and changes
    /// nothing.
    #[test]
    fn insert_and_contains() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut set = new_set::<u32>();

        let (present, mut token) = insert!(set, 1, token);
        assert!(!present);

        token = insert!(set, 2, token).1;

        assert!(set.contains(&1));
        assert_eq!(set.get(&2).copied(), Some(2));
        assert!(!set.contains(&3));
        assert_eq!(set.len(), 2);

        let (present, token) = insert!(set, 1, token);
        assert!(present);
        assert_eq!(set.len(), 2);

        let token = set.clear(token);
        level.leave(token);
    }

    /// Removing says whether the element was there.
    #[test]
    fn remove_takes_the_element() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, token) = set_of!([1, 2, 3], token);

        let (removed, token) = set.remove(&2, token);
        assert!(removed);
        assert_eq!(values(&set), [1, 3]);

        let (removed, token) = set.remove(&2, token);
        assert!(!removed);
        assert_eq!(set.len(), 2);

        let token = set.clear(token);
        level.leave(token);
    }

    /// The elements come out sorted, whatever order they went in.
    #[test]
    fn iteration_is_ordered() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (set, token) = set_of!([5, 1, 4, 2, 3], token);

        assert_eq!(values(&set), [1, 2, 3, 4, 5]);
        assert_eq!(set.iter().len(), 5);
        assert_eq!(set.into_iter().copied().sum::<u32>(), 15);
        assert_eq!(std::format!("{:?}", new_set::<u32>()), "{}");

        let mut set = set;
        let token = set.clear(token);
        level.leave(token);
    }

    /// A comparator of the caller's own searches by something other than a
    /// whole element.
    #[test]
    fn find_with_a_comparator() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, token) = set_of!([10, 20, 30], token);

        assert_eq!(set.find(|value: &u32| 20.cmp(value)).copied(), Some(20));
        assert!(set.find(|value: &u32| 25.cmp(value)).is_none());

        let token = set.clear(token);
        level.leave(token);
    }

    /// The relations walk both sets once, and hold for the empty set as the
    /// definitions say.
    #[test]
    fn set_relations() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut small, token) = set_of!([1, 2], token);
        let (mut large, token) = set_of!([1, 2, 3], token);
        let (mut other, token) = set_of!([4, 5], token);
        let (mut empty, token) = (new_set::<u32>(), token);

        assert!(small.is_subset(&large));
        assert!(!large.is_subset(&small));
        assert!(large.is_superset(&small));
        assert!(small.is_subset(&small));

        assert!(small.is_disjoint(&other));
        assert!(!small.is_disjoint(&large));

        assert!(empty.is_subset(&small));
        assert!(empty.is_disjoint(&small));

        let token = small.clear(token);
        let token = large.clear(token);
        let token = other.clear(token);
        let token = empty.clear(token);
        level.leave(token);
    }

    /// Dropping an emptied set is what `clear` leaves behind, and is allowed.
    ///
    /// The other way round — dropping a set that still holds elements — panics,
    /// but it is not tested here: `RbTreeSet` and the `RbTree` inside it both
    /// panic in their `Drop`, and a panic inside a panic aborts the process
    /// instead of unwinding into `#[should_panic]`. The kernel aborts on the
    /// first panic either way.
    #[test]
    fn an_emptied_set_may_be_dropped() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, token) = set_of!([1, 2], token);
        let token = set.clear(token);

        drop(set);

        level.leave(token);
    }
}
