//! An unordered set of unique values.
//!
//! # Overview
//!
//! [`HashSet`] tests membership, inserts and removes in `O(1)` on average by
//! hashing its elements, and holds all of them in one block of memory taken
//! from an [`Allocator`] at the `MemoryManagement` level — the kernel [`Heap`]
//! unless another one is named, which is what `HashSet<T>` resolves to.
//!
//! The set is a [`HashMap`] whose values are all `()`: this module reuses it
//! and only drops the value from every signature, adds the set relations, and
//! gives the type the name a reader looks for. Everything the map's
//! [module documentation](crate::kernel::hashmap) says about the table, the
//! probes, the markers left by removed elements and the hasher holds here
//! unchanged. Where the elements have to stay sorted, or only have an [`Ord`],
//! a [`BTreeSet`](crate::kernel::btreeset::BTreeSet) is the better fit.
//!
//! # Tokens and dropping
//!
//! Only the operations that may allocate or free take a token (see
//! [`crate::kernel::locking`]): [`try_insert`](HashSet::try_insert),
//! [`try_reserve`](HashSet::try_reserve) and [`clear`](HashSet::clear).
//! Everything that stays inside the table it already has —
//! [`remove`](HashSet::remove), [`retain`](HashSet::retain), the membership
//! tests, the relations and the iterator — needs none.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot return the block: a
//! set that still holds elements *or* a table has to be given to
//! [`clear`](HashSet::clear) before it goes out of scope, and dropping one that
//! was not panics — the [`HashMap`] inside it does, so the panic names that
//! rather than this set. An *empty* set that was filled once still owns its
//! table and still has to be cleared.

use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::iter::FusedIterator;

use crate::{
    kernel::{
        hashmap::{DefaultHashBuilder, HashMap},
        locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    },
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// An unordered set of `T`, backed by a [`HashMap`].
///
/// # Type parameters
///
/// - `T` — element type; [`Hash`] and [`Eq`] are required for everything that
///   looks an element up, and the two must agree: equal elements have to hash
///   equally.
/// - `S` — what builds the hasher for each element. Defaults to
///   [`DefaultHashBuilder`].
/// - `A` — allocator for the one block holding the table. Defaults to the
///   kernel [`Heap`].
///
/// # Dropping
///
/// A set holding elements or a table must be given to [`clear`](HashSet::clear)
/// before it goes out of scope; dropping one that was not panics. See the
/// [module documentation](self).
pub struct HashSet<T, S = DefaultHashBuilder, A: Allocator<MemoryManagementLevelID> = Heap> {
    inner: HashMap<T, (), S, A>,
}

impl<T> HashSet<T, DefaultHashBuilder, Heap> {
    /// Creates an empty set on the kernel [`Heap`].
    ///
    /// Allocates nothing: the first element pays for the first table.
    pub const fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Creates an empty set on the kernel [`Heap`] with room for `capacity`
    /// elements.
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

impl<T> Default for HashSet<T, DefaultHashBuilder, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S> HashSet<T, S, Heap> {
    /// Creates an empty set on the kernel [`Heap`], hashing with `hasher`.
    pub const fn with_hasher(hasher: S) -> Self {
        Self::with_hasher_in(hasher, Heap)
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> HashSet<T, DefaultHashBuilder, A> {
    /// Creates an empty set backed by `alloc`.
    ///
    /// Allocates nothing: the first element pays for the first table.
    pub const fn new_in(alloc: A) -> Self {
        Self {
            inner: HashMap::new_in(alloc),
        }
    }

    /// Creates an empty set backed by `alloc`, with room for `capacity`
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
        match HashMap::try_with_capacity_in(capacity, alloc, token) {
            Ok((inner, token)) => Ok((Self { inner }, token)),
            Err(error) => Err(error),
        }
    }
}

impl<T, S, A: Allocator<MemoryManagementLevelID>> HashSet<T, S, A> {
    /// Creates an empty set backed by `alloc`, hashing with `hasher`.
    ///
    /// Allocates nothing: the first element pays for the first table.
    pub const fn with_hasher_in(hasher: S, alloc: A) -> Self {
        Self {
            inner: HashMap::with_hasher_in(hasher, alloc),
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

    /// Returns how many elements fit before the table is grown; see
    /// [`HashMap::capacity`].
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Borrows what builds the hasher for each element.
    #[inline]
    pub fn hasher(&self) -> &S {
        self.inner.hasher()
    }

    /// Borrows the map the elements live in.
    ///
    /// For the parts of [`HashMap`] this set does not forward.
    #[inline]
    pub fn as_hash_map(&self) -> &HashMap<T, (), S, A> {
        &self.inner
    }

    // --- removing ------------------------------------------------------------

    /// Drops every element `keep` returns `false` for, keeping the table.
    ///
    /// The predicate sees each element exactly once, in no particular order.
    pub fn retain<Keep>(&mut self, mut keep: Keep)
    where
        Keep: FnMut(&T) -> bool,
    {
        self.inner.retain(|value, ()| keep(value));
    }

    /// Drops every element and returns the table to the allocator, leaving an
    /// empty set that may be dropped or filled again.
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

    // --- capacity ------------------------------------------------------------

    /// Makes sure `additional` more elements fit without the table being grown;
    /// see [`HashMap::try_reserve`].
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if a large enough table cannot be had. The set is
    /// unchanged in that case.
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
        self.inner.try_reserve(additional, token)
    }

    // --- iteration -----------------------------------------------------------

    /// Returns an iterator over the elements, in no particular order.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.inner.iter(),
        }
    }
}

impl<T: Hash + Eq, S: BuildHasher, A: Allocator<MemoryManagementLevelID>> HashSet<T, S, A> {
    // --- lookup --------------------------------------------------------------

    /// Returns `true` if `value` is in the set.
    ///
    /// Takes any borrowed form of the element, so a `&str` finds an element of
    /// a set of owned strings — as long as the two hash alike, which
    /// [`Borrow`] requires.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.contains_key(value)
    }

    /// Returns a shared borrow of the element equal to `value`, or `None` if
    /// there is none.
    ///
    /// Useful where the stored element carries more than what it is compared
    /// by.
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get_key_value(value).map(|(value, ())| value)
    }

    // --- insert and remove ---------------------------------------------------

    /// Puts `value` into the set.
    ///
    /// Returns whether the element was already there, in which case the stored
    /// one is kept, `value` is dropped and nothing is allocated.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the table had to grow for a new element and
    /// could not; the set is unchanged and `value` is dropped in that case.
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
        match self.inner.try_insert(value, (), token) {
            Ok((old, token)) => Ok((old.is_some(), token)),
            Err(error) => Err(error),
        }
    }

    /// Takes `value` out of the set, returning whether it was there.
    ///
    /// The bucket is marked rather than emptied, so the table is kept and this
    /// needs no token; see [`HashMap::remove`].
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    /// Takes the element equal to `value` out and returns it, or `None` if it
    /// was not there.
    ///
    /// The table is kept, so this needs no token; see [`remove`](Self::remove).
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove_entry(value).map(|(value, ())| value)
    }

    // --- relations -----------------------------------------------------------

    /// Returns `true` if every element of this set is also in `other`.
    ///
    /// An empty set is a subset of every set. Looks each element of this set up
    /// in `other`, so this is `O(n)` on average rather than a walk of both.
    /// `other` may be backed by a different hasher and a different allocator.
    pub fn is_subset<S2, A2>(&self, other: &HashSet<T, S2, A2>) -> bool
    where
        S2: BuildHasher,
        A2: Allocator<MemoryManagementLevelID>,
    {
        // A set cannot be a subset of a smaller one, and this spares the walk.
        self.len() <= other.len() && self.iter().all(|value| other.contains(value))
    }

    /// Returns `true` if every element of `other` is also in this set.
    pub fn is_superset<S2, A2>(&self, other: &HashSet<T, S2, A2>) -> bool
    where
        S2: BuildHasher,
        A2: Allocator<MemoryManagementLevelID>,
    {
        other.is_subset(self)
    }

    /// Returns `true` if the two sets share no element.
    pub fn is_disjoint<S2, A2>(&self, other: &HashSet<T, S2, A2>) -> bool
    where
        S2: BuildHasher,
        A2: Allocator<MemoryManagementLevelID>,
    {
        // Walking the smaller of the two and looking each element up in the
        // larger is the same answer for less work.
        if self.len() <= other.len() {
            self.iter().all(|value| !other.contains(value))
        } else {
            other.iter().all(|value| !self.contains(value))
        }
    }
}

impl<'a, T, S, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a HashSet<T, S, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T, S, A> fmt::Debug for HashSet<T, S, A>
where
    T: fmt::Debug,
    A: Allocator<MemoryManagementLevelID>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// An iterator over the elements of a [`HashSet`].
///
/// Returned by [`HashSet::iter`]. The elements come in the order the hashes put
/// them in, which is no order at all as far as a caller is concerned.
pub struct Iter<'a, T> {
    /// The entries of the map the elements are the keys of.
    inner: crate::kernel::hashmap::Iter<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.inner.next().map(|(value, ())| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> FusedIterator for Iter<'_, T> {}

#[cfg(test)]
mod tests {
    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type TestSet<T> = HashSet<T, DefaultHashBuilder, HeapAllocator>;

    /// `HashSet<T>` has to resolve to the kernel heap and the default hasher
    /// without naming either. Never called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut set: HashSet<u32> = HashSet::new();

        let token = match set.try_insert(0, token) {
            Ok((_, token)) => token,
            Err((_, token)) => token,
        };

        set.clear(token)
    }

    fn new_set<T>() -> TestSet<T> {
        HashSet::new_in(HeapAllocator)
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

    /// Builds a set from a list of elements.
    macro_rules! set_of {
        ([$($value:expr),*], $token:expr) => {{
            let mut set = new_set();
            let mut token = $token;
            $(token = insert!(set, $value, token).1;)*
            (set, token)
        }};
    }

    /// The elements, sorted — the set itself keeps no order, so the test has to
    /// put one on them.
    fn values<T: Copy + Ord>(set: &TestSet<T>) -> std::vec::Vec<T> {
        let mut values: std::vec::Vec<T> = set.iter().copied().collect();

        values.sort_unstable();
        values
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
        assert_eq!(set.capacity(), 0);
        assert!(!set.contains(&1));
        assert!(set.get(&1).is_none());
        assert!(!set.remove(&1));

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

    /// Removing says whether the element was there, and needs no token: the
    /// bucket is only marked, so nothing is freed.
    #[test]
    fn remove_takes_the_element() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, token) = set_of!([1, 2, 3], token);

        assert!(set.remove(&2));
        assert_eq!(values(&set), [1, 3]);

        assert!(!set.remove(&2));
        assert_eq!(set.len(), 2);

        assert_eq!(set.take(&3), Some(3));
        assert_eq!(values(&set), [1]);

        let token = set.clear(token);
        level.leave(token);
    }

    /// Every element is reached exactly once, and the table is kept across a
    /// growth.
    #[test]
    fn iteration_reaches_every_element() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut set = new_set::<u32>();

        for value in 0..100 {
            token = insert!(set, value, token).1;
        }

        assert_eq!(set.len(), 100);
        assert_eq!(set.iter().len(), 100);
        assert_eq!(values(&set), (0..100).collect::<std::vec::Vec<u32>>());
        assert_eq!((&set).into_iter().copied().sum::<u32>(), 4950);

        for value in 0..100 {
            assert!(set.contains(&value));
        }

        assert_eq!(std::format!("{:?}", new_set::<u32>()), "{}");

        let token = set.clear(token);
        level.leave(token);
    }

    /// `retain` drops the rejected elements and keeps the table.
    #[test]
    fn retain_drops_the_rest() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, token) = set_of!([1, 2, 3, 4, 5, 6], token);

        let capacity = set.capacity();

        set.retain(|value| value % 2 == 0);

        assert_eq!(values(&set), [2, 4, 6]);
        assert_eq!(set.capacity(), capacity);
        assert!(set.contains(&4));

        set.retain(|_| false);
        assert!(set.is_empty());

        let token = set.clear(token);
        level.leave(token);
    }

    /// A capacity asked for up front is there, and the set fills it without
    /// growing.
    #[test]
    fn with_capacity_does_not_grow_early() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, mut token): (TestSet<u32>, _) =
            match HashSet::try_with_capacity_in(16, HeapAllocator, token) {
                Ok(result) => result,
                Err(_) => panic!("try_with_capacity_in() returned an allocation error"),
            };

        let capacity = set.capacity();
        assert!(capacity >= 16);

        for value in 0..capacity as u32 {
            token = insert!(set, value, token).1;
        }

        assert_eq!(set.capacity(), capacity);

        let token = set.clear(token);
        level.leave(token);
    }

    /// The relations hold as their definitions say, including for the empty
    /// set.
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
        assert!(!large.is_disjoint(&small));

        assert!(empty.is_subset(&small));
        assert!(empty.is_disjoint(&small));

        let token = small.clear(token);
        let token = large.clear(token);
        let token = other.clear(token);
        let token = empty.clear(token);
        level.leave(token);
    }

    /// Dropping an emptied set is what `clear` leaves behind, and is allowed.
    #[test]
    fn an_emptied_set_may_be_dropped() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut set, token) = set_of!([1, 2], token);
        let token = set.clear(token);

        drop(set);

        level.leave(token);
    }

    /// Dropping a set that still holds elements cannot free them, so it panics
    /// rather than leaking the table. The map inside it is what panics, and
    /// says so.
    #[test]
    #[should_panic(expected = "A HashMap holding entries or a table must never be dropped")]
    fn implicit_drop_of_a_non_empty_set_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (set, token) = set_of!([1], token);

        // The level guard panics when it is dropped as well, which would turn
        // the panic below into a double panic and abort the test process.
        core::mem::forget(level);
        core::mem::forget(token);

        drop(set);
    }
}
