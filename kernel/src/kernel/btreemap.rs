//! An ordered map from keys to values.
//!
//! # Overview
//!
//! [`BTreeMap`] keeps its entries sorted by key and finds, inserts and removes
//! one in `O(log n)`, with one allocation per entry from an [`Allocator`] at
//! the `MemoryManagement` level — the kernel [`Heap`] unless another one is
//! named, which is what `BTreeMap<K, V>` resolves to.
//!
//! The map itself is [`RbTree`], which the kernel already has: this module
//! reuses it and only pins down the two type parameters a caller would
//! otherwise have to spell out, the lock level and the allocator. What it adds
//! on top is the [`new`](BTreeMap::new) that needs no allocator argument, and
//! the names — `BTreeMap` is what a reader looks for.
//!
//! The name follows the interface, not the structure behind it: the entries
//! live in a red-black tree, so the ordering and the `O(log n)` bounds hold,
//! but not the wide nodes a real B-tree gets its cache behaviour from.
//!
//! # Tokens and dropping
//!
//! Only the operations that allocate or free take a token (see
//! [`crate::kernel::locking`]): [`try_insert`](BTreeMap::try_insert),
//! [`remove`](BTreeMap::remove) and [`clear`](BTreeMap::clear). Looking up and
//! iterating need none.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot free the entries: a
//! non-empty map has to be emptied with [`clear`](BTreeMap::clear) before it
//! goes out of scope, and dropping one that is not panics — the [`RbTree`]
//! inside it does, so the panic names the tree rather than the map.

use core::borrow::Borrow;
use core::cmp::Ordering;

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::{
        allocator::{Allocator, Error},
        rbtree::RbTree,
    },
};

pub use crate::utils::rbtree::{Iter, IterMut};

/// An ordered map from `K` to `V`, backed by an [`RbTree`].
///
/// # Type parameters
///
/// - `K` — key type; [`Ord`] is required for everything but the accessors that
///   do not look at a key.
/// - `V` — value type.
/// - `A` — allocator; one node-sized allocation is made per entry. Defaults to
///   the kernel [`Heap`].
///
/// # Dropping
///
/// A non-empty map must be emptied with [`clear`](BTreeMap::clear) before it
/// goes out of scope; dropping one that still holds entries panics. See the
/// [module documentation](self).
pub struct BTreeMap<K, V, A: Allocator<MemoryManagementLevelID> = Heap> {
    inner: RbTree<K, V, MemoryManagementLevelID, A>,
}

impl<K, V> BTreeMap<K, V, Heap> {
    /// Creates an empty map on the kernel [`Heap`].
    ///
    /// Allocates nothing: the first entry pays for the first node.
    pub const fn new() -> Self {
        Self::new_in(Heap)
    }
}

impl<K, V> Default for BTreeMap<K, V, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, A: Allocator<MemoryManagementLevelID>> BTreeMap<K, V, A> {
    /// Creates an empty map backed by `alloc`.
    ///
    /// Allocates nothing: the first entry pays for the first node.
    pub const fn new_in(alloc: A) -> Self {
        Self {
            inner: RbTree::new_in(alloc),
        }
    }

    /// Returns the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the map holds no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Borrows the tree the entries live in.
    ///
    /// For the parts of [`RbTree`] this map does not forward.
    #[inline]
    pub fn as_tree(&self) -> &RbTree<K, V, MemoryManagementLevelID, A> {
        &self.inner
    }
}

impl<K: Ord, V, A: Allocator<MemoryManagementLevelID>> BTreeMap<K, V, A> {
    // --- lookup --------------------------------------------------------------

    /// Returns a shared borrow of the value stored for `key`, or `None` if
    /// there is none.
    ///
    /// Takes any borrowed form of the key, so a `&str` finds an entry of a map
    /// keyed by an owned string.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get(key)
    }

    /// Returns an exclusive borrow of the value stored for `key`, or `None` if
    /// there is none.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get_mut(key)
    }

    /// Returns `true` if `key` has an entry.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contains_key(key)
    }

    /// Walks the tree with a comparator of the caller's own, which says for
    /// each key it is shown whether what is looked for sorts before it, after
    /// it, or is it.
    ///
    /// This is the way to search by something other than a whole key — the
    /// start of a range, a field of a compound key — without holding the key
    /// itself.
    pub fn find<Q, CB>(&self, cb: CB) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        CB: FnMut(&Q) -> Ordering,
    {
        self.inner.find(cb)
    }

    // --- insert and remove ---------------------------------------------------

    /// Stores `value` under `key`.
    ///
    /// Returns the value that was stored for `key` before, or `None` if the
    /// key is new. Replacing a value allocates nothing.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if a node for a new key could not be allocated;
    /// the map is unchanged and `key` and `value` are dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_insert<Token>(
        &mut self,
        key: K,
        value: V,
        token: Token,
    ) -> Result<(Option<V>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        self.inner.try_insert(key, value, token)
    }

    /// Takes the entry for `key` out, returning its value, or `None` if there
    /// was none.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned, whether an entry was there or not.
    pub fn remove<Q, Token>(&mut self, key: &Q, token: Token) -> (Option<V>, Token)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        self.inner.remove(key, token)
    }

    /// Drops every entry and returns all node memory to the allocator.
    ///
    /// This is what a map has to end with: see the
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

    // --- iteration -----------------------------------------------------------

    /// Returns an iterator over the entries as `(&K, &V)`, by ascending key.
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.inner.iter()
    }

    /// Returns an iterator over the entries as `(&K, &mut V)`, by ascending
    /// key.
    ///
    /// The keys stay shared: changing one would break the order the tree is
    /// built on.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.inner.iter_mut()
    }

    /// Returns an iterator over the keys, ascending.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// Returns an iterator over the values, by ascending key.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }

    /// Returns an iterator over the values as exclusive borrows, by ascending
    /// key.
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.iter_mut().map(|(_, value)| value)
    }
}

impl<'a, K: Ord, V, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a BTreeMap<K, V, A> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K: Ord, V, A: Allocator<MemoryManagementLevelID>> IntoIterator
    for &'a mut BTreeMap<K, V, A>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V, A> core::fmt::Debug for BTreeMap<K, V, A>
where
    K: Ord + core::fmt::Debug,
    V: core::fmt::Debug,
    A: Allocator<MemoryManagementLevelID>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type TestMap<K, V> = BTreeMap<K, V, HeapAllocator>;

    /// `BTreeMap<K, V>` has to resolve to the kernel heap without naming it.
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();

        let token = match map.try_insert(0, 0, token) {
            Ok((_, token)) => token,
            Err((_, token)) => token,
        };

        map.clear(token)
    }

    fn new_map<K, V>() -> TestMap<K, V> {
        BTreeMap::new_in(HeapAllocator)
    }

    /// Inserts, panicking if the allocation fails. A macro rather than a
    /// function because the error arm cannot be unwrapped: a token is not
    /// [`Debug`], so `expect` is unavailable.
    macro_rules! insert {
        ($map:expr, $key:expr, $value:expr, $token:expr) => {
            match $map.try_insert($key, $value, $token) {
                Ok(result) => result,
                Err(_) => panic!("try_insert() returned an allocation error"),
            }
        };
    }

    /// The entries as `(key, value)` pairs, by ascending key.
    fn entries<K: Copy + Ord, V: Copy>(map: &TestMap<K, V>) -> std::vec::Vec<(K, V)> {
        map.iter().map(|(key, value)| (*key, *value)).collect()
    }

    /// A fresh map holds nothing and owns nothing, so it may be dropped as it
    /// is.
    #[test]
    fn empty_map() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert!(map.get(&1).is_none());
        assert!(!map.contains_key(&1));

        let (removed, token) = map.remove(&1, token);
        assert!(removed.is_none());

        let token = map.clear(token);
        drop(map);

        level.leave(token);
    }

    /// An entry can be found, read and written by its key.
    #[test]
    fn insert_and_look_up() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        let (old, mut token) = insert!(map, 1, 10, token);
        assert!(old.is_none());

        token = insert!(map, 2, 20, token).1;

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1).copied(), Some(10));
        assert!(map.contains_key(&2));
        assert!(map.get(&3).is_none());

        *map.get_mut(&1).expect("entry for 1") = 11;
        assert_eq!(map.get(&1).copied(), Some(11));

        // A second insert under the same key replaces the value and allocates
        // nothing.
        let (old, token) = insert!(map, 1, 12, token);
        assert_eq!(old, Some(11));
        assert_eq!(map.len(), 2);

        let token = map.clear(token);
        level.leave(token);
    }

    /// Removing takes the value out and shortens the map.
    #[test]
    fn remove_takes_the_entry() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in 1..=3 {
            token = insert!(map, key, key * 10, token).1;
        }

        let (removed, token) = map.remove(&2, token);
        assert_eq!(removed, Some(20));
        assert_eq!(entries(&map), [(1, 10), (3, 30)]);

        // Removing what is not there changes nothing.
        let (removed, token) = map.remove(&2, token);
        assert!(removed.is_none());
        assert_eq!(map.len(), 2);

        let token = map.clear(token);
        level.leave(token);
    }

    /// The entries come out sorted by key, whatever order they went in.
    #[test]
    fn iteration_is_ordered() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in [5, 1, 4, 2, 3] {
            token = insert!(map, key, key * 10, token).1;
        }

        assert_eq!(entries(&map), [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        assert_eq!(
            map.keys().copied().collect::<std::vec::Vec<u32>>(),
            [1, 2, 3, 4, 5]
        );
        assert_eq!(
            map.values().copied().collect::<std::vec::Vec<u32>>(),
            [10, 20, 30, 40, 50]
        );
        assert_eq!(map.iter().len(), 5);

        for value in map.values_mut() {
            *value += 1;
        }

        for (_, value) in &mut map {
            *value += 1;
        }

        assert_eq!(entries(&map), [(1, 12), (2, 22), (3, 32), (4, 42), (5, 52)]);
        assert_eq!(std::format!("{:?}", new_map::<u32, u32>()), "{}");

        let token = map.clear(token);
        level.leave(token);
    }

    /// A comparator of the caller's own searches by something other than a
    /// whole key.
    #[test]
    fn find_with_a_comparator() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in [10, 20, 30] {
            token = insert!(map, key, key, token).1;
        }

        // Looks for the entry whose key is 20, without holding a `20` as a key.
        let found = map.find(|key: &u32| 20.cmp(key));
        assert_eq!(found.map(|(key, value)| (*key, *value)), Some((20, 20)));

        assert!(map.find(|key: &u32| 25.cmp(key)).is_none());

        let token = map.clear(token);
        level.leave(token);
    }

    /// Dropping a map that still holds entries cannot free them, so it panics
    /// rather than leaking every node.
    #[test]
    #[should_panic(expected = "must never be dropped")]
    fn implicit_drop_of_a_non_empty_map_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        let (_, token) = insert!(map, 1, 1, token);

        // The level guard panics when it is dropped as well, which would turn
        // the panic below into a double panic and abort the test process.
        core::mem::forget(level);
        core::mem::forget(token);

        drop(map);
    }
}
