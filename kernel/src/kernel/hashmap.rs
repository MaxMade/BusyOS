//! An unordered map from keys to values.
//!
//! # Overview
//!
//! [`HashMap`] finds, inserts and removes an entry in `O(1)` on average by
//! hashing its key, and holds every entry in one block of memory taken from an
//! [`Allocator`] at the `MemoryManagement` level — the kernel [`Heap`] unless
//! another one is named, which is what `HashMap<K, V>` resolves to.
//!
//! The block is a table of buckets, a power of two of them, and an entry sits
//! at the first free bucket from the one its hash points at — open addressing
//! with linear probing. That keeps the whole map in a single allocation and the
//! probes in one cache line after another, at the cost of the two things such a
//! table cannot do: the entries come out in no order at all, and a table that
//! fills past three quarters is grown, which rehashes every entry into a new
//! block and invalidates every pointer into the old one. Where the entries have
//! to stay sorted, or a key only has an [`Ord`], a
//! [`BTreeMap`](crate::kernel::btreemap::BTreeMap) is the better fit.
//!
//! A removed entry leaves its bucket marked rather than empty — a probe that
//! ran past it before must still run past it now — and such a marker is taken
//! up again by the next insert that lands on it, or dropped by the next growth.
//!
//! Keys are hashed with [`FnvHasher`] unless another [`BuildHasher`] is named.
//! There is no randomness anywhere in the kernel to seed a hasher with, so two
//! maps of the same keys lay them out the same way; nothing here may be fed
//! keys chosen by a user to collide.
//!
//! # Tokens and dropping
//!
//! Only the operations that may allocate or free take a token (see
//! [`crate::kernel::locking`]): [`try_insert`](HashMap::try_insert),
//! [`try_reserve`](HashMap::try_reserve) and [`clear`](HashMap::clear).
//! Everything that stays inside the table it already has —
//! [`remove`](HashMap::remove), [`retain`](HashMap::retain), the lookups and
//! the iterators — needs none, because a removed entry only marks its bucket
//! and frees no memory.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot return the block: a
//! map that still holds entries *or* a table has to be given to
//! [`clear`](HashMap::clear) before it goes out of scope, and dropping one that
//! was not panics rather than leaking the block. Note the difference to
//! [`BTreeMap`](crate::kernel::btreemap::BTreeMap), which only holds memory
//! while it holds entries: an *empty* `HashMap` that was filled once — after
//! [`remove`](HashMap::remove), [`retain`](HashMap::retain) or
//! [`try_with_capacity`](HashMap::try_with_capacity) — still owns its table and
//! still has to be cleared.

use core::alloc::Layout;
use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;
use core::ptr::{self, NonNull};
use core::slice;

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// Number of buckets the first allocation holds.
///
/// A power of two, as every table size is: the bucket a hash points at is the
/// hash masked down to the size of the table, which only works that way.
const MIN_BUCKETS: usize = 8;

/// Offset basis of the 64-bit FNV-1a hash.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// Prime of the 64-bit FNV-1a hash.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The hasher a [`HashMap`] uses unless another one is named: FNV-1a over the
/// bytes it is shown, with a final mix.
///
/// Small enough to be worth its own state — one word, no table — and good
/// enough for the short keys the kernel hashes: a number, a name, a handle.
/// It is not a keyed hash and nothing seeds it, so it says nothing about keys
/// an attacker may choose; see the [module documentation](self).
#[derive(Clone, Copy, Debug)]
pub struct FnvHasher {
    /// The hash of everything written so far.
    state: u64,
}

impl FnvHasher {
    /// Creates a hasher that has been shown nothing yet.
    pub const fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= *byte as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        // FNV leaves its best bits at the top of the word, while the table
        // masks off the bottom ones — the two ends have to be swapped over.
        // This is the final mix of MurmurHash3, which spreads every bit of the
        // state over the whole result.
        let mut hash = self.state;

        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^= hash >> 33;

        hash
    }
}

/// The [`BuildHasher`] a [`HashMap`] and a
/// [`HashSet`](crate::kernel::hashset::HashSet) use unless another one is
/// named.
pub type DefaultHashBuilder = BuildHasherDefault<FnvHasher>;

/// One bucket of the table.
///
/// A real `enum` rather than a raw slot and a state byte: the discriminant is
/// what the probe reads anyway, and letting it stand for the entry means the
/// compiler drops a key and a value wherever a bucket is dropped or written
/// over, instead of this module having to.
enum Bucket<K, V> {
    /// Never held an entry. Ends a probe: what is looked for cannot be past
    /// it, or it would have been put here.
    Empty,
    /// Held an entry that was taken out again. Does not end a probe, because
    /// the entries that were put down after it are past it.
    Deleted,
    /// Holds an entry, together with the hash of its key — kept so that a
    /// growth can place it again without hashing it, and so that a probe can
    /// reject a bucket without comparing keys.
    Occupied { hash: u64, key: K, value: V },
}

/// An unordered map from `K` to `V`, held in one hash table.
///
/// # Type parameters
///
/// - `K` — key type; [`Hash`] and [`Eq`] are required for everything that
///   looks a key up, and the two must agree: equal keys have to hash equally.
/// - `V` — value type.
/// - `S` — what builds the hasher for each key. Defaults to
///   [`DefaultHashBuilder`].
/// - `A` — allocator for the one block holding the table. Defaults to the
///   kernel [`Heap`].
///
/// # Dropping
///
/// A map holding entries or a table must be given to [`clear`](HashMap::clear)
/// before it goes out of scope; dropping one that was not panics. See the
/// [module documentation](self).
pub struct HashMap<K, V, S = DefaultHashBuilder, A: Allocator<MemoryManagementLevelID> = Heap> {
    /// Start of the table, dangling while nothing is allocated.
    ptr: NonNull<Bucket<K, V>>,
    /// Buckets in the table: a power of two, or zero while there is none.
    cap: usize,
    /// Buckets holding an entry.
    len: usize,
    /// Buckets an entry was taken out of, which still lengthen a probe.
    tombstones: usize,
    hasher: S,
    alloc: A,
    /// Marks the entries as owned, so that drop checking sees `K` and `V` as
    /// types this map may drop.
    phantom: PhantomData<(K, V)>,
}

// SAFETY: the map owns its entries and hands out borrows of them only through
// borrows of itself, so sending and sharing follow the key, the value, the
// hasher and the allocator, exactly as they do for each held directly.
unsafe impl<K: Send, V: Send, S: Send, A: Allocator<MemoryManagementLevelID> + Send> Send
    for HashMap<K, V, S, A>
{
}

// SAFETY: as above.
unsafe impl<K: Sync, V: Sync, S: Sync, A: Allocator<MemoryManagementLevelID> + Sync> Sync
    for HashMap<K, V, S, A>
{
}

impl<K, V> HashMap<K, V, DefaultHashBuilder, Heap> {
    /// Creates an empty map on the kernel [`Heap`].
    ///
    /// Allocates nothing: the first insert pays for the first table.
    pub const fn new() -> Self {
        Self::with_hasher_in(DefaultHashBuilder::new(), Heap)
    }

    /// Creates an empty map on the kernel [`Heap`] with room for `capacity`
    /// entries.
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

impl<K, V> Default for HashMap<K, V, DefaultHashBuilder, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> HashMap<K, V, S, Heap> {
    /// Creates an empty map on the kernel [`Heap`], hashing with `hasher`.
    pub const fn with_hasher(hasher: S) -> Self {
        Self::with_hasher_in(hasher, Heap)
    }
}

impl<K, V, A: Allocator<MemoryManagementLevelID>> HashMap<K, V, DefaultHashBuilder, A> {
    /// Creates an empty map backed by `alloc`.
    ///
    /// Allocates nothing: the first insert pays for the first table.
    pub const fn new_in(alloc: A) -> Self {
        Self::with_hasher_in(DefaultHashBuilder::new(), alloc)
    }

    /// Creates an empty map backed by `alloc`, with room for `capacity`
    /// entries.
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
        let mut map = Self::new_in(alloc);

        // Nothing was allocated on failure, and the map is empty, so dropping
        // it here is allowed.
        match map.try_reserve(capacity, token) {
            Ok(token) => Ok((map, token)),
            Err(error) => Err(error),
        }
    }
}

impl<K, V, S, A: Allocator<MemoryManagementLevelID>> HashMap<K, V, S, A> {
    /// Creates an empty map backed by `alloc`, hashing with `hasher`.
    ///
    /// Allocates nothing: the first insert pays for the first table.
    pub const fn with_hasher_in(hasher: S, alloc: A) -> Self {
        Self {
            ptr: NonNull::dangling(),
            cap: 0,
            len: 0,
            tombstones: 0,
            hasher,
            alloc,
            phantom: PhantomData,
        }
    }

    /// Returns the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map holds no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns how many entries fit before the table is grown.
    ///
    /// Three quarters of the buckets, which is where a linear probe is still
    /// short. Removed entries are not subtracted: their markers take the same
    /// room, and an insert that runs out of it grows the table even though the
    /// map is nowhere near this many entries.
    #[inline]
    pub fn capacity(&self) -> usize {
        // The number of buckets is a power of two, so this division is exact.
        self.cap / 4 * 3
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Borrows what builds the hasher for each key.
    #[inline]
    pub fn hasher(&self) -> &S {
        &self.hasher
    }

    // --- the table -----------------------------------------------------------

    /// Returns the table as a slice, empty while nothing is allocated.
    #[inline]
    fn table(&self) -> &[Bucket<K, V>] {
        // SAFETY: every one of the `cap` buckets of the block was initialised
        // when it was allocated, and they stay put while the map is borrowed.
        // A dangling `ptr` only ever comes with `cap == 0`, which is a valid
        // empty slice.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    /// Returns the table as an exclusive slice.
    #[inline]
    fn table_mut(&mut self) -> &mut [Bucket<K, V>] {
        // SAFETY: as in `table`, and the map is borrowed exclusively for as
        // long as the returned slice lives.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }

    /// Returns the layout of the block currently held, or `None` if there is
    /// none — an untouched map or a cleared one.
    fn block_layout(&self) -> Option<Layout> {
        if self.cap == 0 {
            return None;
        }

        // The block was allocated with this very layout, so it cannot have
        // overflowed.
        Some(Layout::array::<Bucket<K, V>>(self.cap).expect("layout of the table in hand"))
    }

    /// Returns the smallest table that holds `entries` without being grown, or
    /// `None` if no table is that large.
    fn buckets_for(entries: usize) -> Option<usize> {
        let mut buckets = MIN_BUCKETS;

        while buckets / 4 * 3 < entries {
            buckets = buckets.checked_mul(2)?;
        }

        Some(buckets)
    }

    // --- capacity ------------------------------------------------------------

    /// Makes sure `additional` more entries fit without the table being grown.
    ///
    /// Allocates a larger table and places every entry in it again if they do
    /// not, which invalidates every pointer into the old one. Markers left by
    /// removed entries are dropped on the way.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if a large enough table cannot be had, including
    /// the case of the required number of buckets not fitting into a [`Layout`]
    /// at all. The map is unchanged in that case.
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
        // A map of more than `usize::MAX` entries cannot be reached, let alone
        // allocated.
        let Some(required) = self.len.checked_add(additional) else {
            return Err((Error::OutOfMemory, token));
        };

        if required <= self.capacity() {
            return Ok(token);
        }

        let Some(buckets) = Self::buckets_for(required) else {
            return Err((Error::OutOfMemory, token));
        };

        self.try_rehash(buckets, token)
    }

    /// Moves every entry into a table of `buckets` buckets and returns the old
    /// one to the allocator.
    ///
    /// `buckets` must be a power of two larger than [`len`](Self::len), which
    /// is what leaves every probe an empty bucket to stop at.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the new table cannot be had; the map is
    /// unchanged in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn try_rehash<Token>(&mut self, buckets: usize, token: Token) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        debug_assert!(buckets.is_power_of_two() && buckets > self.len);

        let Ok(layout) = Layout::array::<Bucket<K, V>>(buckets) else {
            return Err((Error::OutOfMemory, token));
        };

        let (ptr, token) = match self.alloc.allocate(layout, token) {
            Ok((ptr, token)) => (ptr.cast::<Bucket<K, V>>(), token),
            Err(error) => return Err(error),
        };

        // SAFETY: the block holds `buckets` buckets, none of them initialised
        // yet, so writing to each initialises the whole table without dropping
        // anything.
        for index in 0..buckets {
            unsafe { ptr.as_ptr().add(index).write(Bucket::Empty) };
        }

        let mask = buckets - 1;

        for index in 0..self.cap {
            // SAFETY: every bucket of the old table is initialised, and the
            // block is released below without being read again, so this moves
            // the bucket out of it for good.
            let bucket = unsafe { self.ptr.as_ptr().add(index).read() };

            let Bucket::Occupied { hash, .. } = &bucket else {
                // An empty bucket or a marker holds nothing, so dropping it
                // here is what leaves the marker behind in the old table.
                continue;
            };

            let mut slot = *hash as usize & mask;

            // The new table has more buckets than the map has entries, so an
            // empty one is always found; nothing can be a duplicate, because
            // the old table held no two equal keys.
            //
            // SAFETY: `slot` is masked to the table, so it is inside the block,
            // and every bucket of it was initialised above.
            while !matches!(unsafe { &*ptr.as_ptr().add(slot) }, Bucket::Empty) {
                slot = (slot + 1) & mask;
            }

            // SAFETY: the bucket at `slot` is empty, so it holds nothing that
            // writing over it would have to drop.
            unsafe { ptr.as_ptr().add(slot).write(bucket) };
        }

        let token = match self.block_layout() {
            // SAFETY: the old block came from this allocator with exactly this
            // layout, its entries have been moved out, and `self.ptr` is
            // replaced right below, so nothing reaches it again.
            Some(layout) => unsafe { self.alloc.deallocate(self.ptr.cast(), layout, token) },
            None => token,
        };

        self.ptr = ptr;
        self.cap = buckets;
        self.tombstones = 0;

        Ok(token)
    }

    // --- removing ------------------------------------------------------------

    /// Drops every entry `keep` returns `false` for, keeping the table.
    ///
    /// The predicate sees each entry exactly once, in no particular order, and
    /// may change the values it keeps.
    pub fn retain<Keep>(&mut self, mut keep: Keep)
    where
        Keep: FnMut(&K, &mut V) -> bool,
    {
        for index in 0..self.cap {
            let drop_entry = match &mut self.table_mut()[index] {
                Bucket::Occupied { key, value, .. } => !keep(key, value),
                Bucket::Empty | Bucket::Deleted => false,
            };

            if drop_entry {
                // Writing over the bucket is what drops the key and the value.
                self.table_mut()[index] = Bucket::Deleted;
                self.len -= 1;
                self.tombstones += 1;
            }
        }
    }

    /// Drops every entry and returns the table to the allocator, leaving an
    /// empty map that may be dropped or filled again.
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
        let Some(layout) = self.block_layout() else {
            // No table, so no entries either.
            return token;
        };

        let buckets = self.cap;

        // Nothing counts the entries from here on, so a `Drop` that panics
        // cannot lead to one of them being dropped a second time.
        self.len = 0;
        self.tombstones = 0;

        // SAFETY: every bucket of the block is initialised, and `ptr` and `cap`
        // are reset below, so nothing reaches them again.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), buckets));
        }

        // SAFETY: the block came from this allocator with exactly this layout,
        // its buckets have just been dropped, and `ptr` and `cap` are reset
        // below, so nothing reaches it again.
        let token = unsafe { self.alloc.deallocate(self.ptr.cast(), layout, token) };

        self.ptr = NonNull::dangling();
        self.cap = 0;

        token
    }

    // --- iteration -----------------------------------------------------------

    /// Returns an iterator over the entries as `(&K, &V)`, in no particular
    /// order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            buckets: self.table().iter(),
            remaining: self.len,
        }
    }

    /// Returns an iterator over the entries as `(&K, &mut V)`, in no particular
    /// order.
    ///
    /// The keys stay shared: changing one would leave it in the bucket its old
    /// hash pointed at.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let remaining = self.len;

        IterMut {
            buckets: self.table_mut().iter_mut(),
            remaining,
        }
    }

    /// Returns an iterator over the keys, in no particular order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// Returns an iterator over the values, in no particular order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }

    /// Returns an iterator over the values as exclusive borrows, in no
    /// particular order.
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.iter_mut().map(|(_, value)| value)
    }
}

impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator<MemoryManagementLevelID>> HashMap<K, V, S, A> {
    // --- lookup --------------------------------------------------------------

    /// Returns the hash `key` is placed by.
    fn hash_of<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hasher.hash_one(key)
    }

    /// Returns the bucket holding `key`, or `None` if no bucket does.
    ///
    /// `hash` must be the hash of `key`, which is where the probe starts.
    fn find<Q>(&self, hash: u64, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if self.cap == 0 {
            return None;
        }

        let mask = self.cap - 1;
        let mut index = hash as usize & mask;

        // A table always keeps an empty bucket for the probe to stop at, so
        // this cannot walk the whole table; the bound is what makes that a
        // wrong answer rather than a hang if it ever fails to hold.
        for _ in 0..self.cap {
            match &self.table()[index] {
                // What is looked for would have been put here.
                Bucket::Empty => return None,
                // The hash is compared first: it rejects a bucket that only
                // shares a probe with this key, without touching the key at
                // all.
                Bucket::Occupied {
                    hash: stored,
                    key: stored_key,
                    ..
                } if *stored == hash && stored_key.borrow() == key => return Some(index),
                _ => index = (index + 1) & mask,
            }
        }

        None
    }

    /// Returns the entry in the bucket at `index`, which
    /// [`find`](Self::find) has just reported as occupied.
    fn entry_at(&self, index: usize) -> (&K, &V) {
        match &self.table()[index] {
            Bucket::Occupied { key, value, .. } => (key, value),
            _ => unreachable!("an occupied bucket"),
        }
    }

    /// Returns the entry in the bucket at `index` for writing; see
    /// [`entry_at`](Self::entry_at).
    fn entry_at_mut(&mut self, index: usize) -> (&K, &mut V) {
        match &mut self.table_mut()[index] {
            Bucket::Occupied { key, value, .. } => (key, value),
            _ => unreachable!("an occupied bucket"),
        }
    }

    /// Returns a shared borrow of the value stored for `key`, or `None` if
    /// there is none.
    ///
    /// Takes any borrowed form of the key, so a `&str` finds an entry of a map
    /// keyed by an owned string — as long as the two hash alike, which
    /// [`Borrow`] requires.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find(self.hash_of(key), key)?;

        Some(self.entry_at(index).1)
    }

    /// Returns an exclusive borrow of the value stored for `key`, or `None` if
    /// there is none.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find(self.hash_of(key), key)?;

        Some(self.entry_at_mut(index).1)
    }

    /// Returns the stored key and its value, or `None` if `key` has no entry.
    ///
    /// Useful where the stored key carries more than what it is compared by.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find(self.hash_of(key), key)?;

        Some(self.entry_at(index))
    }

    /// Returns `true` if `key` has an entry.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find(self.hash_of(key), key).is_some()
    }

    // --- insert and remove ---------------------------------------------------

    /// Stores `value` under `key`.
    ///
    /// Returns the value that was stored for `key` before, or `None` if the key
    /// is new. Replacing a value allocates nothing and keeps the key that is
    /// already stored, dropping `key`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the table had to grow for a new key and could
    /// not; the map is unchanged and `key` and `value` are dropped in that
    /// case.
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
        let hash = self.hash_of(&key);

        if let Some(index) = self.find(hash, &key) {
            let stored = self.entry_at_mut(index).1;

            return Ok((Some(mem::replace(stored, value)), token));
        }

        // The key is new, so it needs a bucket of its own. Markers count
        // towards the load: they lengthen a probe exactly as an entry does,
        // and a table of nothing but markers would never end one.
        let token = if self.len + self.tombstones + 1 > self.capacity() {
            // Sized for the entries alone, so a table whose room went to
            // markers is cleaned rather than doubled.
            let Some(buckets) = Self::buckets_for(self.len + 1) else {
                return Err((Error::OutOfMemory, token));
            };

            match self.try_rehash(buckets, token) {
                Ok(token) => token,
                Err(error) => return Err(error),
            }
        } else {
            token
        };

        let mask = self.cap - 1;
        let mut index = hash as usize & mask;

        // The check above left at least one bucket free, and a marker is as
        // good as an empty bucket here: the key is known to be in neither the
        // rest of this probe nor anywhere else.
        let reused = loop {
            match &self.table()[index] {
                Bucket::Occupied { .. } => index = (index + 1) & mask,
                Bucket::Deleted => break true,
                Bucket::Empty => break false,
            }
        };

        self.table_mut()[index] = Bucket::Occupied { hash, key, value };
        self.len += 1;

        if reused {
            self.tombstones -= 1;
        }

        Ok((None, token))
    }

    /// Takes the entry for `key` out, returning its value, or `None` if there
    /// was none.
    ///
    /// The bucket is marked rather than emptied, so the table is kept and this
    /// needs no token; the room the entry took is given back by the next
    /// growth, or to the next entry that lands on the same bucket.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(key).map(|(_, value)| value)
    }

    /// Takes the entry for `key` out, returning the stored key along with its
    /// value, or `None` if there was none.
    ///
    /// The table is kept, so this needs no token; see [`remove`](Self::remove).
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find(self.hash_of(key), key)?;

        match mem::replace(&mut self.table_mut()[index], Bucket::Deleted) {
            Bucket::Occupied { key, value, .. } => {
                self.len -= 1;
                self.tombstones += 1;

                Some((key, value))
            }
            _ => unreachable!("an occupied bucket"),
        }
    }
}

impl<K, V, S, A: Allocator<MemoryManagementLevelID>> Drop for HashMap<K, V, S, A> {
    /// Panics if the map still holds entries or a table.
    ///
    /// Neither can be released here because `Drop::drop` cannot accept the lock
    /// token the allocator needs. Call [`HashMap::clear`] before the map goes
    /// out of scope — an empty map that was filled once owns its table just the
    /// same.
    fn drop(&mut self) {
        if !self.is_empty() || self.block_layout().is_some() {
            panic!(
                "A HashMap holding entries or a table must never be dropped. Use HashMap::clear(...) instead!"
            );
        }
    }
}

impl<K, V, S, A> fmt::Debug for HashMap<K, V, S, A>
where
    K: fmt::Debug,
    V: fmt::Debug,
    A: Allocator<MemoryManagementLevelID>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V, S, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a HashMap<K, V, S, A> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S, A: Allocator<MemoryManagementLevelID>> IntoIterator
    for &'a mut HashMap<K, V, S, A>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

/// An iterator over the entries of a [`HashMap`] as `(&K, &V)`.
///
/// Returned by [`HashMap::iter`]. Walks the table bucket by bucket and skips
/// the ones holding no entry, so the entries come in the order the hashes put
/// them in — which is no order at all as far as a caller is concerned.
pub struct Iter<'a, K, V> {
    /// The buckets left to look at.
    buckets: slice::Iter<'a, Bucket<K, V>>,
    /// The entries left to return, which the empty buckets among them do not
    /// count towards.
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        for bucket in self.buckets.by_ref() {
            if let Bucket::Occupied { key, value, .. } = bucket {
                self.remaining -= 1;

                return Some((key, value));
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

/// An iterator over the entries of a [`HashMap`] as `(&K, &mut V)`.
///
/// Returned by [`HashMap::iter_mut`]; see [`Iter`] for the order.
pub struct IterMut<'a, K, V> {
    /// The buckets left to look at.
    buckets: slice::IterMut<'a, Bucket<K, V>>,
    /// The entries left to return.
    remaining: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        for bucket in self.buckets.by_ref() {
            if let Bucket::Occupied { key, value, .. } = bucket {
                self.remaining -= 1;

                return Some((key, value));
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}

impl<K, V> FusedIterator for IterMut<'_, K, V> {}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type TestMap<K, V> = HashMap<K, V, DefaultHashBuilder, HeapAllocator>;

    /// Puts every key in the same bucket, so that one probe has to walk all of
    /// them.
    #[derive(Default)]
    struct CollidingHasher;

    impl Hasher for CollidingHasher {
        fn write(&mut self, _: &[u8]) {}

        fn finish(&self) -> u64 {
            0
        }
    }

    type CollidingMap<K, V> = HashMap<K, V, BuildHasherDefault<CollidingHasher>, HeapAllocator>;

    /// `HashMap<K, V>` has to resolve to the kernel heap and the default hasher
    /// without naming either. Never called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut map: HashMap<u32, u32> = HashMap::new();

        let token = match map.try_insert(0, 0, token) {
            Ok((_, token)) => token,
            Err((_, token)) => token,
        };

        map.clear(token)
    }

    fn new_map<K, V>() -> TestMap<K, V> {
        HashMap::new_in(HeapAllocator)
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

    /// Creates a map with a capacity; see [`insert`].
    macro_rules! with_capacity {
        ($capacity:expr, $token:expr) => {
            match HashMap::try_with_capacity_in($capacity, HeapAllocator, $token) {
                Ok(result) => result,
                Err(_) => panic!("try_with_capacity_in() returned an allocation error"),
            }
        };
    }

    /// The entries as `(key, value)` pairs, sorted — the map itself keeps no
    /// order, so the test has to put one on them.
    fn entries<K, V, S>(map: &HashMap<K, V, S, HeapAllocator>) -> std::vec::Vec<(K, V)>
    where
        K: Copy + Ord,
        V: Copy,
    {
        let mut entries: std::vec::Vec<(K, V)> =
            map.iter().map(|(key, value)| (*key, *value)).collect();

        entries.sort_unstable_by_key(|(key, _)| *key);
        entries
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
        assert_eq!(map.capacity(), 0);
        assert!(map.get(&1).is_none());
        assert!(!map.contains_key(&1));
        assert!(map.remove(&1).is_none());
        assert_eq!(map.iter().count(), 0);

        // Nothing to free, so clearing is a no-op and dropping is allowed.
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
        assert_eq!(map.get_key_value(&2), Some((&2, &20)));
        assert!(map.contains_key(&2));
        assert!(map.get(&3).is_none());

        *map.get_mut(&1).expect("entry for 1") = 11;
        assert_eq!(map.get(&1).copied(), Some(11));

        // A second insert under the same key replaces the value and allocates
        // nothing.
        let capacity = map.capacity();
        let (old, token) = insert!(map, 1, 12, token);
        assert_eq!(old, Some(11));
        assert_eq!(map.len(), 2);
        assert_eq!(map.capacity(), capacity);

        let token = map.clear(token);
        level.leave(token);
    }

    /// Removing takes the value out and needs no token: the bucket is only
    /// marked, so nothing is freed.
    #[test]
    fn remove_takes_the_entry() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in 1..=3 {
            token = insert!(map, key, key * 10, token).1;
        }

        assert_eq!(map.remove(&2), Some(20));
        assert_eq!(entries(&map), [(1, 10), (3, 30)]);

        // Removing what is not there changes nothing.
        assert!(map.remove(&2).is_none());
        assert_eq!(map.len(), 2);

        assert_eq!(map.remove_entry(&1), Some((1, 10)));
        assert_eq!(entries(&map), [(3, 30)]);

        let token = map.clear(token);
        level.leave(token);
    }

    /// The first table holds `MIN_BUCKETS` buckets, and growing keeps every
    /// entry reachable under its own key.
    #[test]
    fn growth_keeps_the_entries() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        token = insert!(map, 0, 0, token).1;
        assert_eq!(map.capacity(), MIN_BUCKETS / 4 * 3);

        for key in 1..1000 {
            token = insert!(map, key, key * 10, token).1;
        }

        assert_eq!(map.len(), 1000);
        assert!(map.capacity() >= 1000);

        for key in 0..1000 {
            assert_eq!(map.get(&key).copied(), Some(key * 10));
        }

        assert_eq!(map.iter().len(), 1000);
        assert_eq!(map.keys().count(), 1000);

        let token = map.clear(token);
        level.leave(token);
    }

    /// A marker left by a removed entry is taken up again, so filling and
    /// emptying a map over and over does not grow it.
    #[test]
    fn markers_are_reused() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in 0..6 {
            token = insert!(map, key, key, token).1;
        }

        let capacity = map.capacity();
        assert_eq!(capacity, MIN_BUCKETS / 4 * 3);

        for round in 0..100 {
            for key in 0..6 {
                assert_eq!(map.remove(&key), Some(key + round));
            }

            assert!(map.is_empty());

            for key in 0..6 {
                token = insert!(map, key, key + round + 1, token).1;
            }
        }

        assert_eq!(map.len(), 6);
        assert_eq!(map.capacity(), capacity);

        let token = map.clear(token);
        level.leave(token);
    }

    /// A probe walks past the markers of the entries it shares a bucket with,
    /// however many of them there are.
    #[test]
    fn probing_walks_past_markers() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map: CollidingMap<u32, u32> =
            HashMap::with_hasher_in(BuildHasherDefault::default(), HeapAllocator);

        // Every one of these lands on bucket zero and is pushed along by the
        // one before it.
        for key in 0..6 {
            token = insert!(map, key, key * 10, token).1;
        }

        assert_eq!(map.len(), 6);

        // Taking the front and the middle of the chain out must not cut the
        // rest of it off.
        assert_eq!(map.remove(&0), Some(0));
        assert_eq!(map.remove(&3), Some(30));

        for key in [1, 2, 4, 5] {
            assert_eq!(map.get(&key).copied(), Some(key * 10));
        }

        assert!(map.get(&0).is_none());
        assert!(map.get(&3).is_none());
        assert_eq!(entries(&map), [(1, 10), (2, 20), (4, 40), (5, 50)]);

        // The freed buckets take new entries, which are found past the markers
        // in turn.
        token = insert!(map, 6, 60, token).1;
        token = insert!(map, 7, 70, token).1;

        assert_eq!(map.get(&6).copied(), Some(60));
        assert_eq!(map.get(&7).copied(), Some(70));
        assert_eq!(map.len(), 6);

        let token = map.clear(token);
        level.leave(token);
    }

    /// `retain` drops the rejected entries and keeps the table.
    #[test]
    fn retain_drops_the_rest() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in 1..=6 {
            token = insert!(map, key, key, token).1;
        }

        let capacity = map.capacity();

        map.retain(|key, value| {
            *value += 100;
            key % 2 == 0
        });

        assert_eq!(entries(&map), [(2, 102), (4, 104), (6, 106)]);
        assert_eq!(map.len(), 3);
        assert_eq!(map.capacity(), capacity);

        // What is left is still reachable by key, past the markers.
        assert_eq!(map.get(&4).copied(), Some(104));

        map.retain(|_, _| false);
        assert!(map.is_empty());

        let token = map.clear(token);
        level.leave(token);
    }

    /// The iterators reach every entry exactly once and say how many are left.
    #[test]
    fn iteration_reaches_every_entry() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        for key in 1..=5 {
            token = insert!(map, key, key * 10, token).1;
        }

        assert_eq!(map.iter().len(), 5);
        assert_eq!(map.keys().copied().sum::<u32>(), 15);
        assert_eq!(map.values().copied().sum::<u32>(), 150);

        for value in map.values_mut() {
            *value += 1;
        }

        for (_, value) in &mut map {
            *value += 1;
        }

        assert_eq!(entries(&map), [(1, 12), (2, 22), (3, 32), (4, 42), (5, 52)]);

        // A removed entry is gone from the iterators as well, marker or not.
        assert_eq!(map.remove(&3), Some(32));
        assert_eq!((&map).into_iter().count(), 4);
        assert_eq!(map.iter().len(), 4);

        assert_eq!(std::format!("{:?}", new_map::<u32, u32>()), "{}");

        let token = map.clear(token);
        level.leave(token);
    }

    /// A capacity asked for up front is there, and is used up before the table
    /// is grown again.
    #[test]
    fn with_capacity_does_not_grow_early() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (mut map, mut token): (TestMap<u32, u32>, _) = with_capacity!(16, token);

        assert!(map.capacity() >= 16);
        assert!(map.is_empty());

        let capacity = map.capacity();

        for key in 0..capacity as u32 {
            token = insert!(map, key, key, token).1;
        }

        assert_eq!(map.capacity(), capacity);

        token = insert!(map, capacity as u32, 0, token).1;
        assert!(map.capacity() > capacity);

        // Reserving what is already there changes nothing.
        let capacity = map.capacity();
        let token = match map.try_reserve(1, token) {
            Ok(token) => token,
            Err(_) => panic!("try_reserve() returned an allocation error"),
        };
        assert_eq!(map.capacity(), capacity);

        let token = map.clear(token);
        level.leave(token);
    }

    /// A key is looked up by anything it borrows as, as long as the two hash
    /// alike.
    #[test]
    fn lookup_by_a_borrowed_key() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut map = HashMap::new_in(HeapAllocator);

        let (_, token) = insert!(map, std::string::String::from("driver"), 1, token);

        assert_eq!(map.get("driver").copied(), Some(1));
        assert!(!map.contains_key("device"));
        assert_eq!(map.remove("driver"), Some(1));

        let token = map.clear(token);
        level.leave(token);
    }

    /// Every key and value is dropped exactly once, whichever way its entry
    /// leaves the map — including the ones a growth moved to another table.
    #[test]
    fn entries_are_dropped_once() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountingDrop;

        impl Drop for CountingDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, CountingDrop>();

        // Past the first table, so the entries are moved to a second one.
        for key in 0..20 {
            token = insert!(map, key, CountingDrop, token).1;
        }

        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);

        // A removed value belongs to the caller, so nothing is dropped yet.
        let removed = map.remove(&0);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);
        drop(removed);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 1);

        // The value an insert replaces is handed over the same way.
        let (replaced, mut token) = insert!(map, 1, CountingDrop, token);
        drop(replaced);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 2);

        // `retain` and `clear` drop what they take out.
        map.retain(|key, _| *key >= 10);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 11);

        token = insert!(map, 0, CountingDrop, token).1;
        let token = map.clear(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 22);

        level.leave(token);
    }

    /// The default hasher spreads keys that differ in one bit or one byte, so
    /// that they do not all land on the same bucket.
    #[test]
    fn the_default_hasher_spreads() {
        let build = DefaultHashBuilder::new();

        let hash_of = |value: u64| {
            let mut hasher = build.build_hasher();
            value.hash(&mut hasher);
            hasher.finish()
        };

        // The same input hashes the same way; the kernel seeds nothing.
        assert_eq!(hash_of(42), hash_of(42));
        assert_ne!(hash_of(42), hash_of(43));

        // What a table of 16 buckets would mask off has to differ as well.
        let buckets: std::collections::BTreeSet<u64> =
            (0..16).map(|key| hash_of(key) % 16).collect();
        assert!(buckets.len() >= 10, "{} of 16 buckets used", buckets.len());
    }

    /// Dropping a map that still holds entries cannot free them, so it panics
    /// rather than leaking the table.
    #[test]
    #[should_panic(expected = "A HashMap holding entries or a table must never be dropped")]
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

    /// An empty map that still owns its table panics just the same: the table
    /// is what `Drop` cannot give back.
    #[test]
    #[should_panic(expected = "A HashMap holding entries or a table must never be dropped")]
    fn implicit_drop_of_an_emptied_map_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut map = new_map::<u32, u32>();

        let (_, token) = insert!(map, 1, 1, token);
        assert_eq!(map.remove(&1), Some(1));

        assert!(map.is_empty());
        assert!(map.capacity() > 0);

        core::mem::forget(level);
        core::mem::forget(token);

        drop(map);
    }
}
