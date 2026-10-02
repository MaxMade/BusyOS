//! A fixed-size set of small integer-like ids, one bit per id.
//!
//! [`BitSet`] generalises [`CPUSet`](crate::arch::generic::cpu::CPUSet): the
//! same array of words, but keyed by any `T` that converts [`Into<usize>`],
//! so a set of core ids, interrupt vectors or any other dense numbering is the
//! same type with a different parameter.
//!
//! # Sizing
//!
//! The set holds `WORDS` machine words and so `WORDS * usize::BITS` ids. The
//! word count is the parameter, rather than the number of ids, because the
//! array length cannot be computed from a generic parameter without the
//! unstable `generic_const_exprs`. [`words`] does that computation for a
//! concrete number of ids:
//!
//! ```ignore
//! type CPUs = BitSet<CPUID, { bitset::words(CPU::CPUID_BITS) }>;
//! ```
//!
//! # Converting back
//!
//! Adding and testing ids only needs `T: Into<usize>`. Whatever hands ids
//! back out, such as [`iter`](BitSet::iter), [`first`](BitSet::first) and
//! [`Debug`], also needs `T: TryFrom<usize>` to turn a bit index into a `T`
//! again. An index that does not convert, which only [`all`](BitSet::all) or
//! [`Not`] can set, is skipped there.

use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::PhantomData,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign},
};

/// Bits in one word of a [`BitSet`].
const WORD_BITS: usize = usize::BITS as usize;

/// The number of words a [`BitSet`] needs to hold ids `0..bits`.
pub const fn words(bits: usize) -> usize {
    bits.div_ceil(WORD_BITS)
}

/// Splits a raw id into the index of its word and the mask of its bit.
const fn locate(raw: usize) -> (usize, usize) {
    (raw / WORD_BITS, 1 << (raw % WORD_BITS))
}

/// A set of ids of type `T`, stored as one bit per id in `WORDS` words.
///
/// See the [module documentation](self) for how to size it and which bounds
/// `T` needs for what.
///
/// # Panics
///
/// Every method taking a `T` panics if the id is not below
/// [`capacity`](Self::capacity).
pub struct BitSet<T, const WORDS: usize>
where
    T: Into<usize>,
{
    words: [usize; WORDS],

    /// The set stores no `T`, so it is `Send`, `Sync`, `Copy` and so on
    /// whatever `T` is.
    _marker: PhantomData<fn() -> T>,
}

impl<T, const WORDS: usize> BitSet<T, WORDS>
where
    T: Into<usize>,
{
    /// The set with no ids in it.
    pub const fn empty() -> Self {
        Self {
            words: [0; WORDS],
            _marker: PhantomData,
        }
    }

    /// The set with every bit set, all [`capacity`](Self::capacity) of them.
    ///
    /// That may include indices no `T` names, if `T` has fewer values than
    /// the set has bits. They count towards [`len`](Self::len), but are
    /// skipped wherever ids are handed back out.
    pub const fn all() -> Self {
        Self {
            words: [usize::MAX; WORDS],
            _marker: PhantomData,
        }
    }

    /// The number of ids the set has room for.
    pub const fn capacity(&self) -> usize {
        WORDS * WORD_BITS
    }

    /// The number of ids in the set.
    pub fn len(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Whether the set holds no ids.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    /// Whether every bit of the set is set.
    pub fn is_full(&self) -> bool {
        self.words.iter().all(|&word| word == usize::MAX)
    }

    /// Adds `value`, returning whether it was not in the set before.
    pub fn insert(&mut self, value: T) -> bool {
        let (idx, mask) = Self::locate(value);
        let absent = self.words[idx] & mask == 0;
        self.words[idx] |= mask;
        absent
    }

    /// Removes `value`, returning whether it was in the set.
    pub fn remove(&mut self, value: T) -> bool {
        let (idx, mask) = Self::locate(value);
        let present = self.words[idx] & mask != 0;
        self.words[idx] &= !mask;
        present
    }

    /// Adds `value` if it is absent and removes it otherwise, returning
    /// whether it is in the set afterwards.
    pub fn toggle(&mut self, value: T) -> bool {
        let (idx, mask) = Self::locate(value);
        self.words[idx] ^= mask;
        self.words[idx] & mask != 0
    }

    /// Whether `value` is in the set.
    pub fn contains(&self, value: T) -> bool {
        let (idx, mask) = Self::locate(value);
        self.words[idx] & mask != 0
    }

    /// Removes every id.
    pub fn clear(&mut self) {
        self.words = [0; WORDS];
    }

    /// Whether every id in `self` is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.words
            .iter()
            .zip(other.words.iter())
            .all(|(&a, &b)| a & !b == 0)
    }

    /// Whether every id in `other` is also in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Whether `self` and `other` have no id in common.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.words
            .iter()
            .zip(other.words.iter())
            .all(|(&a, &b)| a & b == 0)
    }

    /// The raw words, lowest ids first and each id at bit `id % usize::BITS`
    /// of word `id / usize::BITS`.
    pub const fn as_words(&self) -> &[usize; WORDS] {
        &self.words
    }

    /// Builds a set from raw words laid out as in
    /// [`as_words`](Self::as_words).
    pub const fn from_words(words: [usize; WORDS]) -> Self {
        Self {
            words,
            _marker: PhantomData,
        }
    }

    /// Splits `value` into the index of its word and the mask of its bit.
    ///
    /// # Panics
    ///
    /// If `value` is not below [`capacity`](Self::capacity).
    fn locate(value: T) -> (usize, usize) {
        let raw: usize = value.into();
        assert!(
            raw < WORDS * WORD_BITS,
            "id {raw} out of range for a set of {} ids",
            WORDS * WORD_BITS
        );
        locate(raw)
    }
}

impl<T, const WORDS: usize> BitSet<T, WORDS>
where
    T: Into<usize> + TryFrom<usize>,
{
    /// Walks the ids in the set, lowest first.
    pub fn iter(&self) -> Iter<'_, T, WORDS> {
        Iter {
            words: &self.words,
            front: 0,
            back: WORDS * WORD_BITS,
            _marker: PhantomData,
        }
    }

    /// The lowest id in the set, or [`None`] if it is empty.
    pub fn first(&self) -> Option<T> {
        self.iter().next()
    }

    /// The highest id in the set, or [`None`] if it is empty.
    pub fn last(&self) -> Option<T> {
        self.iter().next_back()
    }
}

/// An iterator over the ids in a [`BitSet`], see [`BitSet::iter`].
pub struct Iter<'a, T, const WORDS: usize> {
    words: &'a [usize; WORDS],

    /// The lowest bit index not yet looked at.
    front: usize,

    /// One past the highest bit index not yet looked at.
    back: usize,

    _marker: PhantomData<fn() -> T>,
}

impl<T, const WORDS: usize> Iter<'_, T, WORDS> {
    /// The lowest set bit in `front..back`, if any.
    fn next_set(&self) -> Option<usize> {
        let mut raw = self.front;
        while raw < self.back {
            let (idx, offset) = (raw / WORD_BITS, raw % WORD_BITS);
            let word = self.words[idx] >> offset;
            if word == 0 {
                // Nothing left in this word, go to the start of the next one.
                raw = (idx + 1) * WORD_BITS;
                continue;
            }
            raw += word.trailing_zeros() as usize;
            return (raw < self.back).then_some(raw);
        }
        None
    }

    /// The highest set bit in `front..back`, if any.
    fn prev_set(&self) -> Option<usize> {
        let mut end = self.back;
        while end > self.front {
            let last = end - 1;
            let (idx, offset) = (last / WORD_BITS, last % WORD_BITS);
            // Keep the bits up to and including `offset`.
            let word = self.words[idx] & (usize::MAX >> (WORD_BITS - 1 - offset));
            if word == 0 {
                end = idx * WORD_BITS;
                continue;
            }
            let raw = idx * WORD_BITS + (WORD_BITS - 1 - word.leading_zeros() as usize);
            return (raw >= self.front).then_some(raw);
        }
        None
    }
}

impl<T, const WORDS: usize> Iterator for Iter<'_, T, WORDS>
where
    T: TryFrom<usize>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while let Some(raw) = self.next_set() {
            self.front = raw + 1;
            if let Ok(value) = T::try_from(raw) {
                return Some(value);
            }
        }
        self.front = self.back;
        None
    }
}

impl<T, const WORDS: usize> DoubleEndedIterator for Iter<'_, T, WORDS>
where
    T: TryFrom<usize>,
{
    fn next_back(&mut self) -> Option<T> {
        while let Some(raw) = self.prev_set() {
            self.back = raw;
            if let Ok(value) = T::try_from(raw) {
                return Some(value);
            }
        }
        self.back = self.front;
        None
    }
}

impl<T, const WORDS: usize> FusedIterator for Iter<'_, T, WORDS> where T: TryFrom<usize> {}

impl<'a, T, const WORDS: usize> IntoIterator for &'a BitSet<T, WORDS>
where
    T: Into<usize> + TryFrom<usize>,
{
    type Item = T;
    type IntoIter = Iter<'a, T, WORDS>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const WORDS: usize> FromIterator<T> for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::empty();
        set.extend(iter);
        set
    }
}

impl<T, const WORDS: usize> Extend<T> for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

// Written out rather than derived: a derive would require `T` to implement
// each trait, although the set stores no `T`.

impl<T, const WORDS: usize> Clone for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const WORDS: usize> Copy for BitSet<T, WORDS> where T: Into<usize> {}

impl<T, const WORDS: usize> Default for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn default() -> Self {
        Self::empty()
    }
}

impl<T, const WORDS: usize> PartialEq for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

impl<T, const WORDS: usize> Eq for BitSet<T, WORDS> where T: Into<usize> {}

impl<T, const WORDS: usize> Hash for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words.hash(state);
    }
}

/// Lists the ids in the set, as `{a, b, ...}`.
impl<T, const WORDS: usize> Debug for BitSet<T, WORDS>
where
    T: Into<usize> + TryFrom<usize> + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// Prints one digit per bit, lowest id first, as `[0110...]`, like
/// [`CPUSet`](crate::arch::generic::cpu::CPUSet) does.
impl<T, const WORDS: usize> Display for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("[")?;
        for word in self.words {
            for offset in 0..WORD_BITS {
                f.write_str(if word & (1 << offset) != 0 { "1" } else { "0" })?;
            }
        }
        f.write_str("]")
    }
}

/// Defines a set operator and its assigning form from the word operation
/// `$op`, which takes the two words of the operands at the same index.
macro_rules! set_operator {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        impl<T, const WORDS: usize> $trait for BitSet<T, WORDS>
        where
            T: Into<usize>,
        {
            type Output = Self;

            fn $method(mut self, rhs: Self) -> Self {
                self.$assign_method(rhs);
                self
            }
        }

        #[doc = $doc]
        impl<T, const WORDS: usize> $trait for &BitSet<T, WORDS>
        where
            T: Into<usize>,
        {
            type Output = BitSet<T, WORDS>;

            fn $method(self, rhs: Self) -> BitSet<T, WORDS> {
                $trait::$method(*self, *rhs)
            }
        }

        #[doc = $doc]
        impl<T, const WORDS: usize> $assign_trait for BitSet<T, WORDS>
        where
            T: Into<usize>,
        {
            fn $assign_method(&mut self, rhs: Self) {
                let op: fn(usize, usize) -> usize = $op;
                for (a, b) in self.words.iter_mut().zip(rhs.words) {
                    *a = op(*a, b);
                }
            }
        }
    };
}

set_operator!(
    BitOr,
    bitor,
    BitOrAssign,
    bitor_assign,
    "Union: the ids in either set.",
    |a, b| a | b
);
set_operator!(
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    "Intersection: the ids in both sets.",
    |a, b| a & b
);
set_operator!(
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    "Symmetric difference: the ids in exactly one of the sets.",
    |a, b| a ^ b
);
set_operator!(
    Sub,
    sub,
    SubAssign,
    sub_assign,
    "Difference: the ids in the left set but not the right one.",
    |a, b| a & !b
);

/// Complement: every bit not in the set, over the whole
/// [`capacity`](BitSet::capacity).
impl<T, const WORDS: usize> Not for BitSet<T, WORDS>
where
    T: Into<usize>,
{
    type Output = Self;

    fn not(mut self) -> Self {
        for word in &mut self.words {
            *word = !*word;
        }
        self
    }
}

/// Complement, see the owned form.
impl<T, const WORDS: usize> Not for &BitSet<T, WORDS>
where
    T: Into<usize>,
{
    type Output = BitSet<T, WORDS>;

    fn not(self) -> BitSet<T, WORDS> {
        !*self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    use std::{format, vec::Vec};

    /// Two words, so that every test also crosses a word boundary.
    type Set = BitSet<usize, 2>;

    #[test]
    fn words_rounds_up() {
        assert_eq!(words(0), 0);
        assert_eq!(words(1), 1);
        assert_eq!(words(WORD_BITS), 1);
        assert_eq!(words(WORD_BITS + 1), 2);
    }

    #[test]
    fn empty_and_all() {
        let empty = Set::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.first(), None);

        let all = Set::all();
        assert!(all.is_full());
        assert_eq!(all.len(), 2 * WORD_BITS);
        assert_eq!(all.capacity(), 2 * WORD_BITS);
        assert_eq!(Set::default(), empty);
    }

    #[test]
    fn insert_remove_toggle_contains() {
        let mut set = Set::empty();
        assert!(set.insert(3));
        assert!(!set.insert(3));
        assert!(set.insert(WORD_BITS + 1));
        assert!(set.contains(3));
        assert!(set.contains(WORD_BITS + 1));
        assert!(!set.contains(4));
        assert_eq!(set.len(), 2);

        assert!(set.remove(3));
        assert!(!set.remove(3));
        assert!(!set.contains(3));

        assert!(set.toggle(7));
        assert!(!set.toggle(7));

        set.clear();
        assert!(set.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_panics() {
        Set::empty().insert(2 * WORD_BITS);
    }

    #[test]
    fn iteration_in_both_directions() {
        let ids = [0, 5, WORD_BITS - 1, WORD_BITS, 2 * WORD_BITS - 1];
        let set: Set = ids.into_iter().collect();

        assert_eq!(set.iter().collect::<Vec<_>>(), ids);
        assert_eq!(
            set.iter().rev().collect::<Vec<_>>(),
            ids.into_iter().rev().collect::<Vec<_>>()
        );
        assert_eq!(set.first(), Some(0));
        assert_eq!(set.last(), Some(2 * WORD_BITS - 1));

        // Meeting in the middle hands every id out exactly once.
        let mut iter = set.iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(2 * WORD_BITS - 1));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next_back(), Some(WORD_BITS));
        assert_eq!(iter.next(), Some(WORD_BITS - 1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iteration_skips_indices_without_an_id() {
        // `u8` names 256 ids, a set of 5 words has room for 320 bits.
        let set = BitSet::<u8, 5>::all();
        assert_eq!(set.iter().count(), 256);
        assert_eq!(set.last(), Some(u8::MAX));
    }

    #[test]
    fn set_operators() {
        let a: Set = [1, 2, WORD_BITS].into_iter().collect();
        let b: Set = [2, 3, WORD_BITS].into_iter().collect();

        assert_eq!((a | b).iter().collect::<Vec<_>>(), [1, 2, 3, WORD_BITS]);
        assert_eq!((a & b).iter().collect::<Vec<_>>(), [2, WORD_BITS]);
        assert_eq!((a ^ b).iter().collect::<Vec<_>>(), [1, 3]);
        assert_eq!((a - b).iter().collect::<Vec<_>>(), [1]);
        assert_eq!(&a | &b, a | b);

        let mut c = a;
        c |= b;
        c -= a;
        assert_eq!(c.iter().collect::<Vec<_>>(), [3]);

        assert_eq!(!Set::empty(), Set::all());
        assert_eq!((!a).len(), 2 * WORD_BITS - 3);
    }

    #[test]
    fn set_relations() {
        let a: Set = [1, 2].into_iter().collect();
        let b: Set = [1, 2, 3].into_iter().collect();
        let c: Set = [4].into_iter().collect();

        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
        assert!(a.is_disjoint(&c));
        assert!(!a.is_disjoint(&b));
        assert!(Set::empty().is_subset(&a));
    }

    #[test]
    fn words_round_trip() {
        let set: Set = [0, WORD_BITS + 2].into_iter().collect();
        assert_eq!(set.as_words(), &[1, 1 << 2]);
        assert_eq!(Set::from_words(*set.as_words()), set);
    }

    #[test]
    fn formatting() {
        let set: BitSet<u8, 1> = [0u8, 2].into_iter().collect();
        assert_eq!(format!("{set:?}"), "{0, 2}");

        let display = format!("{set}");
        assert_eq!(display.len(), WORD_BITS + 2);
        assert!(display.starts_with("[101"));
    }
}
