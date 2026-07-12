//! Half-open interval arithmetic and a binary-tree-ready range type.
//!
//! # Overview
//!
//! A [`Range`] represents the half-open interval `[base, base + length[`.
//! Using half-open intervals means that two adjacent ranges share exactly one
//! boundary value, making adjacency and splitting arithmetic clean and free of
//! off-by-one errors.
//!
//! # Type parameters
//!
//! Every type in this module is generic over two parameters:
//!
//! - `Base`: the scalar type used for addresses or offsets (e.g. `u64`,
//!   `usize`, or a newtype wrapping a raw pointer).
//! - `Length`: the type used for sizes (e.g. `u64`, `usize`).

use crate::utils::allocator::{Allocator, Error as AllocatorError};
use core::cmp::Ordering;
use core::fmt::{Debug, Display, Pointer};
use core::ops::{Add, Sub};

use crate::kernel::locking::{CanAcquire, LockId, PreviousToken};
use crate::utils::rbtree::set::RbTreeSet;

/// A half-open interval `[base, base + length[`.
///
/// The interval includes `base` and excludes `base + length`.  All arithmetic
/// is performed with half-open semantics so that, for example, two adjacent
/// ranges whose combined length equals the sum of their individual lengths have
/// no gap and no overlap between them.
///
/// # Invariants
///
/// The caller is responsible for constructing ranges with a non-zero `length`.
/// A zero-length range is a valid value but will not overlap or be adjacent to
/// any other range, and splitting off a zero-length range leaves the original
/// unchanged.
pub struct Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    base: Base,
    length: Length,
}

// Derived manually so we don't impose extra bounds beyond what the type already
// requires.
impl<Base, Length> Clone for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Base, Length> Copy for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
}

impl<Base, Length> PartialEq for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base && self.length == other.length
    }
}

impl<Base, Length> Eq for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
}

impl<Base, Length> PartialOrd for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.base.partial_cmp(&other.base)
    }
}

impl<Base, Length> Ord for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.cmp(&other.base)
    }
}

/// The result of removing one range from another via [`Range::split_off`].
pub enum SplittedRange<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    /// The subtracted range completely covers the original; no bytes remain.
    None,

    /// The subtracted range does not overlap the original; the original is
    /// returned unchanged.
    Original(Range<Base, Length>),

    /// The subtracted range overlapped one end of the original, leaving a
    /// single contiguous remainder.
    One(Range<Base, Length>),

    /// The subtracted range punched a hole in the middle of the original,
    /// splitting it into two disjoint remainders.  The first element is the
    /// left (lower-address) piece; the second is the right (higher-address)
    /// piece.
    Two(Range<Base, Length>, Range<Base, Length>),
}

impl<Base, Length> Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    /// Creates a new range `[base, base + length[`.
    ///
    /// The constructor is `const` so ranges can be defined in static or
    /// constant contexts.
    ///
    /// # Parameters
    ///
    /// - `base`   — the inclusive lower bound of the range.
    /// - `length` — the size of the range; must be positive for the range to
    ///              contain any elements.
    pub const fn new(base: Base, length: Length) -> Self {
        Self { base, length }
    }

    /// Returns the inclusive lower bound of the range.
    #[inline]
    pub fn base(&self) -> Base {
        self.base
    }

    /// Returns the length (size) of the range.
    #[inline]
    pub fn length(&self) -> Length {
        self.length
    }

    /// Returns the exclusive upper bound of the range (`base + length`).
    #[inline]
    pub fn end(&self) -> Base {
        self.base + self.length
    }

    /// Returns `true` if ranges `a` and `b` share at least one point.
    ///
    /// Two ranges overlap when their intersection is non-empty:
    ///
    /// ```text
    /// max(a.base, b.base) < min(a.end, b.end)
    /// ```
    ///
    /// Adjacent ranges (where one ends exactly where the other begins) are
    /// **not** considered overlapping.
    ///
    /// Both arguments are taken by value because [`Range`] implements [`Copy`].
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Range::new(0u64, 10u64);  // [0, 10[
    /// let b = Range::new(5u64, 10u64);  // [5, 15[
    /// assert!(Range::overlap(&a, &b));    // share [5, 10[
    ///
    /// let c = Range::new(10u64, 5u64);  // [10, 15[
    /// assert!(!Range::overlap(&a, &c));   // adjacent, not overlapping
    /// ```
    #[inline]
    pub fn overlap(a: &Self, b: &Self) -> bool {
        let a_end = a.base + a.length;
        let b_end = b.base + b.length;
        Base::max(a.base, b.base) < Base::min(a_end, b_end)
    }

    /// Returns `true` if ranges `a` and `b` share exactly one boundary point
    /// without overlapping.
    ///
    /// Two ranges are adjacent when one ends exactly where the other begins:
    ///
    /// ```text
    /// max(a.base, b.base) == min(a.end, b.end)
    /// ```
    ///
    /// This is useful when coalescing a sequence of free ranges: two free
    /// ranges that are adjacent can be merged into one via [`try_merge`].
    ///
    /// Both arguments are taken by value because [`Range`] implements [`Copy`].
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Range::new(0u64, 10u64);    // [0,  10[
    /// let b = Range::new(10u64, 10u64);   // [10, 20[
    /// assert!(Range::adjecent(&a, &b));
    ///
    /// let c = Range::new(11u64, 10u64);   // [11, 21[ — gap, not adjacent
    /// assert!(!Range::adjecent(&a, &c));
    /// ```
    ///
    /// [`try_merge`]: Range::try_merge
    #[inline]
    pub fn adjecent(a: &Self, b: &Self) -> bool {
        let a_end = a.base + a.length;
        let b_end = b.base + b.length;
        Base::max(a.base, b.base) == Base::min(a_end, b_end)
    }

    /// Subtracts `other` from `original`, returning the portion(s) of
    /// `original` that are not covered by `other`.
    ///
    /// The result depends on the geometric relationship between the two ranges:
    ///
    /// ```text
    /// No overlap:
    ///   original: [==========]
    ///   other:                   [====]
    ///   result:   Original([==========])
    ///
    /// Complete cover (other ⊇ original):
    ///   original: [==========]
    ///   other:  [==============]
    ///   result:   None
    ///
    /// Left trim (other overlaps the start of original):
    ///   original: [==========]
    ///   other:  [=====]
    ///   result:   One(      [====])
    ///
    /// Right trim (other overlaps the end of original):
    ///   original: [==========]
    ///   other:          [=====]
    ///   result:   One([=====]      )
    ///
    /// Middle hole (other is strictly inside original):
    ///   original: [==========]
    ///   other:       [====]
    ///   result:   Two([==], [===])
    /// ```
    ///
    /// All boundary comparisons use half-open semantics: `original` ends at
    /// `original.base + original.length` (exclusive), and so does `other`.
    /// A range that begins exactly where `original` ends does not overlap and
    /// will therefore return [`SplittedRange::Original`].
    ///
    /// Both arguments are taken by value because [`Range`] implements [`Copy`].
    ///
    /// # Parameters
    ///
    /// - `original` — the range to subtract from.
    /// - `other`    — the range to remove.
    ///
    /// # Returns
    ///
    /// A [`SplittedRange`] describing the zero, one, or two pieces of
    /// `original` that survive after removing `other`.
    pub fn split_off(original: Self, other: Self) -> SplittedRange<Base, Length> {
        let o_start = original.base;
        let o_end = original.base + original.length;
        let x_start = other.base;
        let x_end = other.base + other.length;

        // Clamp the cut region to the actual overlap with `original`.
        let cut_start = x_start.max(o_start);
        let cut_end = x_end.min(o_end);

        // No overlap at all — original is unchanged.
        if cut_start >= cut_end {
            return SplittedRange::Original(original);
        }

        // `other` completely covers `original` — nothing survives.
        if cut_start <= o_start && cut_end >= o_end {
            return SplittedRange::None;
        }

        let has_left = cut_start > o_start;
        let has_right = cut_end < o_end;

        match (has_left, has_right) {
            (false, true) => {
                // `other` consumed the left side; right piece survives.
                SplittedRange::One(Range::new(cut_end, o_end - cut_end))
            }
            (true, false) => {
                // `other` consumed the right side; left piece survives.
                SplittedRange::One(Range::new(o_start, cut_start - o_start))
            }
            (true, true) => {
                // `other` punched a hole in the middle; two pieces survive.
                let left = Range::new(o_start, cut_start - o_start);
                let right = Range::new(cut_end, o_end - cut_end);
                SplittedRange::Two(left, right)
            }
            // (false, false) is the complete-cover case handled above.
            (false, false) => unreachable!(),
        }
    }

    /// Attempts to merge `self` and `other` into a single contiguous range.
    ///
    /// Two ranges can be merged when there is no gap between them — that is,
    /// when they are either **adjacent** (one ends exactly where the other
    /// begins) or **overlapping** (they share at least one point).
    ///
    /// ```text
    /// Mergeable — adjacent:
    ///   self:  [======]
    ///   other:         [======]
    ///   result:[=============]
    ///
    /// Mergeable — overlapping:
    ///   self:  [======]
    ///   other:     [======]
    ///   result:[=========]
    ///
    /// Mergeable — one contains the other:
    ///   self:  [==========]
    ///   other:    [====]
    ///   result:[==========]
    ///
    /// Not mergeable — gap:
    ///   self:  [======]
    ///   other:            [======]
    ///   result: Err(self)
    /// ```
    ///
    /// # Returns
    ///
    /// - `Ok(merged)` — the smallest range that covers both `self` and
    ///   `other`, with:
    ///   - `merged.base   = min(self.base, other.base)`
    ///   - `merged.end    = max(self.end, other.end)`
    ///   - `merged.length = merged.end - merged.base`
    /// - `Err(self)` — the ranges have a gap between them and cannot be
    ///   merged. `self` is returned unchanged so the caller can recover it
    ///   without an extra copy. `other` is dropped.
    pub fn try_merge(self, other: Self) -> Result<Self, Self> {
        let self_end = self.base + self.length;
        let other_end = other.base + other.length;

        // The overlap/adjacency window is [max(bases), min(ends)].
        // A strict gap exists when max(bases) > min(ends).
        if self.base.max(other.base) > self_end.min(other_end) {
            return Err(self);
        }

        let merged_base = self.base.min(other.base);
        let merged_end = self_end.max(other_end);
        Ok(Range::new(merged_base, merged_end - merged_base))
    }
}

impl<Base, Length> Debug for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>
        + Debug,
    Length: Clone + Copy + PartialEq,
{
    /// Formats the range as `[start, end[` using `{:?}` for the endpoints.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{:?}, {:?}[", self.base, self.base + self.length)
    }
}

impl<Base, Length> Display for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>
        + Display,
    Length: Clone + Copy + PartialEq,
{
    /// Formats the range as `[start, end[` using `{}` for the endpoints.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{}, {}[", self.base, self.base + self.length)
    }
}

impl<Base, Length> Pointer for Range<Base, Length>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>
        + Pointer,
    Length: Clone + Copy + PartialEq,
{
    /// Formats the range as `[start, end[` using `{:p}` for the endpoints,
    /// suitable for pointer-valued base types.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{:p}, {:p}[", self.base, self.base + self.length)
    }
}

pub struct RangeTree<Base, Length, ID: LockId, A: Allocator<ID>>(
    RbTreeSet<Range<Base, Length>, ID, A>,
)
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq;

impl<Base, Length, ID: LockId, A: Allocator<ID>> RangeTree<Base, Length, ID, A>
where
    Base: PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Clone
        + Copy
        + Add<Length, Output = Base>
        + Sub<Base, Output = Length>,
    Length: Clone + Copy + PartialEq,
{
    pub const fn new_in(alloc: A) -> Self {
        Self(RbTreeSet::new_in(alloc))
    }

    /// Checks if [`RangeTree`] contains no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Adds a [`Range`] to the [`RangeTree`].
    pub fn add<Token>(
        &mut self,
        range: Range<Base, Length>,
        token: Token,
    ) -> Result<Token, (AllocatorError, Token)>
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        let mut token = token;
        let mut range = range;

        loop {
            match self.0.find(|other| {
                if Range::overlap(&range, other) {
                    return Ordering::Equal;
                }
                range.base.cmp(&other.base)
            }) {
                Some(other) => {
                    let success;
                    let other = *other;
                    (success, token) = self.0.remove(&other, token);
                    assert!(success);

                    range = match Range::try_merge(range, other) {
                        Ok(range) => range,
                        Err(_) => panic!("Must never happen..."),
                    };
                }
                None => {
                    // Early out: No overlapping region found
                    let failure;
                    (failure, token) = self.0.try_insert(range, token)?;
                    assert!(!failure);
                    return Ok(token);
                }
            };
        }
    }

    /// Adds a [`Range`] to the [`RangeTree`].
    pub fn remove<Token>(
        &mut self,
        range: Range<Base, Length>,
        token: Token,
    ) -> Result<Token, (AllocatorError, Token)>
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        let mut token = token;

        match self.0.find(|other| {
            if Range::overlap(&range, other) {
                return Ordering::Equal;
            }
            range.base.cmp(&other.base)
        }) {
            Some(other) => {
                let success;
                let other = *other;
                (success, token) = self.0.remove(&other, token);
                assert!(success);

                // FIX: split_off(other, range) — subtract the query FROM the
                // stored entry to obtain the surviving pieces.
                match Range::split_off(other, range) {
                    SplittedRange::None => {
                        // The query completely covered the stored entry;
                        // nothing survives.
                        Ok(token)
                    }
                    SplittedRange::Original(_) => {
                        // The query had no overlap with the stored entry.
                        // This cannot happen because we found the entry via
                        // the overlap predicate above.
                        panic!("Must never happen...");
                    }
                    SplittedRange::One(range) => {
                        let (present, token) = self.0.try_insert(range, token)?;
                        assert!(!present);
                        Ok(token)
                    }
                    SplittedRange::Two(range_0, range_1) => {
                        let (present, token) = self.0.try_insert(range_0, token)?;
                        assert!(!present);

                        let (present, token) = self.0.try_insert(range_1, token)?;
                        assert!(!present);

                        Ok(token)
                    }
                }
            }
            None => {
                // No stored entry overlaps the query — nothing to do.
                Ok(token)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        kernel::locking::{EpilogueLevel, MemoryManagementLevelID, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type R = Range<u64, u64>;

    fn r(base: u64, length: u64) -> R {
        R::new(base, length)
    }

    fn expect_original(s: SplittedRange<u64, u64>) -> R {
        match s {
            SplittedRange::Original(r) => r,
            _ => panic!("expected Original, got a different variant"),
        }
    }
    fn expect_none(s: SplittedRange<u64, u64>) {
        assert!(matches!(s, SplittedRange::None), "expected None");
    }
    fn expect_one(s: SplittedRange<u64, u64>) -> R {
        match s {
            SplittedRange::One(r) => r,
            _ => panic!("expected One, got a different variant"),
        }
    }
    fn expect_two(s: SplittedRange<u64, u64>) -> (R, R) {
        match s {
            SplittedRange::Two(a, b) => (a, b),
            _ => panic!("expected Two, got a different variant"),
        }
    }

    type RT = RangeTree<u64, u64, MemoryManagementLevelID, HeapAllocator>;

    fn new_tree() -> RT {
        RangeTree::new_in(HeapAllocator)
    }

    /// Collect the stored ranges in sorted order as `(base, length)` pairs.
    /// Used for concise assertions throughout the test suite.
    fn entries(t: &RT) -> std::vec::Vec<(u64, u64)> {
        t.0.iter().map(|r| (r.base(), r.length())).collect()
    }

    macro_rules! add {
        ($tree:expr, $base:expr, $len:expr, $tok:expr) => {
            $tree
                .add(Range::new($base as u64, $len as u64), $tok)
                .unwrap_or_else(|_| panic!("add() returned AllocError"))
        };
    }

    macro_rules! rm {
        ($tree:expr, $base:expr, $len:expr, $tok:expr) => {
            $tree
                .remove(Range::new($base as u64, $len as u64), $tok)
                .unwrap_or_else(|_| panic!("remove() returned AllocError"))
        };
    }

    #[test]
    fn accessors() {
        let range = r(100, 50);
        assert_eq!(range.base(), 100);
        assert_eq!(range.length(), 50);
        assert_eq!(range.end(), 150);
    }

    #[test]
    fn overlap_disjoint_left() {
        assert!(!R::overlap(&r(0, 50), &r(100, 100)));
    }

    #[test]
    fn overlap_disjoint_right() {
        assert!(!R::overlap(&r(200, 50), &r(100, 100)));
    }

    #[test]
    fn overlap_adjacent_left() {
        // Adjacent — NOT overlapping.
        assert!(!R::overlap(&r(50, 50), &r(100, 100)));
    }

    #[test]
    fn overlap_adjacent_right() {
        assert!(!R::overlap(&r(200, 50), &r(100, 100)));
    }

    #[test]
    fn overlap_partial_left() {
        assert!(R::overlap(&r(50, 100), &r(100, 100)));
    }

    #[test]
    fn overlap_partial_right() {
        assert!(R::overlap(&r(150, 100), &r(100, 100)));
    }

    #[test]
    fn overlap_exact_match() {
        assert!(R::overlap(&r(100, 100), &r(100, 100)));
    }

    #[test]
    fn overlap_contained() {
        assert!(R::overlap(&r(110, 80), &r(100, 100)));
    }

    #[test]
    fn overlap_contains() {
        assert!(R::overlap(&r(100, 100), &r(110, 80)));
    }

    #[test]
    fn overlap_single_point_boundary() {
        // [99, 100[ and [100, 101[ share only the boundary point — NOT overlapping.
        assert!(!R::overlap(&r(99, 1), &r(100, 1)));
    }

    #[test]
    fn adjacent_left_touches_right() {
        assert!(R::adjecent(&r(50, 50), &r(100, 100)));
    }

    #[test]
    fn adjacent_right_touches_left() {
        assert!(R::adjecent(&r(200, 50), &r(100, 100)));
    }

    #[test]
    fn adjacent_with_gap() {
        assert!(!R::adjecent(&r(50, 49), &r(100, 100)));
    }

    #[test]
    fn adjacent_overlapping_not_adjacent() {
        assert!(!R::adjecent(&r(50, 100), &r(100, 100)));
    }

    #[test]
    fn adjacent_exact_same_range() {
        assert!(!R::adjecent(&r(100, 100), &r(100, 100)));
    }

    // -------------------------------------------------------------------------
    // split_off
    // -------------------------------------------------------------------------

    #[test]
    fn split_off_no_overlap_left() {
        let orig = r(100, 100);
        let got = expect_original(R::split_off(orig, r(0, 50)));
        assert_eq!(got, orig);
    }

    #[test]
    fn split_off_no_overlap_right() {
        let orig = r(100, 100);
        let got = expect_original(R::split_off(orig, r(200, 50)));
        assert_eq!(got, orig);
    }

    #[test]
    fn split_off_adjacent_left_is_original() {
        let orig = r(100, 100);
        let got = expect_original(R::split_off(orig, r(50, 50)));
        assert_eq!(got, orig);
    }

    #[test]
    fn split_off_adjacent_right_is_original() {
        let orig = r(100, 100);
        let got = expect_original(R::split_off(orig, r(200, 50)));
        assert_eq!(got, orig);
    }

    #[test]
    fn split_off_exact_cover() {
        expect_none(R::split_off(r(100, 100), r(100, 100)));
    }

    #[test]
    fn split_off_over_cover() {
        expect_none(R::split_off(r(100, 100), r(50, 200)));
    }

    #[test]
    fn split_off_cover_left_flush() {
        expect_none(R::split_off(r(100, 100), r(100, 200)));
    }

    #[test]
    fn split_off_cover_right_flush() {
        expect_none(R::split_off(r(100, 100), r(0, 200)));
    }

    #[test]
    fn split_off_left_trim_exact() {
        let got = expect_one(R::split_off(r(100, 100), r(100, 50)));
        assert_eq!(got, r(150, 50));
    }

    #[test]
    fn split_off_left_trim_overrun() {
        let got = expect_one(R::split_off(r(100, 100), r(80, 70)));
        assert_eq!(got, r(150, 50));
    }

    #[test]
    fn split_off_right_trim_exact() {
        let got = expect_one(R::split_off(r(100, 100), r(150, 50)));
        assert_eq!(got, r(100, 50));
    }

    #[test]
    fn split_off_right_trim_overrun() {
        let got = expect_one(R::split_off(r(100, 100), r(150, 80)));
        assert_eq!(got, r(100, 50));
    }

    #[test]
    fn split_off_left_trim_single_unit() {
        let got = expect_one(R::split_off(r(100, 100), r(100, 1)));
        assert_eq!(got, r(101, 99));
    }

    #[test]
    fn split_off_right_trim_single_unit() {
        let got = expect_one(R::split_off(r(100, 100), r(199, 1)));
        assert_eq!(got, r(100, 99));
    }

    #[test]
    fn split_off_middle() {
        let (left, right) = expect_two(R::split_off(r(100, 100), r(130, 40)));
        assert_eq!(left, r(100, 30));
        assert_eq!(right, r(170, 30));
    }

    #[test]
    fn split_off_middle_1_wide_pieces() {
        let (left, right) = expect_two(R::split_off(r(100, 100), r(101, 98)));
        assert_eq!(left, r(100, 1));
        assert_eq!(right, r(199, 1));
    }

    #[test]
    fn split_off_middle_almost_full() {
        let (left, right) = expect_two(R::split_off(r(0, 1000), r(1, 998)));
        assert_eq!(left, r(0, 1));
        assert_eq!(right, r(999, 1));
    }

    // -------------------------------------------------------------------------
    // try_merge
    // -------------------------------------------------------------------------

    #[test]
    fn merge_adjacent_right() {
        // [100, 200[ + [200, 300[ → [100, 300[
        let merged = r(100, 100).try_merge(r(200, 100)).unwrap();
        assert_eq!(merged, r(100, 200));
    }

    #[test]
    fn merge_adjacent_left() {
        // [200, 300[ + [100, 200[ → [100, 300[
        let merged = r(200, 100).try_merge(r(100, 100)).unwrap();
        assert_eq!(merged, r(100, 200));
    }

    #[test]
    fn merge_overlapping() {
        // [100, 200[ + [150, 250[ → [100, 250[
        let merged = r(100, 100).try_merge(r(150, 100)).unwrap();
        assert_eq!(merged, r(100, 150));
    }

    #[test]
    fn merge_overlapping_reverse() {
        // [150, 250[ + [100, 200[ → [100, 250[
        let merged = r(150, 100).try_merge(r(100, 100)).unwrap();
        assert_eq!(merged, r(100, 150));
    }

    #[test]
    fn merge_identical() {
        // [100, 200[ + [100, 200[ → [100, 200[
        let merged = r(100, 100).try_merge(r(100, 100)).unwrap();
        assert_eq!(merged, r(100, 100));
    }

    #[test]
    fn merge_self_contains_other() {
        // [100, 200[ fully contains [120, 150[ → [100, 200[
        let merged = r(100, 100).try_merge(r(120, 30)).unwrap();
        assert_eq!(merged, r(100, 100));
    }

    #[test]
    fn merge_other_contains_self() {
        // [120, 150[ is inside [100, 200[ → [100, 200[
        let merged = r(120, 30).try_merge(r(100, 100)).unwrap();
        assert_eq!(merged, r(100, 100));
    }

    #[test]
    fn merge_gap_returns_err_and_recovers_self() {
        // [100, 200[ and [201, 300[ — gap of 1 byte.
        let original = r(100, 100);
        let err = original.try_merge(r(201, 99));
        assert!(err.is_err(), "gap must return Err");
        // Err recovers self unchanged so the caller can keep using it.
        assert_eq!(err.unwrap_err(), original);
    }

    #[test]
    fn merge_gap_left_returns_err() {
        // [200, 300[ and [100, 198[ — gap of 2 bytes.
        let err = r(200, 100).try_merge(r(100, 98));
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), r(200, 100));
    }

    #[test]
    fn merge_single_unit_adjacent() {
        // [99, 100[ + [100, 101[ → [99, 101[
        let merged = r(99, 1).try_merge(r(100, 1)).unwrap();
        assert_eq!(merged, r(99, 2));
    }

    #[test]
    fn merge_single_unit_gap() {
        // [99, 100[ and [101, 102[ — gap of 1 unit.
        let err = r(99, 1).try_merge(r(101, 1));
        assert!(err.is_err());
    }

    // -------------------------------------------------------------------------
    // add()
    // -------------------------------------------------------------------------

    /// Inserting into an empty tree stores the range verbatim.
    #[test]
    fn add_into_empty_tree() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        let token = add!(t, 100, 100, token); // [100, 200[

        assert_eq!(entries(&t), [(100, 100)]);
        let token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Two non-overlapping, non-adjacent ranges are stored independently.
    #[test]
    fn add_two_disjoint_ranges() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = add!(t, 300, 100, token); // [300, 400[

        assert_eq!(entries(&t), [(100, 100), (300, 100)]);
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// `overlap()` uses a strict `<`, so adjacent ranges (which share only a
    /// single boundary point) are stored as two separate entries rather than
    /// being merged.
    #[test]
    fn add_adjacent_ranges_stay_separate() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = add!(t, 200, 100, token); // [200, 300[ — touches at 200, no overlap

        assert_eq!(entries(&t), [(100, 100), (200, 100)]);
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A new range that partially overlaps the right side of a stored entry
    /// causes the two to be merged.
    #[test]
    fn add_right_overlap_merges() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = add!(t, 150, 150, token); // [150, 300[ — overlaps right part

        assert_eq!(entries(&t), [(100, 200)]); // [100, 300[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A new range that partially overlaps the left side of a stored entry
    /// causes the two to be merged.
    #[test]
    fn add_left_overlap_merges() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 200, 100, token); // [200, 300[
        token = add!(t, 100, 150, token); // [100, 250[ — overlaps left part

        assert_eq!(entries(&t), [(100, 200)]); // [100, 300[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Adding a range that lies entirely inside an existing stored entry is a
    /// no-op: the stored entry already covers it.
    #[test]
    fn add_inside_existing_is_absorbed() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 200, token); // [100, 300[
        token = add!(t, 150, 50, token); // [150, 200[ — entirely inside

        assert_eq!(entries(&t), [(100, 200)]);
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A new range that completely covers an existing entry absorbs it.
    #[test]
    fn add_completely_covers_existing() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 150, 50, token); // [150, 200[
        token = add!(t, 100, 200, token); // [100, 300[ — covers existing

        assert_eq!(entries(&t), [(100, 200)]); // [100, 300[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Adding a range that overlaps two previously separate entries merges all
    /// three into one.
    #[test]
    fn add_bridges_two_existing_entries() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = add!(t, 300, 100, token); // [300, 400[
        token = add!(t, 150, 200, token); // [150, 350[ — overlaps both

        assert_eq!(entries(&t), [(100, 300)]); // [100, 400[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Adding a single range that overlaps three separate entries merges all
    /// of them in one call.
    #[test]
    fn add_bridges_three_existing_entries() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 0, 100, token); // [0,   100[
        token = add!(t, 200, 100, token); // [200, 300[
        token = add!(t, 400, 100, token); // [400, 500[
        token = add!(t, 50, 400, token); // [50,  450[ — overlaps all three

        assert_eq!(entries(&t), [(0, 500)]); // [0, 500[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    // -----------------------------------------------------------------------=-
    // remove()
    // -----------------------------------------------------------------------=-

    /// Removing a range with no overlap against any stored entry is a no-op.
    #[test]
    fn remove_no_match_is_noop() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = rm!(t, 300, 50, token); // [300, 350[ — no overlap

        assert_eq!(entries(&t), [(100, 100)]);
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Removing from an empty tree does nothing.
    #[test]
    fn remove_from_empty_tree_is_noop() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        let token = rm!(t, 100, 100, token);

        assert!(t.is_empty());
        drop(t);
        level.leave(token);
    }

    /// A query adjacent to a stored entry (but not overlapping) leaves it
    /// untouched.
    #[test]
    fn remove_adjacent_query_is_noop() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = rm!(t, 200, 50, token); // [200, 250[ — adjacent, not overlapping

        assert_eq!(entries(&t), [(100, 100)]);
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Removing a range that exactly matches a stored entry empties the tree.
    #[test]
    fn remove_exact_match_removes_entry() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token);
        token = rm!(t, 100, 100, token);

        assert!(t.is_empty());
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A query that completely covers a stored entry removes it entirely.
    #[test]
    fn remove_query_covers_stored_entry() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = rm!(t, 50, 200, token); // [50, 250[ — covers entirely

        assert!(t.is_empty());
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A query that overlaps only the left part of a stored entry leaves the
    /// right remainder.
    #[test]
    fn remove_trims_left_end() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = rm!(t, 100, 50, token); // remove [100, 150[

        assert_eq!(entries(&t), [(150, 50)]); // [150, 200[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A query that overlaps only the right part of a stored entry leaves the
    /// left remainder.
    #[test]
    fn remove_trims_right_end() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = rm!(t, 150, 50, token); // remove [150, 200[

        assert_eq!(entries(&t), [(100, 50)]); // [100, 150[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A query strictly inside a stored entry splits it into two remainders.
    #[test]
    fn remove_punches_hole_in_middle() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = rm!(t, 130, 40, token); // remove [130, 170[

        assert_eq!(entries(&t), [(100, 30), (170, 30)]); // [100,130[ + [170,200[
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// A removal only affects the entry that overlaps the query; all other
    /// stored entries are untouched.
    #[test]
    fn remove_leaves_other_entries_intact() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token); // [100, 200[
        token = add!(t, 300, 100, token); // [300, 400[
        token = rm!(t, 130, 40, token); // only overlaps [100, 200[

        assert_eq!(entries(&t), [(100, 30), (170, 30), (300, 100)]);
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    // -----------------------------------------------------------------------=-
    // round-trips
    // -----------------------------------------------------------------------=-

    /// add followed by remove of the exact same range restores the empty state.
    #[test]
    fn add_then_remove_exact_roundtrip() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 100, 100, token);
        token = rm!(t, 100, 100, token);

        assert!(t.is_empty());
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }

    /// Removing a large range in three non-overlapping slices ultimately
    /// empties the tree.
    #[test]
    fn remove_all_in_three_slices() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut t = new_tree();

        token = add!(t, 0, 300, token); // [0, 300[
        token = rm!(t, 0, 100, token); // remove [0,   100[
        token = rm!(t, 100, 100, token); // remove [100, 200[
        token = rm!(t, 200, 100, token); // remove [200, 300[

        assert!(t.is_empty());
        token = t.0.clear(token);
        drop(t);
        level.leave(token);
    }
}
