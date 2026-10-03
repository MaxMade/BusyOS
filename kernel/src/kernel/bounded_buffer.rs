//! A bounded ring buffer for a single owner, with a choice of what a push
//! into a full buffer does.
//!
//! [`BoundedBuffer`] is the counterpart of [`SPSC`](crate::kernel::spsc::SPSC)
//! and [`MPSC`](crate::kernel::mpsc::MPSC) for data that is not shared: it
//! takes `&mut self` for every change, so it needs no atomics and no `unsafe`
//! at the call site, and pushing and popping are plain loads and stores.
//! Whoever shares one has to lock it, or keep it core-local.
//!
//! # Modes
//!
//! What happens to a push into a full buffer is part of the type, chosen by
//! the `M` parameter:
//!
//! - [`Dropping`]: the new value is refused and handed back, the buffer keeps
//!   what it has. [`push`](BoundedBuffer::push) returns `Result<(), T>`, as
//!   [`SPSC::try_push`](crate::kernel::spsc::SPSC::try_push) does.
//! - [`Overwriting`]: the oldest value is evicted to make room and handed
//!   back, so the buffer always keeps the newest `N` values.
//!   [`push`](BoundedBuffer::push) returns `Option<T>`.
//!
//! Either way the displaced value goes back to the caller rather than being
//! dropped here, which matters for a value whose `Drop` needs a token.
//!
//! # Sizing
//!
//! The ring is an array of `N` slots inside the buffer, so building one
//! allocates nothing and needs no token, and a buffer may be a `static` or a
//! core-local variable. Unlike the lock-free queues, `N` need not be a power of
//! two, only at least one.

use core::{
    fmt::{Debug, Formatter, Result as FmtResult},
    iter::FusedIterator,
    marker::PhantomData,
    mem::MaybeUninit,
};

mod sealed {
    pub trait Sealed {}
}

/// What a push into a full [`BoundedBuffer`] does, see the
/// [module documentation](self#modes).
///
/// Sealed: [`Dropping`] and [`Overwriting`] are the only modes.
pub trait Mode: sealed::Sealed {
    /// What [`push`](BoundedBuffer::push) returns.
    type Output<T>;

    /// Appends `value` to `buffer`, which may be full.
    fn push<T, const N: usize>(buffer: &mut BoundedBuffer<T, N, Self>, value: T) -> Self::Output<T>
    where
        Self: Sized;
}

/// A full buffer refuses a new value, see the [module documentation](self#modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dropping {}

/// A full buffer evicts its oldest value, see the
/// [module documentation](self#modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Overwriting {}

impl sealed::Sealed for Dropping {}
impl sealed::Sealed for Overwriting {}

impl Mode for Dropping {
    /// `Err` with the value itself if the buffer is full.
    type Output<T> = Result<(), T>;

    fn push<T, const N: usize>(buffer: &mut BoundedBuffer<T, N, Self>, value: T) -> Result<(), T> {
        if buffer.is_full() {
            return Err(value);
        }

        buffer.push_unchecked(value);
        Ok(())
    }
}

impl Mode for Overwriting {
    /// The oldest value, if the buffer was full and it had to make room.
    type Output<T> = Option<T>;

    fn push<T, const N: usize>(buffer: &mut BoundedBuffer<T, N, Self>, value: T) -> Option<T> {
        let evicted = if buffer.is_full() { buffer.pop() } else { None };

        buffer.push_unchecked(value);
        evicted
    }
}

/// A ring buffer of up to `N` values, whose behaviour when full is set by
/// `M`.
///
/// See the [module documentation](self) for the modes and how to size it.
pub struct BoundedBuffer<T, const N: usize, M: Mode> {
    /// The ring. The `len` slots starting at `head`, wrapping around, hold
    /// values, the others are uninitialised.
    slots: [MaybeUninit<T>; N],

    /// Index of the oldest value, the one [`pop`](Self::pop) takes next.
    head: usize,

    /// Number of values in the buffer.
    len: usize,

    _mode: PhantomData<M>,
}

impl<T, const N: usize, M: Mode> BoundedBuffer<T, N, M> {
    /// Rejects a buffer with no room, in which a push could do nothing useful.
    const NON_EMPTY: () = assert!(N >= 1, "BoundedBuffer capacity must be at least one");

    /// Creates an empty buffer.
    ///
    /// Allocates nothing and may be used to build a `static`.
    pub const fn new() -> Self {
        // Evaluates the assertion, so that a bad `N` fails here rather than on
        // the first push.
        let () = Self::NON_EMPTY;

        Self {
            slots: [const { MaybeUninit::uninit() }; N],
            head: 0,
            len: 0,
            _mode: PhantomData,
        }
    }

    /// The number of values the buffer can hold at once.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// The number of values in the buffer.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no values.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer holds [`capacity`](Self::capacity) values, so that
    /// the next push refuses or evicts one.
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Appends `value`, doing what the mode says if the buffer is full.
    ///
    /// Returns `Result<(), T>` for [`Dropping`] and `Option<T>` for
    /// [`Overwriting`], see the [module documentation](self#modes).
    pub fn push(&mut self, value: T) -> M::Output<T> {
        M::push(self, value)
    }

    /// Takes the oldest value out of the buffer, or returns [`None`] if it is
    /// empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // SAFETY: the buffer is not empty, so the slot at `head` holds a
        // value. Advancing `head` below makes it uninitialised again, so the
        // value is moved out exactly once.
        let value = unsafe { self.slots[self.head].assume_init_read() };

        self.head = Self::wrap(self.head + 1);
        self.len -= 1;

        Some(value)
    }

    /// The oldest value, the one [`pop`](Self::pop) would take, without
    /// taking it.
    pub fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// The oldest value, mutably.
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// The newest value, the one pushed last.
    pub fn back(&self) -> Option<&T> {
        self.len.checked_sub(1).and_then(|last| self.get(last))
    }

    /// The newest value, mutably.
    pub fn back_mut(&mut self) -> Option<&mut T> {
        self.len.checked_sub(1).and_then(|last| self.get_mut(last))
    }

    /// The value `index` places behind the oldest one, or [`None`] if there
    /// are not that many.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        // SAFETY: `index < len`, so the slot holds a value.
        Some(unsafe { self.slots[Self::wrap(self.head + index)].assume_init_ref() })
    }

    /// The value `index` places behind the oldest one, mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }

        // SAFETY: `index < len`, so the slot holds a value.
        Some(unsafe { self.slots[Self::wrap(self.head + index)].assume_init_mut() })
    }

    /// Walks the values from the oldest to the newest, without taking them.
    pub fn iter(&self) -> Iter<'_, T, N, M> {
        Iter {
            buffer: self,
            front: 0,
            back: self.len,
        }
    }

    /// Drops every value in the buffer.
    ///
    /// A value whose `Drop` needs more than it can be given here, such as the
    /// last [`Arc`](crate::kernel::arc::Arc) on a value, has to be popped and
    /// handed back properly instead.
    pub fn clear(&mut self) {
        while let Some(value) = self.pop() {
            drop(value);
        }
    }

    /// Appends `value` behind the newest one.
    ///
    /// The buffer must not be full, which the modes check before calling.
    fn push_unchecked(&mut self, value: T) {
        debug_assert!(!self.is_full());

        let tail = Self::wrap(self.head + self.len);
        self.slots[tail].write(value);
        self.len += 1;
    }

    /// Maps a position past the end of the array back to its start.
    ///
    /// The positions handed in are below `2 * N`, so one subtraction is
    /// enough and no division is needed.
    const fn wrap(position: usize) -> usize {
        if position >= N {
            position - N
        } else {
            position
        }
    }
}

impl<T, const N: usize> BoundedBuffer<T, N, Dropping> {
    /// Appends `value`, or hands it back if the buffer is full.
    ///
    /// The same as [`push`](Self::push) in this mode, under the name
    /// [`SPSC`](crate::kernel::spsc::SPSC) and
    /// [`MPSC`](crate::kernel::mpsc::MPSC) use.
    ///
    /// # Errors
    ///
    /// `value` itself, if the buffer is full. The buffer is unchanged in that
    /// case.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        self.push(value)
    }
}

impl<T, const N: usize, M: Mode> Default for BoundedBuffer<T, N, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize, M: Mode> Drop for BoundedBuffer<T, N, M> {
    /// Drops every value still in the buffer, see [`clear`](Self::clear).
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Clone, const N: usize, M: Mode> Clone for BoundedBuffer<T, N, M> {
    fn clone(&self) -> Self {
        let mut clone = Self::new();
        for value in self {
            // The clone has the same capacity, so it never fills up here.
            clone.push_unchecked(value.clone());
        }
        clone
    }
}

/// Lists the values from the oldest to the newest.
impl<T: Debug, const N: usize, M: Mode> Debug for BoundedBuffer<T, N, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// An iterator over the values in a [`BoundedBuffer`], see
/// [`BoundedBuffer::iter`].
pub struct Iter<'a, T, const N: usize, M: Mode> {
    buffer: &'a BoundedBuffer<T, N, M>,

    /// Index, from the oldest value, of the next value from the front.
    front: usize,

    /// One past the index of the next value from the back.
    back: usize,
}

impl<'a, T, const N: usize, M: Mode> Iterator for Iter<'a, T, N, M> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.front == self.back {
            return None;
        }

        let value = self.buffer.get(self.front);
        self.front += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl<'a, T, const N: usize, M: Mode> DoubleEndedIterator for Iter<'a, T, N, M> {
    fn next_back(&mut self) -> Option<&'a T> {
        if self.front == self.back {
            return None;
        }

        self.back -= 1;
        self.buffer.get(self.back)
    }
}

impl<T, const N: usize, M: Mode> ExactSizeIterator for Iter<'_, T, N, M> {}

impl<T, const N: usize, M: Mode> FusedIterator for Iter<'_, T, N, M> {}

impl<'a, T, const N: usize, M: Mode> IntoIterator for &'a BoundedBuffer<T, N, M> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    use std::{format, rc::Rc, vec::Vec};

    #[test]
    fn empty_buffer() {
        let mut buffer = BoundedBuffer::<u32, 3, Dropping>::new();
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 3);
        assert_eq!(buffer.pop(), None);
        assert_eq!(buffer.front(), None);
        assert_eq!(buffer.back(), None);
    }

    #[test]
    fn values_come_out_in_order() {
        let mut buffer = BoundedBuffer::<u32, 3, Dropping>::new();
        for i in 0..3 {
            assert_eq!(buffer.try_push(i), Ok(()));
        }
        assert!(buffer.is_full());
        for i in 0..3 {
            assert_eq!(buffer.pop(), Some(i));
        }
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn dropping_refuses_when_full() {
        let mut buffer = BoundedBuffer::<u32, 2, Dropping>::new();
        assert_eq!(buffer.push(1), Ok(()));
        assert_eq!(buffer.push(2), Ok(()));
        assert_eq!(buffer.push(3), Err(3));

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.push(3), Ok(()));
        assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn overwriting_evicts_the_oldest() {
        let mut buffer = BoundedBuffer::<u32, 3, Overwriting>::new();
        for i in 0..3 {
            assert_eq!(buffer.push(i), None);
        }
        assert_eq!(buffer.push(3), Some(0));
        assert_eq!(buffer.push(4), Some(1));
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), [2, 3, 4]);
        assert_eq!(buffer.pop(), Some(2));
    }

    #[test]
    fn capacity_of_one() {
        let mut dropping = BoundedBuffer::<u32, 1, Dropping>::new();
        assert_eq!(dropping.push(1), Ok(()));
        assert_eq!(dropping.push(2), Err(2));
        assert_eq!(dropping.pop(), Some(1));

        let mut overwriting = BoundedBuffer::<u32, 1, Overwriting>::new();
        assert_eq!(overwriting.push(1), None);
        assert_eq!(overwriting.push(2), Some(1));
        assert_eq!(overwriting.pop(), Some(2));
    }

    #[test]
    fn wraps_around_a_capacity_that_is_not_a_power_of_two() {
        let mut buffer = BoundedBuffer::<usize, 3, Dropping>::new();
        for i in 0..1000 {
            assert_eq!(buffer.push(i), Ok(()));
            assert_eq!(buffer.push(i + 1), Ok(()));
            assert_eq!(buffer.pop(), Some(i));
            assert_eq!(buffer.pop(), Some(i + 1));
        }
    }

    #[test]
    fn front_back_and_indexing() {
        let mut buffer = BoundedBuffer::<u32, 3, Overwriting>::new();
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4);

        assert_eq!(buffer.front(), Some(&2));
        assert_eq!(buffer.back(), Some(&4));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(3), None);

        *buffer.front_mut().unwrap() = 20;
        *buffer.back_mut().unwrap() = 40;
        *buffer.get_mut(1).unwrap() = 30;
        assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), [20, 30, 40]);
    }

    #[test]
    fn iteration_in_both_directions() {
        let mut buffer = BoundedBuffer::<u32, 4, Overwriting>::new();
        for i in 0..6 {
            buffer.push(i);
        }

        assert_eq!(buffer.iter().len(), 4);
        assert_eq!(
            buffer.iter().rev().copied().collect::<Vec<_>>(),
            [5, 4, 3, 2]
        );

        let mut iter = buffer.iter();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next_back(), Some(&4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn values_are_dropped_once() {
        let value = Rc::new(());
        {
            let mut buffer = BoundedBuffer::<Rc<()>, 2, Overwriting>::new();
            buffer.push(value.clone());
            buffer.push(value.clone());
            // Evicted and handed back, then dropped here.
            drop(buffer.push(value.clone()));
            assert_eq!(Rc::strong_count(&value), 3);

            buffer.clear();
            assert_eq!(Rc::strong_count(&value), 1);

            buffer.push(value.clone());
        }
        assert_eq!(Rc::strong_count(&value), 1);
    }

    #[test]
    fn clone_and_debug() {
        let mut buffer = BoundedBuffer::<u32, 3, Overwriting>::new();
        for i in 0..5 {
            buffer.push(i);
        }

        let clone = buffer.clone();
        assert_eq!(clone.iter().copied().collect::<Vec<_>>(), [2, 3, 4]);
        assert_eq!(format!("{buffer:?}"), "[2, 3, 4]");
    }
}
