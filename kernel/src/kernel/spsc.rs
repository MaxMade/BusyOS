//! A bounded, lock-free queue with one producer and one consumer.
//!
//! [`SPSC`] is a ring of `N` slots and two positions: `head`, which only the
//! producer moves, and `tail`, which only the consumer moves. Each side reads
//! the other's position to learn how much room or how many values there are,
//! and publishes its own once it is done with a slot, so neither side ever
//! writes what the other writes and nothing has to be locked or retried.
//!
//! # Why the shape it has
//!
//! - **Bounded, and fixed at compile time.** The ring is an array inside the
//!   queue, so building one allocates nothing and needs no token, and a queue
//!   may be a `static`. A full queue refuses a value rather than growing: the
//!   caller gets it back from [`try_push`](SPSC::try_push).
//! - **Wait-free on both sides.** A push or a pop is a load of the other
//!   side's position, one slot access and a store of its own position. There
//!   is no loop, so neither side can be held up by the other: a value is
//!   visible to the consumer as soon as its push has returned, and a slot is
//!   free for the producer as soon as the pop that emptied it has returned.
//! - **One of each.** With a single writer per position, neither needs a
//!   compare-and-swap, which is what the queue saves over
//!   [`MPSC`](crate::kernel::mpsc::MPSC). The price is that both
//!   [`try_push`](SPSC::try_push) and [`pop`](SPSC::pop) are `unsafe`: sharing
//!   the queue does not make a second producer or a second consumer sound.

use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

/// A bounded, lock-free queue of up to `N` values, with one producer and one
/// consumer.
///
/// `N` has to be a power of two, so that a position maps to its slot by a mask
/// and stays consistent when the position counters wrap around; a queue of any
/// other size does not compile.
///
/// See the [module documentation](self) for how the queue works and what it
/// guarantees.
pub struct SPSC<T, const N: usize> {
    /// The ring. The slots from `tail` up to `head` hold values, the others
    /// are uninitialised.
    slots: [UnsafeCell<MaybeUninit<T>>; N],

    /// Position the producer writes next. `head % N` is its slot. Written only
    /// by the producer.
    head: AtomicUsize,

    /// Position the consumer reads next. `tail % N` is its slot. Written only
    /// by the consumer.
    tail: AtomicUsize,
}

// SAFETY: sending the queue sends the values in it.
unsafe impl<T: Send, const N: usize> Send for SPSC<T, N> {}

// SAFETY: a shared queue only ever moves values in and out of it, from the
// core that pushes to the core that pops, and never hands out a reference to
// one. So `T: Send` is all a value needs. The positions keep every value
// written by one side before the other reads it.
unsafe impl<T: Send, const N: usize> Sync for SPSC<T, N> {}

impl<T, const N: usize> SPSC<T, N> {
    /// Masks a position down to the index of its slot.
    const MASK: usize = {
        assert!(N.is_power_of_two(), "SPSC capacity must be a power of two");
        N - 1
    };

    /// Creates an empty queue.
    ///
    /// Allocates nothing and may be used to build a `static`.
    pub const fn new() -> Self {
        // Evaluates the assertion in `MASK`, so that a bad `N` fails here
        // rather than on the first push.
        let _ = Self::MASK;

        Self {
            slots: [const { UnsafeCell::new(MaybeUninit::uninit()) }; N],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// The number of values the queue can hold at once.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Appends `value` to the queue.
    ///
    /// Never waits: a full queue is reported, not waited out.
    ///
    /// # Errors
    ///
    /// `value` itself, if the queue is full. The queue is unchanged in that
    /// case.
    ///
    /// # Safety
    ///
    /// There is one producer: no other core may call this at the same time,
    /// and it must not be re-entered from an interrupt that hit a call already
    /// in progress. Two producers would share `head` and could write the same
    /// slot.
    pub unsafe fn try_push(&self, value: T) -> Result<(), T> {
        // Only the producer writes `head`, so it reads back its own store.
        let head = self.head.load(Ordering::Relaxed);

        // Acquire, pairing with the release in `pop`: once a slot reads free,
        // the consumer is done moving its previous value out.
        let tail = self.tail.load(Ordering::Acquire);

        // Wrapping, since the positions do. `N` being a power of two keeps the
        // distance right across the wrap.
        if head.wrapping_sub(tail) == N {
            return Err(value);
        }

        // SAFETY: the slot lies outside `tail..head`, so it holds no value and
        // the consumer stays out of it until `head` below covers it. This is
        // the only producer, so nothing else writes it either.
        unsafe { (*self.slots[head & Self::MASK].get()).write(value) };

        // Release, so that the consumer reading the new `head` sees the value
        // written above.
        self.head.store(head.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Takes the oldest value out of the queue, or returns [`None`] if it is
    /// empty.
    ///
    /// # Safety
    ///
    /// There is one consumer: no other core may call this at the same time,
    /// and it must not be re-entered from an interrupt that hit a call already
    /// in progress. Two consumers would share `tail` and could take the same
    /// value twice.
    pub unsafe fn pop(&self) -> Option<T> {
        // Only the consumer writes `tail`, so it reads back its own store.
        let tail = self.tail.load(Ordering::Relaxed);

        // Acquire, pairing with the release in `try_push`, so that the value
        // read below is the one the producer wrote.
        let head = self.head.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        // SAFETY: the slot lies inside `tail..head`, so it holds a value, and
        // the producer stays out of it until `tail` below frees it. Reading it
        // moves it out, and the slot counts as empty from here on.
        let value = unsafe { (*self.slots[tail & Self::MASK].get()).assume_init_read() };

        // Release, so that the producer reusing the slot only writes it after
        // the value has been moved out above.
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        Some(value)
    }

    /// The number of values in the queue.
    ///
    /// A snapshot: either side may change it right after the check. Only the
    /// producer may rely on it not shrinking below what it saw being freed,
    /// and only the consumer on it not dropping below what it saw being
    /// filled.
    pub fn len(&self) -> usize {
        // `tail` first: read the other way round, a push and a pop in between
        // could make `tail` overtake the `head` already read.
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);

        // In this order the distance cannot go negative, but pops and pushes
        // between the two loads can stretch it past `N`.
        head.wrapping_sub(tail).min(N)
    }

    /// Whether the queue holds no values.
    ///
    /// A snapshot, see [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize> Default for SPSC<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for SPSC<T, N> {
    /// Drops every value still in the queue.
    ///
    /// A value whose own `Drop` needs more than it can be given here, such as
    /// the last [`Arc`](crate::kernel::arc::Arc) on a value, has to be popped
    /// and handed back properly before the queue goes.
    fn drop(&mut self) {
        // SAFETY: `&mut self` rules out the producer and any other consumer.
        while let Some(value) = unsafe { self.pop() } {
            drop(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    use std::{sync::Arc, thread};

    #[test]
    fn pop_on_empty_returns_none() {
        let queue = SPSC::<u32, 4>::new();
        assert!(queue.is_empty());
        assert_eq!(unsafe { queue.pop() }, None);
    }

    #[test]
    fn values_come_out_in_order() {
        let queue = SPSC::<u32, 4>::new();
        for i in 0..4 {
            assert_eq!(unsafe { queue.try_push(i) }, Ok(()));
        }
        assert_eq!(queue.len(), 4);
        for i in 0..4 {
            assert_eq!(unsafe { queue.pop() }, Some(i));
        }
        assert_eq!(unsafe { queue.pop() }, None);
    }

    #[test]
    fn full_queue_hands_the_value_back() {
        let queue = SPSC::<u32, 2>::new();
        assert_eq!(unsafe { queue.try_push(1) }, Ok(()));
        assert_eq!(unsafe { queue.try_push(2) }, Ok(()));
        assert_eq!(unsafe { queue.try_push(3) }, Err(3));

        assert_eq!(unsafe { queue.pop() }, Some(1));
        assert_eq!(unsafe { queue.try_push(3) }, Ok(()));
        assert_eq!(unsafe { queue.pop() }, Some(2));
        assert_eq!(unsafe { queue.pop() }, Some(3));
    }

    #[test]
    fn capacity_of_one() {
        let queue = SPSC::<u32, 1>::new();
        assert_eq!(unsafe { queue.try_push(1) }, Ok(()));
        assert_eq!(unsafe { queue.try_push(2) }, Err(2));
        assert_eq!(unsafe { queue.pop() }, Some(1));
        assert_eq!(unsafe { queue.pop() }, None);
    }

    #[test]
    fn wraps_around_many_laps() {
        let queue = SPSC::<usize, 4>::new();
        for i in 0..1000 {
            assert_eq!(unsafe { queue.try_push(i) }, Ok(()));
            assert_eq!(unsafe { queue.pop() }, Some(i));
        }
    }

    #[test]
    fn wraps_around_the_position_counters() {
        let queue = SPSC::<usize, 4>::new();
        queue.head.store(usize::MAX - 1, Ordering::Relaxed);
        queue.tail.store(usize::MAX - 1, Ordering::Relaxed);

        for i in 0..4 {
            assert_eq!(unsafe { queue.try_push(i) }, Ok(()));
        }
        assert_eq!(unsafe { queue.try_push(4) }, Err(4));
        assert_eq!(queue.len(), 4);
        for i in 0..4 {
            assert_eq!(unsafe { queue.pop() }, Some(i));
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn drop_drops_remaining_values() {
        let value = Arc::new(());
        {
            let queue = SPSC::<Arc<()>, 4>::new();
            unsafe { queue.try_push(value.clone()) }.unwrap();
            unsafe { queue.try_push(value.clone()) }.unwrap();
            assert_eq!(Arc::strong_count(&value), 3);
        }
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    fn concurrent_producer_and_consumer_lose_nothing() {
        const COUNT: usize = 100_000;

        let queue = Arc::new(SPSC::<usize, 64>::new());

        let producer = {
            let queue = queue.clone();
            thread::spawn(move || {
                for i in 0..COUNT {
                    let mut value = i;
                    while let Err(v) = unsafe { queue.try_push(value) } {
                        value = v;
                        thread::yield_now();
                    }
                }
            })
        };

        let mut next = 0;
        while next < COUNT {
            match unsafe { queue.pop() } {
                Some(i) => {
                    assert_eq!(i, next);
                    next += 1;
                }
                None => thread::yield_now(),
            }
        }

        producer.join().unwrap();
        assert_eq!(unsafe { queue.pop() }, None);
    }
}
