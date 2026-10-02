//! A bounded, lock-free queue with many producers and one consumer.
//!
//! [`MPSC`] is a ring of `N` slots in which every slot carries a sequence
//! number next to its value, after Dmitry Vyukov's bounded queue. The number
//! says which lap of the ring the slot is on and whether its value has been
//! written yet, so a producer and the consumer agree on the state of a slot by
//! one load of it, and nothing has to be locked.
//!
//! # Why the shape it has
//!
//! - **Bounded, and fixed at compile time.** The ring is an array inside the
//!   queue, so building one allocates nothing and needs no token, and a queue
//!   may be a `static`. A full queue refuses a value rather than growing: the
//!   caller gets it back from [`try_push`](MPSC::try_push).
//! - **Producers never wait.** A producer claims a slot with one
//!   compare-and-swap on `head`, writes its value and publishes it. It never
//!   waits for another producer or for the consumer, so pushing from an
//!   interrupt that hit a push already in progress on the same core is sound:
//!   the two simply claim different slots.
//! - **One consumer.** Only the consumer moves `tail`, so it needs no
//!   compare-and-swap, and [`pop`](MPSC::pop) is `unsafe` because sharing the
//!   queue does not make a second consumer sound.
//!
//! # A producer between claiming and publishing
//!
//! A slot is claimed before its value is written. Should a producer stop in
//! between, the consumer finds the slot claimed but not yet published and
//! [`pop`](MPSC::pop) reports the queue as empty until it is, even if later
//! slots are already filled. Entries thus come out strictly in the order their
//! slots were claimed, at the cost of one stalled producer holding up the
//! entries behind it. A producer that stalls this way is one that was
//! interrupted, so the wait lasts as long as the interrupt does.

use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

/// One entry of the ring.
struct Slot<T> {
    /// Where the slot is in its cycle, relative to the position `pos` that
    /// names it on the current lap:
    ///
    /// - `pos`: empty, and free for the producer that claims `pos`.
    /// - `pos + 1`: holds the value pushed at `pos`, ready for the consumer.
    /// - `pos + N`: emptied by the consumer, free for the next lap.
    seq: AtomicUsize,

    /// The value, initialised exactly while `seq` says `pos + 1`.
    value: UnsafeCell<MaybeUninit<T>>,
}

/// A bounded, lock-free queue of up to `N` values, with any number of
/// producers and one consumer.
///
/// `N` has to be a power of two, so that a position maps to its slot by a mask
/// and stays consistent when the position counters wrap around, and at least
/// two; a queue of any other size does not compile.
///
/// See the [module documentation](self) for how the queue works and what it
/// guarantees.
pub struct MPSC<T, const N: usize> {
    /// The ring.
    slots: [Slot<T>; N],

    /// Position the next producer tries to claim. `head % N` is its slot.
    head: AtomicUsize,

    /// Position the consumer reads next. `tail % N` is its slot. Written only
    /// by the consumer, and atomic only because [`pop`](Self::pop) takes
    /// `&self`.
    tail: AtomicUsize,
}

// SAFETY: sending the queue sends the values in it.
unsafe impl<T: Send, const N: usize> Send for MPSC<T, N> {}

// SAFETY: a shared queue only ever moves values in and out of it, from
// whichever core pushes to whichever core pops, and never hands out a
// reference to one. So `T: Send` is all a value needs. The slot protocol keeps
// every value written by one side before the other reads it.
unsafe impl<T: Send, const N: usize> Sync for MPSC<T, N> {}

impl<T, const N: usize> MPSC<T, N> {
    /// Masks a position down to the index of its slot.
    const MASK: usize = {
        assert!(N.is_power_of_two(), "MPSC capacity must be a power of two");
        // With one slot, `pos + 1` would mean both "holds a value" and "free
        // for the next lap", so a push could overwrite an unread value.
        assert!(N >= 2, "MPSC capacity must be at least two");
        N - 1
    };

    /// Creates an empty queue.
    ///
    /// Allocates nothing and may be used to build a `static`.
    pub const fn new() -> Self {
        // Evaluates the assertion in `MASK`, so that a bad `N` fails here
        // rather than on the first push.
        let _ = Self::MASK;

        let mut slots = [const {
            Slot {
                seq: AtomicUsize::new(0),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            }
        }; N];

        // Slot `i` starts out free for the producer claiming position `i`.
        let mut i = 0;
        while i < N {
            slots[i].seq = AtomicUsize::new(i);
            i += 1;
        }

        Self {
            slots,
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
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let mut pos = self.head.load(Ordering::Relaxed);

        loop {
            let slot = &self.slots[pos & Self::MASK];

            // Acquire, pairing with the release in `pop`: once the slot reads
            // free, the consumer is done moving the previous value out of it.
            let seq = slot.seq.load(Ordering::Acquire);

            // Wrapping, since the positions do; read as signed, it says how far
            // the slot is ahead of or behind the lap `pos` is on.
            let diff = seq.wrapping_sub(pos) as isize;

            if diff == 0 {
                // The slot is free for `pos`, so try to claim `pos`. Relaxed:
                // the claim publishes nothing, `seq` below does.
                match self.head.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: winning the claim on `pos` makes this the
                        // only producer writing the slot on this lap, and the
                        // consumer stays out of it until `seq` below says
                        // `pos + 1`.
                        unsafe { (*slot.value.get()).write(value) };

                        // Release, so that the consumer reading `pos + 1` sees
                        // the value written above.
                        slot.seq.store(pos.wrapping_add(1), Ordering::Release);

                        return Ok(());
                    }
                    // Another producer claimed `pos` first. Retry from where
                    // `head` is now.
                    Err(current) => pos = current,
                }
            } else if diff < 0 {
                // The slot still holds the value from the previous lap, which
                // the consumer has not taken: the queue is full.
                return Err(value);
            } else {
                // Another producer claimed `pos` and moved on since `head` was
                // read. Catch up.
                pos = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// Takes the oldest value out of the queue, or returns [`None`] if there
    /// is none ready.
    ///
    /// [`None`] also covers a slot that has been claimed but whose producer has
    /// not published its value yet, see the
    /// [module documentation](self#a-producer-between-claiming-and-publishing).
    ///
    /// # Safety
    ///
    /// There is one consumer: no other core may call this at the same time,
    /// and it must not be re-entered from an interrupt that hit a call already
    /// in progress. Two consumers would share `tail` and could take the same
    /// value twice.
    pub unsafe fn pop(&self) -> Option<T> {
        // Only the consumer writes `tail`, so it reads back its own store.
        let pos = self.tail.load(Ordering::Relaxed);
        let slot = &self.slots[pos & Self::MASK];

        // Acquire, pairing with the release in `try_push`, so that the value
        // read below is the one the producer wrote.
        if slot.seq.load(Ordering::Acquire) != pos.wrapping_add(1) {
            return None;
        }

        // SAFETY: `seq` says `pos + 1`, so the value is initialised, and no
        // producer touches the slot again until `seq` below frees it. Reading
        // it moves it out, and the slot counts as empty from here on.
        let value = unsafe { (*slot.value.get()).assume_init_read() };

        // Release, so that the producer taking the slot on the next lap only
        // writes it after the value has been moved out above.
        slot.seq.store(pos.wrapping_add(N), Ordering::Release);
        self.tail.store(pos.wrapping_add(1), Ordering::Relaxed);

        Some(value)
    }

    /// Whether the queue holds no values ready to be popped.
    ///
    /// A snapshot: a producer may push right after the check. Only the
    /// consumer's view is stable, since only it removes values.
    pub fn is_empty(&self) -> bool {
        let pos = self.tail.load(Ordering::Relaxed);
        let slot = &self.slots[pos & Self::MASK];

        slot.seq.load(Ordering::Acquire) != pos.wrapping_add(1)
    }
}

impl<T, const N: usize> Default for MPSC<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for MPSC<T, N> {
    /// Drops every value still in the queue.
    ///
    /// A value whose own `Drop` needs more than it can be given here, such as
    /// the last [`Arc`](crate::kernel::arc::Arc) on a value, has to be popped
    /// and handed back properly before the queue goes.
    fn drop(&mut self) {
        // SAFETY: `&mut self` rules out every producer and any other consumer.
        while let Some(value) = unsafe { self.pop() } {
            drop(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    use std::{sync::Arc, thread, vec::Vec};

    #[test]
    fn pop_on_empty_returns_none() {
        let queue = MPSC::<u32, 4>::new();
        assert!(queue.is_empty());
        assert_eq!(unsafe { queue.pop() }, None);
    }

    #[test]
    fn values_come_out_in_order() {
        let queue = MPSC::<u32, 4>::new();
        for i in 0..4 {
            assert_eq!(queue.try_push(i), Ok(()));
        }
        assert!(!queue.is_empty());
        for i in 0..4 {
            assert_eq!(unsafe { queue.pop() }, Some(i));
        }
        assert_eq!(unsafe { queue.pop() }, None);
    }

    #[test]
    fn full_queue_hands_the_value_back() {
        let queue = MPSC::<u32, 2>::new();
        assert_eq!(queue.try_push(1), Ok(()));
        assert_eq!(queue.try_push(2), Ok(()));
        assert_eq!(queue.try_push(3), Err(3));

        assert_eq!(unsafe { queue.pop() }, Some(1));
        assert_eq!(queue.try_push(3), Ok(()));
        assert_eq!(unsafe { queue.pop() }, Some(2));
        assert_eq!(unsafe { queue.pop() }, Some(3));
    }

    #[test]
    fn wraps_around_many_laps() {
        let queue = MPSC::<usize, 4>::new();
        for i in 0..1000 {
            assert_eq!(queue.try_push(i), Ok(()));
            assert_eq!(unsafe { queue.pop() }, Some(i));
        }
    }

    #[test]
    fn drop_drops_remaining_values() {
        let value = Arc::new(());
        {
            let queue = MPSC::<Arc<()>, 4>::new();
            queue.try_push(value.clone()).unwrap();
            queue.try_push(value.clone()).unwrap();
            assert_eq!(Arc::strong_count(&value), 3);
        }
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    fn concurrent_producers_lose_nothing() {
        const PRODUCERS: usize = 4;
        const PER_PRODUCER: usize = 10_000;

        let queue = Arc::new(MPSC::<(usize, usize), 64>::new());

        let producers: Vec<_> = (0..PRODUCERS)
            .map(|p| {
                let queue = queue.clone();
                thread::spawn(move || {
                    for i in 0..PER_PRODUCER {
                        let mut value = (p, i);
                        while let Err(v) = queue.try_push(value) {
                            value = v;
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();

        // Each producer's values have to come out in the order it pushed them.
        let mut next = [0usize; PRODUCERS];
        let mut received = 0;
        while received < PRODUCERS * PER_PRODUCER {
            match unsafe { queue.pop() } {
                Some((p, i)) => {
                    assert_eq!(i, next[p]);
                    next[p] += 1;
                    received += 1;
                }
                None => thread::yield_now(),
            }
        }

        for producer in producers {
            producer.join().unwrap();
        }
        assert!(next.iter().all(|&n| n == PER_PRODUCER));
        assert_eq!(unsafe { queue.pop() }, None);
    }
}
