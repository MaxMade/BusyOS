//! A fair mutex whose waiters block rather than spin, with a level in the
//! locking hierarchy.
//!
//! Lives below [`locking`](super), as it mints its level tokens itself rather
//! than through [`Lock`](super::Lock): taking it may sleep, which acquires the
//! scheduler's lock, so acquiring and releasing it take a token that also
//! reaches [`level::Scheduler`].

use core::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicIsize, Ordering as AtomicOrdering},
};

use kernel_derive::lock_id;

use super::{Above, CanAcquire, Exclusive, LockId, PreviousToken, SchedulerLevelID, Token, level};
use crate::{
    kernel::{
        arc::Arc,
        compiler::likely::{likely, unlikely},
        linked_list::{LinkedList, Node},
        ticketlock::{ThreadTicketlock, Ticketlock},
    },
    mem::{cache::Cache, heap::Heap},
    sched::{
        scheduler::{CURRENT, Scheduler, SchedulerDriver},
        thread::Thread,
    },
};

/// How often [`Mutex::acquire`] tries to take the mutex before it blocks.
const SPIN_ITERATIONS: usize = 1024;

/// The threads blocked in [`Mutex::acquire`], in the order they blocked.
///
/// The nodes come from a [`Cache`], as in the run queue, so that blocking and
/// waking reuse the node just freed instead of going to the heap.
type WaitQueue = LinkedList<Arc<Thread>, Cache<Node<Arc<Thread>>, Heap>>;

/// What the slow path of [`Mutex::lock`] found under the scheduler's lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wait {
    /// The mutex was free after all and is taken.
    Acquired,
    /// The thread is queued and blocks, until a [`Mutex::unlock`] hands the
    /// mutex over.
    Blocked,
    /// The thread could not be queued for lack of memory, and tries again.
    OutOfMemory,
}

#[lock_id(Driver)]
pub struct MutexDriverID;

/// A fair mutex around a `T`, as a blocking lock at the level of `Id`.
///
/// [`acquire`](Mutex::acquire) returns a `(guard, token)` pair, as
/// [`Lock::acquire`](super::Lock::acquire) does, and the guard's `release`
/// consumes both. A thread that finds the mutex taken spins for a while, and
/// then blocks until the holder hands the mutex over.
///
/// Only the levels above [`level::Scheduler`] may sleep. Of those, only
/// `Driver` has a lock identity here: `Epilogue`, the top of the hierarchy,
/// could only be taken with a bare root token.
///
/// # Fairness
///
/// A semaphore with a single unit: the counter is one while the mutex is
/// free, and minus the threads committed to block while it is taken. A
/// blocking thread takes the mutex from the counter in advance, and a release
/// that finds the counter negative hands the mutex straight to the thread
/// that has waited longest, instead of freeing it. The mutex thus never shows
/// as free while threads wait, and a thread that comes later, spinning or
/// not, cannot take it from them.
///
/// The decrement of a blocking thread and its queueing are one step under
/// the queue's lock, so a release that saw the decrement finds the thread
/// queued once it has the lock, and the scheduler's lock waits for the thread
/// to have stopped running, see [`Scheduler::try_block`].
pub struct Mutex<T, Id: LockId> {
    /// One while free, minus the threads committed to block while taken.
    counter: AtomicIsize,
    /// The blocked threads, each holding a handle on itself until a release
    /// takes it out to wake it.
    waiting: ThreadTicketlock<WaitQueue>,
    /// The value the mutex protects.
    data: UnsafeCell<T>,
    _id: PhantomData<Id>,
}

// SAFETY: sending the mutex sends the T inside it.
unsafe impl<T: Send, Id: LockId> Send for Mutex<T, Id> {}

// SAFETY: only exclusive guards exist, which allow moving values out via
// `&mut T` (T: Send). Identical bounds to std's Mutex.
unsafe impl<T: Send, Id: LockId> Sync for Mutex<T, Id> {}

impl<T, Id> Mutex<T, Id>
where
    Id: LockId,
    Id::Level: Above<level::Scheduler>,
{
    /// Creates an unlocked mutex around `value`.
    ///
    /// Allocates nothing.
    pub const fn new(value: T) -> Self {
        Self {
            counter: AtomicIsize::new(1),
            waiting: ThreadTicketlock::new(Ticketlock::new(), LinkedList::new_in(Cache::new())),
            data: UnsafeCell::new(value),
            _id: PhantomData,
        }
    }

    /// Consumes the mutex, returning the inner value.
    ///
    /// Safe without a token: `self` by value proves no guards exist.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Returns a raw pointer to the underlying data, without locking, see
    /// [`Lock::data_ptr`](super::Lock::data_ptr).
    pub const fn data_ptr(&self) -> *mut T {
        self.data.get()
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// The mutable borrow statically guarantees no guard exists, so no
    /// locking needs to take place.
    pub const fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Exclusive access, blocking until the mutex is free.
    ///
    /// Returns the data guard and the target-level token as a pair. Release
    /// them together via [`MutexGuard::release`].
    ///
    /// Must be called on the epilogue level, as all kernel code, see
    /// [`Epilogue`](crate::kernel::handler::Epilogue).
    ///
    /// # Panics
    ///
    /// If no scheduler has registered.
    pub fn acquire<From>(&self, token: From) -> (MutexGuard<'_, T, Id>, Token<Id, From, Exclusive>)
    where
        From:
            CanAcquire<Id::Level> + CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        // The token is embedded in the type of the forged one from here on.
        core::mem::forget(self.lock(token));

        // SAFETY: the mutex has just been taken.
        let token = unsafe { Token::forge() };

        (MutexGuard { mutex: self }, token)
    }

    /// Takes the mutex, blocking until it is free.
    ///
    /// Tries to take it for [`SPIN_ITERATIONS`] rounds first, with nothing
    /// but the counter. Then decides under the scheduler's lock, see
    /// [`Scheduler::try_block`]: takes it if it has come free by now, and
    /// blocks otherwise, behind every thread already blocked, until an
    /// [`unlock`](Mutex::unlock) hands it over. If the thread cannot be
    /// queued for lack of memory, it starts over.
    ///
    /// A caller with an [`Init`](super::Init) token, from start-up, and an
    /// idle thread never block, but spin until the mutex is free, as on a
    /// spinlock.
    fn lock<From>(&self, token: From) -> From
    where
        From: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        loop {
            // Fast path: spin trying to take the mutex. The counter is not
            // positive while threads wait, so this never overtakes them.
            for _ in 0..SPIN_ITERATIONS {
                if self.counter.load(AtomicOrdering::Relaxed) <= 0 {
                    core::hint::spin_loop();
                    continue;
                }

                if likely(
                    self.counter
                        .compare_exchange_weak(
                            1,
                            0,
                            AtomicOrdering::Acquire,
                            AtomicOrdering::Relaxed,
                        )
                        .is_ok(),
                ) {
                    return token;
                }
            }

            // Only a thread that may stop running can block: not start-up,
            // which holds an init token, as the core does not run threads
            // yet, and not an idle thread, which must always stay runnable.
            // Those keep spinning, as on a spinlock, until the holder,
            // running on another core or scheduled there, releases the mutex.
            if From::INIT || unlikely(CURRENT.with(|current| current.is_idle())) {
                continue;
            }

            // Slow path: block until the mutex is handed over.
            let (scheduler, t) = match SchedulerDriver::get(token) {
                (Some(scheduler), t) => (scheduler, t),
                (None, _) => panic!("Trying to acquire mutex without a scheduler"),
            };

            // Decide under the scheduler's lock whether the mutex came free
            // after all, or the thread blocks.
            let mut wait = Wait::OutOfMemory;
            token = scheduler.try_block(t, |current, token| {
                let (mut waiting, token) = self.waiting.acquire(token);

                // Take the mutex in advance: either while free, or the
                // hand-over of a release to come, see `Mutex`.
                if self.counter.fetch_sub(1, AtomicOrdering::Acquire) > 0 {
                    wait = Wait::Acquired;
                    return (false, waiting.release(token));
                }

                match waiting.try_push_back(current.clone(), token) {
                    Ok(token) => {
                        wait = Wait::Blocked;
                        (true, waiting.release(token))
                    }
                    Err((_, token)) => {
                        // The handle is dropped with the failed push.
                        // `CURRENT` holds another, so it was not the last one.
                        //
                        // Undo the decrement. A release that saw it and waits
                        // for the lock finds either another thread to hand
                        // over to, or, with the queue empty, the mutex freed
                        // by this.
                        self.counter.fetch_add(1, AtomicOrdering::Relaxed);
                        wait = Wait::OutOfMemory;
                        (false, waiting.release(token))
                    }
                }
            });

            // The registered scheduler keeps a handle of its own.
            drop(scheduler);

            match wait {
                // A blocked thread is woken with the mutex handed over, see
                // `unlock`.
                Wait::Acquired | Wait::Blocked => return token,
                Wait::OutOfMemory => continue,
            }
        }
    }

    /// Releases the mutex, handing it straight to the thread that has been
    /// blocked in [`lock`](Mutex::lock) for longest, if any, and waking it.
    ///
    /// # Panics
    ///
    /// If a thread is to be woken without a scheduler, or cannot be queued to
    /// run for lack of memory.
    fn unlock<From>(&self, token: From) -> From
    where
        From: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        // Not negative: nobody waits, and the mutex is free now.
        if likely(self.counter.fetch_add(1, AtomicOrdering::Release) >= 0) {
            return token;
        }

        // Negative: a thread has taken the mutex in advance, and is queued by
        // the time the lock is free, see `Mutex`.
        let (mut waiting, token) = self.waiting.acquire(token);
        let (thread, token) = waiting.pop_front(token);
        let token = waiting.release(token);

        // Empty only if the thread that took the mutex in advance could not
        // be queued: it undid that, and with it, the mutex is free.
        let Some(thread) = thread else {
            return token;
        };

        let (scheduler, token) = match SchedulerDriver::get(token) {
            (Some(scheduler), token) => (scheduler, token),
            (None, _) => panic!("Trying to wake thread blocked on mutex without a scheduler"),
        };

        match scheduler.enqueue(thread, token) {
            Ok(token) => token,
            Err((error, _)) => panic!("Unable to wake thread blocked on mutex: {}", error),
        }
    }
}

/// Exclusive data guard of a [`Mutex`]. Dereferences to `T` (mutably).
///
/// Release explicitly via [`MutexGuard::release`], together with the level
/// token.
pub struct MutexGuard<'a, T, Id>
where
    Id: LockId,
    Id::Level: Above<level::Scheduler>,
{
    mutex: &'a Mutex<T, Id>,
}

impl<'a, T, Id> MutexGuard<'a, T, Id>
where
    Id: LockId,
    Id::Level: Above<level::Scheduler>,
{
    /// Unlocks, consuming the guard and the level token, recovering the
    /// previous-level token, and waking the thread that has waited longest,
    /// if any.
    ///
    /// # Panics
    ///
    /// If a thread is to be woken without a scheduler, or cannot be queued to
    /// run for lack of memory.
    pub fn release<P>(self, token: Token<Id, P, Exclusive>) -> P
    where
        P: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        core::mem::forget(token);

        // SAFETY: the consumed token proves the mutex is held and that
        // nothing below it is any more; `P` was embedded at acquire time and
        // is still held by this thread.
        let previous = unsafe { P::forge_previous() };

        self.mutex.unlock(previous)
    }
}

impl<'a, T, Id> Deref for MutexGuard<'a, T, Id>
where
    Id: LockId,
    Id::Level: Above<level::Scheduler>,
{
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: exclusive hold for the guard's lifetime.
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T, Id> DerefMut for MutexGuard<'a, T, Id>
where
    Id: LockId,
    Id::Level: Above<level::Scheduler>,
{
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: exclusive hold; `&mut self` ensures uniqueness.
        unsafe { &mut *self.mutex.data.get() }
    }
}

/// Driver-level mutex: the blocking lock a driver holds while it waits for
/// its device.
pub type DriverMutex<T> = Mutex<T, MutexDriverID>;

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, thread, vec::Vec};

    use super::*;
    use crate::kernel::{
        locking::{EpilogueLevel, Init, RootToken},
        spinlock::{MemoryManagementSpinlock, Spinlock},
    };

    fn is_init<T: PreviousToken>(_: &T) -> bool {
        T::INIT
    }

    /// The `Driver` level sits between `Epilogue` and `MemoryManagement`: an
    /// epilogue may take a driver mutex, and a driver mutex may be held while
    /// descending to the memory levels below.
    #[test]
    fn usage_driver_level() {
        let root_token = unsafe { RootToken::forge() };

        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let driver = DriverMutex::new(0);
        let memory_management = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let (mut driver_guard, token) = driver.acquire(token);
        let (memory_management_guard, token) = memory_management.acquire(token);
        *driver_guard += 1;

        let token = memory_management_guard.release(token);
        let token = driver_guard.release(token);

        assert_eq!(driver.into_inner(), 1);

        epilogue_level.leave(token);
    }

    /// Every token acquired on top of an init token is one too, whichever
    /// lock it comes from, and an init token takes spinlocks and ticketlocks
    /// as any other.
    #[test]
    fn usage_init_token() {
        let root_token = unsafe { RootToken::forge() };

        let (epilogue_level, token) = EpilogueLevel::enter(root_token);
        assert!(!is_init(&token));

        let token = Init(token);
        assert!(is_init(&token));

        let driver = DriverMutex::new(0);
        let thread = ThreadTicketlock::new(Ticketlock::new(), 0);
        let memory_management = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let (mut driver_guard, token) = driver.acquire(token);
        let (thread_guard, token) = thread.acquire(token);
        let (memory_management_guard, token) = memory_management.acquire(token);
        assert!(is_init(&token));
        *driver_guard += 1;

        let token = memory_management_guard.release(token);
        let token = thread_guard.release(token);
        let token = driver_guard.release(token);
        assert!(is_init(&token));

        let Init(token) = token;
        assert!(!is_init(&token));

        assert_eq!(driver.into_inner(), 1);

        epilogue_level.leave(token);
    }

    /// Contended by init tokens, a mutex spins as a spinlock does, and still
    /// excludes: nothing ever reaches the scheduler, of which there is none.
    #[test]
    fn stress_init_token() {
        const NUM_THREADS: usize = 8;
        const ITERATIONS: usize = 100_000;

        let counter = std::sync::Arc::new(DriverMutex::new(0));
        let barrier = std::sync::Arc::new(Barrier::new(NUM_THREADS));

        let mut threads = Vec::new();
        for _ in 0..NUM_THREADS {
            let counter = counter.clone();
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                let root_token = unsafe { RootToken::forge() };
                let (epilogue_level, base_token) = EpilogueLevel::enter(root_token);
                let mut base_token = Init(base_token);

                barrier.wait();
                for _ in 0..ITERATIONS {
                    let (mut counter, token) = counter.acquire(base_token);
                    *counter += 1;
                    base_token = counter.release(token);
                }

                let Init(base_token) = base_token;
                epilogue_level.leave(base_token);
            });
            threads.push(handle);
        }

        for handle in threads {
            handle.join().unwrap();
        }

        let counter = std::sync::Arc::into_inner(counter).unwrap();
        assert_eq!(counter.into_inner(), NUM_THREADS * ITERATIONS);
    }
}
