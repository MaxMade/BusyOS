use core::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
};

use kernel_derive::lock_id;

use crate::kernel::locking::{HierarchicalLock, Lock, LockId, level};
use crate::kernel::prologue_lock::PrologueLock;

#[lock_id(Driver)]
pub struct SpinlockDriverID;

#[lock_id(Thread)]
pub struct SpinlockThreadID;

#[lock_id(MemoryManagement)]
pub struct SpinlockMemoryManagementID;

#[lock_id(Memory)]
pub struct SpinlockMemoryID;

#[lock_id(Prologue)]
pub struct SpinlockPrologueID;

pub struct Spinlock<Id: LockId> {
    state: AtomicBool,
    _id: PhantomData<Id>,
}

impl<Id: LockId> Spinlock<Id> {
    pub const fn new() -> Self {
        Self {
            state: AtomicBool::new(false),
            _id: PhantomData,
        }
    }
}

impl<Id: LockId> HierarchicalLock for Spinlock<Id> {
    type Id = Id;

    unsafe fn raw_lock(&self) {
        while self.state.swap(true, AtomicOrdering::Acquire) {
            core::hint::spin_loop();
        }
    }
    unsafe fn raw_try_lock(&self) -> bool {
        // A compare-exchange rather than a swap: a failed attempt leaves the
        // cache line alone instead of writing back the value it already had.
        self.state
            .compare_exchange(
                false,
                true,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
    }
    unsafe fn raw_unlock(&self) {
        self.state.store(false, AtomicOrdering::Release);
    }
    unsafe fn raw_lock_shared(&self) {
        panic!();
    }
    unsafe fn raw_try_lock_shared(&self) -> bool {
        panic!();
    }
    unsafe fn raw_unlock_shared(&self) {
        panic!();
    }
    unsafe fn raw_lock_shared_nested(&self) {
        panic!();
    }

    unsafe fn raw_unlock_shared_nested(&self) {
        panic!();
    }
}

/// Driver-level spinlock.
///
/// TODO(@MaxMade): a lock at [`level::Driver`] is meant to be a blocking
/// lock — a driver sleeps while it waits for its device — and none exists
/// yet, so this spinning one stands in. Until it is replaced, a hold must
/// not sleep: every waiter spins for as long as it does.
pub type DriverSpinlock<T> = Lock<T, Spinlock<SpinlockDriverID>>;

/// Thread-level spinlock.
///
/// [`level::Thread`] is below the scheduler, so this is a spinning lock in
/// earnest: a hold may not sleep, and the scheduler may take one while it
/// holds its own lock.
pub type ThreadSpinlock<T> = Lock<T, Spinlock<SpinlockThreadID>>;

pub type MemoryManagementSpinlock<T> = Lock<T, Spinlock<SpinlockMemoryManagementID>>;

pub type MemorySpinlock<T> = Lock<T, Spinlock<SpinlockMemoryID>>;

/// Prologue-level spinlock. Masks interrupts for the duration of the hold.
pub type PrologueSpinlock<T> = PrologueLock<T, Spinlock<SpinlockPrologueID>>;

#[lock_id(Driver)]
pub struct RWSpinlockDriverID;

#[lock_id(Thread)]
pub struct RWSpinlockThreadID;

#[lock_id(MemoryManagement)]
pub struct RWSpinlockMemoryManagementID;

#[lock_id(Memory)]
pub struct RWSpinlockMemoryID;

#[lock_id(Prologue)]
pub struct RWSpinlockPrologueID;

pub struct RWSpinlock<Id: LockId> {
    state: AtomicUsize,
    _id: PhantomData<Id>,
}

impl<Id: LockId> RWSpinlock<Id> {
    pub const fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
            _id: PhantomData,
        }
    }
}

impl<Id: LockId> HierarchicalLock for RWSpinlock<Id> {
    type Id = Id;

    unsafe fn raw_lock(&self) {
        while let Err(_) = self.state.compare_exchange_weak(
            0,
            usize::MAX,
            AtomicOrdering::Acquire,
            AtomicOrdering::Relaxed,
        ) {
            core::hint::spin_loop();
        }
    }
    unsafe fn raw_try_lock(&self) -> bool {
        self.state
            .compare_exchange(
                0,
                usize::MAX,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
    }
    unsafe fn raw_unlock(&self) {
        self.state.store(0, AtomicOrdering::Release);
    }
    unsafe fn raw_lock_shared(&self) {
        loop {
            let state = self.state.load(AtomicOrdering::Relaxed);
            if state != usize::MAX
                && self
                    .state
                    .compare_exchange_weak(
                        state,
                        state + 1,
                        AtomicOrdering::Acquire,
                        AtomicOrdering::Relaxed,
                    )
                    .is_ok()
            {
                break;
            }

            core::hint::spin_loop();
        }
    }
    unsafe fn raw_try_lock_shared(&self) -> bool {
        // Only a writer (`usize::MAX`) turns the attempt away; the retry loop
        // is for a concurrently changing reader count, not for waiting.
        let mut state = self.state.load(AtomicOrdering::Relaxed);
        loop {
            if state == usize::MAX {
                return false;
            }

            match self.state.compare_exchange_weak(
                state,
                state + 1,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(updated) => state = updated,
            }
        }
    }
    unsafe fn raw_unlock_shared(&self) {
        self.state.fetch_sub(1, AtomicOrdering::Release);
    }

    unsafe fn raw_lock_shared_nested(&self) {
        self.state.fetch_add(1, AtomicOrdering::Acquire);
    }

    unsafe fn raw_unlock_shared_nested(&self) {
        self.state.fetch_sub(1, AtomicOrdering::Release);
    }
}

/// Driver-level reader-writer spinlock.
///
/// TODO(@MaxMade): a lock at [`level::Driver`] is meant to be a blocking
/// lock — a driver sleeps while it waits for its device — and none exists
/// yet, so this spinning one stands in. Until it is replaced, a hold must
/// not sleep: every waiter spins for as long as it does.
pub type DriverRWSpinlock<T> = Lock<T, RWSpinlock<RWSpinlockDriverID>>;

/// Thread-level reader-writer spinlock.
///
/// As [`ThreadSpinlock`], a spinning lock: see [`level::Thread`].
pub type ThreadRWSpinlock<T> = Lock<T, RWSpinlock<RWSpinlockThreadID>>;

pub type MemoryManagementRWSpinlock<T> = Lock<T, RWSpinlock<RWSpinlockMemoryManagementID>>;

pub type MemoryRWSpinlock<T> = Lock<T, RWSpinlock<RWSpinlockMemoryID>>;

/// Prologue-level reader-writer spinlock. Masks interrupts for the duration
/// of the hold.
pub type PrologueRWSpinlock<T> = PrologueLock<T, RWSpinlock<RWSpinlockPrologueID>>;

#[cfg(test)]
mod test {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use crate::kernel::locking::{
        CanAcquire, DriverLevel, EpilogueLevel, PreviousToken, RootToken, SyscallLevel, ThreadLevel,
    };

    use super::*;

    extern crate std;

    #[test]
    fn usage_exclusive() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let memory = MemorySpinlock::new(Spinlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire(token);

        let (memory_guard, token) = memory.acquire(token);

        let token = memory_guard.release(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn usage_shared() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0);

        let memory = MemoryRWSpinlock::new(RWSpinlock::new(), 0);

        let (epilogue_guard_0, token) = epilogue.acquire_shared(token);

        let (epilogue_guard_1, token) = epilogue.acquire_shared_nested(token);

        let (memory_guard, token) = memory.acquire(token);

        let token = memory_guard.release(token);

        let token = epilogue_guard_1.release(token);

        let token = epilogue_guard_0.release(token);

        syscall_level.leave(token);
    }

    fn do_memory_work<From>(token: From) -> From
    where
        From: CanAcquire<level::Memory> + PreviousToken,
    {
        let memory = MemorySpinlock::new(Spinlock::new(), 0);
        let (memory_guard, token) = memory.acquire(token);

        memory_guard.release(token)
    }

    #[test]
    fn usage_exclusive_subroutine() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire(token);

        let token = do_memory_work(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn usage_shared_subroutine() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire_shared(token);

        let token = do_memory_work(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn stress_exclusive() {
        const NUM_EXCLUSIVE: usize = 8;
        const ITERATIONS: usize = 1_000_000;

        let counter = Arc::new(MemoryManagementSpinlock::new(Spinlock::new(), 0));
        let barrier = Arc::new(Barrier::new(NUM_EXCLUSIVE));

        let mut threads = Vec::new();
        for _ in 0..NUM_EXCLUSIVE {
            let counter = counter.clone();
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                let root_token = unsafe { RootToken::forge() };
                let (epilogue_level, mut base_token) = EpilogueLevel::enter(root_token);

                barrier.wait();
                for _ in 0..ITERATIONS {
                    let (mut counter, token) = counter.acquire(base_token);
                    *counter += 1;
                    base_token = counter.release(token);
                }

                epilogue_level.leave(base_token);
            });
            threads.push(handle);
        }

        for handle in threads {
            handle.join().unwrap();
        }

        let mut counter = Arc::into_inner(counter).unwrap();
        assert!(*counter.get_mut() == NUM_EXCLUSIVE * ITERATIONS);
    }

    #[test]
    fn stress_shared() {
        const NUM_EXCLUSIVE: usize = 4;
        const NUM_SHARED: usize = 4;
        const ITERATIONS: usize = 1_000_000;

        let counter = Arc::new(MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0));
        let barrier = Arc::new(Barrier::new(NUM_EXCLUSIVE + NUM_SHARED));

        let mut threads = Vec::new();
        for _ in 0..NUM_EXCLUSIVE {
            let counter = counter.clone();
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                let root_token = unsafe { RootToken::forge() };
                let (epilogue_level, mut base_token) = EpilogueLevel::enter(root_token);

                barrier.wait();
                for _ in 0..ITERATIONS {
                    let (mut counter, token) = counter.acquire(base_token);
                    *counter += 1;
                    base_token = counter.release(token);
                }

                epilogue_level.leave(base_token);
            });
            threads.push(handle);
        }

        for _ in 0..NUM_SHARED {
            let counter = counter.clone();
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                let root_token = unsafe { RootToken::forge() };
                let (epilogue_level, mut base_token) = EpilogueLevel::enter(root_token);

                let mut prev = None;
                barrier.wait();
                for _ in 0..ITERATIONS {
                    let (counter, token) = counter.acquire_shared(base_token);
                    if let Some(prev) = prev {
                        assert!(prev <= *counter);
                    }
                    prev = Some(*counter);
                    base_token = counter.release(token);
                }

                epilogue_level.leave(base_token);
            });
            threads.push(handle);
        }

        for handle in threads {
            handle.join().unwrap();
        }

        let mut counter = Arc::into_inner(counter).unwrap();
        assert!(*counter.get_mut() == NUM_EXCLUSIVE * ITERATIONS);
    }

    /// The uncontended path: an attempt on a free lock behaves exactly like
    /// `acquire`, and the lock is free again after the release.
    #[test]
    fn usage_try_exclusive() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let lock = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let (guard, token) = lock
            .try_acquire(token)
            .unwrap_or_else(|_| panic!("uncontended lock must be free"));
        let token = guard.release(token);

        let (guard, token) = lock
            .try_acquire(token)
            .unwrap_or_else(|_| panic!("lock was released"));
        let token = guard.release(token);

        syscall_level.leave(token);
    }

    /// As `usage_try_exclusive`, in shared mode: the token handed out still
    /// takes a nested shared hold and still descends to lower levels.
    #[test]
    fn usage_try_shared() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let shared = MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0);
        let memory = MemoryRWSpinlock::new(RWSpinlock::new(), 0);

        let (shared_guard, token) = shared
            .try_acquire_shared(token)
            .unwrap_or_else(|_| panic!("uncontended lock must be free"));

        let (nested_guard, token) = shared.acquire_shared_nested(token);

        let (memory_guard, token) = memory
            .try_acquire(token)
            .unwrap_or_else(|_| panic!("uncontended lock must be free"));
        let token = memory_guard.release(token);

        let token = nested_guard.release(token);
        let token = shared_guard.release(token);

        syscall_level.leave(token);
    }

    /// Runs `attempt` on this thread while another thread holds `lock`.
    ///
    /// The holder takes the lock in whatever mode `hold` chooses and calls
    /// the `pause` it is handed while still holding it; `pause` returns only
    /// once `attempt` is done, so the hold provably spans the attempt.
    ///
    /// Contention has to come from a second thread: within one thread the
    /// hierarchy already rules a second hold at the same level out at compile
    /// time, since the first one consumed the token.
    fn while_held<T, Hold, Attempt>(lock: Arc<T>, hold: Hold, attempt: Attempt)
    where
        T: Send + Sync + 'static,
        Hold: FnOnce(&T, &dyn Fn()) + Send + 'static,
        Attempt: FnOnce(&T),
    {
        let held = Arc::new(Barrier::new(2));
        let checked = Arc::new(Barrier::new(2));

        let holder = {
            let lock = lock.clone();
            let held = held.clone();
            let checked = checked.clone();

            thread::spawn(move || {
                hold(&lock, &|| {
                    held.wait();
                    checked.wait();
                })
            })
        };

        held.wait();
        attempt(&lock);
        checked.wait();

        holder.join().unwrap();
    }

    fn hold_exclusive<L>(lock: &Lock<usize, L>, pause: &dyn Fn())
    where
        L: HierarchicalLock,
        L::Id: LockId<Level = level::MemoryManagement>,
    {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let (guard, token) = lock.acquire(token);
        pause();
        let token = guard.release(token);

        epilogue_level.leave(token);
    }

    fn hold_shared<L>(lock: &Lock<usize, L>, pause: &dyn Fn())
    where
        L: HierarchicalLock,
        L::Id: LockId<Level = level::MemoryManagement>,
    {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let (guard, token) = lock.acquire_shared(token);
        pause();
        let token = guard.release(token);

        epilogue_level.leave(token);
    }

    /// A failed attempt hands the caller's token back unchanged — same level,
    /// so it still acquires a *different* lock at that level.
    #[test]
    fn usage_try_exclusive_contended() {
        let lock = Arc::new(MemoryManagementSpinlock::new(Spinlock::new(), 0));

        while_held(lock, hold_exclusive, |lock| {
            let root_token = unsafe { RootToken::forge() };
            let (epilogue_level, token) = EpilogueLevel::enter(root_token);

            let (rejected, token) = match lock.try_acquire(token) {
                Ok((guard, token)) => (false, guard.release(token)),
                Err(token) => (true, token),
            };

            let other = MemoryManagementSpinlock::new(Spinlock::new(), 0);
            let (reusable, token) = match other.try_acquire(token) {
                Ok((guard, token)) => (true, guard.release(token)),
                Err(token) => (false, token),
            };

            epilogue_level.leave(token);

            assert!(rejected, "a held lock must turn an attempt away");
            assert!(reusable, "the token must survive a failed attempt");
        });
    }

    /// A writer turns a writer and a reader away alike.
    #[test]
    fn usage_try_contended_by_writer() {
        let lock = Arc::new(MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0));

        while_held(lock, hold_exclusive, |lock| {
            let root_token = unsafe { RootToken::forge() };
            let (epilogue_level, token) = EpilogueLevel::enter(root_token);

            let (writer_rejected, token) = match lock.try_acquire(token) {
                Ok((guard, token)) => (false, guard.release(token)),
                Err(token) => (true, token),
            };

            let (reader_rejected, token) = match lock.try_acquire_shared(token) {
                Ok((guard, token)) => (false, guard.release(token)),
                Err(token) => (true, token),
            };

            epilogue_level.leave(token);

            assert!(writer_rejected, "a writer excludes a writer");
            assert!(reader_rejected, "a writer excludes a reader");
        });
    }

    /// A reader turns a writer away, but lets another reader in.
    #[test]
    fn usage_try_contended_by_reader() {
        let lock = Arc::new(MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0));

        while_held(lock, hold_shared, |lock| {
            let root_token = unsafe { RootToken::forge() };
            let (epilogue_level, token) = EpilogueLevel::enter(root_token);

            let (writer_rejected, token) = match lock.try_acquire(token) {
                Ok((guard, token)) => (false, guard.release(token)),
                Err(token) => (true, token),
            };

            let (reader_admitted, token) = match lock.try_acquire_shared(token) {
                Ok((guard, token)) => (true, guard.release(token)),
                Err(token) => (false, token),
            };

            epilogue_level.leave(token);

            assert!(writer_rejected, "a reader excludes a writer");
            assert!(reader_admitted, "readers do not exclude each other");
        });
    }

    /// Under contention an attempt either takes the lock or takes nothing:
    /// the counter matches the number of successes exactly.
    #[test]
    fn stress_try_exclusive() {
        const NUM_EXCLUSIVE: usize = 8;
        const ITERATIONS: usize = 100_000;

        let counter = Arc::new(MemoryManagementSpinlock::new(Spinlock::new(), 0));
        let barrier = Arc::new(Barrier::new(NUM_EXCLUSIVE));

        let mut threads = Vec::new();
        for _ in 0..NUM_EXCLUSIVE {
            let counter = counter.clone();
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                let root_token = unsafe { RootToken::forge() };
                let (epilogue_level, mut base_token) = EpilogueLevel::enter(root_token);

                barrier.wait();
                let mut acquired = 0;
                for _ in 0..ITERATIONS {
                    base_token = match counter.try_acquire(base_token) {
                        Ok((mut counter, token)) => {
                            *counter += 1;
                            acquired += 1;
                            counter.release(token)
                        }
                        Err(token) => token,
                    };
                }

                epilogue_level.leave(base_token);
                acquired
            });
            threads.push(handle);
        }

        let acquired: usize = threads.into_iter().map(|h| h.join().unwrap()).sum();

        let mut counter = Arc::into_inner(counter).unwrap();
        assert!(*counter.get_mut() == acquired);
    }

    /// The `Driver` level sits between `Epilogue` and `MemoryManagement`: an
    /// epilogue may take a driver lock, and a driver lock may be held while
    /// descending to the memory levels below.
    #[test]
    fn usage_driver_level() {
        let root_token = unsafe { RootToken::forge() };

        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let driver = DriverSpinlock::new(Spinlock::new(), 0);
        let memory_management = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let (driver_guard, token) = driver.acquire(token);
        let (memory_management_guard, token) = memory_management.acquire(token);

        let token = memory_management_guard.release(token);
        let token = driver_guard.release(token);

        epilogue_level.leave(token);
    }

    /// A context entering at the `Driver` level holds it, so it reaches
    /// everything below — the levels a driver actually works on — and, as with
    /// every other level, not its own.
    #[test]
    fn usage_driver_entry_level() {
        let root_token = unsafe { RootToken::forge() };

        let (driver_level, token) = DriverLevel::enter(root_token);

        let memory_management = MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0);
        let memory = MemorySpinlock::new(Spinlock::new(), 0);

        let (memory_management_guard, token) = memory_management.acquire_shared(token);
        let (memory_guard, token) = memory.acquire(token);

        let token = memory_guard.release(token);
        let token = memory_management_guard.release(token);

        driver_level.leave(token);
    }

    /// The `Thread` level sits below the scheduler: a driver — or anything
    /// else above it — may take a thread lock, and a thread lock may be held
    /// while descending to the memory levels below.
    #[test]
    fn usage_thread_level() {
        let root_token = unsafe { RootToken::forge() };

        let (driver_level, token) = DriverLevel::enter(root_token);

        let thread = ThreadSpinlock::new(Spinlock::new(), 0);
        let memory_management = MemoryManagementSpinlock::new(Spinlock::new(), 0);

        let (thread_guard, token) = thread.acquire(token);
        let (memory_management_guard, token) = memory_management.acquire(token);

        let token = memory_management_guard.release(token);
        let token = thread_guard.release(token);

        driver_level.leave(token);
    }

    /// A context entering at the `Thread` level reaches the levels below it —
    /// allocating a control block is the ordinary case — and, as with every
    /// other level, not its own.
    #[test]
    fn usage_thread_entry_level() {
        let root_token = unsafe { RootToken::forge() };

        let (thread_level, token) = ThreadLevel::enter(root_token);

        let memory_management = MemoryManagementRWSpinlock::new(RWSpinlock::new(), 0);
        let memory = MemorySpinlock::new(Spinlock::new(), 0);

        let (memory_management_guard, token) = memory_management.acquire_shared(token);
        let (memory_guard, token) = memory.acquire(token);

        let token = memory_guard.release(token);
        let token = memory_management_guard.release(token);

        thread_level.leave(token);
    }
}
