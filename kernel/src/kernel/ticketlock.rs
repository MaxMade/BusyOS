use core::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
};

use kernel_derive::lock_id;

use crate::kernel::locking::{HierarchicalLock, Lock, LockId, level};
use crate::kernel::prologue_lock::PrologueLock;

#[lock_id(Driver)]
pub struct TicketlockDriverID;

#[lock_id(Thread)]
pub struct TicketlockThreadID;

#[lock_id(MemoryManagement)]
pub struct TicketlockMemoryManagementID;

#[lock_id(Memory)]
pub struct TicketlockMemoryID;

#[lock_id(Prologue)]
pub struct TicketlockPrologueID;

pub struct Ticketlock<Id: LockId> {
    ticket: AtomicUsize,
    serving: AtomicUsize,
    _id: PhantomData<Id>,
}

impl<Id: LockId> Ticketlock<Id> {
    pub const fn new() -> Self {
        Self {
            ticket: AtomicUsize::new(0),
            serving: AtomicUsize::new(0),
            _id: PhantomData,
        }
    }
}

impl<Id: LockId> HierarchicalLock for Ticketlock<Id> {
    type Id = Id;

    unsafe fn raw_lock(&self) {
        let ticket = self.ticket.fetch_add(1, AtomicOrdering::Relaxed);
        while self.serving.load(AtomicOrdering::Acquire) != ticket {
            core::hint::spin_loop();
        }
    }
    unsafe fn raw_unlock(&self) {
        self.serving.fetch_add(1, AtomicOrdering::Release);
    }
    unsafe fn raw_lock_shared(&self) {
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

/// Driver-level ticketlock.
///
/// TODO(@MaxMade): a lock at [`level::Driver`] is meant to be a blocking
/// lock — a driver sleeps while it waits for its device — and none exists
/// yet, so this spinning one stands in. Until it is replaced, a hold must
/// not sleep: every waiter spins for as long as it does.
pub type DriverTicketlock<T> = Lock<T, Ticketlock<TicketlockDriverID>>;

/// Thread-level ticketlock.
///
/// [`level::Thread`] is below the scheduler, so this is a spinning lock in
/// earnest: a hold may not sleep, and the scheduler may take one while it
/// holds its own lock.
pub type ThreadTicketlock<T> = Lock<T, Ticketlock<TicketlockThreadID>>;

pub type MemoryManagementTicketlock<T> = Lock<T, Ticketlock<TicketlockMemoryManagementID>>;

pub type MemoryTicketlock<T> = Lock<T, Ticketlock<TicketlockMemoryID>>;

/// Prologue-level ticketlock. Masks interrupts for the duration of the hold.
pub type PrologueTicketlock<T> = PrologueLock<T, Ticketlock<TicketlockPrologueID>>;

#[lock_id(Driver)]
pub struct RWTicketlockDriverID;

#[lock_id(Thread)]
pub struct RWTicketlockThreadID;

#[lock_id(MemoryManagement)]
pub struct RWTicketlockMemoryManagementID;

#[lock_id(Memory)]
pub struct RWTicketlockMemoryID;

#[lock_id(Prologue)]
pub struct RWTicketlockPrologueID;

pub struct RWTicketlock<Id: LockId> {
    ticket: AtomicUsize,
    serving: AtomicUsize,
    state: AtomicUsize,
    _id: PhantomData<Id>,
}

impl<Id: LockId> RWTicketlock<Id> {
    pub const fn new() -> Self {
        Self {
            ticket: AtomicUsize::new(0),
            serving: AtomicUsize::new(0),
            state: AtomicUsize::new(0),
            _id: PhantomData,
        }
    }

    #[inline]
    fn take_ticket(&self) -> usize {
        self.ticket.fetch_add(1, AtomicOrdering::Relaxed)
    }

    #[inline]
    fn wait_turn(&self, ticket: usize) {
        while self.serving.load(AtomicOrdering::Acquire) != ticket {
            core::hint::spin_loop();
        }
    }

    #[inline]
    fn try_join_batch(&self) -> bool {
        let mut state = self.state.load(AtomicOrdering::Relaxed);
        loop {
            if state == 0 || state == usize::MAX {
                return false;
            }
            match self.state.compare_exchange_weak(
                state,
                state + 1,
                AtomicOrdering::Acquire, // entering a critical section
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(updated) => state = updated,
            }
        }
    }
}

impl<Id: LockId> HierarchicalLock for RWTicketlock<Id> {
    type Id = Id;

    unsafe fn raw_lock(&self) {
        let ticket = self.take_ticket();

        self.wait_turn(ticket);

        while let Err(_) = self.state.compare_exchange_weak(
            0,
            usize::MAX,
            AtomicOrdering::Acquire,
            AtomicOrdering::Relaxed,
        ) {
            core::hint::spin_loop();
        }
    }
    unsafe fn raw_unlock(&self) {
        self.state.store(0, AtomicOrdering::Relaxed);

        self.serving.fetch_add(1, AtomicOrdering::Release);
    }
    unsafe fn raw_lock_shared(&self) {
        if self.try_join_batch() {
            return;
        }

        let ticket = self.take_ticket();

        self.wait_turn(ticket);

        self.state.fetch_add(1, AtomicOrdering::Acquire);

        self.serving.fetch_add(1, AtomicOrdering::Release);
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

/// Driver-level reader-writer ticketlock.
///
/// TODO(@MaxMade): a lock at [`level::Driver`] is meant to be a blocking
/// lock — a driver sleeps while it waits for its device — and none exists
/// yet, so this spinning one stands in. Until it is replaced, a hold must
/// not sleep: every waiter spins for as long as it does.
pub type DriverRWTicketlock<T> = Lock<T, RWTicketlock<RWTicketlockDriverID>>;

/// Thread-level reader-writer ticketlock.
///
/// As [`ThreadTicketlock`], a spinning lock: see [`level::Thread`].
pub type ThreadRWTicketlock<T> = Lock<T, RWTicketlock<RWTicketlockThreadID>>;

pub type MemoryManagementRWTicketlock<T> = Lock<T, RWTicketlock<RWTicketlockMemoryManagementID>>;

pub type MemoryRWTicketlock<T> = Lock<T, RWTicketlock<RWTicketlockMemoryID>>;

/// Prologue-level reader-writer ticketlock. Masks interrupts for the duration
/// of the hold.
pub type PrologueRWTicketlock<T> = PrologueLock<T, RWTicketlock<RWTicketlockPrologueID>>;

#[cfg(test)]
mod test {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use crate::kernel::locking::{
        CanAcquire, DriverLevel, EpilogueLevel, PreviousToken, RootToken, SyscallLevel,
    };

    use super::*;

    extern crate std;

    #[test]
    fn usage_exclusive() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementTicketlock::new(Ticketlock::new(), 0);

        let memory = MemoryTicketlock::new(Ticketlock::new(), 0);

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

        let epilogue = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);

        let memory = MemoryRWTicketlock::new(RWTicketlock::new(), 0);

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
        let memory = MemoryTicketlock::new(Ticketlock::new(), 0);
        let (memory_guard, token) = memory.acquire(token);

        memory_guard.release(token)
    }

    #[test]
    fn usage_exclusive_subroutine() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementTicketlock::new(Ticketlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire(token);

        let token = do_memory_work(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn usage_shared_subroutine() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire_shared(token);

        let token = do_memory_work(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn stress_exclusive() {
        const NUM_EXCLUSIVE: usize = 8;
        const ITERATIONS: usize = 1_000_000;

        let counter = Arc::new(MemoryManagementTicketlock::new(Ticketlock::new(), 0));
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

        let counter = Arc::new(MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0));
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

    /// The `Driver` level sits between `Epilogue` and `MemoryManagement`, and
    /// a ticketlock is available at it like at the levels below.
    #[test]
    fn usage_driver_level() {
        let root_token = unsafe { RootToken::forge() };

        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let driver = DriverTicketlock::new(Ticketlock::new(), 0);
        let memory_management = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);

        let (driver_guard, token) = driver.acquire(token);
        let (memory_management_guard, token) = memory_management.acquire(token);

        let token = memory_management_guard.release(token);
        let token = driver_guard.release(token);

        epilogue_level.leave(token);
    }

    /// The `Thread` level sits below the scheduler, and a ticketlock is
    /// available at it like at the levels below.
    #[test]
    fn usage_thread_level() {
        let root_token = unsafe { RootToken::forge() };

        let (driver_level, token) = DriverLevel::enter(root_token);

        let thread = ThreadTicketlock::new(Ticketlock::new(), 0);
        let memory_management = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);

        let (thread_guard, token) = thread.acquire(token);
        let (memory_management_guard, token) = memory_management.acquire(token);

        let token = memory_management_guard.release(token);
        let token = thread_guard.release(token);

        driver_level.leave(token);
    }
}
