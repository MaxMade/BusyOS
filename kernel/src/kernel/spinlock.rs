use core::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
};

use kernel_derive::lock_id;

use crate::kernel::locking::{HierarchicalLock, Lock, LockId, level};
use crate::kernel::prologue_lock::PrologueLock;

#[lock_id(Driver)]
pub struct SpinlockDriverID;

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
    unsafe fn raw_unlock(&self) {
        self.state.store(false, AtomicOrdering::Release);
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

/// Driver-level spinlock.
///
/// TODO(@MaxMade): a lock at [`level::Driver`] is meant to be a blocking
/// lock — a driver sleeps while it waits for its device — and none exists
/// yet, so this spinning one stands in. Until it is replaced, a hold must
/// not sleep: every waiter spins for as long as it does.
pub type DriverSpinlock<T> = Lock<T, Spinlock<SpinlockDriverID>>;

pub type MemoryManagementSpinlock<T> = Lock<T, Spinlock<SpinlockMemoryManagementID>>;

pub type MemorySpinlock<T> = Lock<T, Spinlock<SpinlockMemoryID>>;

/// Prologue-level spinlock. Masks interrupts for the duration of the hold.
pub type PrologueSpinlock<T> = PrologueLock<T, Spinlock<SpinlockPrologueID>>;

#[lock_id(Driver)]
pub struct RWSpinlockDriverID;

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
        CanAcquire, DriverLevel, EpilogueLevel, PreviousToken, RootToken, SyscallLevel,
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
}
