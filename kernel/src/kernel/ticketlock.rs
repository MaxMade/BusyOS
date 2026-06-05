use core::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
};

use kernel_derive::lock_id;

use crate::kernel::locking::{HierarchicalLock, Lock, LockId, level};

#[lock_id(MemoryManagement)]
pub struct TicketlockMemoryManagementID;

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

pub type MemoryManagementTicketlock<T> = Lock<T, Ticketlock<TicketlockMemoryManagementID>>;

pub type PrologueTicketlock<T> = Lock<T, Ticketlock<TicketlockPrologueID>>;

#[lock_id(MemoryManagement)]
pub struct RWTicketlockMemoryManagementID;

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
        self.state.fetch_add(1, AtomicOrdering::Release);
    }

    unsafe fn raw_unlock_shared_nested(&self) {
        self.state.fetch_sub(1, AtomicOrdering::Release);
    }
}

pub type MemoryManagementRWTicketlock<T> = Lock<T, RWTicketlock<RWTicketlockMemoryManagementID>>;

pub type PrologueRWTicketlock<T> = Lock<T, RWTicketlock<RWTicketlockPrologueID>>;

#[cfg(test)]
mod test {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use crate::kernel::locking::{
        CanAcquire, EpilogueLevel, PreviousToken, RootToken, SyscallLevel,
    };

    use super::*;

    extern crate std;

    #[test]
    fn usage_exclusive() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementTicketlock::new(Ticketlock::new(), 0);

        let prologue = PrologueTicketlock::new(Ticketlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire(token);

        let (prologue_guard, token) = prologue.acquire(token);

        let token = prologue_guard.release(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn usage_shared() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);

        let prologue = PrologueRWTicketlock::new(RWTicketlock::new(), 0);

        let (epilogue_guard_0, token) = epilogue.acquire_shared(token);

        let (epilogue_guard_1, token) = epilogue.acquire_shared_nested(token);

        let (prologue_guard, token) = prologue.acquire(token);

        let token = prologue_guard.release(token);

        let token = epilogue_guard_1.release(token);

        let token = epilogue_guard_0.release(token);

        syscall_level.leave(token);
    }

    fn do_prologue_work<From>(token: From) -> From
    where
        From: CanAcquire<level::Prologue> + PreviousToken,
    {
        let prologue = PrologueTicketlock::new(Ticketlock::new(), 0);
        let (prologue_guard, token) = prologue.acquire(token);

        prologue_guard.release(token)
    }

    #[test]
    fn usage_exclusive_subroutine() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementTicketlock::new(Ticketlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire(token);

        let token = do_prologue_work(token);

        let token = epilogue_guard.release(token);

        syscall_level.leave(token);
    }

    #[test]
    fn usage_shared_subroutine() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let epilogue = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);

        let (epilogue_guard, token) = epilogue.acquire_shared(token);

        let token = do_prologue_work(token);

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
}
