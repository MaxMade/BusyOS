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
    unsafe fn raw_try_lock(&self) -> bool {
        // The lock is free exactly when nobody is queued ahead of us, i.e.
        // when the next ticket handed out is the one currently being served.
        // Claiming that ticket both takes the lock and keeps the queue
        // consistent, so `raw_unlock` needs no special case.
        let serving = self.serving.load(AtomicOrdering::Acquire);

        self.ticket
            .compare_exchange(
                serving,
                serving + 1,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
    }
    unsafe fn raw_unlock(&self) {
        self.serving.fetch_add(1, AtomicOrdering::Release);
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

    /// Claims the turn only if the queue is empty, i.e. if the next ticket
    /// to be handed out is the one currently being served. Never waits.
    ///
    /// A caller that succeeds owns the turn exactly as if it had taken a
    /// ticket and waited for it, and must hand it on — through `raw_unlock`,
    /// or by bumping `serving` itself if it gives up.
    #[inline]
    fn try_take_turn(&self) -> bool {
        let serving = self.serving.load(AtomicOrdering::Acquire);

        self.ticket
            .compare_exchange(
                serving,
                serving + 1,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
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
    unsafe fn raw_try_lock(&self) -> bool {
        if !self.try_take_turn() {
            return false;
        }

        if self
            .state
            .compare_exchange(
                0,
                usize::MAX,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
        {
            return true;
        }

        // The turn is ours but readers of an earlier batch are still inside.
        // `raw_lock` would spin them out; give the turn up instead so the
        // next waiter is not stuck behind an attempt that never took hold.
        self.serving.fetch_add(1, AtomicOrdering::Release);

        false
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
    unsafe fn raw_try_lock_shared(&self) -> bool {
        if self.try_join_batch() {
            return true;
        }

        if !self.try_take_turn() {
            return false;
        }

        // Owning the turn rules out a writer: one holds `state` at
        // `usize::MAX` only between taking its own turn and releasing it, so
        // the count can be joined unconditionally, exactly as in
        // `raw_lock_shared`.
        self.state.fetch_add(1, AtomicOrdering::Acquire);

        self.serving.fetch_add(1, AtomicOrdering::Release);

        true
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

    /// The uncontended path: an attempt on a free lock behaves exactly like
    /// `acquire`, and the lock is free again after the release.
    #[test]
    fn usage_try_exclusive() {
        let root_token = unsafe { RootToken::forge() };

        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let lock = MemoryManagementTicketlock::new(Ticketlock::new(), 0);

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

        let shared = MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0);
        let memory = MemoryRWTicketlock::new(RWTicketlock::new(), 0);

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
        let lock = Arc::new(MemoryManagementTicketlock::new(Ticketlock::new(), 0));

        while_held(lock, hold_exclusive, |lock| {
            let root_token = unsafe { RootToken::forge() };
            let (epilogue_level, token) = EpilogueLevel::enter(root_token);

            let (rejected, token) = match lock.try_acquire(token) {
                Ok((guard, token)) => (false, guard.release(token)),
                Err(token) => (true, token),
            };

            let other = MemoryManagementTicketlock::new(Ticketlock::new(), 0);
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
        let lock = Arc::new(MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0));

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
        let lock = Arc::new(MemoryManagementRWTicketlock::new(RWTicketlock::new(), 0));

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
