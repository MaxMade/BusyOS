//! Relevant traits to enforce strict locking rules (mostly) at compile time.

use kernel_derive::{Locking, lock_id};

/// Base lock level hierarchy
#[derive(Locking)]
pub enum Level {
    // System initialisation, from the bootstrap stub until the first thread
    // is scheduled. The top of the hierarchy, so init code may acquire
    // anything; no lock of its own lives here, and nothing can acquire it.
    Init,
    Syscall,
    // Top half of interrupt handling.
    Epilogue,
    // Device drivers, e.g. a driver's own state or the device it talks to.
    //
    // Below `Epilogue`, so the top half of an interrupt may take a driver's
    // lock to hand a completed transfer over, and above `Scheduler`, so a
    // driver may sleep while it holds one — waiting for a device is the
    // ordinary case. Per the note below, that makes a lock at this level a
    // blocking one.
    //
    // TODO(@MaxMade): no blocking lock exists yet, so the `Driver` locks in
    // `spinlock` and `ticketlock` stand in for one. Until they are replaced, a
    // hold at this level must not sleep after all: every waiter spins for as
    // long as it does.
    Driver,
    // The scheduler's own lock, and the boundary between blocking and
    // spinning locks.
    //
    // Sleeping *is* acquiring this lock, so a blocking operation is written
    // `fn sleep<T: CanAcquire<level::Scheduler>>(token: T) -> T`. Everything
    // strictly above may therefore sleep and is a blocking lock (Mutex); this
    // level and everything below may not and is a spinning lock (Spinlock,
    // Ticketlock).
    Scheduler,
    // Thread state, e.g. a thread's control block or the table they live in.
    //
    // Below `Scheduler`, so the scheduler may take a thread's lock while it
    // holds its own — picking the next thread reads and writes the thread it
    // picks — and above `MemoryManagement`, so a control block may be
    // allocated or freed under one. Per the note above, a lock at this level
    // is a spinning one: a hold may not sleep, which is what keeps a thread
    // from being descheduled while holding the state its waker needs. Code
    // that has to block therefore takes the scheduler's lock first.
    Thread,
    // Memory Management, e.g. creating/removing page mappings or
    // allocating/freeing heap memory.
    MemoryManagement,
    /// Raw memory, e.g. allocating physical memory ranges for paging or
    /// using the heap. Sits below `MemoryManagement`, which builds on it.
    Memory,
    // Bottom half of interrupt handling.
    Prologue,
}

#[lock_id(Init)]
pub struct InitLevelID;

pub struct InitLevel;

impl InitLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<InitLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<InitLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for InitLevel {
    fn drop(&mut self) {
        panic!("Init level must never be left implicitly! Use InitLevel::leave(...) instead!");
    }
}

#[lock_id(Syscall)]
pub struct SyscallLevelID;

pub struct SyscallLevel;

impl SyscallLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<SyscallLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<SyscallLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for SyscallLevel {
    fn drop(&mut self) {
        panic!(
            "Syscall level must never be left implicitly! Use SyscallLevel::leave(...) instead!"
        );
    }
}

#[lock_id(Epilogue)]
pub struct EpilogueLevelID;

pub struct EpilogueLevel;

impl EpilogueLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<EpilogueLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<EpilogueLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for EpilogueLevel {
    fn drop(&mut self) {
        panic!(
            "Epilogue level must never be left implicitly! Use EpilogueLevel::leave(...) instead!"
        );
    }
}

#[lock_id(Driver)]
pub struct DriverLevelID;

pub struct DriverLevel;

impl DriverLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<DriverLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<DriverLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for DriverLevel {
    fn drop(&mut self) {
        panic!("Driver level must never be left implicitly! Use DriverLevel::leave(...) instead!");
    }
}

#[lock_id(Thread)]
pub struct ThreadLevelID;

pub struct ThreadLevel;

impl ThreadLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<ThreadLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<ThreadLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for ThreadLevel {
    fn drop(&mut self) {
        panic!("Thread level must never be left implicitly! Use ThreadLevel::leave(...) instead!");
    }
}

#[lock_id(MemoryManagement)]
pub struct MemoryManagementLevelID;

pub struct MemoryManagementLevel;

impl MemoryManagementLevel {
    pub fn enter(
        root_token: RootToken,
    ) -> (Self, Token<MemoryManagementLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<MemoryManagementLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for MemoryManagementLevel {
    fn drop(&mut self) {
        panic!(
            "MemoryManagement level must never be left implicitly! Use MemoryManagementLevel::leave(...) instead!"
        );
    }
}

#[lock_id(Memory)]
pub struct MemoryLevelID;

pub struct MemoryLevel;

impl MemoryLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<MemoryLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<MemoryLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for MemoryLevel {
    fn drop(&mut self) {
        panic!("Memory level must never be left implicitly! Use MemoryLevel::leave(...) instead!");
    }
}

#[lock_id(Prologue)]
pub struct PrologueLevelID;

pub struct PrologueLevel;

impl PrologueLevel {
    pub fn enter(root_token: RootToken) -> (Self, Token<PrologueLevelID, RootToken, Shared>) {
        core::mem::forget(root_token);

        let token = unsafe { Token::forge() };
        (Self, token)
    }

    pub fn leave(self, token: Token<PrologueLevelID, RootToken, Shared>) {
        core::mem::forget(self);
        core::mem::forget(token);
    }
}

impl Drop for PrologueLevel {
    fn drop(&mut self) {
        panic!(
            "Prologue level must never be left implicitly! Use PrologueLevel::leave(...) instead!"
        );
    }
}
