//! Relevant traits to enforce strict locking rules (mostly) at compile time.

use kernel_derive::{Locking, lock_id};

/// Base lock level hierarchy
#[derive(Locking)]
pub enum Level {
    Syscall,
    Epilogue,
    MemoryManagement,
    Prologue,
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
