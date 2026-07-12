//! Common helpers for testing

use core::{alloc::Layout, ptr::NonNull};

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    utils::allocator::{Allocator, Error as AllocatorError},
};

/// Heap-backed allocator for testing using the token system.
pub struct HeapAllocator;

impl HeapAllocator {
    /// Create a new [`HeapAllocator`].
    pub const fn new() -> Self {
        Self
    }
}

unsafe impl Allocator<MemoryManagementLevelID> for HeapAllocator {
    fn allocate<Token>(
        &self,
        layout: Layout,
        token: Token,
    ) -> Result<(NonNull<u8>, Token), (AllocatorError, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        match unsafe { NonNull::new(std::alloc::alloc(layout)) } {
            Some(ptr) => Ok((ptr, token)),
            None => Err((AllocatorError::OutOfMemory, token)),
        }
    }

    unsafe fn deallocate<Token>(&self, ptr: NonNull<u8>, layout: Layout, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        token
    }
}
