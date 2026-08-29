use core::ffi::c_void;

use crate::{
    arch::{
        REGULAR_PAGE_SIZE,
        generic::paging::{Error as PagingError, PageFrameAllocator, PhysicalAddress},
    },
    kernel::{
        bootinfo::BOOTINFO,
        locking::{CanAcquire, PreviousToken, level},
        spinlock::{MemorySpinlock, Spinlock},
    },
};

const EARLY_PAGE_FRAMES_SIZE: usize = 1024 * REGULAR_PAGE_SIZE;

#[repr(C, align(4096))]
pub struct EarlyPageFrames {
    pages: [u8; EARLY_PAGE_FRAMES_SIZE],
    offset: usize,
}

static EARLY_PAGE_FRAMES: MemorySpinlock<EarlyPageFrames> = MemorySpinlock::new(
    Spinlock::new(),
    EarlyPageFrames {
        pages: [0; EARLY_PAGE_FRAMES_SIZE],
        offset: 0,
    },
);

pub struct EarlyPageFrameAllocator;

impl PageFrameAllocator for EarlyPageFrames {
    fn allocate<Token>(
        token: Token,
    ) -> Result<(PhysicalAddress<c_void>, Token), (PagingError, Token)>
    where
        Token: CanAcquire<level::Memory> + PreviousToken,
    {
        let phys_virt_shift = unsafe { BOOTINFO.assume_init_ref().kernel_virt_phys_offset };

        let (mut early_page_frames, token) = EARLY_PAGE_FRAMES.acquire(token);
        let addr = early_page_frames.pages.as_ptr().addr();
        let offset = early_page_frames.offset;
        early_page_frames.offset = early_page_frames.offset.saturating_add(REGULAR_PAGE_SIZE);
        let token = early_page_frames.release(token);

        if offset >= EARLY_PAGE_FRAMES_SIZE {
            return Err((PagingError::OutOfMemory, token));
        }

        let phys_addr = PhysicalAddress::new((addr + offset - phys_virt_shift) as _);

        Ok((phys_addr, token))
    }

    unsafe fn deallocate<Token>(phys_addr: PhysicalAddress<c_void>, _: Token) -> Token
    where
        Token: CanAcquire<level::Memory> + PreviousToken,
    {
        panic!("Early page frames must never be freed: {:p}", phys_addr);
    }
}
