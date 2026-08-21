use core::ffi::c_void;

use busyos::{
    arch::{
        generic::paging::{Error as PagingError, PageFrameAllocator, PageSize, PhysicalAddress},
    },
    kernel::locking::{CanAcquire, PreviousToken, level::MemoryManagement},
};

/// A static [`PageFrameAllocator`].
pub struct UEFIPageFrameAllocator;

impl PageFrameAllocator for UEFIPageFrameAllocator {
    fn allocate<Token>(
        page_size: PageSize,
        token: Token,
    ) -> Result<(PhysicalAddress<c_void>, Token), (PagingError, Token)>
    where
        Token: CanAcquire<MemoryManagement> + PreviousToken,
    {
        // UEFI offers (seemingly) only 4 KiB page frames
        if page_size != PageSize::Regular {
            return Err((PagingError::OutOfMemory, token));
        }

        let phys_addr = match uefi::boot::allocate_pages(
            uefi::boot::AllocateType::AnyPages,
            uefi::boot::MemoryType::LOADER_DATA,
            1,
        ) {
            Ok(mem) => PhysicalAddress::new(mem.as_ptr().cast()),
            Err(_) => {
                return Err((PagingError::OutOfMemory, token));
            }
        };

        Ok((phys_addr, token))
    }

    unsafe fn deallocate<Token>(_: PhysicalAddress<c_void>, _: PageSize, token: Token) -> Token
    where
        Token: CanAcquire<MemoryManagement> + PreviousToken,
    {
        // XXX: Leak memory...
        //
        // The `UEFIPageFrameAllocator` is only available as part of the boot
        // services. Therefore, we even try to free the pages...

        token
    }
}

pub type Paging = busyos::arch::Paging<UEFIPageFrameAllocator>;

pub fn prepare_handover<Token>(token: Token) -> (Paging, Token)
where
    Token: CanAcquire<MemoryManagement> + PreviousToken,
{
    match unsafe { Paging::temporary_upper_half(token) } {
        Ok((paging, token)) => (paging, token),
        Err((error, _)) => {
            panic!("Unable to prepare page tables for hand-over: {}", error);
        }
    }
}
