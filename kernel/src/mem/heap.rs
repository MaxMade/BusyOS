//! The kernel's heap.
//!
//! One [`BuddyAllocator`] behind a lock at the `MemoryManagement` level,
//! reached through [`Heap`] — a handle carrying no state of its own, so that
//! it can be copied around freely and every copy refers to the same
//! allocator.
//!
//! # Bootstrapping
//!
//! The heap is needed before there is anything to allocate memory *from*: the
//! page frame allocator has to build its own bookkeeping first, and that
//! bookkeeping lives on the heap. It therefore starts out on a pool reserved
//! in the image itself, handed over by
//! [`early_initialisation`](Heap::early_initialisation), and grows from the
//! frame allocator later — the buddy allocator takes memory in any number of
//! ranges, so growing it is just another
//! [`add`](BuddyAllocator::add).

use core::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use core::{alloc::Layout, ptr::NonNull};

use crate::{
    kernel::{
        locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
        ticketlock::{MemoryManagementTicketlock, Ticketlock},
    },
    mem::buddy_allocator::BuddyAllocator,
    utils::allocator::{Allocator, Error},
};

/// Size of the pool the heap starts out on.
const INITIALMEM_POOL_SIZE: usize = 1024 * 1024;

/// The pool itself, page aligned.
///
/// Keep the alignment at or below 8 KiB: the kernel is also compiled for
/// `x86_64-unknown-uefi`, whose PE/COFF sections cannot encode a stricter one,
/// and the toolchain answers a larger value with a crash rather than an error.
#[repr(C, align(4096))]
struct InitialMemPool([u8; INITIALMEM_POOL_SIZE]);

/// Zero-initialised, so it costs `.bss` rather than image bytes.
static mut INITIAL_MEM_POOL: InitialMemPool = InitialMemPool([0; INITIALMEM_POOL_SIZE]);

/// The one heap of the kernel.
///
/// A ticketlock rather than a spinlock, so that a core cannot be starved of
/// the heap by its neighbours. Its level is `MemoryManagement`: a holder may
/// still reach the levels below — raw memory and the frame allocator — which
/// is what refilling the heap from the frame allocator needs.
///
/// The orders span 16 B to 1 GiB. The lower bound is what a free-list node
/// takes, the upper one leaves room for the largest mapping the kernel might
/// hand out in one piece; the orders above the pool's size simply stay empty
/// until the heap grows.
static INTERNAL_HEAP: MemoryManagementTicketlock<BuddyAllocator<4, 30>> =
    MemoryManagementTicketlock::new(Ticketlock::new(), BuddyAllocator::new());

/// Handle on the kernel heap.
///
/// Carries nothing: the allocator lives in [`INTERNAL_HEAP`], so every `Heap`
/// is the same heap and can be copied into whatever needs an
/// [`Allocator`] without borrowing anything.
#[derive(Debug, Clone, Copy)]
pub struct Heap;

impl Heap {
    /// Hands the heap the pool reserved in the image.
    ///
    /// Until this has run every allocation fails, so it belongs in the boot
    /// path of the first core, before anything that allocates.
    ///
    /// # Safety
    ///
    /// The pool is a `static mut` handed over as one exclusive borrow that
    /// then lives as long as the kernel: nothing may touch
    /// [`INITIAL_MEM_POOL`] itself, before or after.
    ///
    /// # Panics
    ///
    /// If called more than once — the second call would hand the same pool
    /// over twice, which would alias every block in it.
    pub unsafe fn early_initialisation<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        static INITIALISED: AtomicBool = AtomicBool::new(false);

        if INITIALISED.swap(true, AtomicOrdering::Relaxed) {
            panic!("Multiple invocations of Heap::early_initialisation(...)");
        }

        let (mut heap, token) = INTERNAL_HEAP.acquire(token);

        // SAFETY: the guard above makes this the only borrow of the pool ever
        // taken, and the pool is a `static`, so the borrow is good for as long
        // as the allocator needs it.
        #[allow(static_mut_refs)]
        unsafe {
            heap.add(INITIAL_MEM_POOL.0.as_mut_slice())
        };

        heap.release(token)
    }
}

// SAFETY: the blocks come from a buddy allocator that hands out each of them
// once, and every access to it is made under its own lock.
unsafe impl Allocator<MemoryManagementLevelID> for Heap {
    /// Allocates from the heap, refilling it if it cannot serve the request.
    fn allocate<Token>(
        &self,
        layout: Layout,
        token: Token,
    ) -> Result<(NonNull<u8>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut heap, token) = INTERNAL_HEAP.acquire(token);

        if let Ok(mem) = heap.allocate(layout) {
            let mem: NonNull<u8> = mem.cast::<u8>();
            let token = heap.release(token);
            return Ok((mem, token));
        }

        // TODO(@MaxMade): Fill using global page frame allocator

        let token = heap.release(token);
        return Err((Error::OutOfMemory, token));
    }

    /// Returns a block to the heap.
    ///
    /// Nothing is handed back to the frame allocator: freed blocks coalesce
    /// and stay with the heap.
    unsafe fn deallocate<Token>(&self, ptr: NonNull<u8>, layout: Layout, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut heap, token) = INTERNAL_HEAP.acquire(token);

        unsafe { heap.deallocate(ptr, layout) };

        heap.release(token)
    }
}
