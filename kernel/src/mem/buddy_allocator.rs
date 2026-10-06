//! Buddy allocator over the memory it is handed.
//!
//! # Overview
//!
//! Memory is carved into blocks whose sizes are the powers of two from
//! `2^MIN_SIZE_LOG` up to `2^MAX_SIZE_LOG2`; a block of size `2^(MIN+k)` is
//! said to be of *order* `k`. Every free block sits in the free list of its
//! order, and a request is served from the smallest order that can hold it,
//! splitting larger blocks in half until one of the right size falls out.
//!
//! Freeing walks the same ladder back up: a block whose *buddy* — the other
//! half of the block they were split from — is also free merges with it into
//! one block of the next order, repeatedly, so that a run of frees leaves
//! memory as unfragmented as it started.
//!
//! Two halves of a block of size `s` differ in exactly the bit `s` of their
//! address, which is what makes both directions cheap to name: the buddy of
//! the block at `addr` is the one at `addr ^ s`. A block of size `s` always
//! starts on a multiple of `s`, which is what that identity rests on, and
//! [`add`](BuddyAllocator::add) is careful to cut only such blocks.
//!
//! # Metadata
//!
//! The allocator keeps nothing but the free-list heads: no bitmap, no bounds,
//! no counters. The links live *inside* the free blocks — a free block has no
//! other use for its bytes — which is why `MIN_SIZE_LOG` must leave room for a
//! [`BuddyAllocatorNode`].
//!
//! The price is that "is my buddy free?" is answered by walking the free list
//! of that order, so freeing costs time linear in the number of free blocks
//! rather than constant time. In exchange the allocator holds no memory of its
//! own, needs no bounds fixed up front, and so takes memory in any number of
//! ranges, wherever they happen to lie.
//!
//! # Handing over memory
//!
//! Memory arrives through [`add`](BuddyAllocator::add), one range at a time
//! and at any point in the allocator's life, which lets a memory map be fed in
//! piece by piece as it becomes known. Ranges that turn out to be neighbours
//! merge as if they had arrived together.

use core::alloc::Layout;
use core::fmt::{Display, Formatter, Result as FmtResult};
use core::ptr::NonNull;

use crate::user::errno::{Errno, ToErrno};

type BuddyAllocatorNodePtr = Option<NonNull<BuddyAllocatorNode>>;

const fn null() -> BuddyAllocatorNodePtr {
    None
}

/// Links of one free block, stored in the block itself.
///
/// Doubly linked so that coalescing can take a block out of the middle of its
/// order's list, which is where the buddy usually sits.
struct BuddyAllocatorNode {
    prev: BuddyAllocatorNodePtr,
    next: BuddyAllocatorNodePtr,
}

impl BuddyAllocatorNode {
    pub const fn new() -> Self {
        BuddyAllocatorNode {
            prev: null(),
            next: null(),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    /// The memory allocator has no suitable free memory range.
    OutOfMemory,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Error::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn core::error::Error> {
        self.source()
    }
}

impl ToErrno for Error {
    fn to_errno(&self) -> Errno {
        match self {
            Error::OutOfMemory => Errno::ENOMEM,
        }
    }
}

/// Upper bound on the number of orders, so that `heads` needs no arithmetic on
/// the const parameters — an array length may not depend on one.
///
/// A block of `2^64` bytes cannot be addressed anyway, so nothing is lost.
const MAX_ORDERS: usize = usize::BITS as usize;

/// A buddy allocator for blocks of `2^MIN_SIZE_LOG` up to `2^MAX_SIZE_LOG2`
/// bytes.
///
/// See the [module documentation](self) for how it works.
pub struct BuddyAllocator<const MIN_SIZE_LOG: usize, const MAX_SIZE_LOG2: usize> {
    /// Free list per order; index `k` holds the blocks of order `k`.
    heads: [BuddyAllocatorNodePtr; MAX_ORDERS],
}

impl<const MIN_SIZE_LOG: usize, const MAX_SIZE_LOG2: usize>
    BuddyAllocator<MIN_SIZE_LOG, MAX_SIZE_LOG2>
{
    /// Number of orders, i.e. the number of free lists in use.
    const ORDERS: usize = MAX_SIZE_LOG2 - MIN_SIZE_LOG + 1;

    /// Size of a block of the lowest order.
    const MIN_SIZE: usize = 1 << MIN_SIZE_LOG;

    /// Creates an allocator with no memory.
    ///
    /// Every allocation fails with [`Error::OutOfMemory`] until memory is
    /// handed over with [`add`](Self::add) — the split that lets the allocator
    /// live in a `static` from the start and be given memory once the memory
    /// map is known.
    pub const fn new() -> Self {
        Self {
            heads: [null(); MAX_ORDERS],
        }
    }

    /// Creates an allocator holding `mem`.
    ///
    /// # Panics
    ///
    /// As [`add`](Self::add).
    pub fn new_with(mem: &'static mut [u8]) -> Self {
        let mut allocator = Self::new();
        allocator.add(mem);
        allocator
    }

    /// Hands `mem` over to the allocator.
    ///
    /// The range is cut into the largest blocks it can hold — greedily from
    /// its start, and only ever a block that starts on a multiple of its own
    /// size, which is the invariant the buddy arithmetic rests on — and each
    /// of them is handed to the free path. Going through
    /// [`deallocate`](Self::deallocate) rather than straight into the lists is
    /// what makes a range merge with whatever is already free next to it: a
    /// range added beside an earlier one leaves blocks as large as if the two
    /// had been added at once.
    ///
    /// Up to `2^MIN_SIZE_LOG - 1` bytes at either end can be left out, since a
    /// block has to start on a multiple of its size and fit whole.
    ///
    /// Taking `&'static mut [u8]` is what keeps this safe: the allocator ends
    /// up owning the bytes it hands out, and the same range cannot be given
    /// twice.
    ///
    /// # Panics
    ///
    /// If the orders are not a sensible range, or if a block of the lowest
    /// order could not hold a [`BuddyAllocatorNode`].
    pub fn add(&mut self, mem: &'static mut [u8]) {
        assert!(MIN_SIZE_LOG <= MAX_SIZE_LOG2, "empty range of orders");
        assert!(
            MAX_SIZE_LOG2 < MAX_ORDERS,
            "block size exceeds the address space"
        );
        assert!(
            Self::MIN_SIZE >= size_of::<BuddyAllocatorNode>(),
            "a block of the lowest order cannot hold its own free-list links"
        );

        let start = mem.as_mut_ptr();
        let end = start.addr() + mem.len();

        // Walked as an offset into `mem`, so that every block keeps the
        // provenance of the range it came from.
        //
        // The first block of the lowest order has to start on a multiple of
        // its size like every other one, so a range that begins between two
        // such boundaries loses its first few bytes.
        let mut offset = start.addr().next_multiple_of(Self::MIN_SIZE) - start.addr();

        if offset >= mem.len() {
            return;
        }

        while end - (start.addr() + offset) >= Self::MIN_SIZE {
            let addr = start.addr() + offset;
            let mut order = Self::ORDERS - 1;

            while order > 0 {
                let size = Self::block_size(order);

                if size <= end - addr && addr & (size - 1) == 0 {
                    break;
                }

                order -= 1;
            }

            let size = Self::block_size(order);
            let layout = Layout::from_size_align(size, size).expect("block layout");

            debug_assert!(
                addr & (size - 1) == 0,
                "the greedy cut left a block that is not aligned to its size"
            );

            // SAFETY: `addr` is a multiple of `size` and the block fits in
            // `mem`, which the allocator now owns alone, so nothing else can
            // hold or have freed it.
            unsafe { self.deallocate(NonNull::new_unchecked(start.add(offset)), layout) };

            offset += size;
        }
    }

    /// Bytes currently free.
    ///
    /// Walks every free list, so this is a diagnostic rather than something to
    /// call on a hot path.
    pub fn free(&self) -> usize {
        let mut free = 0;

        for order in 0..Self::ORDERS {
            let mut cursor = self.heads[order];

            while let Some(node) = cursor {
                free += Self::block_size(order);

                // SAFETY: every node in a free list is live.
                cursor = unsafe { node.as_ref().next };
            }
        }

        free
    }

    /// Allocates a block for `layout`.
    ///
    /// The returned block is the whole buddy block, which is the requested
    /// size rounded up to a power of two and to at least `2^MIN_SIZE_LOG`; its
    /// address is aligned to its own size, so any alignment up to the block
    /// size is satisfied.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if no free block of the required order can be
    /// made, and equally if `layout` is larger — or more strictly aligned —
    /// than a block of the highest order.
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, Error> {
        let order = self.order_of(layout).ok_or(Error::OutOfMemory)?;

        // Smallest order at or above `order` that has something to give.
        let mut split = order;
        while split < Self::ORDERS && self.heads[split].is_none() {
            split += 1;
        }

        if split == Self::ORDERS {
            return Err(Error::OutOfMemory);
        }

        // SAFETY: the loop above stopped at a non-empty list.
        let block = unsafe { self.pop(split) };

        // Halve down to the requested order, keeping the lower half and
        // returning the upper one to the list one order down.
        while split > order {
            split -= 1;

            // SAFETY: the upper half of a block that was just taken out of
            // the lists is memory the allocator owns, aligned to its own
            // size, and unreachable through any other block.
            unsafe {
                let buddy = block.byte_add(Self::block_size(split));

                self.push(split, buddy);
            }
        }

        Ok(NonNull::slice_from_raw_parts(
            block,
            Self::block_size(order),
        ))
    }

    /// Returns a block obtained from [`allocate`](Self::allocate).
    ///
    /// # Safety
    ///
    /// - `ptr` must have come from [`allocate`](Self::allocate) on **this**
    ///   allocator, or name a block of a range being handed over by
    ///   [`add`](Self::add), and `layout` must be the one it was allocated
    ///   with — the layout is what names the block's order, so a different one
    ///   frees a block of the wrong size.
    /// - The block must not be freed twice, and must not be read or written
    ///   afterwards. Its first bytes become free-list links immediately.
    ///
    /// Keeping no bounds means a pointer into memory the allocator was never
    /// given cannot be recognised as such; it would simply become a free
    /// block, and be handed out later.
    ///
    /// # Panics
    ///
    /// If `ptr` is not the start of a block of the order `layout` names, or if
    /// the block — or a larger one it has since merged into — is still free.
    /// The latter covers a double free unless the memory was handed out again
    /// in between, and nothing catches a free through a *different* layout, so
    /// the checks are a courtesy rather than part of the contract above.
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let mut order = match self.order_of(layout) {
            Some(order) => order,
            None => panic!("layout {layout:?} names no block of this allocator"),
        };

        let mut addr = ptr.addr().get();
        assert!(
            addr & (Self::block_size(order) - 1) == 0,
            "{ptr:p} is not the start of a block of order {order}"
        );
        assert!(!self.is_available(order, addr), "{ptr:p} is already free");

        // Climb as long as the other half is free as well.
        while order < Self::ORDERS - 1 {
            let size = Self::block_size(order);

            let buddy = match self.find(order, addr ^ size) {
                Some(buddy) => buddy,
                None => break,
            };

            // SAFETY: `find` returned a node of this list, so it is live and
            // linked.
            unsafe { self.unlink(order, buddy.cast()) };

            // The merged block starts at whichever half comes first.
            addr &= !size;
            order += 1;
        }

        // SAFETY: `addr` names a block of `order` that the allocator owns and
        // that is in no free list — either the block being freed, or the
        // result of merging it with buddies that were just unlinked. Its
        // provenance is the one that came in with `ptr`.
        unsafe { self.push(order, ptr.byte_sub(ptr.addr().get() - addr)) };
    }

    // --- Layout and geometry ---------------------------------------------

    /// Size of a block of `order`.
    const fn block_size(order: usize) -> usize {
        1 << (MIN_SIZE_LOG + order)
    }

    /// The order a `layout` has to be served from, or `None` if it exceeds the
    /// highest one.
    ///
    /// Alignment is folded into the size: a block is aligned to its own size,
    /// so asking for a block at least as large as the alignment is what makes
    /// the result aligned.
    fn order_of(&self, layout: Layout) -> Option<usize> {
        let size = layout
            .size()
            .max(layout.align())
            .max(Self::MIN_SIZE)
            .checked_next_power_of_two()?;

        let order = size.trailing_zeros() as usize - MIN_SIZE_LOG;

        (order < Self::ORDERS).then_some(order)
    }

    // --- Free lists --------------------------------------------------------

    /// The block of `order` at `addr`, if it is in that order's free list.
    ///
    /// Linear in the length of the list — the price of keeping no bitmap.
    fn find(&self, order: usize, addr: usize) -> BuddyAllocatorNodePtr {
        let mut cursor = self.heads[order];

        while let Some(node) = cursor {
            if node.addr().get() == addr {
                return Some(node);
            }

            // SAFETY: every node in a free list is live.
            cursor = unsafe { node.as_ref().next };
        }

        None
    }

    /// Whether the block of `order` at `addr` is free — in its own right, or
    /// because a larger block it is part of is.
    ///
    /// A block is only ever split out of a larger one by taking that one out
    /// of its list, so nothing enclosing a handed-out block can be free: this
    /// answers "is this block available?" without a false positive.
    fn is_available(&self, order: usize, addr: usize) -> bool {
        let mut enclosing = order;

        while enclosing < Self::ORDERS {
            let size = Self::block_size(enclosing);

            if self.find(enclosing, addr & !(size - 1)).is_some() {
                return true;
            }

            enclosing += 1;
        }

        false
    }

    /// Puts `block` into the free list of `order`.
    ///
    /// # Safety
    ///
    /// `block` must be the start of a block of `order` that the allocator owns
    /// and that is in no free list. Its first
    /// `size_of::<BuddyAllocatorNode>()` bytes are overwritten.
    unsafe fn push(&mut self, order: usize, block: NonNull<u8>) {
        let node = block.cast::<BuddyAllocatorNode>();

        let mut links = BuddyAllocatorNode::new();
        links.next = self.heads[order];

        // SAFETY: the caller vouches for `block`, which is free memory of at
        // least `MIN_SIZE` bytes and therefore large enough and aligned for a
        // node.
        unsafe { node.write(links) };

        if let Some(mut next) = self.heads[order] {
            // SAFETY: the old head is a live node of this list.
            unsafe { next.as_mut().prev = Some(node) };
        }

        self.heads[order] = Some(node);
    }

    /// Takes the first block out of the free list of `order`.
    ///
    /// # Safety
    ///
    /// The list must not be empty.
    unsafe fn pop(&mut self, order: usize) -> NonNull<u8> {
        let node = match self.heads[order] {
            Some(node) => node,
            None => unreachable!("pop from the empty free list of order {order}"),
        };

        let block = node.cast::<u8>();

        // SAFETY: the head of a non-empty list is a live node.
        unsafe { self.unlink(order, block) };

        block
    }

    /// Takes `block` out of the free list of `order`.
    ///
    /// # Safety
    ///
    /// `block` must currently be in that list.
    unsafe fn unlink(&mut self, order: usize, block: NonNull<u8>) {
        let node = block.cast::<BuddyAllocatorNode>();

        // SAFETY: the caller vouches that the node is live and linked.
        let (prev, next) = unsafe {
            let node = node.as_ref();
            (node.prev, node.next)
        };

        match prev {
            // SAFETY: a linked predecessor is a live node.
            Some(mut prev) => unsafe { prev.as_mut().next = next },
            None => self.heads[order] = next,
        }

        if let Some(mut next) = next {
            // SAFETY: a linked successor is a live node.
            unsafe { next.as_mut().prev = prev };
        }
    }
}

impl<const MIN_SIZE_LOG: usize, const MAX_SIZE_LOG2: usize> Default
    for BuddyAllocator<MIN_SIZE_LOG, MAX_SIZE_LOG2>
{
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: the free-list pointers are what makes this `!Send` by default, and
// they name memory that belongs to no particular core: the allocator can be
// moved to another one and still means the same thing.
unsafe impl<const MIN_SIZE_LOG: usize, const MAX_SIZE_LOG2: usize> Send
    for BuddyAllocator<MIN_SIZE_LOG, MAX_SIZE_LOG2>
{
}

// SAFETY: every method that touches a free list takes `&mut self`, so sharing
// an allocator hands out no way to disturb it. This exists so that one can
// live in a `static` behind a lock, which is the only way it is meant to be
// shared.
unsafe impl<const MIN_SIZE_LOG: usize, const MAX_SIZE_LOG2: usize> Sync
    for BuddyAllocator<MIN_SIZE_LOG, MAX_SIZE_LOG2>
{
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16 B blocks up to 1 KiB, i.e. seven orders.
    type Buddy = BuddyAllocator<4, 10>;

    const MIN: usize = 16;
    const MAX: usize = 1024;

    /// Leaks a zeroed region for an allocator to own.
    ///
    /// Aligned to the largest block, so that the geometry a test reasons about
    /// does not depend on where the host allocator happens to put it.
    fn region(len: usize) -> &'static mut [u8] {
        let layout = layout(len, MAX);

        // SAFETY: the layout has a non-zero size, and the region is leaked on
        // purpose, so handing it out for `'static` is sound.
        unsafe {
            let mem = std::alloc::alloc_zeroed(layout);
            assert!(!mem.is_null(), "the host allocator refused {len} bytes");

            std::slice::from_raw_parts_mut(mem, len)
        }
    }

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    /// Allocates until the allocator is empty, returning every block.
    fn drain(allocator: &mut Buddy, size: usize) -> std::vec::Vec<NonNull<[u8]>> {
        let mut blocks = std::vec::Vec::new();

        while let Ok(block) = allocator.allocate(layout(size, 1)) {
            blocks.push(block);
        }

        blocks
    }

    /// Frees every block of `blocks`, which must all be of `size`.
    fn release(allocator: &mut Buddy, blocks: &[NonNull<[u8]>], size: usize) {
        for block in blocks {
            // SAFETY: every block came from this allocator with this layout,
            // and `drain` hands out each of them exactly once.
            unsafe { allocator.deallocate(block.cast(), layout(size, 1)) };
        }
    }

    #[test]
    fn an_allocator_without_memory_is_out_of_memory() {
        let mut allocator = Buddy::new();

        assert_eq!(allocator.free(), 0);
        assert!(allocator.allocate(layout(MIN, 1)).is_err());
    }

    #[test]
    fn a_region_is_taken_in_whole() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        // Nothing is held back: no bitmap, and the region is aligned and a
        // multiple of the largest block.
        assert_eq!(allocator.free(), 16 * 1024);

        let blocks = drain(&mut allocator, MAX);
        assert_eq!(blocks.len() * MAX, 16 * 1024);
        assert_eq!(allocator.free(), 0);
    }

    #[test]
    fn a_freed_block_comes_back() {
        let mut allocator = Buddy::new_with(region(16 * 1024));
        let capacity = allocator.free();

        let block = allocator.allocate(layout(MIN, 1)).unwrap();
        assert_eq!(block.len(), MIN);
        assert_eq!(allocator.free(), capacity - MIN);

        // SAFETY: `block` came from this allocator with this layout and has
        // not been freed.
        unsafe { allocator.deallocate(block.cast(), layout(MIN, 1)) };

        assert_eq!(allocator.free(), capacity);
        assert_eq!(
            allocator.allocate(layout(MIN, 1)).unwrap().cast::<u8>(),
            block.cast::<u8>(),
            "the free list is LIFO, so the same block is handed out again"
        );
    }

    #[test]
    fn a_request_is_rounded_up_to_a_power_of_two() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        assert_eq!(allocator.allocate(layout(1, 1)).unwrap().len(), MIN);
        assert_eq!(
            allocator.allocate(layout(MIN + 1, 1)).unwrap().len(),
            2 * MIN
        );
        assert_eq!(allocator.allocate(layout(MAX - 1, 1)).unwrap().len(), MAX);
    }

    #[test]
    fn a_block_is_aligned_to_its_own_size() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        for align in [MIN, 64, 256, MAX] {
            let block = allocator.allocate(layout(MIN, align)).unwrap();

            assert_eq!(block.addr().get() % align, 0, "alignment {align}");
            assert!(block.len() >= align, "alignment {align} forces the size");
        }
    }

    #[test]
    fn a_layout_beyond_the_highest_order_fails() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        assert!(allocator.allocate(layout(MAX + 1, 1)).is_err());
        assert!(allocator.allocate(layout(MIN, 2 * MAX)).is_err());
        assert_eq!(allocator.free(), 16 * 1024, "a failure costs nothing");
    }

    #[test]
    fn freeing_everything_coalesces_back_to_the_start() {
        let mut allocator = Buddy::new_with(region(16 * 1024));
        let capacity = allocator.free();

        // Split the region all the way down.
        let blocks = drain(&mut allocator, MIN);
        assert_eq!(blocks.len() * MIN, capacity);
        assert_eq!(allocator.free(), 0);

        release(&mut allocator, &blocks, MIN);
        assert_eq!(allocator.free(), capacity);

        // Only full coalescing can serve blocks of the highest order again.
        assert_eq!(drain(&mut allocator, MAX).len() * MAX, capacity);
    }

    #[test]
    fn blocks_never_overlap() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        // A mix of sizes, so that the run splits blocks of several orders.
        let mut blocks = std::vec::Vec::new();
        for size in [MIN, 4 * MIN, MIN, MAX, 2 * MIN, MAX / 2, MIN] {
            while let Ok(block) = allocator.allocate(layout(size, 1)) {
                blocks.push((block.addr().get(), block.len()));

                if blocks.len() > 64 {
                    break;
                }
            }
        }

        blocks.sort();

        for pair in blocks.windows(2) {
            let (base, len) = pair[0];
            let (next, _) = pair[1];

            assert!(base + len <= next, "{base:#x}+{len:#x} runs into {next:#x}");
        }
    }

    #[test]
    fn a_region_that_is_not_a_power_of_two_is_used_as_far_as_it_goes() {
        // 5000 bytes from an aligned start: four blocks of the highest order,
        // then 512, 256 and 128, and the last eight bytes are too few for a
        // block of the lowest order.
        let mut allocator = Buddy::new_with(region(5000));

        assert_eq!(allocator.free(), 4992);

        let blocks = drain(&mut allocator, MIN);
        assert_eq!(blocks.len() * MIN, 4992);

        release(&mut allocator, &blocks, MIN);
        assert_eq!(allocator.free(), 4992);
    }

    #[test]
    fn writing_to_a_block_does_not_disturb_the_allocator() {
        let mut allocator = Buddy::new_with(region(16 * 1024));
        let capacity = allocator.free();

        let blocks = drain(&mut allocator, 2 * MIN);

        for (pattern, block) in blocks.iter().enumerate() {
            // SAFETY: the block is ours for as long as it is not freed, and
            // `block.len()` bytes long.
            unsafe { block.cast::<u8>().write_bytes(pattern as u8, block.len()) };
        }

        release(&mut allocator, &blocks, 2 * MIN);

        assert_eq!(allocator.free(), capacity);
        assert_eq!(drain(&mut allocator, MAX).len() * MAX, capacity);
    }

    #[test]
    fn a_round_trip_is_exact_at_any_alignment() {
        // A range handed over at an arbitrary offset ends on an arbitrary one,
        // so its tail is a run of blocks below the highest order — the case
        // where splitting and coalescing have the least symmetry to lean on.
        // Whatever the allocator took in must still come back exactly.
        for skew in [0usize, 16, 80, 336, 576, 688, 1000] {
            let mem = region(16 * 1024 + MAX);
            let (_, mem) = mem.split_at_mut(skew);

            let mut allocator = Buddy::new_with(mem);
            let capacity = allocator.free();

            let blocks = drain(&mut allocator, MIN);
            assert_eq!(blocks.len() * MIN, capacity, "skew {skew}");

            release(&mut allocator, &blocks, MIN);
            assert_eq!(allocator.free(), capacity, "skew {skew}");

            assert_eq!(
                drain(&mut allocator, MIN).len() * MIN,
                capacity,
                "skew {skew}"
            );
        }
    }

    #[test]
    fn a_range_added_later_is_handed_out() {
        let mut allocator = Buddy::new();

        allocator.add(region(MAX));
        assert_eq!(allocator.free(), MAX);
        assert_eq!(allocator.allocate(layout(MAX, 1)).unwrap().len(), MAX);

        // A second range, nowhere near the first, is taken in just the same.
        allocator.add(region(4 * MAX));
        assert_eq!(allocator.free(), 4 * MAX);
        assert_eq!(drain(&mut allocator, MAX).len() * MAX, 4 * MAX);
    }

    #[test]
    fn ranges_added_apart_merge_if_they_are_neighbours() {
        let mut allocator = Buddy::new();

        let (lower, upper) = region(MAX).split_at_mut(MAX / 2);

        // The upper half first, so that the merge has to happen downwards.
        allocator.add(upper);
        allocator.add(lower);

        assert_eq!(allocator.free(), MAX);

        // Only a block spanning both ranges can serve the highest order.
        assert_eq!(allocator.allocate(layout(MAX, 1)).unwrap().len(), MAX);
        assert_eq!(allocator.free(), 0);
    }

    #[test]
    #[should_panic(expected = "is already free")]
    fn a_double_free_is_caught() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        let block = allocator.allocate(layout(MIN, 1)).unwrap();

        // SAFETY: the first call is sound; the second is the bug under test,
        // and the allocator is expected to catch it rather than corrupt its
        // lists.
        unsafe {
            allocator.deallocate(block.cast(), layout(MIN, 1));
            allocator.deallocate(block.cast(), layout(MIN, 1));
        }
    }

    #[test]
    #[should_panic(expected = "is not the start of a block")]
    fn a_pointer_into_the_middle_of_a_block_is_caught() {
        let mut allocator = Buddy::new_with(region(16 * 1024));

        let block = allocator.allocate(layout(MAX, 1)).unwrap();

        // SAFETY: deliberately not the start of the block, to check that it is
        // refused instead of splitting the free lists across it.
        unsafe {
            allocator.deallocate(block.cast::<u8>().byte_add(MIN), layout(MAX, 1));
        }
    }
}
