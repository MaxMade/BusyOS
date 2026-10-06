//! The kernel's physical page frame allocators.
//!
//! # Two allocators, one after the other
//!
//! Page tables have to be allocated before there is anywhere to keep track of
//! free memory, so frames come from two places over the kernel's life:
//!
//! - [`EarlyPageFrames`], a bump allocator over a fixed array in `.bss`, used
//!   to build the kernel's first page tables. It never frees.
//! - [`PageFrames`], the buddy allocators below, which take over once the
//!   memory map is known and the kernel runs on its own page tables.
//!
//! Both implement [`PageFrameAllocator`], so [`Paging`] is parameterized by
//! whichever is in charge at the time.
//!
//! The changeover is two calls, in this order, once the kernel's own page
//! tables are active: [`init_from_bootinfo`](PageFrames::init_from_bootinfo)
//! hands the usable physical memory to the buddy allocators, and
//! [`handover_from_early`](PageFrames::handover_from_early) then drains what
//! [`EarlyPageFrames`] never got round to handing out into them as well.
//!
//! # Why three buddy allocators
//!
//! A page frame's *size* is a property of the mapping it will be used for, and
//! an allocator that hands out 4 KiB blocks cannot promise the 1 GiB alignment
//! a gigantic mapping needs. So each page size gets its own
//! [`BuddyAllocator`], parameterized by the log of that page size, and each is
//! capped one order below the next size up:
//!
//! | Allocator | Block sizes      | Serves               |
//! |-----------|------------------|----------------------|
//! | `regular` | 4 KiB – 1 MiB    | [`PageSize::Regular`]  |
//! | `huge`    | 2 MiB – 512 MiB  | [`PageSize::Huge`]     |
//! | `gigantic`| 1 GiB and up     | [`PageSize::Gigantic`] |
//!
//! Splitting the ranges rather than running one allocator across all of them
//! is what keeps a run of regular allocations from eating the memory a
//! gigantic mapping would need, and it is why a request for more contiguous
//! regular pages than fit in one block of the highest regular order fails
//! instead of being carved out of huge-page memory: ask for the larger page
//! size instead.
//!
//! # Splitting bigger pages
//!
//! Keeping the page sizes apart would otherwise mean running out of regular
//! pages with gigabytes still free for gigantic ones, so an allocator that has
//! nothing large enough left borrows from the size above it before it gives
//! up: [`split_larger_page`](PageFrames::split_larger_page) allocates one page
//! of the next size up and hands it to the smaller allocator, which is enough
//! to serve any request that allocator could ever serve. It happens only on
//! the way to failing, and it cascades — a regular allocation can break a
//! gigantic page into huge ones and one of those into regular ones — so the
//! memory map's whole free memory is available to every page size, in the
//! order the kernel actually asks for it.
//!
//! The split does not come back: each allocator stops one order below the
//! page it was given, so the halves it cuts that page into can never merge
//! into it again, and a page that has once been broken up stays with the
//! smaller size.
//!
//! # Physical addresses and the direct map
//!
//! [`BuddyAllocator`] keeps its free lists inside the free blocks, so it works
//! in virtual addresses. Everything here therefore goes through
//! [`ReversePaging`](GenericReversePaging) — the kernel's direct map of
//! physical memory — on the way in and out, which is also why none of it works
//! before the kernel's own page tables are active.

use core::alloc::Layout;
use core::ffi::c_void;
use core::ptr::NonNull;

use crate::{
    arch::{
        Paging, REGULAR_PAGE_SIZE,
        generic::paging::{
            Error as PagingError, PageFrameAllocator, PageSize, Paging as GenericPaging,
            PhysicalAddress, ReversePaging as GenericReversePaging, VirtualAddress,
        },
    },
    kernel::{
        bootinfo::BOOTINFO,
        locking::{CanAcquire, PreviousToken, level},
        spinlock::{MemorySpinlock, Spinlock},
        ticketlock::{MemoryTicketlock, Ticketlock},
    },
    mem::buddy_allocator::{BuddyAllocator, Error as BuddyAllocatorError},
    utils::range_tree::Range,
};

/// Bytes of `.bss` reserved for early page frames: 1024 regular pages.
///
/// Sized for the page tables of early boot and nothing beyond them; once these
/// run out, [`allocate`](PageFrameAllocator::allocate) fails and there is no
/// way to get more.
const EARLY_PAGE_FRAMES_SIZE: usize = 1024 * REGULAR_PAGE_SIZE;

/// A bump allocator over a fixed array of page frames in `.bss`.
///
/// Serves the page tables built before there is a memory map to allocate from,
/// which is why it is page aligned: every block it hands out has to be a page
/// frame. Frames are handed out in order and never reused — see
/// [`deallocate`](PageFrameAllocator::deallocate), which panics — so the array
/// is sized for the whole of early boot and simply runs out afterwards.
#[repr(C, align(4096))]
pub struct EarlyPageFrames {
    /// The frames themselves, handed out from the front.
    pages: [u8; EARLY_PAGE_FRAMES_SIZE],
    /// Offset of the next frame to hand out; past the end once exhausted.
    offset: usize,
}

static EARLY_PAGE_FRAMES: MemorySpinlock<EarlyPageFrames> = MemorySpinlock::new(
    Spinlock::new(),
    EarlyPageFrames {
        pages: [0; EARLY_PAGE_FRAMES_SIZE],
        offset: 0,
    },
);

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

/// A range of physical memory, as the memory map hands it over.
type PhysicalRange = Range<PhysicalAddress<c_void>, usize>;

/// `page_shift` itself, or a degenerate shift if the architecture has no such
/// page size.
///
/// A page size the architecture does not have still needs *some* pair of
/// orders to instantiate [`BuddyAllocator`] with, since the field exists
/// either way. `usize::BITS - 1` names the largest block a `usize` can
/// address, so an allocator built from it is never handed memory and never
/// serves a request — [`PageFrames::allocate`] turns the missing page size
/// into an error long before it gets that far.
const fn page_shift(page_shift: Option<usize>) -> usize {
    match page_shift {
        Some(page_shift) => page_shift,
        None => usize::BITS as usize - 1,
    }
}

/// One shift below `page_shift`, i.e. the highest order of the allocator
/// *below* the one that page size is served from.
///
/// Degenerate as [`page_shift`] when the architecture has no such page size.
const fn page_shift_below(page_shift: Option<usize>) -> usize {
    match page_shift {
        Some(page_shift) => page_shift - 1,
        None => usize::BITS as usize - 1,
    }
}

/// The kernel's physical memory, one [`BuddyAllocator`] per page size.
///
/// See the [module documentation](self) for why the page sizes are kept apart.
/// Created empty; memory is handed to the individual allocators with
/// [`BuddyAllocator::add`] once the memory map is known.
pub struct PageFrames {
    /// Blocks of a gigantic page and up.
    gigantic: BuddyAllocator<
        { page_shift(<Paging<PageFrames> as GenericPaging<PageFrames>>::GIGANTIC_PAGE_SHIFT) },
        { usize::BITS as usize - 1 },
    >,
    /// Blocks from a huge page up to one order below a gigantic one.
    huge: BuddyAllocator<
        { page_shift(<Paging<PageFrames> as GenericPaging<PageFrames>>::HUGE_PAGE_SHIFT) },
        {
            page_shift_below(<Paging<PageFrames> as GenericPaging<PageFrames>>::GIGANTIC_PAGE_SHIFT)
        },
    >,
    /// Blocks from a regular page up to one order below a huge one.
    regular: BuddyAllocator<
        { <Paging<PageFrames> as GenericPaging<PageFrames>>::REGULAR_PAGE_SHIFT },
        { page_shift_below(<Paging<PageFrames> as GenericPaging<PageFrames>>::HUGE_PAGE_SHIFT) },
    >,
}

impl PageFrames {
    /// Creates the allocators with no memory at all.
    ///
    /// Every allocation fails with [`Error::OutOfMemory`](BuddyAllocatorError)
    /// until memory is handed over, which is what lets this live in a `static`
    /// from the start.
    pub const fn new() -> Self {
        Self {
            gigantic: BuddyAllocator::new(),
            huge: BuddyAllocator::new(),
            regular: BuddyAllocator::new(),
        }
    }

    /// Drains what is left of [`EarlyPageFrames`] into these allocators.
    ///
    /// The early bump allocator never frees, so the frames it has already
    /// handed out — the kernel's first page tables — stay where they are.
    /// Everything it has *not* handed out is free memory sitting in `.bss`,
    /// and this walks it to the end one frame at a time so that it is not lost
    /// for the rest of the kernel's life.
    ///
    /// [`EarlyPageFrames`] is left exhausted, which is the point: from here on
    /// every allocation from it fails and frames come from [`PageFrames`].
    ///
    /// Call once, and only once the kernel's own page tables are active —
    /// [`deallocate`](PageFrameAllocator::deallocate) reaches the frames
    /// through the direct map. Calling it twice is harmless: the second call
    /// finds the early allocator empty and returns immediately.
    ///
    /// # Panics
    ///
    /// As [`BuddyAllocator::deallocate`], which sees a double free if a frame
    /// somehow reaches this twice.
    pub fn handover_from_early<Token>(token: Token) -> Token
    where
        Token: CanAcquire<level::Memory> + PreviousToken,
    {
        let mut token = token;

        loop {
            match EarlyPageFrames::allocate(token) {
                // SAFETY: the frame has just been taken out of the early
                // allocator, which never hands the same one out twice and
                // cannot take it back, so nothing else holds it.
                Ok((phys_addr, t)) => {
                    token = unsafe { <PageFrames as PageFrameAllocator>::deallocate(phys_addr, t) };
                }
                Err((_, token)) => return token,
            }
        }
    }

    /// Hands the usable physical memory of [`BOOTINFO`] to the allocators.
    ///
    /// Each range of the memory map is aligned up to a regular page and then
    /// offered to the allocators from the largest page size down: the whole
    /// number of gigantic pages at its front goes to `gigantic`, whatever
    /// follows goes to `huge` the same way, and the rest to `regular`. A range
    /// whose base is not aligned to a page size is not offered to that
    /// allocator at all — a block has to start on a multiple of its own size —
    /// so a range starting mid-gigantic-page is served entirely by the smaller
    /// allocators, and the last few bytes below a regular page are dropped.
    ///
    /// Ranges are cut into blocks by [`BuddyAllocator::add`], so a count that
    /// is not a power of two is fine: it becomes several blocks, and ranges
    /// that turn out to be neighbours merge as if they had arrived together.
    ///
    /// Call once, before [`handover_from_early`](Self::handover_from_early).
    ///
    /// # Safety
    ///
    /// - [`BOOTINFO`] must be initialised, and its
    ///   [`memory_ranges`](crate::kernel::bootinfo::Bootinfo::memory_ranges)
    ///   must name physical memory that is free for the kernel to hand out —
    ///   not firmware tables, not the kernel image, not the boot loader's own
    ///   memory.
    /// - The kernel's own page tables must be active, since the allocators
    ///   reach their memory through the direct map.
    /// - Call this once. A range handed over twice is a double free.
    ///
    /// # Panics
    ///
    /// If a range lies beyond what the direct map covers, as
    /// [`phys_to_virt`](GenericReversePaging::phys_to_virt).
    pub unsafe fn init_from_bootinfo<Token>(token: Token) -> Token
    where
        Token: CanAcquire<level::Memory> + PreviousToken,
    {
        let mut token = token;

        // SAFETY: the caller vouches that the boot information is there.
        let bootinfo = unsafe { BOOTINFO.assume_init_ref() };

        for range in bootinfo.memory_ranges.iter().cloned() {
            // Unused slots of the memory map are empty ranges, which `align`
            // rejects along with anything too short to hold a page frame.
            let mut range =
                match range.align(<Paging<Self> as GenericPaging<Self>>::REGULAR_PAGE_SIZE) {
                    Some(range) => range,
                    None => continue,
                };

            let (mut page_frames, t) = PAGE_FRAMES.acquire(token);

            if let Some(size) = <Paging<Self> as GenericPaging<Self>>::GIGANTIC_PAGE_SIZE
                && let Some((pages, rest)) = Self::split_pages(range, size)
            {
                // SAFETY: the caller vouches that the range is free physical
                // memory, and `split_pages` cut this piece out of it, so no
                // other allocator is offered the same bytes.
                unsafe { Self::add(&mut page_frames.gigantic, pages) };
                range = rest;
            }

            if let Some(size) = <Paging<Self> as GenericPaging<Self>>::HUGE_PAGE_SIZE
                && let Some((pages, rest)) = Self::split_pages(range, size)
            {
                // SAFETY: as above; this piece is what the gigantic allocator
                // left behind.
                unsafe { Self::add(&mut page_frames.huge, pages) };
                range = rest;
            }

            let size = <Paging<Self> as GenericPaging<Self>>::REGULAR_PAGE_SIZE;
            if let Some((pages, _)) = Self::split_pages(range, size) {
                // SAFETY: as above. Whatever is left over is shorter than a
                // page frame and is simply dropped.
                unsafe { Self::add(&mut page_frames.regular, pages) };
            }

            token = page_frames.release(t);
        }

        token
    }

    /// Splits the whole pages of `size` at the front of `range` off from the
    /// rest, as `(pages, rest)`.
    ///
    /// Returns `None` if `range` does not start on a multiple of `size`, or if
    /// it is not even one page long — in both cases there is nothing an
    /// allocator of that page size could take, since a block has to start on a
    /// multiple of its own size and fit whole.
    fn split_pages(range: PhysicalRange, size: usize) -> Option<(PhysicalRange, PhysicalRange)> {
        if !range.base().addr().is_multiple_of(size) {
            return None;
        }

        let length = (range.length() / size) * size;
        if length == 0 {
            return None;
        }

        // SAFETY: `length <= range.length()`, so the split point is inside the
        // range — one past its end at the most. Nothing is dereferenced.
        let rest = unsafe { range.base().byte_add(length) };

        Some((
            Range::new(range.base(), length),
            Range::new(rest, range.length() - length),
        ))
    }

    /// Hands the physical memory `range` names to `allocator`.
    ///
    /// The range is passed on as the slice of the direct map that covers it,
    /// so [`BuddyAllocator::add`] does the cutting: greedily, into the largest
    /// aligned blocks the range can hold, merging with whatever is free beside
    /// it already.
    ///
    /// # Safety
    ///
    /// - `range` must name physical memory that is free for the allocator to
    ///   own, and must be handed over exactly once.
    /// - The kernel's own page tables must be active, so that the direct map
    ///   the range is reached through is there.
    ///
    /// # Panics
    ///
    /// If `range` lies beyond what the direct map covers.
    unsafe fn add<const MIN_SIZE_LOG: usize, const MAX_SIZE_LOG2: usize>(
        allocator: &mut BuddyAllocator<MIN_SIZE_LOG, MAX_SIZE_LOG2>,
        range: PhysicalRange,
    ) {
        // SAFETY: the caller vouches that the kernel's page tables are active,
        // which is what makes the direct map's addresses dereferenceable.
        let virt_addr = unsafe { Paging::<Self>::phys_to_virt(range.base()) };

        // SAFETY: the caller vouches that these bytes are free physical memory
        // handed over exactly once, so the allocator owns them alone for the
        // rest of the kernel's life — which is what `'static` claims here.
        let mem = unsafe {
            core::slice::from_raw_parts_mut(virt_addr.as_ptr().cast::<u8>(), range.length())
        };

        allocator.add(mem);
    }

    /// Allocates `num_pages` contiguous page frames of `page_size`.
    ///
    /// The frames come from the allocator that page size belongs to, so the
    /// range is aligned to at least `page_size` — which is what a mapping of
    /// that size requires of it — and, `num_pages` not being a power of two,
    /// may be rounded up to the next power-of-two block. See
    /// [`BuddyAllocator::allocate`].
    ///
    /// If that allocator has nothing large enough left, one page of the next
    /// size up is broken into it first — see
    /// [`split_larger_page`](Self::split_larger_page) — so running out of one
    /// page size while a bigger one still has memory costs a split rather than
    /// an error.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`](BuddyAllocatorError) if the architecture has no
    /// such page size, if `num_pages` of them cannot be named at all, or if no
    /// block that large is free and none can be split out of a bigger page
    /// either. Note that the allocator for a page size stops one order below
    /// the next size up, so a run longer than that fails even with memory to
    /// spare — no amount of splitting helps, since the blocks a split hands
    /// over are of exactly that order: ask for the larger page size instead.
    ///
    /// # Panics
    ///
    /// If `num_pages` is zero.
    pub fn allocate(
        &mut self,
        page_size: PageSize,
        num_pages: usize,
    ) -> Result<PhysicalPageFrames, BuddyAllocatorError> {
        let layout = Self::layout(page_size, num_pages).ok_or(BuddyAllocatorError::OutOfMemory)?;

        let ptr = match self.allocate_block(page_size, layout) {
            Ok(ptr) => ptr,
            // Nothing that large is free here, so break a page of the next
            // size up into this allocator and ask it once more. Asking twice
            // is enough: the page arrives as two blocks of the highest order
            // this allocator has, and a request bigger than one of those is
            // one it could never serve, however much memory it were given.
            Err(BuddyAllocatorError::OutOfMemory) => {
                self.split_larger_page(page_size)?;
                self.allocate_block(page_size, layout)?
            }
        };

        // SAFETY: the block came out of an allocator holding memory of the
        // direct map, so it has a physical address, and it is ours until it is
        // handed back to `deallocate`.
        let start =
            unsafe { Paging::<PageFrames>::virt_to_phys(VirtualAddress::new(ptr.as_ptr().cast())) };

        Ok(PhysicalPageFrames {
            start,
            size: page_size,
            num: num_pages,
        })
    }

    /// Allocates one block for `layout` from the allocator `page_size` belongs
    /// to, without splitting anything.
    ///
    /// The plain lookup behind [`allocate`](Self::allocate): it fails as soon
    /// as that one allocator has nothing large enough, which is what tells
    /// `allocate` that it is time to break up a bigger page.
    fn allocate_block(
        &mut self,
        page_size: PageSize,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, BuddyAllocatorError> {
        match page_size {
            PageSize::Regular => self.regular.allocate(layout),
            PageSize::Huge => self.huge.allocate(layout),
            PageSize::Gigantic => self.gigantic.allocate(layout),
        }
    }

    /// Moves one page of the next size up into the allocator for `page_size`.
    ///
    /// The page is allocated from the larger allocator like any other — so
    /// this recurses at most as far as `gigantic`, refilling `huge` from it on
    /// the way if that is what a regular allocation needs — and is then handed
    /// to the smaller allocator with [`add`](Self::add), which cuts it into
    /// blocks of the highest order that one has. Two of them, since each
    /// allocator stops exactly one order below the next page size up.
    ///
    /// This is a one-way street: the smaller allocator caps out one order
    /// below the page it was given, so the halves can never merge back into
    /// one and the memory stays with the smaller page size for the rest of the
    /// kernel's life. Only ever splitting when an allocation would otherwise
    /// fail is what keeps that from eroding the larger sizes.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`](BuddyAllocatorError) if there is no larger page
    /// size to split — nothing is bigger than a gigantic page, and an
    /// architecture may have neither — or if a page of it cannot be had
    /// either.
    fn split_larger_page(&mut self, page_size: PageSize) -> Result<(), BuddyAllocatorError> {
        match page_size {
            PageSize::Regular => {
                let page = self.take_page(PageSize::Huge)?;

                // SAFETY: the page has just been allocated, so it is free
                // physical memory belonging to nobody else, and it is handed
                // over exactly once — it is never returned to the allocator it
                // came from. The caller of `allocate` vouches for the page
                // tables, which that allocation went through already.
                unsafe { Self::add(&mut self.regular, page) };
            }
            PageSize::Huge => {
                let page = self.take_page(PageSize::Gigantic)?;

                // SAFETY: as above.
                unsafe { Self::add(&mut self.huge, page) };
            }
            // Nothing to split a gigantic page out of.
            PageSize::Gigantic => return Err(BuddyAllocatorError::OutOfMemory),
        }

        Ok(())
    }

    /// Allocates a single page of `page_size` and returns the physical range
    /// it covers, ready to be handed to a smaller allocator.
    ///
    /// The range is given up for good: it is deliberately not a
    /// [`PhysicalPageFrames`], since nothing will ever free it again.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`](BuddyAllocatorError) if the architecture has no
    /// such page size, or if no page of it is free.
    fn take_page(&mut self, page_size: PageSize) -> Result<PhysicalRange, BuddyAllocatorError> {
        let size = Paging::<Self>::page_size(page_size).ok_or(BuddyAllocatorError::OutOfMemory)?;

        let page = self.allocate(page_size, 1)?;

        Ok(Range::new(page.start, size))
    }

    /// Returns page frames obtained from [`allocate`](Self::allocate).
    ///
    /// # Safety
    ///
    /// - `page_frames` must have come from [`allocate`](Self::allocate) on
    ///   **this** `PageFrames`, unchanged: it names the block's allocator and
    ///   order, so a doctored one frees a block of the wrong size.
    /// - The frames must not be freed twice, and must be referenced by no page
    ///   table entry afterwards. Their first bytes become free-list links
    ///   immediately.
    ///
    /// # Panics
    ///
    /// As [`BuddyAllocator::deallocate`].
    pub unsafe fn deallocate(&mut self, page_frames: PhysicalPageFrames) {
        let layout = Self::layout(page_frames.size, page_frames.num)
            .expect("page frames of a size that cannot have been allocated");

        // SAFETY: the frames were allocated out of the direct map, so this is
        // the virtual address they were handed out at.
        let virt_addr = unsafe { Paging::<PageFrames>::phys_to_virt(page_frames.start) };

        // SAFETY: `phys_to_virt` offsets into the direct map, which does not
        // start at zero, so the result cannot be null.
        let ptr = unsafe { NonNull::new_unchecked(virt_addr.as_ptr().cast()) };

        // SAFETY: the caller vouches that these frames came from `allocate`,
        // and `layout` is recomputed from the size and count it recorded, so
        // it is the one they were allocated with. The recorded size also picks
        // the allocator they came from.
        match page_frames.size {
            PageSize::Regular => unsafe { self.regular.deallocate(ptr, layout) },
            PageSize::Huge => unsafe { self.huge.deallocate(ptr, layout) },
            PageSize::Gigantic => unsafe { self.gigantic.deallocate(ptr, layout) },
        }
    }

    /// The layout `num_pages` frames of `page_size` are (de)allocated with, or
    /// `None` if the architecture has no such page size or the range cannot be
    /// named.
    ///
    /// Alignment is the page size, not the size of the whole range: a mapping
    /// cares that each page is aligned, and [`BuddyAllocator`] aligns a block
    /// to its own size regardless.
    ///
    /// # Panics
    ///
    /// If `num_pages` is zero.
    fn layout(page_size: PageSize, num_pages: usize) -> Option<Layout> {
        assert!(num_pages != 0, "asked for no page frames at all");

        let size = Paging::<PageFrames>::page_size(page_size)?;

        Layout::from_size_align(size.checked_mul(num_pages)?, size).ok()
    }
}

impl Default for PageFrames {
    fn default() -> Self {
        Self::new()
    }
}

/// A range of contiguous physical page frames handed out by
/// [`PageFrames::allocate`].
///
/// Carries the page size and count along with the address because that pair is
/// what names the block to free again — [`PageFrames::deallocate`] keeps no
/// record of its own.
pub struct PhysicalPageFrames {
    /// Physical address of the first frame, aligned to at least `size`.
    start: PhysicalAddress<c_void>,
    /// Page size the frames were allocated for.
    size: PageSize,
    /// Number of frames of `size`.
    num: usize,
}

impl PhysicalPageFrames {
    /// Physical address of the first frame, aligned to at least
    /// [`page_size`](Self::page_size).
    pub const fn start(&self) -> PhysicalAddress<c_void> {
        self.start
    }

    /// Number of frames of [`page_size`](Self::page_size) the range holds.
    pub const fn num_frames(&self) -> usize {
        self.num
    }

    /// Page size the frames were allocated for.
    pub const fn page_size(&self) -> PageSize {
        self.size
    }
}

/// The kernel's page frames.
///
/// A [`Ticketlock`] rather than a [`Spinlock`]: it serves waiters in the order
/// they arrived, so a core cannot be starved out of physical memory by others
/// that keep winning the race. Every allocator above this one bottoms out
/// here, which is what makes that worth paying for.
///
/// Public because the allocators above it reach it directly rather than
/// through [`PageFrameAllocator`], which only ever hands out single regular
/// frames: the [`Heap`](crate::mem::heap::Heap) grows by whole pages of
/// whatever size fits.
pub static PAGE_FRAMES: MemoryTicketlock<PageFrames> =
    MemoryTicketlock::new(Ticketlock::new(), PageFrames::new());

impl PageFrameAllocator for PageFrames {
    fn allocate<Token>(
        token: Token,
    ) -> Result<(PhysicalAddress<c_void>, Token), (PagingError, Token)>
    where
        Token: CanAcquire<level::Memory> + PreviousToken,
    {
        let (mut page_frames, token) = PAGE_FRAMES.acquire(token);

        match page_frames.allocate(PageSize::Regular, 1) {
            Ok(physical_page_range) => {
                let token = page_frames.release(token);
                Ok((physical_page_range.start.cast(), token))
            }
            Err(BuddyAllocatorError::OutOfMemory) => {
                let token = page_frames.release(token);
                Err((PagingError::OutOfMemory, token))
            }
        }
    }

    unsafe fn deallocate<Token>(phys_addr: PhysicalAddress<c_void>, token: Token) -> Token
    where
        Token: CanAcquire<level::Memory> + PreviousToken,
    {
        // Reconstructed rather than remembered: `allocate` only ever asks for
        // a single regular frame, so this is the range it handed out.
        let physical_page_frames = PhysicalPageFrames {
            start: phys_addr,
            size: PageSize::Regular,
            num: 1,
        };

        let (mut page_frames, token) = PAGE_FRAMES.acquire(token);

        // SAFETY: the caller vouches that `phys_addr` came from `allocate`,
        // which allocated it out of `PAGE_FRAMES` as exactly this range.
        unsafe { page_frames.deallocate(physical_page_frames) };

        page_frames.release(token)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::arch::x86_64::paging::{
        GIGANTIC_PAGE_SHIFT, GIGANTIC_PAGE_SIZE, HUGE_PAGE_SHIFT, HUGE_PAGE_SIZE,
        REGULAR_PAGE_SHIFT,
    };

    fn range(base: usize, length: usize) -> PhysicalRange {
        PhysicalRange::new(PhysicalAddress::new(base as *mut c_void), length)
    }

    // --- The orders each allocator covers ----------------------------------

    /// A page size the architecture has is served from its own shift.
    #[test]
    fn a_page_size_is_served_from_its_own_shift() {
        assert_eq!(page_shift(Some(REGULAR_PAGE_SHIFT)), REGULAR_PAGE_SHIFT);
        assert_eq!(page_shift_below(Some(HUGE_PAGE_SHIFT)), HUGE_PAGE_SHIFT - 1);
    }

    /// A page size the architecture does not have still needs some pair of
    /// orders; both degenerate to the largest a `usize` can name, so the
    /// allocator built from them is never handed anything.
    #[test]
    fn a_missing_page_size_degenerates() {
        assert_eq!(page_shift(None), usize::BITS as usize - 1);
        assert_eq!(page_shift_below(None), usize::BITS as usize - 1);
    }

    /// The block sizes the module documentation tabulates: each allocator
    /// stops one order below the next page size up, so a run of regular
    /// allocations cannot eat the memory a gigantic mapping would need.
    #[test]
    fn the_allocators_cover_the_documented_block_sizes() {
        assert_eq!(1usize << REGULAR_PAGE_SHIFT, 4 * 1024);
        assert_eq!(
            1usize << page_shift_below(Some(HUGE_PAGE_SHIFT)),
            1024 * 1024
        );

        assert_eq!(1usize << page_shift(Some(HUGE_PAGE_SHIFT)), HUGE_PAGE_SIZE);
        assert_eq!(
            1usize << page_shift_below(Some(GIGANTIC_PAGE_SHIFT)),
            512 * 1024 * 1024
        );

        assert_eq!(
            1usize << page_shift(Some(GIGANTIC_PAGE_SHIFT)),
            GIGANTIC_PAGE_SIZE
        );
    }

    // --- layout ------------------------------------------------------------

    /// The layout spans every frame but is aligned only to one of them: a
    /// mapping cares that each page is aligned, not the range as a whole.
    #[test]
    fn a_layout_spans_the_frames_and_is_aligned_to_one() {
        let layout = PageFrames::layout(PageSize::Regular, 1).unwrap();
        assert_eq!(layout.size(), REGULAR_PAGE_SIZE);
        assert_eq!(layout.align(), REGULAR_PAGE_SIZE);

        let layout = PageFrames::layout(PageSize::Huge, 3).unwrap();
        assert_eq!(layout.size(), 3 * HUGE_PAGE_SIZE);
        assert_eq!(layout.align(), HUGE_PAGE_SIZE);

        let layout = PageFrames::layout(PageSize::Gigantic, 2).unwrap();
        assert_eq!(layout.size(), 2 * GIGANTIC_PAGE_SIZE);
        assert_eq!(layout.align(), GIGANTIC_PAGE_SIZE);
    }

    /// More frames than an address can name is not a layout at all, which is
    /// what turns into `OutOfMemory` rather than an overflow.
    #[test]
    fn a_layout_that_cannot_be_named_is_none() {
        assert!(PageFrames::layout(PageSize::Gigantic, usize::MAX).is_none());
    }

    #[test]
    #[should_panic(expected = "asked for no page frames at all")]
    fn a_layout_of_no_frames_at_all_panics() {
        PageFrames::layout(PageSize::Regular, 0);
    }

    // --- split_pages -------------------------------------------------------

    /// A range of whole pages is taken in full, leaving nothing behind.
    #[test]
    fn a_range_of_whole_pages_is_taken_in_full() {
        let (pages, rest) = PageFrames::split_pages(range(0, 4 * 4096), 4096).unwrap();

        assert_eq!(pages, range(0, 4 * 4096));
        assert_eq!(rest.length(), 0);
        assert_eq!(rest.base().addr(), 4 * 4096);
    }

    /// Whatever is short of a whole page stays behind for a smaller allocator.
    #[test]
    fn a_partial_page_at_the_end_stays_behind() {
        let (pages, rest) = PageFrames::split_pages(range(0, 2 * 4096 + 100), 4096).unwrap();

        assert_eq!(pages, range(0, 2 * 4096));
        assert_eq!(rest, range(2 * 4096, 100));
    }

    /// The two pieces are disjoint and together cover the original, so no byte
    /// is offered to two allocators and none is invented.
    #[test]
    fn the_pieces_partition_the_range() {
        let original = range(HUGE_PAGE_SIZE, 3 * HUGE_PAGE_SIZE + 4096);
        let (pages, rest) = PageFrames::split_pages(original, HUGE_PAGE_SIZE).unwrap();

        assert_eq!(pages.base(), original.base());
        assert_eq!(pages.end(), rest.base());
        assert_eq!(rest.end(), original.end());
        assert_eq!(pages.length() + rest.length(), original.length());
    }

    /// A block has to start on a multiple of its own size, so a range that
    /// does not is left to the smaller allocators entirely.
    #[test]
    fn a_misaligned_range_is_not_offered_at_all() {
        assert!(PageFrames::split_pages(range(4096 + 1, 8 * 4096), 4096).is_none());
        assert!(
            PageFrames::split_pages(
                range(HUGE_PAGE_SIZE, 4 * GIGANTIC_PAGE_SIZE),
                GIGANTIC_PAGE_SIZE
            )
            .is_none()
        );
    }

    /// A range shorter than one page has nothing to offer either, empty or
    /// not.
    #[test]
    fn a_range_shorter_than_a_page_is_not_offered_at_all() {
        assert!(PageFrames::split_pages(range(0, 0), 4096).is_none());
        assert!(PageFrames::split_pages(range(0, 4095), 4096).is_none());
    }

    /// Exactly one page is still a page.
    #[test]
    fn a_range_of_exactly_one_page_is_taken() {
        let (pages, rest) = PageFrames::split_pages(range(4096, 4096), 4096).unwrap();

        assert_eq!(pages, range(4096, 4096));
        assert_eq!(rest.length(), 0);
    }

    // --- PhysicalPageFrames ------------------------------------------------

    /// The accessors report the address, size and count the allocation
    /// recorded — the triple that names the block to free again, and all a
    /// caller has to go on.
    #[test]
    fn page_frames_report_what_they_were_allocated_as() {
        let page_frames = PhysicalPageFrames {
            start: PhysicalAddress::new(HUGE_PAGE_SIZE as *mut c_void),
            size: PageSize::Huge,
            num: 3,
        };

        assert_eq!(page_frames.start().addr(), HUGE_PAGE_SIZE);
        assert_eq!(page_frames.page_size(), PageSize::Huge);
        assert_eq!(page_frames.num_frames(), 3);
    }
}
