//! An intrusive, lock-free LIFO stack.
//!
//! # Overview
//!
//! [`LockFreeStack`] chains [`LockFreeStackNode`]s into a singly-linked list
//! whose head is a single atomic word. Both [`push`](LockFreeStack::push) and
//! [`pop`](LockFreeStack::pop) are one compare-exchange loop over that word:
//! nothing is ever parked and no operation has to finish what another one
//! started, so the stack works where a lock must not be taken — inside an
//! interrupt handler, on the panic path, or in code that holds no lock-level
//! token (see [`crate::kernel::locking`]). It is lock-free, not wait-free: an
//! individual push or pop retries for as long as other cores keep winning the
//! race.
//!
//! The stack is intrusive and owns nothing. A node is a link field the caller
//! embeds in its own structure, and the storage stays the caller's; the two
//! operations only move ownership of one node — into the stack on a push,
//! back out of it on a pop.
//!
//! # The tagged head
//!
//! `#[repr(align(64))]` on the node leaves the low six bits of every node
//! address zero, and the head word carries a counter there:
//!
//! ```text
//!  63                                      6   5       0
//! ┌──────────────────────────────────────────┬───────────┐
//! │  address of the top node (64B aligned)   │  counter  │
//! └──────────────────────────────────────────┴───────────┘
//! ```
//!
//! Every successful push and pop bumps the counter, which is what keeps a pop
//! safe against the ABA problem: a pop reads the top node's successor and then
//! swings the head over to it, and in between the top node may have been
//! popped, handed out and pushed again. The address alone would be unchanged,
//! so the stale successor would be installed as the new head and every node
//! pushed in the meantime would be lost. The counter has moved on, the
//! compare-exchange fails, and the pop retries.
//!
//! Six bits is a mitigation rather than a proof: a pop whose head word is
//! exactly a multiple of 64 successful operations old, and whose top node
//! happens to be that same node again, still succeeds with a stale successor.
//! Keeping the number of nodes well above 64 keeps that coincidence away.
//!
//! Because the counter shares the word with the address, `mask_cnt` has to be
//! applied to everything read out of the head — a node address is never used
//! as it was loaded, and an empty stack is a head whose *address half* is
//! null, not a head word that is null.
//!
//! # Node lifetime
//!
//! [`pop`](LockFreeStack::pop) dereferences the top node before its
//! compare-exchange has decided whether that node was still on top, so it can
//! read a node that another core has already popped and given away. The link
//! field is atomic, so that read is well defined and merely stale, and a failed
//! compare-exchange throws the value away — but the read does happen. The
//! storage behind a node that has ever been pushed must therefore stay mapped
//! and readable for as long as the stack is in use, even after the node has
//! come back out. Static storage, or memory that is never handed back,
//! satisfies this; a node freed to an allocator that may unmap it does not.
//!
//! # The allocator
//!
//! [`LockFreeAllocator`] puts one stack behind each power-of-two size class
//! from 64 B to 4 KiB and serves a request the way a buddy allocator does:
//! from the class that fits it best, or — if that one is empty — by taking a
//! block of a larger class and halving it until one of the right size falls
//! out, the upper halves going onto the stacks they pass on the way down.
//!
//! What it does not do is the other half of a buddy allocator. Merging a freed
//! block with its buddy means finding that buddy and taking it out of the
//! middle of a list, which is not something a stack can do; a free only pushes
//! the block back onto the stack of its own class. A block therefore never
//! grows back, and a run of small requests that has cut every 4 KiB block into
//! 64 B ones leaves the allocator with nothing large to hand out, however much
//! of it is free. New memory for the classes that ran dry comes from the
//! kernel heap instead — see [`allocate`](LockFreeAllocator::allocate), which
//! is also the one operation of the three that is not lock-free.
//!
//! The free blocks are the nodes: a block's first 64 bytes are its link while
//! it is free. Nothing is freed back to the heap, which is exactly what the
//! node lifetime above asks for.
//!
//! # The queue
//!
//! [`MPSC`] carries the filled buffers from wherever they were written to
//! whoever drains them. Any number of cores push; one drains. It is a ring of a
//! fixed number of slots and it never blocks a producer: a ring that has filled
//! up is one whose oldest entry gets overwritten, and
//! [`push`](MPSC::push) hands that entry back to the producer that displaced it
//! so the buffer can go straight back to the allocator.
//!
//! Losing the oldest messages under overload is the deliberate trade. The
//! alternative — a producer that waits for the consumer — is what a printk must
//! not do: it runs in interrupt handlers and on the panic path, where there may
//! be no consumer left to wait for. The loss is not silent, though:
//! [`pop`](MPSC::pop) hands the consumer a flag beside each entry saying whether
//! anything was overwritten ahead of it, so a reader can mark the gap.

use core::{
    alloc::Layout,
    fmt::{self, Arguments, Write},
    marker::PhantomData,
    mem::align_of,
    ptr::{self, NonNull},
    str,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicUsize, Ordering},
};

use crate::{
    arch::{CPU, generic::cpu::CPU as _},
    driver::console::{ConsoleOutput, ConsoleOutputDriver, Consoles},
    kernel::locking::{CanAcquire, DriverLevelID, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::Allocator,
};

/// The link field of one stack element.
///
/// Embed it in the structure that is to be pushed and recover that structure
/// from a popped pointer; the stack itself never looks past this field.
///
/// The 64-byte alignment is what frees the low bits of a node address for the
/// counter in the head word (see the [module documentation](self)) and, as it
/// matches a cache line, keeps one node's link out of another's line.
#[repr(align(64))]
struct LockFreeStackNode {
    /// The node below this one, or null for the bottom of the stack. Never
    /// carries a counter, so a popper can install it as a head address as it
    /// is.
    next: AtomicPtr<Self>,
}

impl LockFreeStackNode {
    /// A node that is on no stack.
    const fn new() -> Self {
        Self {
            next: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

/// A lock-free LIFO stack of [`LockFreeStackNode`]s.
///
/// See the [module documentation](self) for how the head word is encoded and
/// for how long the nodes have to stay readable.
struct LockFreeStack {
    /// Address of the top node with a counter in its low bits; a null address
    /// means empty. Never dereferenced without [`Self::mask_cnt`].
    head: AtomicPtr<LockFreeStackNode>,
}

// SAFETY: the head word is only ever touched atomically, which is what the
// compare-exchange loops are written for, and the stack passes node pointers
// around rather than handing out references to them. Whether a node itself may
// travel between cores is the caller's contract, not the stack's.
//
// `AtomicPtr<T>` is `Send` and `Sync` for every `T` already, so these two impls
// only record that reasoning; they grant nothing the auto traits would not.
unsafe impl Send for LockFreeStack {}

// SAFETY: as above.
unsafe impl Sync for LockFreeStack {}

impl LockFreeStack {
    /// The bits of the head word that hold the counter, i.e. the bits an
    /// aligned node address leaves free.
    const CNT_MASK: usize = align_of::<LockFreeStackNode>() - 1;

    /// An empty stack.
    const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Puts `node` on top of the stack.
    ///
    /// Retries until its compare-exchange wins; every retry means another core
    /// got an operation of its own through in the meantime.
    ///
    /// # Safety
    ///
    /// The caller gives `node` up: it must own the node exclusively, the node
    /// must not already be on this or any other stack, and nothing may touch it
    /// until it comes back out of [`pop`](Self::pop).
    ///
    /// `node` must be a real node — a fabricated pointer that is not 64-byte
    /// aligned would run into the counter, which the debug assertion catches
    /// before anything is written.
    ///
    /// The node's storage has to stay readable for as long as the stack is
    /// used, even once it has been popped again; see the
    /// [module documentation](self).
    unsafe fn push(&self, node: NonNull<LockFreeStackNode>) {
        let node_ptr = node.as_ptr();
        debug_assert_eq!(node_ptr as usize & Self::CNT_MASK, 0);

        let mut head = self.head.load(Ordering::Relaxed);

        loop {
            // Untagged, so that a popper can take this successor as an address
            // without masking it first. Relaxed is enough: the node is still
            // private until the compare-exchange below publishes it, and that
            // one releases this store along with it.
            unsafe {
                (*node_ptr)
                    .next
                    .store(Self::mask_cnt(head), Ordering::Relaxed)
            };

            let new_head = Self::set_cnt(node_ptr, Self::get_cnt(head).wrapping_add(1));

            match self.head.compare_exchange_weak(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => head = actual,
            }
        }
    }

    /// Takes the top node off the stack, or returns `None` if it is empty.
    ///
    /// As with [`push`](Self::push), retries for as long as other cores win the
    /// race for the head word.
    ///
    /// # Safety
    ///
    /// Ownership of the returned node passes to the caller: it is no longer on
    /// the stack, and pushing it again is the only way to put it back.
    ///
    /// Every node that has ever been pushed — including ones already popped by
    /// somebody else — must still be readable, because this may read one of
    /// them before finding out that it is gone. See the
    /// [module documentation](self).
    unsafe fn pop(&self) -> Option<NonNull<LockFreeStackNode>> {
        // Acquire, so that everything the pusher of this node did before
        // publishing it is visible here.
        let mut head = self.head.load(Ordering::Acquire);

        loop {
            let node_ptr = Self::mask_cnt(head);
            let node = NonNull::new(node_ptr)?;

            // May be stale — the node can have been popped and reused by now,
            // in which case the counter in `head` has moved on and the
            // compare-exchange below discards this value.
            let next = Self::mask_cnt(unsafe { node.as_ref().next.load(Ordering::Relaxed) });
            let new_head = Self::set_cnt(next, Self::get_cnt(head).wrapping_add(1));

            match self.head.compare_exchange_weak(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(node),
                Err(actual) => head = actual,
            }
        }
    }

    /// Whether the stack held nothing at the moment of the load.
    ///
    /// A snapshot only: a push or pop on another core can invalidate the answer
    /// before it is even returned. Note that an empty stack does not have a
    /// null head word — the counter stays behind — so emptiness is the address
    /// half being null.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        Self::mask_cnt(self.head.load(Ordering::Acquire)).is_null()
    }

    /// The address held in a head word, with the counter cleared.
    fn mask_cnt(tagged: *mut LockFreeStackNode) -> *mut LockFreeStackNode {
        ((tagged as usize) & !Self::CNT_MASK) as *mut _
    }

    /// The counter held in a head word.
    fn get_cnt(tagged: *mut LockFreeStackNode) -> usize {
        (tagged as usize) & Self::CNT_MASK
    }

    /// A head word from an address and a counter, dropping whatever does not
    /// fit into the bits the alignment leaves free.
    fn set_cnt(node: *mut LockFreeStackNode, cnt: usize) -> *mut LockFreeStackNode {
        (((node as usize) & !Self::CNT_MASK) | (cnt & Self::CNT_MASK)) as *mut _
    }
}

/// Number of size classes: one for every power of two from
/// [`LockFreeAllocator::MIN_SIZE`] to [`LockFreeAllocator::MAX_SIZE`].
const BINS: usize = LockFreeAllocator::MAX_SIZE_LOG - LockFreeAllocator::MIN_SIZE_LOG + 1;

/// A lock-free allocator for blocks of 64 B up to 4 KiB.
///
/// One [`LockFreeStack`] of free blocks per size class, the smallest class
/// first. See the [module documentation](self) for how a request is served and
/// for what the missing coalescing costs.
///
/// Minimum Size: 64 Byte: 6
/// Maximium Size: 4096 Bytes: 12
struct LockFreeAllocator([LockFreeStack; BINS]);

impl LockFreeAllocator {
    const MIN_SIZE_LOG: usize = 6;
    const MIN_SIZE: usize = 1 << Self::MIN_SIZE_LOG;

    const MAX_SIZE_LOG: usize = 12;
    const MAX_SIZE: usize = 1 << Self::MAX_SIZE_LOG;

    /// An allocator holding no memory.
    ///
    /// [`allocate_lock_free`](Self::allocate_lock_free) fails until something
    /// has been freed to it; [`allocate`](Self::allocate) fills the classes
    /// from the kernel heap as it goes.
    const fn new() -> Self {
        Self([const { LockFreeStack::new() }; BINS])
    }

    /// The class a block of `size` bytes belongs to.
    ///
    /// `size` has to be a power of two between [`Self::MIN_SIZE`] and
    /// [`Self::MAX_SIZE`], which is what both allocation paths round a request
    /// to before they get here.
    fn size_to_idx(size: usize) -> usize {
        size.ilog2() as usize - Self::MIN_SIZE_LOG
    }

    /// The size of the blocks in class `idx`.
    fn idx_to_size(idx: usize) -> usize {
        1 << (idx + Self::MIN_SIZE_LOG)
    }

    /// The block a request of `size` bytes is served from: `size` rounded up to
    /// a power of two and to at least [`Self::MIN_SIZE`], or `None` if no class
    /// is large enough.
    ///
    /// The one place the rounding happens. A caller that has to work out what to
    /// hand back to [`deallocate`](Self::deallocate) — because it kept the
    /// length of what it asked for rather than the block it got — asks here, and
    /// so asks exactly the question the allocation asked.
    fn block_size(size: usize) -> Option<usize> {
        if size == 0 || size > Self::MAX_SIZE {
            return None;
        }

        Some(usize::max(size, Self::MIN_SIZE).next_power_of_two())
    }

    /// Takes one block out of class `idx`, splitting a larger one if that class
    /// has nothing.
    ///
    /// The classes are walked upwards from `idx` and the first block found is
    /// halved on the way back down: the lower half is carried on, the upper one
    /// is published to the class below. The block returned belongs to the
    /// caller.
    ///
    /// Returns `None` if every class from `idx` up was seen empty. That is a
    /// snapshot, as with [`LockFreeStack::is_empty`] — a block freed to a class
    /// the walk has already passed is missed — so a `None` means "nothing to be
    /// had just now" rather than "nothing is free".
    fn take(&self, idx: usize) -> Option<NonNull<u8>> {
        for bin in idx..BINS {
            // SAFETY: a block reaches a stack only through `give`, which hands
            // it over for good, and the storage behind it is never returned to
            // the heap — see the module documentation on node lifetime.
            let Some(node) = (unsafe { self.0[bin].pop() }) else {
                continue;
            };

            let block = node.cast::<u8>();

            // Halve down to `idx`, keeping the lower half every time. The upper
            // half of a block of class `split + 1` starts one block of class
            // `split` into it.
            for split in (idx..bin).rev() {
                // SAFETY: the block has just come out of the stacks, so it is
                // the allocator's alone and its upper half is reachable through
                // nothing else. Both halves are aligned to their own size
                // because the block was.
                unsafe { self.give(split, block.byte_add(Self::idx_to_size(split))) };
            }

            return Some(block);
        }

        None
    }

    /// Puts `block`, one block of class `idx`, onto that class's stack.
    ///
    /// # Safety
    ///
    /// `block` must name a block of exactly [`idx_to_size(idx)`](Self::idx_to_size)
    /// bytes that the caller owns and gives up here, aligned to its own size.
    /// It must not already be on a stack, and nothing may touch it until it
    /// comes back out of [`take`](Self::take).
    unsafe fn give(&self, idx: usize, block: NonNull<u8>) {
        let node = block.cast::<LockFreeStackNode>();

        // A free block's first bytes are its link, so the node is laid down
        // before the block is published. Whatever the block held is gone from
        // here on, which is why a free is the point of no return for the
        // caller's data.
        //
        // SAFETY: a block is aligned to its own size and no class is smaller
        // than `MIN_SIZE`, which is exactly the size and the alignment of a
        // node, so the node fits inside the block the caller has given up.
        unsafe { node.write(LockFreeStackNode::new()) };

        // SAFETY: the node has just been laid down in a block nothing else
        // holds, and its storage outlives the allocator — see above.
        unsafe { self.0[idx].push(node) };
    }

    /// Allocates `size` bytes, falling back to the kernel heap.
    ///
    /// The free lists are tried first, exactly as in
    /// [`allocate_lock_free`](Self::allocate_lock_free). If they have nothing,
    /// one block of [`Self::MAX_SIZE`] — the largest class, so that the
    /// fallback refills the allocator rather than serving this one request — is
    /// taken from the [`Heap`] and halved down to the class asked for, the
    /// unused halves staying behind in the free lists.
    ///
    /// Taking the heap's lock is what costs this the "lock-free" of the name:
    /// it needs a token for the `MemoryManagement` level and so cannot be
    /// called where no token can be had. That is what
    /// [`allocate_lock_free`](Self::allocate_lock_free) is for.
    ///
    /// The block is as in [`allocate_lock_free`](Self::allocate_lock_free): the
    /// whole size class, aligned to its own size. The token comes back either
    /// way, so a failed allocation costs the caller nothing but the answer.
    ///
    /// Memory taken from the heap is never given back to it — a block freed to
    /// this allocator stays in its free lists.
    fn allocate<Token>(&self, size: usize, token: Token) -> (Option<NonNull<[u8]>>, Token)
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let Some(size) = Self::block_size(size) else {
            return (None, token);
        };
        let idx = Self::size_to_idx(size);

        if let Some(block) = self.take(idx) {
            return (Some(NonNull::slice_from_raw_parts(block, size)), token);
        }

        // Aligned to its own size like every other block, which is what keeps
        // the halves it is cut into aligned to theirs, down to the 64 bytes a
        // node needs.
        let layout =
            Layout::from_size_align(Self::MAX_SIZE, Self::MAX_SIZE).expect("maximum block layout");

        let (block, token) = match Heap.allocate(layout, token) {
            Ok((block, token)) => (block, token),
            Err((_, token)) => return (None, token),
        };

        // Halve down to `idx` as `take` does, the fresh block being one of the
        // largest class.
        for split in (idx..BINS - 1).rev() {
            // SAFETY: the heap has just handed the block over and nothing else
            // knows of it yet, so its upper halves are the allocator's to give
            // away.
            unsafe { self.give(split, block.byte_add(Self::idx_to_size(split))) };
        }

        (Some(NonNull::slice_from_raw_parts(block, size)), token)
    }

    /// Allocates `size` bytes from the free lists alone.
    ///
    /// Takes no lock and needs no token, which is what makes it callable from
    /// an interrupt handler or a panic path — and what makes it fail whenever
    /// the classes from the requested one up happen to be empty, rather than
    /// growing the allocator the way [`allocate`](Self::allocate) does.
    ///
    /// The block returned is the whole size class, i.e. `size` rounded up to a
    /// power of two and to at least [`Self::MIN_SIZE`], and it is aligned to
    /// its own size, so any alignment up to that is satisfied. Hand it to
    /// [`deallocate`](Self::deallocate) as it was returned: its length is what
    /// names the class it goes back to.
    fn allocate_lock_free(&self, size: usize) -> Option<NonNull<[u8]>> {
        let size = Self::block_size(size)?;

        let block = self.take(Self::size_to_idx(size))?;

        Some(NonNull::slice_from_raw_parts(block, size))
    }

    /// Returns a block obtained from either allocation path.
    ///
    /// The block goes back onto the stack of its own size class and is merged
    /// with nothing; see the [module documentation](self).
    ///
    /// # Safety
    ///
    /// - `mem` must be a block this allocator handed out, exactly as it was
    ///   returned — the length is what names the class it belongs to, so a
    ///   different one frees a block of the wrong size.
    /// - The block must not be freed twice, and must not be read or written
    ///   afterwards: its first bytes become a link immediately.
    ///
    /// The allocator keeps no bounds, so a pointer it never handed out cannot
    /// be recognised as such; it would simply become a free block and be handed
    /// out later.
    unsafe fn deallocate(&self, mem: NonNull<[u8]>) {
        let size = mem.len();

        debug_assert!(
            size.is_power_of_two() && (Self::MIN_SIZE..=Self::MAX_SIZE).contains(&size),
            "not a block of this allocator"
        );

        // SAFETY: the caller gives the block up and it is one this allocator
        // handed out, so it is a whole block of the class its length names and
        // aligned to its own size.
        unsafe { self.give(Self::size_to_idx(size), mem.cast::<u8>()) };
    }
}

/// A bounded multi-producer, single-consumer queue of pointers that overwrites
/// rather than blocks.
///
/// A ring of `LEN` slots with two counters that only ever go up: `head` is the
/// next ticket to hand a producer, `tail` the next entry the consumer wants.
/// Each is an index into the ring modulo its length, so the distance between
/// them is how far the consumer is behind — and once that distance passes
/// `LEN`, a producer is writing over an entry nobody has read.
///
/// That overwrite is the point. A producer never waits for the consumer and
/// never fails: [`push`](Self::push) takes a ticket, swaps its pointer into the
/// slot that ticket names, and hands the caller back whatever was there. See
/// the [module documentation](self) for why that is the shape a printk buffer
/// wants.
struct MPSC {
    /// The ring. A null slot is one the consumer has emptied, or one whose
    /// producer has taken its ticket but not yet deposited its pointer.
    slots: [AtomicPtr<u8>; 512],
    /// Tickets handed out; `head % LEN` is the slot the next producer takes.
    head: AtomicUsize,
    /// Tickets consumed; `tail % LEN` is the slot the consumer reads next.
    /// Written only by the consumer.
    tail: AtomicUsize,
    /// Set by a producer that overwrote an entry nobody had read, cleared by
    /// the consumer when [`pop`](MPSC::pop) reports it.
    overwritten: AtomicBool,
}

// SAFETY: every field is touched atomically and the queue passes pointers
// around rather than handing out references to what they name. Whether a buffer
// itself may travel between cores is the caller's contract, not the queue's.
unsafe impl Send for MPSC {}

// SAFETY: as above. Note that this says nothing about `pop`, which is `unsafe`
// exactly because sharing the queue does not make a second consumer sound.
unsafe impl Sync for MPSC {}

impl MPSC {
    /// Slots in the ring, i.e. how far the consumer may fall behind before the
    /// oldest entries start being overwritten.
    ///
    /// Must match the length of `slots`; [`new`](Self::new) does not compile
    /// otherwise.
    const LEN: usize = 512;

    const fn new() -> Self {
        let slots: [*mut u8; Self::LEN] = [ptr::null_mut(); Self::LEN];
        let slots = unsafe { core::mem::transmute(slots) };

        Self {
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            overwritten: AtomicBool::new(false),
        }
    }

    /// Puts `ptr` into the queue, returning whatever it displaced.
    ///
    /// Never fails and never waits: the ticket is one `fetch_add` and the slot
    /// is one `swap`, so a producer is done in two atomic operations however
    /// many others are pushing at the same time.
    ///
    /// A `Some` is the entry that was `LEN` tickets older than this one and
    /// that the consumer had not taken yet — the ring is full and this push
    /// overwrote it. Ownership of it passes to the caller, which is what keeps
    /// an overrun from leaking: the buffer it names is the caller's to free (or
    /// to reuse for the next message) from here on.
    ///
    /// # Ownership
    ///
    /// The caller gives `ptr` up. Nothing may touch it until it comes back out
    /// — through [`pop`](Self::pop) on the consumer's side, or through the
    /// return value of a later `push` that overwrites it.
    fn push(&self, ptr: NonNull<u8>) -> Option<NonNull<u8>> {
        // A ticket rather than a publication: the slot below is what makes the
        // entry visible, so this only has to hand every producer a different
        // number.
        let idx = self.head.fetch_add(1, Ordering::Relaxed);

        // Release, so that a consumer taking this pointer sees everything
        // written through it beforehand; Acquire, so that everything written
        // through the pointer coming back is visible here.
        let prev = NonNull::new(self.slots[idx % Self::LEN].swap(ptr.as_ptr(), Ordering::AcqRel));

        // A slot that still held something is one the consumer had not reached:
        // this push overwrote a message. Raised here, where an overwrite is
        // exactly what just happened, rather than worked out on the consumer's
        // side — there, a producer that holds a ticket but has not deposited
        // yet is indistinguishable from one that was overwritten.
        //
        // Relaxed: the flag says that something happened, not what, and orders
        // nothing else.
        if prev.is_some() {
            self.overwritten.store(true, Ordering::Relaxed);
        }

        prev
    }

    /// Takes the oldest entry out of the queue, or returns `None` if the ring
    /// holds nothing.
    ///
    /// Ownership of the returned pointer passes to the caller.
    ///
    /// Producers that lapped the ring were handed the entries they overwrote by
    /// [`push`](Self::push) itself, so this starts at the oldest ticket that can
    /// still be in the ring and walks forward from there: what a consumer that
    /// cannot keep up loses is the oldest messages, which is the trade a ring
    /// like this is for.
    ///
    /// # The empty slots it walks over
    ///
    /// A ticket is claimed before the pointer that goes with it is deposited, so
    /// a slot can be empty while the tickets on either side of it are filled —
    /// its producer was descheduled in between. This steps over such a slot
    /// rather than stopping at it, because stopping would hold up every entry
    /// behind it for as long as that one producer is away.
    ///
    /// The cost is that the entry lands in the ring after the consumer has gone
    /// past its ticket. It is not lost: it is handed out either on the next lap,
    /// once `LEN` further tickets have been claimed, or to the producer that
    /// eventually overwrites it. It does come out late and behind newer
    /// messages, and if the ring is retired while a producer is in that window,
    /// its entry stays in `slots` for whoever tears the ring down.
    ///
    /// # Recognising a gap
    ///
    /// The `bool` says whether anything was overwritten since the previous
    /// call: one bit, not a count, which is what a reader needs to mark the gap
    /// in what it prints. It is taken before the entry is looked for, so the
    /// gap it reports is the one *in front of* the entry returned beside it —
    /// mark it first, then print the message.
    ///
    /// It is reported here rather than worked out from `tail`, because from
    /// this side a producer that holds a ticket but has not deposited yet looks
    /// exactly like one that was overwritten. [`push`](Self::push) raises the
    /// flag where an overwrite is unambiguous, and reading it here clears it,
    /// so a gap is marked once rather than on every message from there on.
    ///
    /// The flag comes back even when the entry does not: a `(None, true)` is an
    /// overrun that the consumer has already drained past, and losing it
    /// because there was nothing to return with it would lose the gap.
    ///
    /// # Safety
    ///
    /// There is one consumer: no other core may call this, and this must not be
    /// re-entered from an interrupt that hit a call already in progress. Two
    /// consumers would share `tail` and could hand the same entry out twice.
    /// Whether every ticket handed out so far has been consumed.
    ///
    /// A snapshot, safe to call from anywhere. A producer between taking its
    /// ticket and depositing its pointer already counts as non-empty here.
    fn is_empty(&self) -> bool {
        // SeqCst, pairing with the flag in `drain`, see there.
        self.tail.load(Ordering::SeqCst) >= self.head.load(Ordering::SeqCst)
    }

    #[must_use = "dropping the entry leaks its buffer, and dropping the flag \
                  loses the record of a gap"]
    unsafe fn pop(&self) -> (Option<NonNull<u8>>, bool) {
        // Taken before the walk below, so that it covers the messages that came
        // before the one this call returns. Relaxed: the flag says that
        // something happened, not what, and orders nothing else.
        let overwritten = self.overwritten.swap(false, Ordering::Relaxed);

        // The consumer owns `tail` alone, so plain loads and stores of it are
        // enough; it is atomic only because `push` takes `&self`.
        let mut tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // The oldest entry still in the ring is `LEN` tickets behind the newest
        // one handed out; anything older has been overwritten, and the producer
        // that overwrote it took it back through the return value of `push`.
        //
        // Skipping to it costs nothing: the tickets jumped over name the same
        // slots as the ones that follow, so the walk below still passes every
        // slot of the ring.
        tail = tail.max(head.saturating_sub(Self::LEN));

        while tail < head {
            // Acquire, pairing with the release in `push`.
            let entry = self.slots[tail % Self::LEN].swap(ptr::null_mut(), Ordering::AcqRel);

            tail += 1;

            if let Some(entry) = NonNull::new(entry) {
                self.tail.store(tail, Ordering::Relaxed);
                return (Some(entry), overwritten);
            }
        }

        // Nothing between `tail` and `head`. Each slot along the way was looked
        // at once and `tail` keeps the ground gained, so a queue that is empty
        // costs the single comparison above and this walk costs one step per
        // ticket over the life of the queue, not one per call.
        self.tail.store(tail, Ordering::Relaxed);

        (None, overwritten)
    }
}

/// How important a message is, and so whether [`LOG_LEVEL`] lets it through.
///
/// Ordered least to most important, and `#[repr(u8)]` so that the discriminant
/// is what [`LOG_LEVEL`] holds and compares against.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Panic = u8::MAX,
}

/// The lowest [`LogLevel`] that still gets logged.
///
/// Read on every [`printk!`] and written from wherever the log level is
/// configured, so it is an atomic rather than something behind a lock — a level
/// check must not be able to block.
pub static LOG_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as _);

/// Whether a message at `level` is to be logged at all.
///
/// The check [`printk!`] makes before it formats anything, so that a message
/// that is not wanted costs one relaxed load and no stack buffer.
pub fn enabled(level: LogLevel) -> bool {
    LOG_LEVEL.load(Ordering::Relaxed) <= level as u8
}

static MESSAGE_ALLOCATOR: LockFreeAllocator = LockFreeAllocator::new();

static MESSAGE_QUEUE: MPSC = MPSC::new();

/// Set while a core drains [`MESSAGE_QUEUE`] to the console.
///
/// [`MPSC::pop`] allows a single consumer, and any number of cores may call
/// [`__printk`] at once, so the drain is taken by whoever sets this first. It
/// is a flag to try rather than a lock to wait for: a core that finds it set
/// leaves its message to the core draining, and an epilogue that logs while
/// its own core is in the middle of a drain returns instead of re-entering
/// it.
static DRAINING: AtomicBool = AtomicBool::new(false);

/// What is written to the console where messages were lost, see
/// [`MPSC::pop`].
const GAP_MARKER: &str = "[... messages lost ...]\n";

/// The low bits of a buffer pointer, which carry the size class of its block.
///
/// A block is aligned to its own size and the smallest class is 64 bytes, so
/// those six bits of a block address are always zero — the same room
/// [`LockFreeStack`] borrows for its counter.
///
/// Carrying the class along with the pointer is what lets a buffer be freed
/// without measuring the string inside it. The alternative is a `strlen` of up
/// to 4 KiB on every overrun, and it has to arrive at exactly the rounding the
/// allocation used: a block freed to the wrong class is a block handed out
/// twice.
///
/// Every reader of [`MESSAGE_QUEUE`] has to put a pointer through [`untag`]
/// before touching the string behind it.
const TAG_MASK: usize = LockFreeAllocator::MIN_SIZE - 1;

const _: () = assert!(
    BINS <= TAG_MASK + 1,
    "a size class no longer fits in the low bits of a block address"
);

/// Puts the size class of a block of `size` bytes into the low bits of its
/// address.
fn tag(block: NonNull<u8>, size: usize) -> NonNull<u8> {
    let idx = LockFreeAllocator::size_to_idx(size);

    debug_assert_eq!(
        block.as_ptr() as usize & TAG_MASK,
        0,
        "a block is not aligned to its own size"
    );

    // SAFETY: setting bits in an address that is not null cannot make it null.
    unsafe { NonNull::new_unchecked(block.as_ptr().map_addr(|addr| addr | idx)) }
}

/// Splits a tagged pointer back into its block and the size of that block.
fn untag(tagged: NonNull<u8>) -> (NonNull<u8>, usize) {
    let idx = tagged.as_ptr() as usize & TAG_MASK;

    // SAFETY: clearing the tag restores the block's own address, which is the
    // one the allocator handed out and so is not null.
    let block =
        unsafe { NonNull::new_unchecked(tagged.as_ptr().map_addr(|addr| addr & !TAG_MASK)) };

    (block, LockFreeAllocator::idx_to_size(idx))
}

/// How much of `msg` fits in `max` bytes, cut on a character boundary.
///
/// Never more than `msg` itself. Cutting mid-character would put the trailing
/// half of one in the log as stray bytes, so the cut walks back to the last
/// boundary at or before `max`.
fn floor_char_boundary(msg: &str, max: usize) -> usize {
    if msg.len() <= max {
        return msg.len();
    }

    let mut end = max;

    while !msg.is_char_boundary(end) {
        end -= 1;
    }

    end
}

/// Cuts `msg` down to what the largest block can hold.
///
/// One byte of that block goes to the terminator.
fn truncate(msg: &str) -> &str {
    &msg[..floor_char_boundary(msg, LockFreeAllocator::MAX_SIZE - 1)]
}

/// Writes `msg` into `block` as a NUL terminated string and hands it to
/// [`MESSAGE_QUEUE`], freeing whatever the queue displaced to make room for it.
///
/// The body both entry points share, so that the two cannot drift apart — they
/// differ only in where the block comes from.
fn enqueue(mut block: NonNull<[u8]>, msg: &str) {
    let size = block.len();
    let len = msg.len();

    debug_assert!(
        len < size,
        "block too small for the message and its terminator"
    );

    // SAFETY: the allocator has just handed this block over, so nothing else
    // holds it, and it is longer than the message.
    let buffer = unsafe { block.as_mut() };

    // The block is the whole size class, which is usually larger than the
    // message needs, so only the front of it is written.
    buffer[..len].copy_from_slice(msg.as_bytes());

    // A NUL inside the message would end the string early and hide the rest of
    // it, leaving the tail of the block to be read as a second message. The
    // terminator below is left as the only one.
    for c in &mut buffer[..len] {
        if *c == 0 {
            *c = b' ';
        }
    }

    buffer[len] = 0;

    // SAFETY: the block came from the allocator, so its address is not null.
    let buffer = unsafe { NonNull::new_unchecked(buffer.as_mut_ptr()) };

    let Some(displaced) = MESSAGE_QUEUE.push(tag(buffer, size)) else {
        return;
    };

    // The queue was full and dropped its oldest message to make room for this
    // one. That buffer belongs to this caller now, and freeing it here is what
    // keeps an overrun from draining the allocator.
    let (block, size) = untag(displaced);

    // SAFETY: the queue has given the buffer up, so it is out of the queue and
    // out of everybody else's reach, and the tag names the class it was
    // allocated from.
    unsafe { MESSAGE_ALLOCATOR.deallocate(NonNull::slice_from_raw_parts(block, size)) };
}

/// Queues one message without taking a lock.
///
/// The message is copied into a block of [`MESSAGE_ALLOCATOR`] as a NUL
/// terminated string and the block goes to [`MESSAGE_QUEUE`]. Nothing here waits
/// on anything, which is what makes it callable from an interrupt handler or the
/// panic path — and what makes it give up rather than block when it cannot get
/// memory.
///
/// Three things can cost a message, none of which stop the caller:
///
/// - The allocator has nothing left. It does not grow on this path, so the
///   message is dropped. [`__printk`] is the one that can go to the heap.
/// - The queue is full, in which case [`push`](MPSC::push) hands back the oldest
///   message and this frees it. The reader learns of the gap from
///   [`pop`](MPSC::pop).
/// - The message is longer than the largest block, in which case it is truncated
///   rather than dropped.
///
/// A NUL inside `message` becomes a space: the string has to end where the
/// terminator is put and nowhere else.
pub fn __printk_lock_free<S: AsRef<str>>(message: S) {
    let msg = truncate(message.as_ref());

    if let Some(block) = MESSAGE_ALLOCATOR.allocate_lock_free(msg.len() + 1) {
        enqueue(block, msg);
    }
}

/// Queues one message, growing [`MESSAGE_ALLOCATOR`] from the kernel heap if it
/// has run dry.
///
/// The same work as [`__printk_lock_free`] and with the same handling of a full
/// queue, a long message and an embedded NUL, differing only in where an empty
/// allocator turns for memory: this one takes the heap's lock and refills, so it
/// drops a message only when physical memory is gone.
///
/// # Token
///
/// The bound is `Driver`, which is stricter than the `MemoryManagement` the
/// allocator itself needs: after queueing, this drains the queue to the
/// console, see [`drain`], and the console's lock is at that level. The token
/// is consumed and returned whether or not the message got through, so a
/// caller threads it on regardless.
///
/// Not for an interrupt handler or the panic path: it can block on the heap's
/// lock. Use [`__printk_lock_free`] there.
pub fn __printk<S, Token>(message: S, token: Token) -> Token
where
    S: AsRef<str>,
    Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    let msg = truncate(message.as_ref());

    let (block, token) = MESSAGE_ALLOCATOR.allocate(msg.len() + 1, token);

    if let Some(block) = block {
        enqueue(block, msg);
    }

    drain(token)
}

/// Writes every message in [`MESSAGE_QUEUE`] to the console, oldest first,
/// and frees its buffer.
///
/// Leaves the queue alone while no console has registered, so that what was
/// logged during early boot shows up once one has. The queue overwrites its
/// oldest entries in the meantime, and the gap is marked when the drain
/// finally happens.
///
/// Returns at once if another drain is under way, see [`DRAINING`]. That
/// drain looks at the queue again after it lets go of the flag, so a message
/// queued while it was running is not left behind.
///
/// A message the console fails to show is dropped all the same: there is
/// nowhere left to report the failure to.
///
/// # Token
///
/// The `token` is consumed and returned. It has to reach the `Driver` level,
/// which is what the console's own lock is at.
fn drain<Token>(token: Token) -> Token
where
    Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    let (console, mut token) = Consoles::get(token);
    let Some(console) = console else {
        return token;
    };

    loop {
        // SeqCst on the flag and on the queue check below: a producer that
        // queued its message and then found the flag set relies on this drain
        // seeing that message when it looks again after letting go.
        if DRAINING.swap(true, Ordering::SeqCst) {
            return token;
        }

        loop {
            // SAFETY: holding `DRAINING` makes this the only consumer, on any
            // core, and a drain is never re-entered: a nested one finds the
            // flag set and returns above.
            let (entry, overwritten) = unsafe { MESSAGE_QUEUE.pop() };

            if overwritten {
                token = match console.write(&GAP_MARKER, token) {
                    Ok((_, token)) | Err((_, token)) => token,
                };
            }

            let Some(entry) = entry else {
                break;
            };

            let (block, size) = untag(entry);

            // SAFETY: the queue has given the buffer up, so it is this
            // drain's alone, and the tag names the size of the block it was
            // allocated as.
            if let Some(msg) = unsafe { message(block, size) } {
                token = match console.write(&msg, token) {
                    Ok((_, token)) | Err((_, token)) => token,
                };
            }

            // SAFETY: as above, the buffer is out of the queue and nobody else
            // holds it.
            unsafe { MESSAGE_ALLOCATOR.deallocate(NonNull::slice_from_raw_parts(block, size)) };
        }

        DRAINING.store(false, Ordering::SeqCst);

        if MESSAGE_QUEUE.is_empty() {
            return token;
        }
    }
}

/// The message in a buffer [`enqueue`] filled: the string up to its NUL.
///
/// [`None`] if the bytes are not UTF-8, which `enqueue` never writes.
///
/// # Safety
///
/// `block` must be a block of `size` bytes that came out of
/// [`MESSAGE_QUEUE`], and stay untouched for as long as the string is used.
unsafe fn message<'a>(block: NonNull<u8>, size: usize) -> Option<&'a str> {
    // SAFETY: see the function's contract.
    let bytes = unsafe { core::slice::from_raw_parts(block.as_ptr(), size) };

    // `enqueue` wrote a whole `&str` followed by the only NUL in it.
    let len = bytes.iter().position(|&c| c == 0).unwrap_or(size);

    str::from_utf8(&bytes[..len]).ok()
}

/// Shows everything still queued and then `message` on the console,
/// bypassing every lock, and halts the calling core.
///
/// What [`printk!`] and [`printkln!`] do at [`LogLevel::Panic`], and the way
/// out of the panic path. Interrupts are masked first, the console is looked
/// up without [`Consoles`]'s lock, and two newlines set the output apart
/// from what was on the console. Then [`MESSAGE_QUEUE`] is drained, so that
/// the last messages before the panic are not lost, and `message` comes last.
/// Everything goes through
/// [`emergency_write`](ConsoleOutput::emergency_write). Without a console
/// there is nowhere to show anything, and the core simply halts.
///
/// The drain takes the consumer side of the queue regardless of
/// [`DRAINING`]: a drain the panic interrupted is on this core and never
/// runs again. At worst the one message it had already taken out is lost.
/// Buffers are not freed, since nothing runs afterwards, and the allocator
/// may be what panicked.
pub fn __printk_emergency(message: Arguments<'_>) -> ! {
    // SAFETY: this core halts at the end, so the interrupt state never has to
    // be restored.
    unsafe { CPU::raw_disable_interrupts() };

    // TODO(@MaxMade): Stop the other cores once there are any: send every
    // other core a dedicated IPI whose handler acknowledges and halts, wait
    // for all acknowledgements with a timeout, and then print.

    // SAFETY: interrupts are masked on the only core there is, so nothing
    // registers a console meanwhile.
    if let Some(console) = unsafe { Consoles::emergency_get() } {
        // SAFETY for every `emergency_write` below: interrupts are masked on
        // the only core there is, so nothing else writes to the console.
        unsafe { console.emergency_write(&"\n\n") };

        loop {
            // SAFETY: the only other consumer is a drain this panic
            // interrupted, which never resumes, see above.
            let (entry, overwritten) = unsafe { MESSAGE_QUEUE.pop() };

            if overwritten {
                unsafe { console.emergency_write(&GAP_MARKER) };
            }

            let Some(entry) = entry else {
                break;
            };

            let (block, size) = untag(entry);

            // SAFETY: the queue has given the buffer up, and it is never
            // freed.
            if let Some(msg) = unsafe { self::message(block, size) } {
                unsafe { console.emergency_write(&msg) };
            }
        }

        // Formatted straight onto the console rather than into a `Buffer`,
        // so that a long message, such as one with a register dump, is not
        // cut short.
        let _ = EmergencyWriter(&console).write_fmt(message);
    }

    // SAFETY: there is nothing left to run.
    unsafe { CPU::halt() }
}

/// Formats onto a console through
/// [`emergency_write`](ConsoleOutput::emergency_write), see
/// [`__printk_emergency`].
struct EmergencyWriter<'a>(&'a ConsoleOutputDriver);

impl Write for EmergencyWriter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // SAFETY: only used by `__printk_emergency`, with interrupts masked
        // on the only core there is.
        unsafe { self.0.emergency_write(&s) };

        Ok(())
    }
}

/// Bytes of stack a [`printk!`] formats into.
///
/// The cap on one message: a longer one is cut on a character boundary. Well
/// under what a block can hold — [`__printk_lock_free`] would take up to
/// `MAX_SIZE - 1` — because this sits on the stack of whoever is logging, and
/// that can be an interrupt handler.
pub const BUFFER_SIZE: usize = 512;

/// Formatting scratch space for the [`printk!`] macros.
///
/// A message is built here and only then copied into a block of
/// [`MESSAGE_ALLOCATOR`], so that formatting itself needs neither the allocator
/// nor a lock — `printk!` has to work where neither can be had.
///
/// Writing past the end truncates instead of failing: a cut log line beats an
/// error that nobody is in a position to report.
pub struct Buffer {
    buf: [u8; BUFFER_SIZE],
    len: usize,
}

impl Buffer {
    /// An empty buffer.
    pub const fn new() -> Self {
        Self {
            buf: [0; BUFFER_SIZE],
            len: 0,
        }
    }

    /// Formats `args` into the buffer, after whatever is already in it.
    pub fn format(&mut self, args: Arguments<'_>) {
        // `write_str` never reports an error, so there is none to handle.
        let _ = self.write_fmt(args);
    }

    /// What has been written so far.
    pub fn as_str(&self) -> &str {
        // SAFETY: `write_str` appends whole characters only — a fragment that
        // does not fit is cut on a character boundary — so what is in the buffer
        // is a prefix-wise valid UTF-8 string.
        unsafe { str::from_utf8_unchecked(&self.buf[..self.len]) }
    }
}

impl Default for Buffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Write for Buffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let fit = floor_char_boundary(s, BUFFER_SIZE - self.len);

        self.buf[self.len..self.len + fit].copy_from_slice(&s.as_bytes()[..fit]);
        self.len += fit;

        Ok(())
    }
}

/// Formats `args` and queues the message through [`__printk_lock_free`], or
/// shows it through [`__printk_emergency`] at [`LogLevel::Panic`].
///
/// What [`printk!`] calls without a token. Never inlined, on purpose: the
/// [`Buffer`] the message is formatted into is [`BUFFER_SIZE`] bytes, and
/// here it only takes up stack for the duration of the call. Expanded into
/// the caller instead, it would sit in the caller's frame for as long as the
/// caller runs, once for every `printk!` in it.
#[inline(never)]
pub fn __printk_fmt_lock_free(level: LogLevel, args: Arguments<'_>) {
    if level == LogLevel::Panic {
        __printk_emergency(args);
    }

    let mut buffer = Buffer::new();

    buffer.format(args);

    __printk_lock_free(buffer.as_str());
}

/// Formats `args` and queues the message through [`__printk`], or shows it
/// through [`__printk_emergency`] at [`LogLevel::Panic`].
///
/// What [`printk!`] calls with a token that reaches the `Driver` level, see
/// [`Select`]. Never inlined, for the reason [`__printk_fmt_lock_free`]
/// gives.
///
/// # Token
///
/// The `token` is consumed and returned, as by [`__printk`].
#[inline(never)]
pub fn __printk_fmt<Token>(level: LogLevel, args: Arguments<'_>, token: Token) -> Token
where
    Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    if level == LogLevel::Panic {
        __printk_emergency(args);
    }

    let mut buffer = Buffer::new();

    buffer.format(args);

    __printk(buffer.as_str(), token)
}

/// Picks the path a [`printk!`] with a token may take.
///
/// [`__printk`] wants a token that reaches the `Driver` level, which is what
/// draining the queue to a console will need. Plenty of callers do not have one:
/// everything at `Scheduler` and below, which includes the `Prologue` level an
/// interrupt's bottom half runs at, and code already holding the heap. A message
/// from there still has to go somewhere, and [`__printk_lock_free`] is where.
///
/// So rather than making every caller pick, [`printk!`] asks here. [`ViaDriver`]
/// and [`ViaLockFree`] both offer `emit`; `ViaDriver` is implemented for
/// `Select<T>` and `ViaLockFree` for `&Select<T>`, so method resolution reaches
/// the first without an autoref and the second only with one. A candidate whose
/// bound does not hold is passed over, which leaves `ViaDriver` for the tokens
/// that reach `Driver` and `ViaLockFree` for the rest.
///
/// # What this costs
///
/// The fallback is silent: both paths queue the same message, and they differ
/// only in whether an empty allocator may grow from the heap. What is worth
/// knowing is that the choice is made where the macro is written, from the type
/// in hand. Inside a generic function whose token is not bounded by
/// `CanAcquire<Driver>`, the compiler cannot see the level, so the lock-free
/// path is taken even for a caller whose token would have reached the driver
/// level. Bound the parameter if the message has to take the growing path.
pub struct Select<T>(PhantomData<T>);

impl<T> Select<T> {
    /// A selector for the token's type, without taking the token itself.
    ///
    /// Takes it by reference so that the type is pinned down before the token is
    /// moved into [`emit`](ViaDriver::emit) — nothing is kept borrowed.
    pub fn for_token(_: &T) -> Self {
        Self(PhantomData)
    }
}

/// The path for a token that reaches the `Driver` level. See [`Select`].
pub trait ViaDriver<T> {
    /// Formats and queues the message through [`__printk_fmt`], returning
    /// the token.
    fn emit(self, level: LogLevel, args: Arguments<'_>, token: T) -> T;
}

impl<T> ViaDriver<T> for Select<T>
where
    T: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    #[inline(always)]
    fn emit(self, level: LogLevel, args: Arguments<'_>, token: T) -> T {
        __printk_fmt(level, args, token)
    }
}

/// The path for every other token. See [`Select`].
pub trait ViaLockFree<T> {
    /// Formats and queues the message through [`__printk_fmt_lock_free`],
    /// handing the token straight back untouched.
    fn emit(self, level: LogLevel, args: Arguments<'_>, token: T) -> T;
}

impl<T> ViaLockFree<T> for &Select<T> {
    #[inline(always)]
    fn emit(self, level: LogLevel, args: Arguments<'_>, token: T) -> T {
        __printk_fmt_lock_free(level, args);

        token
    }
}

/// The body of [`printk!`] and [`printkln!`], which differ only in the
/// [`Arguments`] they build.
///
/// Two arms: without a token it is an expression of type `()`, with one it
/// evaluates to the token, so that a caller can thread it on.
///
/// Only the level check is expanded at the call site. Formatting happens in
/// [`__printk_fmt`] or [`__printk_fmt_lock_free`], which are never inlined,
/// so that the caller's frame does not carry a [`Buffer`].
#[doc(hidden)]
#[macro_export]
macro_rules! __printk_emit {
    ($level:expr, $args:expr) => {{
        let level = $level;

        if $crate::kernel::printk::enabled(level) {
            $crate::kernel::printk::__printk_fmt_lock_free(level, $args);
        }
    }};

    ($level:expr, $token:expr, $args:expr) => {{
        // Both are in scope so that resolution can choose between them; whichever
        // one loses is unused, which is what the `allow` is for.
        #[allow(unused_imports)]
        use $crate::kernel::printk::{ViaDriver as _, ViaLockFree as _};

        let level = $level;
        let token = $token;

        if $crate::kernel::printk::enabled(level) {
            $crate::kernel::printk::Select::for_token(&token).emit(level, $args, token)
        } else {
            token
        }
    }};
}

/// Formats a message and queues it, if [`LOG_LEVEL`] lets a message at `level`
/// through.
///
/// ```ignore
/// printk!(LogLevel::Info, "disk {} has {} blocks", name, blocks);
///
/// let token = printk!(LogLevel::Warn, token, "queue {} is full", id);
/// ```
///
/// With a token it evaluates to that token, so it threads through code that has
/// one; without, it evaluates to `()`. Which of [`__printk`] and
/// [`__printk_lock_free`] the token form takes depends on whether the token
/// reaches the `Driver` level — see [`Select`], and note the caveat there about
/// generic functions.
///
/// A message is formatted on the stack, into [`BUFFER_SIZE`] bytes, and cut on a
/// character boundary if it does not fit. Nothing is formatted at all when the
/// level is not being logged.
///
/// At [`LogLevel::Panic`] the message is not queued: it goes to the console
/// straight away through [`__printk_emergency`], after whatever was still
/// queued, and the calling core halts. That needs neither a token nor a lock.
///
/// [`printkln!`] is the same with a newline on the end.
#[macro_export]
macro_rules! printk {
    ($level:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__printk_emit!($level, ::core::format_args!($fmt $(, $arg)*))
    };

    ($level:expr, $token:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__printk_emit!($level, $token, ::core::format_args!($fmt $(, $arg)*))
    };
}

/// [`printk!`] with a newline appended to the format string.
///
/// ```ignore
/// printkln!(LogLevel::Info, "booted in {} ms", millis);
///
/// let token = printkln!(LogLevel::Error, token, "cpu {} did not come up", id);
/// ```
#[macro_export]
macro_rules! printkln {
    ($level:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__printk_emit!(
            $level,
            ::core::format_args!(::core::concat!($fmt, "\n") $(, $arg)*)
        )
    };

    ($level:expr, $token:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__printk_emit!(
            $level,
            $token,
            ::core::format_args!(::core::concat!($fmt, "\n") $(, $arg)*)
        )
    };
}

pub fn init<Token>(level: LogLevel, token: Token) -> Token
where
    Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
{
    let mut token = token;

    LOG_LEVEL.store(level as _, Ordering::Relaxed);

    let layout = unsafe {
        Layout::from_size_align_unchecked(LockFreeAllocator::MAX_SIZE, LockFreeAllocator::MIN_SIZE)
    };
    for _ in 0..16 {
        let ptr;
        (ptr, token) = match Heap::allocate(&Heap, layout, token) {
            Ok(result) => result,
            Err((_, token)) => {
                return token;
            }
        };

        let mem: NonNull<[u8]> = NonNull::slice_from_raw_parts(ptr, LockFreeAllocator::MAX_SIZE);

        unsafe { MESSAGE_ALLOCATOR.deallocate(mem) };
    }

    token
}

#[cfg(test)]
mod tests {
    use std::{boxed::Box, string::String, sync::Barrier, thread, vec::Vec};

    use super::*;
    use crate::kernel::locking::{EpilogueLevelID, PrologueLevelID, RootToken, Shared, Token};

    /// A node pointer that may be handed to another thread.
    ///
    /// [`NonNull`] is deliberately not [`Send`], but every node here lives in
    /// leaked storage that outlives all threads of a test.
    #[derive(Clone, Copy)]
    struct SharedNode(NonNull<LockFreeStackNode>);

    // SAFETY: see above.
    unsafe impl Send for SharedNode {}

    // SAFETY: as above; the node behind the pointer is only ever touched
    // through the stack, which synchronises the accesses itself.
    unsafe impl Sync for SharedNode {}

    /// `count` fresh nodes.
    ///
    /// The storage is leaked rather than freed: a node that has been pushed has
    /// to stay readable for as long as the stack lives, because a pop can read
    /// it after another thread has taken it (see the
    /// [module documentation](self)). `Box` also gives the nodes the 64-byte
    /// alignment the counter in the head word depends on.
    fn leak_nodes(count: usize) -> Vec<SharedNode> {
        (0..count)
            .map(|_| SharedNode(NonNull::from(Box::leak(Box::new(LockFreeStackNode::new())))))
            .collect()
    }

    /// Pops until the stack is empty, collecting the node addresses.
    fn drain(stack: &LockFreeStack) -> Vec<usize> {
        let mut addresses = Vec::new();

        while let Some(node) = unsafe { stack.pop() } {
            addresses.push(node.as_ptr() as usize);
        }

        addresses
    }

    /// The addresses of `nodes` in ascending order, to compare a pool against
    /// what came back out of the stack in some arbitrary concurrent order.
    fn sorted_addresses(nodes: &[SharedNode]) -> Vec<usize> {
        sorted(nodes.iter().map(|node| node.0.as_ptr() as usize).collect())
    }

    /// As [`sorted_addresses`], for addresses that are already collected.
    fn sorted(mut addresses: Vec<usize>) -> Vec<usize> {
        addresses.sort_unstable();
        addresses
    }

    /// A stack that has seen nothing is empty and hands out nothing.
    #[test]
    fn fresh_stack_is_empty() {
        let stack = LockFreeStack::new();

        assert!(stack.is_empty());
        assert!(unsafe { stack.pop() }.is_none());
    }

    /// A pushed node comes back out as it went in, counter bits and all
    /// removed.
    #[test]
    fn push_then_pop_returns_the_same_node() {
        let stack = LockFreeStack::new();
        let node = leak_nodes(1)[0];

        unsafe { stack.push(node.0) };
        assert!(!stack.is_empty());

        assert_eq!(unsafe { stack.pop() }, Some(node.0));
        assert!(stack.is_empty());
    }

    /// Nodes come off in the reverse of the order they went on.
    #[test]
    fn pop_order_is_last_in_first_out() {
        let stack = LockFreeStack::new();
        let nodes = leak_nodes(4);

        for node in &nodes {
            unsafe { stack.push(node.0) };
        }

        for node in nodes.iter().rev() {
            assert_eq!(unsafe { stack.pop() }, Some(node.0));
        }

        assert!(stack.is_empty());
    }

    /// An exhausted stack keeps saying so rather than handing a node out
    /// twice.
    #[test]
    fn pop_on_an_empty_stack_keeps_returning_none() {
        let stack = LockFreeStack::new();
        let node = leak_nodes(1)[0];

        unsafe { stack.push(node.0) };
        assert_eq!(unsafe { stack.pop() }, Some(node.0));

        for _ in 0..4 {
            assert!(unsafe { stack.pop() }.is_none());
        }
    }

    /// The counter left behind in the head word must not read as a node: once a
    /// push and a pop have gone through, the head word is non-null while the
    /// stack is empty.
    #[test]
    fn emptiness_is_not_a_null_head_word() {
        let stack = LockFreeStack::new();
        let node = leak_nodes(1)[0];

        unsafe { stack.push(node.0) };
        assert_eq!(unsafe { stack.pop() }, Some(node.0));

        assert!(
            !stack.head.load(Ordering::Relaxed).is_null(),
            "the counter should have stayed behind in the head word"
        );
        assert!(stack.is_empty());
        assert!(unsafe { stack.pop() }.is_none());
    }

    /// A link between two nodes carries no counter, so a popper can install one
    /// as the new head address without masking it.
    #[test]
    fn links_never_carry_a_counter() {
        let stack = LockFreeStack::new();
        let nodes = leak_nodes(2);

        // The second push sees a head word whose counter is 1, and that counter
        // must not end up in the link.
        unsafe { stack.push(nodes[0].0) };
        unsafe { stack.push(nodes[1].0) };

        let next = unsafe { nodes[1].0.as_ref().next.load(Ordering::Relaxed) };
        assert_eq!(next, nodes[0].0.as_ptr());

        let bottom = unsafe { nodes[0].0.as_ref().next.load(Ordering::Relaxed) };
        assert!(bottom.is_null());
    }

    /// The counter wraps every 64 operations, and the stack has to keep working
    /// across the wrap.
    #[test]
    fn the_counter_wraps_without_losing_a_node() {
        let stack = LockFreeStack::new();
        let node = leak_nodes(1)[0];

        for _ in 0..4 * (LockFreeStack::CNT_MASK + 1) {
            unsafe { stack.push(node.0) };
            assert_eq!(unsafe { stack.pop() }, Some(node.0));
        }

        assert!(stack.is_empty());
    }

    /// A pointer that is not a node address would eat into the counter. Only
    /// checked in a debug build, which is also the only build in which the
    /// assertion stops the push before it writes through that pointer.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn pushing_a_misaligned_pointer_is_caught() {
        let stack = LockFreeStack::new();
        let node = leak_nodes(1)[0];

        // One byte into a real node, so the pointer is not wild — just not a
        // node address.
        let misaligned = unsafe { NonNull::new_unchecked(node.0.as_ptr().byte_add(1)) };

        unsafe { stack.push(misaligned) };
    }

    /// Concurrent pushes all get through: no node is lost and none is linked
    /// twice.
    #[test]
    fn concurrent_pushes_lose_no_node() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 256;

        let stack = LockFreeStack::new();
        let nodes = leak_nodes(THREADS * PER_THREAD);
        let barrier = Barrier::new(THREADS);

        thread::scope(|scope| {
            for chunk in nodes.chunks(PER_THREAD) {
                // Borrows for the worker, so that `move` takes nothing but the
                // chunk this thread is to push.
                let (stack, barrier) = (&stack, &barrier);

                scope.spawn(move || {
                    barrier.wait();

                    for node in chunk {
                        unsafe { stack.push(node.0) };
                    }
                });
            }
        });

        assert_eq!(sorted(drain(&stack)), sorted_addresses(&nodes));
    }

    /// Every node goes to exactly one popper: a stale successor installed by a
    /// racing pop would either lose nodes or hand one out twice.
    #[test]
    fn concurrent_pops_hand_out_each_node_once() {
        const THREADS: usize = 8;
        const NODES: usize = 2048;

        let stack = LockFreeStack::new();
        let nodes = leak_nodes(NODES);

        for node in &nodes {
            unsafe { stack.push(node.0) };
        }

        let barrier = Barrier::new(THREADS);
        let mut taken = Vec::new();

        thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        drain(&stack)
                    })
                })
                .collect();

            for handle in handles {
                taken.extend(handle.join().expect("a popping thread panicked"));
            }
        });

        assert_eq!(sorted(taken), sorted_addresses(&nodes));
        assert!(stack.is_empty());
    }

    /// A pool of nodes cycling through the stack from several threads at once.
    /// The pool has to come out whole, which it does not if a push and a pop
    /// ever tear the chain between them.
    ///
    /// The pool is kept far larger than the 64 values the counter can hold, so
    /// that the test does not sit on the ABA window the
    /// [module documentation](self) describes.
    #[test]
    fn a_pool_survives_concurrent_push_and_pop() {
        const THREADS: usize = 4;
        const POOL: usize = 1024;
        const ROUNDS: usize = 5_000;

        let stack = LockFreeStack::new();
        let nodes = leak_nodes(POOL);

        for node in &nodes {
            unsafe { stack.push(node.0) };
        }

        let barrier = Barrier::new(THREADS);

        thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    barrier.wait();

                    for _ in 0..ROUNDS {
                        // The pool is never exhausted for long: whoever took
                        // the last node is about to put one back.
                        loop {
                            if let Some(node) = unsafe { stack.pop() } {
                                unsafe { stack.push(node) };
                                break;
                            }

                            core::hint::spin_loop();
                        }
                    }
                });
            }
        });

        assert_eq!(sorted(drain(&stack)), sorted_addresses(&nodes));
    }

    /// The stack itself crosses cores, as a `static` one has to.
    ///
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn stack_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<LockFreeStack>();
    }

    /// What a format actually produces.
    #[test]
    fn a_buffer_holds_what_was_formatted_into_it() {
        let mut buffer = Buffer::new();

        buffer.format(format_args!("{} and {}", 1, "two"));

        assert_eq!(buffer.as_str(), "1 and two");
    }

    /// A second format carries on where the first left off, which is what lets
    /// a prefix be written before the caller's message.
    #[test]
    fn formatting_appends() {
        let mut buffer = Buffer::new();

        buffer.format(format_args!("a"));
        buffer.format(format_args!("b{}", 1));

        assert_eq!(buffer.as_str(), "ab1");
    }

    /// Overflowing the buffer cuts the message rather than reporting an error
    /// the caller is in no position to do anything with.
    #[test]
    fn a_buffer_truncates_rather_than_failing() {
        let mut buffer = Buffer::new();
        let long: String = core::iter::repeat('x').take(BUFFER_SIZE * 2).collect();

        buffer.format(format_args!("{long}"));

        assert_eq!(buffer.as_str().len(), BUFFER_SIZE);
    }

    /// The cut walks back to a character boundary: `as_str` hands out a `&str`
    /// without checking, so half a character left in the buffer would be
    /// undefined behaviour rather than a stray byte.
    #[test]
    fn a_buffer_truncates_on_a_character_boundary() {
        let mut buffer = Buffer::new();

        // Three bytes each, and `BUFFER_SIZE` is not a multiple of three, so a
        // cut at the limit lands inside a character.
        let wide: String = core::iter::repeat('\u{20ac}').take(BUFFER_SIZE).collect();

        buffer.format(format_args!("{wide}"));

        let got = buffer.as_str();

        assert!(wide.starts_with(got), "the cut did not land on a boundary");
        assert_eq!(got.len() % 3, 0, "a character was split");
        assert!(got.len() > BUFFER_SIZE - 3, "cut further back than needed");
    }

    /// A message is logged when it is at least as important as the threshold.
    #[test]
    fn the_level_gate_admits_what_is_important_enough() {
        LOG_LEVEL.store(LogLevel::Warn as u8, Ordering::Relaxed);

        assert!(!enabled(LogLevel::Trace));
        assert!(!enabled(LogLevel::Debug));
        assert!(!enabled(LogLevel::Info));
        assert!(enabled(LogLevel::Warn));
        assert!(enabled(LogLevel::Error));
        assert!(enabled(LogLevel::Panic));

        LOG_LEVEL.store(LogLevel::Info as u8, Ordering::Relaxed);
    }

    /// Nothing has been given to [`MESSAGE_ALLOCATOR`], and the lock-free path
    /// does not grow it, so there is nowhere to put this message. Dropping it is
    /// the contract; panicking on the logging path would not be.
    #[test]
    fn printk_with_no_memory_drops_the_message() {
        printk!(LogLevel::Error, "no memory to put {} in", "this");
        printkln!(LogLevel::Error, "nor {}", "this");
    }

    /// Both forms compile with a token that reaches the `Driver` level, and the
    /// token comes back out of the ones that take it.
    ///
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn the_macros_take_a_token_that_reaches_the_driver_level(
        token: Token<EpilogueLevelID, RootToken, Shared>,
    ) -> Token<EpilogueLevelID, RootToken, Shared> {
        printk!(LogLevel::Info, "no token here");
        printkln!(LogLevel::Info, "nor here, but {} argument", 1);

        let token = printk!(LogLevel::Warn, token, "a token and {} argument", 1);

        printkln!(
            LogLevel::Error,
            token,
            "and a newline: {:?}",
            LogLevel::Trace
        )
    }

    /// The same below the `Driver` level, where [`__printk`] cannot be called at
    /// all — `MemoryManagement` sits under `Driver`, so the bound does not hold.
    ///
    /// This is the case the fallback in [`Select`] exists for: without it the
    /// token form is a hard error here rather than quietly taking the lock-free
    /// path.
    ///
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn the_macros_take_a_token_that_cannot_reach_the_driver_level(
        token: Token<MemoryManagementLevelID, RootToken, Shared>,
    ) -> Token<MemoryManagementLevelID, RootToken, Shared> {
        printkln!(LogLevel::Info, token, "below the driver level")
    }

    /// The `Prologue` level is the bottom of the hierarchy, so a token from it
    /// reaches nothing at all — least of all `Driver`. It is also exactly where
    /// a lock must not be taken, an interrupt's bottom half being the case the
    /// lock-free path exists for.
    ///
    /// That this compiles is the proof that [`ViaLockFree`] is selected:
    /// [`__printk`] called with this token directly is rejected with "lock level
    /// `Prologue` is not above `Driver`".
    ///
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn the_macros_send_a_prologue_token_down_the_lock_free_path(
        token: Token<PrologueLevelID, RootToken, Shared>,
    ) -> Token<PrologueLevelID, RootToken, Shared> {
        printkln!(LogLevel::Error, token, "from the bottom half")
    }

    /// A trailing comma is accepted wherever an argument list can end.
    ///
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn the_macros_accept_a_trailing_comma() {
        printk!(LogLevel::Info, "no arguments",);
        printk!(LogLevel::Info, "one {}", 1,);
        printkln!(LogLevel::Info, "one {}", 1,);
    }
}
