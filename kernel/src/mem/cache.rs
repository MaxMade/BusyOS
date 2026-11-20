//! A lock-free stack that keeps its nodes for reuse.
//!
//! [`Cache`] is a Treiber stack: a singly linked list whose head is replaced
//! by one compare-and-swap per [`try_push`](Cache::try_push) and
//! [`pop`](Cache::pop), so no core ever waits for another one holding a lock.
//!
//! # Two lists
//!
//! Every node sits on exactly one of two such stacks:
//!
//! - `used` holds the nodes carrying a value, `Some(value)`, newest on top.
//! - `free` holds the nodes a [`pop`](Cache::pop) has emptied, `None`.
//!
//! A push takes its node from `free` and only goes to the allocator when that
//! is empty, so once the cache has grown to the number of values it holds at
//! its peak, pushing and popping allocate nothing and need no lock at all. The
//! token [`try_push`](Cache::try_push) takes is only used when it does
//! allocate.
//!
//! Nodes are never given back while the cache is shared. That is what makes
//! the stack sound without any other reclamation scheme: a core that read the
//! head just before another core popped that node may still read the node's
//! `next`, so the memory has to stay a node for as long as anyone may pop.
//!
//! # ABA and tags
//!
//! A pop reads the head `A` and its successor `B`, then swaps `A` for `B`.
//! If, between the read and the swap, other cores pop `A`, pop `B` and push
//! `A` again, the head is `A` once more and a plain compare-and-swap would
//! succeed, installing `B`, which is no longer on the stack.
//!
//! The head therefore carries a tag next to the pointer, which every
//! successful swap increments, so in the scenario above the head reads `A`
//! with a different tag and the stale swap fails. The tag lives in two places
//! no address needs:
//!
//! - its low part in the low bits of the pointer, which are always zero
//!   because nodes are aligned to 64 bytes, and
//! - its high part in the bits the architecture leaves untranslated, through
//!   [`CPU::try_encode`](crate::arch::generic::cpu::CPU::try_encode).
//!
//! On x86_64 that is 6 and 16 bits, so the tag wraps after about four million
//! swaps. This makes ABA unlikely rather than impossible: it needs a core to
//! stall over exactly a multiple of that many swaps of the same head and find
//! the same node on top. Six bits alone were not enough, as a stall over a
//! multiple of 64 swaps is easily long enough to happen.
//!
//! # As an allocator
//!
//! [`Cache`] is also an [`Allocator`] of blocks the size of one `T`. Such a
//! block is the element of a node taken off `free`, which then sits on no list
//! until it is deallocated and goes back onto `free`, its element `None` once
//! more. Values and blocks thus share one pool of nodes: a node emptied by a
//! [`pop`](Cache::pop) may be handed out by the next allocation and the other
//! way round.
//!
//! A layout larger or more strictly aligned than `T` does not fit a node and
//! is passed through to the allocator the cache takes its nodes from.
//! Deallocation decides by the same layout, so every block goes back to where
//! it came from.
//!
//! # Tokens and dropping
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot return the nodes: a
//! cache that holds any node, used *or* free, has to be given to
//! [`clear`](Cache::clear) before it goes out of scope, and dropping one that
//! was not panics rather than leaking them. Taking `&mut self`, `clear` also
//! makes sure no core is still in the middle of a push or pop.
//!
//! A block handed out by [`allocate`](Allocator::allocate) is on neither list,
//! so `clear` cannot reach it and dropping does not notice it: it has to be
//! deallocated before the cache is cleared, or its node is leaked.

use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::mem::offset_of;
use core::ptr::{NonNull, null_mut};
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::{
    arch::{
        CPU,
        generic::{cpu::CPU as _, paging::VirtualAddress},
    },
    kernel::compiler::likely::{likely, unlikely},
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// Minimum alignment of a [`Node`]. Has to match its `repr(align)`.
const NODE_ALIGN: usize = 64;

/// Number of low address bits that are always zero in a node pointer and carry
/// the low part of the tag instead.
const TAG_BITS: u32 = NODE_ALIGN.trailing_zeros();

/// Selects the low part of the tag in a tagged pointer.
const TAG_MASK: usize = (1 << TAG_BITS) - 1;

/// One entry of either list.
///
/// The alignment is what frees the low tag bits. 64 bytes is also a cache line, so
/// two nodes never share one.
#[repr(C, align(64))]
struct Node<T> {
    /// The node below this one, untagged. Atomic because a pop may read it
    /// while the core that owns the node writes it: see the
    /// [module documentation](self).
    next: AtomicPtr<Node<T>>,

    /// `Some` while the node is on `used`, `None` while it is on `free`.
    /// Only touched by the core that has taken the node off a list.
    element: UnsafeCell<Option<T>>,
}

const _: () = assert!(align_of::<Node<u8>>() == NODE_ALIGN);

/// A Treiber stack of nodes, whose head carries a tag in its low bits and in
/// the bits [`CPU::try_encode`](crate::arch::generic::cpu::CPU::try_encode)
/// stores values in.
struct Stack<T> {
    head: AtomicPtr<Node<T>>,
}

impl<T> Stack<T> {
    const fn new() -> Self {
        Self {
            head: AtomicPtr::new(null_mut()),
        }
    }

    /// The tag of a head, its high part above the low one.
    fn tag(head: *mut Node<T>) -> usize {
        let Ok(high) = CPU::try_decode(VirtualAddress::new(head)) else {
            unreachable!("Stack head carries a value try_encode cannot store");
        };
        (high << TAG_BITS) | (head.addr() & TAG_MASK)
    }

    /// Strips the tag off a head.
    fn untag(head: *mut Node<T>) -> *mut Node<T> {
        // Encoding `0` cannot exceed the maximum and restores the address.
        let Ok(plain) = CPU::try_encode(VirtualAddress::new(head), 0) else {
            unreachable!("try_encode refused to store 0");
        };
        plain.as_ptr().map_addr(|addr| addr & !TAG_MASK)
    }

    /// Tags `node` to replace `head`, with the tag one past that of `head`.
    fn retag(node: *mut Node<T>, head: *mut Node<T>) -> *mut Node<T> {
        let tag = Self::tag(head) + 1;
        let node = node.map_addr(|addr| addr | (tag & TAG_MASK));

        let tagged = CPU::try_encode(VirtualAddress::new(node), tag >> TAG_BITS);

        // The high part ran past its maximum and wraps around to `0`, which is
        // what the node already carries. Once every few million swaps.
        if unlikely(tagged.is_err()) {
            return node;
        }

        tagged.map_or(node, |tagged| tagged.as_ptr())
    }

    /// Whether the stack holds no node.
    fn is_empty(&self) -> bool {
        Self::untag(self.head.load(Ordering::Relaxed)).is_null()
    }

    /// Pushes `node`, which the caller has to own: it may not be on any list.
    fn push(&self, node: NonNull<Node<T>>) {
        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            // SAFETY: the node is valid and only ours until the swap below
            // publishes it.
            unsafe { node.as_ref() }
                .next
                .store(Self::untag(head), Ordering::Relaxed);

            // Release, so that a pop that sees the node also sees its `next`
            // and its element.
            let swapped = self.head.compare_exchange_weak(
                head,
                Self::retag(node.as_ptr(), head),
                Ordering::Release,
                Ordering::Relaxed,
            );

            // Fails only if another core changed the head in between.
            if likely(swapped.is_ok()) {
                return;
            }

            let (Ok(current) | Err(current)) = swapped;
            head = current;
        }
    }

    /// Pops the top node, which then belongs to the caller.
    fn pop(&self) -> Option<NonNull<Node<T>>> {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            let node = NonNull::new(Self::untag(head))?;

            // SAFETY: nodes are not freed while the cache is shared, so the
            // node is still valid even if another core popped it meanwhile.
            // Its `next` may be stale then, but the tag makes the swap below
            // fail in that case.
            let next = unsafe { node.as_ref() }.next.load(Ordering::Relaxed);

            let swapped = self.head.compare_exchange_weak(
                head,
                Self::retag(next, head),
                Ordering::Acquire,
                Ordering::Acquire,
            );

            // Fails only if another core changed the head in between.
            if likely(swapped.is_ok()) {
                return Some(node);
            }

            let (Ok(current) | Err(current)) = swapped;
            head = current;
        }
    }

    /// Pops the top node without synchronising, for when the stack is no
    /// longer shared.
    fn pop_exclusive(&mut self) -> Option<NonNull<Node<T>>> {
        let head = self.head.get_mut();
        let mut node = NonNull::new(Self::untag(*head))?;

        // SAFETY: `&mut self` rules out every other access to the stack and so
        // to the nodes on it.
        *head = *unsafe { node.as_mut() }.next.get_mut();

        Some(node)
    }
}

/// A lock-free stack of values that keeps the nodes it has emptied for the
/// next push, taking new ones from `A`, the kernel [`Heap`] by default.
///
/// See the [module documentation](self) for how it works and what it costs.
pub struct Cache<T, A: Allocator<MemoryManagementLevelID> = Heap> {
    /// Nodes holding a value.
    used: Stack<T>,

    /// Empty nodes, kept for reuse.
    free: Stack<T>,

    allocator: A,
}

// SAFETY: sending the cache sends the values in it and the allocator.
unsafe impl<T: Send, A: Allocator<MemoryManagementLevelID> + Send> Send for Cache<T, A> {}

// SAFETY: a shared cache only moves values in and out of it and never hands
// out a reference to one, so `T: Send` is all a value needs. Every node is
// taken off a list by one core at a time, and the swaps that move it order the
// writes to its element before the next owner's reads. Pushing may allocate
// through `&A` from any core.
unsafe impl<T: Send, A: Allocator<MemoryManagementLevelID> + Sync> Sync for Cache<T, A> {}

impl<T> Cache<T, Heap> {
    /// Creates an empty cache on the kernel heap.
    ///
    /// Allocates nothing and may be used to build a `static`.
    pub const fn new() -> Self {
        Self::new_in(Heap)
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Cache<T, A> {
    /// Layout every node is allocated with.
    const NODE_LAYOUT: Layout = Layout::new::<Node<T>>();

    /// Creates an empty cache taking its nodes from `allocator`.
    ///
    /// Allocates nothing.
    pub const fn new_in(allocator: A) -> Self {
        Self {
            used: Stack::new(),
            free: Stack::new(),
            allocator,
        }
    }

    /// Whether the cache holds no value. With other cores pushing and popping,
    /// the answer may be stale by the time it is returned.
    pub fn is_empty(&self) -> bool {
        self.used.is_empty()
    }

    /// Pushes `value`, reusing an empty node if there is one and allocating a
    /// new one otherwise.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if no node was left and none could be allocated.
    /// The value is handed back.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_push<Token>(&self, value: T, token: Token) -> Result<Token, (Error, T, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let (node, token) = match self.take_node(token) {
            Ok(taken) => taken,
            Err((error, token)) => return Err((error, value, token)),
        };

        // SAFETY: the node was just taken off `free` or allocated, so it is
        // ours until it is pushed onto `used`. Others may still read its
        // `next`, but never its element.
        unsafe { *node.as_ref().element.get() = Some(value) };

        self.used.push(node);
        Ok(token)
    }

    /// Pops the value pushed last, or `None` if there is none.
    ///
    /// The emptied node is kept for the next push, so this needs no token.
    pub fn pop(&self) -> Option<T> {
        let node = self.used.pop()?;

        // SAFETY: the node was just taken off `used`, so it is ours until it
        // is pushed onto `free`.
        let value = unsafe { (*node.as_ref().element.get()).take() };
        debug_assert!(value.is_some(), "Node on the used list without a value");

        self.free.push(node);
        value
    }

    /// Drops every value and returns every node to the allocator, leaving an
    /// empty cache that may be dropped or filled again.
    ///
    /// This is what a cache has to end with: see the
    /// [module documentation](self).
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn clear<Token>(&mut self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let token = Self::release(&mut self.used, &self.allocator, token);
        Self::release(&mut self.free, &self.allocator, token)
    }

    /// Takes an empty node off `free`, or allocates one if there is none. The
    /// node is on no list then and belongs to the caller.
    fn take_node<Token>(&self, token: Token) -> Result<(NonNull<Node<T>>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let node = self.free.pop();

        // Only until the cache has grown to the most nodes it holds at once.
        if unlikely(node.is_none()) {
            return self.allocate_node(token);
        }

        let Some(node) = node else {
            unreachable!("Checked just above");
        };
        Ok((node, token))
    }

    /// Whether a block of `layout` fits the element of a node.
    fn fits(layout: Layout) -> bool {
        layout.size() <= size_of::<T>() && layout.align() <= align_of::<T>()
    }

    /// Allocates a node holding no value and on no list.
    fn allocate_node<Token>(
        &self,
        token: Token,
    ) -> Result<(NonNull<Node<T>>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let (ptr, token) = self.allocator.allocate(Self::NODE_LAYOUT, token)?;
        let node = ptr.cast::<Node<T>>();

        // SAFETY: the block is fresh, large enough and aligned for a node.
        unsafe {
            node.write(Node {
                next: AtomicPtr::new(null_mut()),
                element: UnsafeCell::new(None),
            })
        };

        Ok((node, token))
    }

    /// Drops every node on `stack` along with its element and returns it to
    /// `allocator`.
    fn release<Token>(stack: &mut Stack<T>, allocator: &A, mut token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        while let Some(node) = stack.pop_exclusive() {
            // SAFETY: the node came from `allocator` with `NODE_LAYOUT` and has
            // just been unlinked from a stack nothing else can reach.
            token = unsafe {
                node.drop_in_place();
                allocator.deallocate(node.cast(), Self::NODE_LAYOUT, token)
            };
        }
        token
    }
}

// SAFETY: a block that fits a node is the element of a node taken off `free`
// or freshly allocated, which is on no list until it is deallocated, so no
// other allocation and no value aliases it. Every other block comes from `A`,
// which upholds the contract for it, and goes back there because the layout
// decides both ways.
unsafe impl<T, A: Allocator<MemoryManagementLevelID>> Allocator<MemoryManagementLevelID>
    for Cache<T, A>
{
    /// Hands out the element of an empty node if `layout` fits a `T`, and
    /// passes the request on to `A` otherwise. See the
    /// [module documentation](self).
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if no node was left and none could be allocated,
    /// or if `A` could not serve a layout that does not fit.
    fn allocate<Token>(
        &self,
        layout: Layout,
        token: Token,
    ) -> Result<(NonNull<u8>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        if unlikely(!Self::fits(layout)) {
            return self.allocator.allocate(layout, token);
        }

        let (node, token) = self.take_node(token)?;

        // SAFETY: the node is valid. The element starts at an offset aligned
        // for `T`, holds at least `size_of::<T>()` bytes, and the pointer is
        // derived from the node's so that it can be turned back into it.
        let element = unsafe {
            NonNull::new_unchecked(UnsafeCell::raw_get(&raw const (*node.as_ptr()).element))
        };

        Ok((element.cast(), token))
    }

    /// Puts the node back onto `free`, or passes the block on to `A` if
    /// `layout` does not fit a `T`.
    unsafe fn deallocate<Token>(&self, ptr: NonNull<u8>, layout: Layout, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        if unlikely(!Self::fits(layout)) {
            // SAFETY: the same layout made `allocate` take the block from `A`.
            return unsafe { self.allocator.deallocate(ptr, layout, token) };
        }

        // SAFETY: `ptr` is the element of a node `allocate` handed out, so
        // stepping back by its offset yields the node, which is ours. Whatever
        // the caller left in the element is no `Option<T>`, so it is
        // overwritten rather than dropped.
        unsafe {
            let node = ptr.byte_sub(offset_of!(Node<T>, element)).cast::<Node<T>>();
            UnsafeCell::raw_get(&raw const (*node.as_ptr()).element).write(None);
            self.free.push(node);
        }

        token
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Drop for Cache<T, A> {
    /// Panics if the cache still holds a node, used or free.
    ///
    /// Nodes cannot be released here because `Drop::drop` cannot accept the
    /// lock token the allocator needs. Call [`Cache::clear`] before the cache
    /// goes out of scope.
    fn drop(&mut self) {
        if !self.used.is_empty() || !self.free.is_empty() {
            panic!("A Cache holding nodes must never be dropped. Use Cache::clear(...) instead!");
        }
    }
}

impl<T> Default for Cache<T, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use crate::{
        kernel::boxed::Box,
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type TestCache<T> = Cache<T, HeapAllocator>;

    /// Pushes, panicking if the allocation fails. A macro rather than a
    /// function because the error arm cannot be unwrapped: a token is not
    /// [`Debug`], so `expect` is unavailable.
    macro_rules! push {
        ($cache:expr, $value:expr, $token:expr) => {
            match $cache.try_push($value, $token) {
                Ok(token) => token,
                Err(_) => panic!("try_push() returned an allocation error"),
            }
        };
    }

    /// An allocator that has no memory.
    struct Exhausted;

    unsafe impl Allocator<MemoryManagementLevelID> for Exhausted {
        fn allocate<Token>(
            &self,
            _layout: Layout,
            token: Token,
        ) -> Result<(NonNull<u8>, Token), (Error, Token)>
        where
            Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
        {
            Err((Error::OutOfMemory, token))
        }

        unsafe fn deallocate<Token>(
            &self,
            _ptr: NonNull<u8>,
            _layout: Layout,
            _token: Token,
        ) -> Token
        where
            Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
        {
            unreachable!("Nothing was allocated")
        }
    }

    /// Allocates, panicking if the allocation fails; see [`push`].
    macro_rules! allocate {
        ($cache:expr, $layout:expr, $token:expr) => {
            match $cache.allocate($layout, $token) {
                Ok(allocated) => allocated,
                Err(_) => panic!("allocate() returned an allocation error"),
            }
        };
    }

    #[test]
    fn allocation_reuses_nodes() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::<u64>::new_in(HeapAllocator);
        let layout = Layout::new::<u64>();

        let (first, token) = allocate!(cache, layout, token);
        assert_eq!(first.as_ptr().addr() % align_of::<u64>(), 0);
        assert!(cache.used.is_empty());
        assert!(cache.free.is_empty());

        unsafe { first.cast::<u64>().write(0xdead_beef) };
        let token = unsafe { cache.deallocate(first, layout, token) };
        assert!(!cache.free.is_empty());

        let (second, token) = allocate!(cache, layout, token);
        assert_eq!(first, second);
        assert!(cache.free.is_empty());

        let token = unsafe { cache.deallocate(second, layout, token) };
        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn values_and_blocks_share_nodes() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::<u64>::new_in(HeapAllocator);
        let layout = Layout::new::<u64>();

        token = push!(cache, 1, token);
        assert_eq!(cache.pop(), Some(1));

        // The block is the element of the node the pop emptied.
        let (block, token) = allocate!(cache, layout, token);
        assert!(cache.free.is_empty());

        // A deallocated block is an empty node again, and the next push takes it.
        let token = unsafe { cache.deallocate(block, layout, token) };
        let token = push!(cache, 2, token);
        assert!(cache.free.is_empty());
        assert_eq!(cache.pop(), Some(2));

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn larger_layouts_pass_through() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::<u32>::new_in(HeapAllocator);

        // Too large, and too strictly aligned.
        for layout in [Layout::new::<[u32; 4]>(), Layout::new::<u64>()] {
            let block;
            (block, token) = allocate!(cache, layout, token);
            token = unsafe { cache.deallocate(block, layout, token) };

            // The block never was a node, so none is cached.
            assert!(cache.free.is_empty());
        }

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn backs_a_box() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::<[u64; 3]>::new_in(HeapAllocator);

        let (boxed, token) = match Box::try_new_in([1, 2, 3], &cache, token) {
            Ok(allocated) => allocated,
            Err(_) => panic!("Box::try_new_in() returned an allocation error"),
        };
        assert_eq!(*boxed, [1, 2, 3]);

        let token = boxed.release(token);
        assert!(!cache.free.is_empty());

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn concurrent_allocations_are_disjoint() {
        const THREADS: usize = 4;
        const PER_THREAD: usize = 10_000;

        let cache = Arc::new(TestCache::<(usize, usize)>::new_in(HeapAllocator));
        let layout = Layout::new::<(usize, usize)>();

        // Every thread writes its mark into each block it holds and checks it
        // is still there before giving the block back: two threads holding the
        // same node would overwrite each other's mark.
        let workers: std::vec::Vec<_> = (0..THREADS)
            .map(|t| {
                let cache = cache.clone();
                thread::spawn(move || {
                    let root = unsafe { RootToken::forge() };
                    let (level, mut token) = EpilogueLevel::enter(root);
                    let mut held = std::vec::Vec::new();

                    for i in 0..PER_THREAD {
                        let block;
                        (block, token) = allocate!(cache, layout, token);
                        let block = block.cast::<(usize, usize)>();
                        unsafe { block.write((t, i)) };
                        held.push((block, i));

                        if i % 4 == 3 {
                            for (block, i) in held.drain(..) {
                                assert_eq!(unsafe { block.read() }, (t, i));
                                token = unsafe { cache.deallocate(block.cast(), layout, token) };
                            }
                        }
                    }

                    for (block, i) in held.drain(..) {
                        assert_eq!(unsafe { block.read() }, (t, i));
                        token = unsafe { cache.deallocate(block.cast(), layout, token) };
                    }

                    level.leave(token);
                })
            })
            .collect();

        for worker in workers {
            worker.join().unwrap();
        }

        let mut cache = Arc::into_inner(cache).unwrap();
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn pops_in_reverse_push_order() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::new_in(HeapAllocator);

        assert!(cache.is_empty());
        assert_eq!(cache.pop(), None);

        for i in 0..5 {
            token = push!(cache, i, token);
        }
        assert!(!cache.is_empty());

        for i in (0..5).rev() {
            assert_eq!(cache.pop(), Some(i));
        }
        assert_eq!(cache.pop(), None);
        assert!(cache.is_empty());

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn reuses_popped_nodes() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::new_in(HeapAllocator);

        token = push!(cache, 1, token);
        let node = Stack::untag(cache.used.head.load(Ordering::Relaxed));

        assert_eq!(cache.pop(), Some(1));
        assert!(cache.used.is_empty());
        assert!(!cache.free.is_empty());

        token = push!(cache, 2, token);
        assert!(cache.free.is_empty());
        assert_eq!(Stack::untag(cache.used.head.load(Ordering::Relaxed)), node);

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn tags_every_swap() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::new_in(HeapAllocator);

        token = push!(cache, 0, token);
        let first = cache.used.head.load(Ordering::Relaxed);

        // Pop and push the same node: the head points to it again, but with
        // another tag.
        assert_eq!(cache.pop(), Some(0));
        token = push!(cache, 0, token);
        let second = cache.used.head.load(Ordering::Relaxed);

        assert_eq!(Stack::untag(first), Stack::untag(second));
        assert_ne!(first, second);
        assert_eq!(Stack::<u32>::tag(second), Stack::<u32>::tag(first) + 2);

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn tag_carries_into_the_address_bits_and_wraps() {
        type S = Stack<u32>;

        // Only compared, never dereferenced.
        let node = core::ptr::without_provenance_mut::<Node<u32>>(0x7fff_dead_b000);

        // Carry: the low part is full, so the next tag moves into the bits
        // above the address.
        let full_low = node.map_addr(|addr| addr | TAG_MASK);
        let carried = S::retag(node, full_low);
        assert_eq!(S::tag(carried), TAG_MASK + 1);
        assert_eq!(carried.addr() & TAG_MASK, 0);
        assert_eq!(S::untag(carried), node);

        // Wrap: the largest tag both parts can hold is followed by `0`.
        let Err(max_high) = CPU::try_encode(VirtualAddress::new(node), usize::MAX) else {
            panic!("try_encode() accepted usize::MAX");
        };
        let max = CPU::try_encode(VirtualAddress::new(full_low), max_high)
            .unwrap()
            .as_ptr();
        assert_eq!(S::tag(max), (max_high << TAG_BITS) | TAG_MASK);
        assert_eq!(S::retag(node, max), node);
    }

    #[test]
    fn failed_push_returns_value() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut cache = Cache::<u32, Exhausted>::new_in(Exhausted);

        let token = match cache.try_push(7, token) {
            Ok(_) => panic!("try_push() succeeded without memory"),
            Err((Error::OutOfMemory, value, token)) => {
                assert_eq!(value, 7);
                token
            }
        };
        assert!(cache.is_empty());

        let token = cache.clear(token);
        level.leave(token);
    }

    #[test]
    fn clear_drops_values() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut cache = TestCache::new_in(HeapAllocator);
        let value = Arc::new(());

        for _ in 0..3 {
            token = push!(cache, value.clone(), token);
        }
        drop(cache.pop());
        assert_eq!(Arc::strong_count(&value), 3);

        let token = cache.clear(token);
        assert_eq!(Arc::strong_count(&value), 1);
        assert!(cache.used.is_empty());
        assert!(cache.free.is_empty());

        level.leave(token);
    }

    #[test]
    #[should_panic(expected = "A Cache holding nodes must never be dropped")]
    fn drop_with_nodes_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let cache = TestCache::new_in(HeapAllocator);

        let token = push!(cache, 0, token);
        core::mem::forget(level);
        core::mem::forget(token);
        drop(cache);
    }

    #[test]
    fn concurrent_push_and_pop_lose_nothing() {
        const THREADS: usize = 4;
        const PER_THREAD: usize = 10_000;

        let cache = Arc::new(TestCache::<(usize, usize)>::new_in(HeapAllocator));

        // Every thread pushes its values and pops as many as it pushed, so
        // that nodes keep moving between the lists while others use them.
        let workers: std::vec::Vec<_> = (0..THREADS)
            .map(|t| {
                let cache = cache.clone();
                thread::spawn(move || {
                    let root = unsafe { RootToken::forge() };
                    let (level, mut token) = EpilogueLevel::enter(root);
                    let mut popped = std::vec::Vec::with_capacity(PER_THREAD);

                    for i in 0..PER_THREAD {
                        token = push!(cache, (t, i), token);
                        if i % 2 == 1 {
                            popped.push(cache.pop().unwrap());
                            popped.push(cache.pop().unwrap());
                        }
                    }

                    level.leave(token);
                    popped
                })
            })
            .collect();

        let mut seen = std::vec![std::vec![false; PER_THREAD]; THREADS];
        for worker in workers {
            for (t, i) in worker.join().unwrap() {
                assert!(!seen[t][i], "({t}, {i}) popped twice");
                seen[t][i] = true;
            }
        }
        assert!(seen.iter().flatten().all(|&s| s));

        let mut cache = Arc::into_inner(cache).unwrap();
        assert!(cache.is_empty());

        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let token = cache.clear(token);
        level.leave(token);
    }
}
