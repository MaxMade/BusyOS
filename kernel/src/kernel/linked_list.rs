//! A doubly-linked list of individually allocated nodes.
//!
//! # Overview
//!
//! [`LinkedList`] owns its elements, one allocation per element, taken from an
//! [`Allocator`] at the `MemoryManagement` level — the kernel [`Heap`] unless
//! another one is named, which is what `LinkedList<T>` resolves to.
//!
//! Both ends are `O(1)`, so the list serves as a queue as well as a stack, and
//! an element can be dropped out of the middle without moving its neighbours
//! ([`retain`](LinkedList::retain)).
//!
//! # Tokens and dropping
//!
//! Allocating and freeing nodes goes through the kernel's lock-level token
//! system (see [`crate::kernel::locking`]), so every operation that adds or
//! removes an element takes a token and hands it back; the accessors and the
//! iterators, which only walk existing nodes, do not.
//!
//! [`Drop::drop`] cannot be handed a token, so it cannot free the nodes: a
//! non-empty list has to be emptied with [`clear`](LinkedList::clear) before it
//! goes out of scope, and dropping one that is not empty panics rather than
//! leaking every node in it.

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

use crate::{
    kernel::locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken},
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// One element, together with its place in the chain.
struct Node<T> {
    /// Towards the head, `None` for the head itself.
    prev: Link<T>,
    /// Towards the tail, `None` for the tail itself.
    next: Link<T>,
    /// The element.
    data: T,
}

/// A pointer to a node, or the end of the chain.
type Link<T> = Option<NonNull<Node<T>>>;

/// A doubly-linked list of `T`.
///
/// # Type parameters
///
/// - `T` — element type.
/// - `A` — allocator; one `Node`-sized allocation is made per element.
///   Defaults to the kernel [`Heap`].
///
/// # Dropping
///
/// A non-empty list must be emptied with [`clear`](LinkedList::clear) before it
/// goes out of scope; dropping one that still holds elements panics. See the
/// [module documentation](self).
pub struct LinkedList<T, A: Allocator<MemoryManagementLevelID> = Heap> {
    head: Link<T>,
    tail: Link<T>,
    len: usize,
    alloc: A,
    /// Marks the nodes as owned, so that drop checking sees `T` as a type this
    /// list may drop.
    phantom: PhantomData<Node<T>>,
}

// SAFETY: the list owns its nodes and hands out borrows of the elements only
// through borrows of itself, so sending and sharing follow `T` and the
// allocator, exactly as they do for a `T` held directly.
unsafe impl<T: Send, A: Allocator<MemoryManagementLevelID> + Send> Send for LinkedList<T, A> {}

// SAFETY: as above.
unsafe impl<T: Sync, A: Allocator<MemoryManagementLevelID> + Sync> Sync for LinkedList<T, A> {}

impl<T> LinkedList<T, Heap> {
    /// Creates an empty list on the kernel [`Heap`].
    ///
    /// Allocates nothing: the first element pays for the first node.
    pub const fn new() -> Self {
        Self::new_in(Heap)
    }
}

impl<T> Default for LinkedList<T, Heap> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> LinkedList<T, A> {
    /// Creates an empty list backed by `alloc`.
    ///
    /// Allocates nothing: the first element pays for the first node.
    pub const fn new_in(alloc: A) -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            alloc,
            phantom: PhantomData,
        }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    // --- accessors -----------------------------------------------------------

    /// Returns a shared borrow of the first element, or `None` if the list is
    /// empty.
    #[inline]
    pub fn front(&self) -> Option<&T> {
        // SAFETY: the head is a live node owned by the list.
        self.head.map(|node| unsafe { &(*node.as_ptr()).data })
    }

    /// Returns an exclusive borrow of the first element, or `None` if the list
    /// is empty.
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        // SAFETY: as above, and the list is borrowed exclusively for as long as
        // the returned borrow lives.
        self.head.map(|node| unsafe { &mut (*node.as_ptr()).data })
    }

    /// Returns a shared borrow of the last element, or `None` if the list is
    /// empty.
    #[inline]
    pub fn back(&self) -> Option<&T> {
        // SAFETY: the tail is a live node owned by the list.
        self.tail.map(|node| unsafe { &(*node.as_ptr()).data })
    }

    /// Returns an exclusive borrow of the last element, or `None` if the list
    /// is empty.
    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        // SAFETY: as above, and the list is borrowed exclusively for as long as
        // the returned borrow lives.
        self.tail.map(|node| unsafe { &mut (*node.as_ptr()).data })
    }

    // --- node handling -------------------------------------------------------

    /// Allocates a node and initialises it with `node`.
    ///
    /// On success returns `(ptr, token)`. On failure returns `(error, token)`
    /// with the list unchanged.
    fn alloc_node<Token>(
        &self,
        node: Node<T>,
        token: Token,
    ) -> Result<(NonNull<Node<T>>, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let layout = Layout::new::<Node<T>>();

        let (ptr, token) = match self.alloc.allocate(layout, token) {
            Ok((ptr, token)) => (ptr.cast::<Node<T>>(), token),
            Err(error) => return Err(error),
        };

        // SAFETY: `ptr` is freshly allocated, sized and aligned for a
        // `Node<T>`, and owned by this call alone.
        unsafe { ptr.as_ptr().write(node) };

        Ok((ptr, token))
    }

    /// Moves the element out of a node and returns the node's memory to the
    /// allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live `Node<T>` that is no longer part of the chain
    /// — [`unlink`](LinkedList::unlink) must have run for it, or it must never
    /// have been linked. `ptr` is dangling afterwards.
    unsafe fn take_node<Token>(&self, ptr: NonNull<Node<T>>, token: Token) -> (T, Token)
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let layout = Layout::new::<Node<T>>();

        // SAFETY: the node is live and unreachable from the list, so this moves
        // the element out for good; the two links are `Copy` and go away with
        // the block.
        let value = unsafe { ptr::read(&ptr.as_ref().data) };

        // SAFETY: the block came from this allocator with exactly this layout,
        // has not been freed before, and nothing touches it afterwards.
        let token = unsafe { self.alloc.deallocate(ptr.cast(), layout, token) };

        (value, token)
    }

    /// Drops a node's element in place and returns its memory to the allocator.
    ///
    /// # Safety
    ///
    /// As [`take_node`](LinkedList::take_node).
    unsafe fn drop_node<Token>(&self, ptr: NonNull<Node<T>>, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // SAFETY: the caller upholds `take_node`'s preconditions; the element
        // comes out and is dropped right here.
        let (value, token) = unsafe { self.take_node(ptr, token) };
        drop(value);
        token
    }

    /// Takes `node` out of the chain and shortens the list by one.
    ///
    /// The node itself is left untouched — it still holds its element and its
    /// stale links — and has to be freed by the caller.
    ///
    /// # Safety
    ///
    /// `node` must be a live node currently linked into *this* list.
    unsafe fn unlink(&mut self, node: NonNull<Node<T>>) {
        // SAFETY: `node` is live per the caller.
        let (prev, next) = unsafe { ((*node.as_ptr()).prev, (*node.as_ptr()).next) };

        match prev {
            // SAFETY: a neighbour of a linked node is itself a live node.
            Some(prev) => unsafe { (*prev.as_ptr()).next = next },
            None => self.head = next,
        }

        match next {
            // SAFETY: as above.
            Some(next) => unsafe { (*next.as_ptr()).prev = prev },
            None => self.tail = prev,
        }

        self.len -= 1;
    }

    // --- push and pop --------------------------------------------------------

    /// Puts `value` at the front of the list.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if no node could be allocated; the list is
    /// unchanged and `value` is dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_push_front<Token>(&mut self, value: T, token: Token) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let node = Node {
            prev: None,
            next: self.head,
            data: value,
        };

        // Allocate before touching the chain, so that a failure leaves the list
        // exactly as it was.
        let (ptr, token) = match self.alloc_node(node, token) {
            Ok((ptr, token)) => (ptr, token),
            Err(error) => return Err(error),
        };

        match self.head {
            // SAFETY: the old head is a live node owned by the list.
            Some(head) => unsafe { (*head.as_ptr()).prev = Some(ptr) },
            // The list was empty, so the new node is both ends of it.
            None => self.tail = Some(ptr),
        }

        self.head = Some(ptr);
        self.len += 1;

        Ok(token)
    }

    /// Puts `value` at the back of the list.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if no node could be allocated; the list is
    /// unchanged and `value` is dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_push_back<Token>(&mut self, value: T, token: Token) -> Result<Token, (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let node = Node {
            prev: self.tail,
            next: None,
            data: value,
        };

        // As in `try_push_front`: allocate first, link afterwards.
        let (ptr, token) = match self.alloc_node(node, token) {
            Ok((ptr, token)) => (ptr, token),
            Err(error) => return Err(error),
        };

        match self.tail {
            // SAFETY: the old tail is a live node owned by the list.
            Some(tail) => unsafe { (*tail.as_ptr()).next = Some(ptr) },
            // The list was empty, so the new node is both ends of it.
            None => self.head = Some(ptr),
        }

        self.tail = Some(ptr);
        self.len += 1;

        Ok(token)
    }

    /// Takes the first element out of the list, or returns `None` if it is
    /// empty.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned, whether an element was there or
    /// not.
    pub fn pop_front<Token>(&mut self, token: Token) -> (Option<T>, Token)
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let Some(head) = self.head else {
            return (None, token);
        };

        // SAFETY: the head is a live node linked into this list.
        unsafe { self.unlink(head) };

        // SAFETY: the node is out of the chain and still holds its element.
        let (value, token) = unsafe { self.take_node(head, token) };

        (Some(value), token)
    }

    /// Takes the last element out of the list, or returns `None` if it is
    /// empty.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned, whether an element was there or
    /// not.
    pub fn pop_back<Token>(&mut self, token: Token) -> (Option<T>, Token)
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let Some(tail) = self.tail else {
            return (None, token);
        };

        // SAFETY: the tail is a live node linked into this list.
        unsafe { self.unlink(tail) };

        // SAFETY: the node is out of the chain and still holds its element.
        let (value, token) = unsafe { self.take_node(tail, token) };

        (Some(value), token)
    }

    // --- bulk removal --------------------------------------------------------

    /// Drops every element `keep` returns `false` for, keeping the order of the
    /// rest.
    ///
    /// The predicate sees each element exactly once, from front to back.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn retain<Keep, Token>(&mut self, mut keep: Keep, token: Token) -> Token
    where
        Keep: FnMut(&T) -> bool,
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;
        let mut current = self.head;

        while let Some(node) = current {
            // Step on before the node can be freed: its links are gone by then.
            // SAFETY: `node` is a live node owned by the list.
            current = unsafe { (*node.as_ptr()).next };

            // SAFETY: as above, and the borrow ends before the node is freed.
            if keep(unsafe { &(*node.as_ptr()).data }) {
                continue;
            }

            // SAFETY: `node` is linked into this list, so unlinking it makes it
            // unreachable, which is what freeing it requires.
            unsafe {
                self.unlink(node);
                token = self.drop_node(node, token);
            }
        }

        token
    }

    /// Drops every element and returns all node memory to the allocator.
    ///
    /// This is what a list has to end with: see the
    /// [module documentation](self).
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn clear<Token>(&mut self, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;
        let mut current = self.head;

        // The chain is walked once and dropped node by node; the list's own
        // fields are reset at the end rather than per node.
        while let Some(node) = current {
            // SAFETY: `node` is a live node owned by the list.
            current = unsafe { (*node.as_ptr()).next };

            // SAFETY: the nodes ahead are reached through `current`, which was
            // read before the node was freed, and the list is emptied below, so
            // nothing reaches this node again.
            token = unsafe { self.drop_node(node, token) };
        }

        self.head = None;
        self.tail = None;
        self.len = 0;

        token
    }

    // --- iteration -----------------------------------------------------------

    /// Returns an iterator over shared borrows of the elements, front to back.
    ///
    /// Also walkable from the back, and its length is known exactly.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            front: self.head,
            back: self.tail,
            len: self.len,
            phantom: PhantomData,
        }
    }

    /// Returns an iterator over exclusive borrows of the elements, front to
    /// back.
    ///
    /// Also walkable from the back, and its length is known exactly.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            front: self.head,
            back: self.tail,
            len: self.len,
            phantom: PhantomData,
        }
    }
}

impl<T, A: Allocator<MemoryManagementLevelID>> Drop for LinkedList<T, A> {
    /// Panics if the list is non-empty.
    ///
    /// The nodes cannot be released here because `Drop::drop` cannot accept the
    /// lock token the allocator needs. Call [`LinkedList::clear`] before the
    /// list goes out of scope.
    fn drop(&mut self) {
        if !self.is_empty() {
            panic!(
                "A non-empty LinkedList must never be dropped. Use LinkedList::clear(...) instead!"
            );
        }
    }
}

impl<T: fmt::Debug, A: Allocator<MemoryManagementLevelID>> fmt::Debug for LinkedList<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a LinkedList<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, A: Allocator<MemoryManagementLevelID>> IntoIterator for &'a mut LinkedList<T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

/// Shared iterator over the elements. Produced by [`LinkedList::iter`].
///
/// The two ends walk towards each other and `len` is what is left between
/// them, so the two directions can be mixed and always meet in the middle.
pub struct Iter<'a, T> {
    front: Link<T>,
    back: Link<T>,
    len: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            return None;
        }

        let node = self.front?;
        self.len -= 1;

        // SAFETY: the node belongs to the borrowed list, which cannot change
        // while the iterator holds a shared borrow of it.
        unsafe {
            self.front = (*node.as_ptr()).next;
            Some(&(*node.as_ptr()).data)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            return None;
        }

        let node = self.back?;
        self.len -= 1;

        // SAFETY: as in `next`.
        unsafe {
            self.back = (*node.as_ptr()).prev;
            Some(&(*node.as_ptr()).data)
        }
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

/// Exclusive iterator over the elements. Produced by
/// [`LinkedList::iter_mut`].
///
/// Walks like [`Iter`], from both ends.
pub struct IterMut<'a, T> {
    front: Link<T>,
    back: Link<T>,
    len: usize,
    phantom: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            return None;
        }

        let node = self.front?;
        self.len -= 1;

        // SAFETY: the node belongs to the borrowed list, which is borrowed
        // exclusively, and each node is handed out exactly once — the two ends
        // stop when `len` runs out — so no two borrows alias.
        unsafe {
            self.front = (*node.as_ptr()).next;
            Some(&mut (*node.as_ptr()).data)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            return None;
        }

        let node = self.back?;
        self.len -= 1;

        // SAFETY: as in `next`.
        unsafe {
            self.back = (*node.as_ptr()).prev;
            Some(&mut (*node.as_ptr()).data)
        }
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    type List<T> = LinkedList<T, HeapAllocator>;

    /// `LinkedList<T>` has to resolve to the kernel heap without naming it.
    /// Never called — this only has to compile.
    #[allow(dead_code)]
    fn default_allocator_is_the_heap<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut list: LinkedList<u32> = LinkedList::new();

        let token = match list.try_push_back(0, token) {
            Ok(token) => token,
            Err((_, token)) => token,
        };

        list.clear(token)
    }

    fn new_list<T>() -> List<T> {
        LinkedList::new_in(HeapAllocator)
    }

    /// The elements front to back. A macro-free helper, so it can be used in
    /// assertions directly.
    fn values<T: Copy>(list: &List<T>) -> std::vec::Vec<T> {
        list.iter().copied().collect()
    }

    /// Pushes at the front, panicking if the allocation fails. A macro rather
    /// than a function because the error arm cannot be unwrapped: a token is
    /// not [`Debug`], so `expect` is unavailable.
    macro_rules! push_front {
        ($list:expr, $value:expr, $token:expr) => {
            match $list.try_push_front($value, $token) {
                Ok(token) => token,
                Err(_) => panic!("try_push_front() returned an allocation error"),
            }
        };
    }

    /// Pushes at the back; see [`push_front`].
    macro_rules! push_back {
        ($list:expr, $value:expr, $token:expr) => {
            match $list.try_push_back($value, $token) {
                Ok(token) => token,
                Err(_) => panic!("try_push_back() returned an allocation error"),
            }
        };
    }

    /// A fresh list holds nothing and has nothing at either end.
    #[test]
    fn empty_list() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.front().is_none());
        assert!(list.back().is_none());
        assert_eq!(values(&list), []);

        let (front, token) = list.pop_front(token);
        assert!(front.is_none());

        let (back, token) = list.pop_back(token);
        assert!(back.is_none());

        let token = list.clear(token);
        level.leave(token);
    }

    /// Pushing at the front reverses, popping at the front takes the newest.
    #[test]
    fn push_and_pop_at_the_front() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        token = push_front!(list, 1, token);
        token = push_front!(list, 2, token);
        token = push_front!(list, 3, token);

        assert_eq!(values(&list), [3, 2, 1]);
        assert_eq!(list.len(), 3);

        let (value, mut token) = list.pop_front(token);
        assert_eq!(value, Some(3));
        assert_eq!(values(&list), [2, 1]);

        token = list.clear(token);
        level.leave(token);
    }

    /// Pushing at the back keeps the order, popping at the back takes the
    /// newest.
    #[test]
    fn push_and_pop_at_the_back() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        token = push_back!(list, 1, token);
        token = push_back!(list, 2, token);
        token = push_back!(list, 3, token);

        assert_eq!(values(&list), [1, 2, 3]);

        let (value, mut token) = list.pop_back(token);
        assert_eq!(value, Some(3));
        assert_eq!(values(&list), [1, 2]);

        token = list.clear(token);
        level.leave(token);
    }

    /// Pushing at the back and popping at the front is a queue.
    #[test]
    fn queue_order() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        for value in 1..=3 {
            token = push_back!(list, value, token);
        }

        for expected in 1..=3 {
            let (value, t) = list.pop_front(token);
            token = t;
            assert_eq!(value, Some(expected));
        }

        assert!(list.is_empty());

        let token = list.clear(token);
        level.leave(token);
    }

    /// Emptying the list one element at a time leaves both ends consistent, so
    /// it can be filled again afterwards.
    #[test]
    fn draining_and_refilling() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        token = push_back!(list, 1, token);
        token = push_back!(list, 2, token);

        let (_, t) = list.pop_front(token);
        let (_, t) = list.pop_back(t);
        token = t;

        assert!(list.is_empty());
        assert!(list.front().is_none());
        assert!(list.back().is_none());

        token = push_back!(list, 7, token);

        assert_eq!(values(&list), [7]);
        assert_eq!(list.front().copied(), Some(7));
        assert_eq!(list.back().copied(), Some(7));

        token = list.clear(token);
        level.leave(token);
    }

    /// The ends can be read and written through their accessors.
    #[test]
    fn front_and_back_accessors() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        token = push_back!(list, 1, token);
        token = push_back!(list, 2, token);

        assert_eq!(list.front().copied(), Some(1));
        assert_eq!(list.back().copied(), Some(2));

        *list.front_mut().expect("front") = 10;
        *list.back_mut().expect("back") = 20;

        assert_eq!(values(&list), [10, 20]);

        token = list.clear(token);
        level.leave(token);
    }

    /// The iterator walks from either end and the two directions meet in the
    /// middle exactly once per element.
    #[test]
    fn iteration_from_both_ends() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        for value in 1..=4 {
            token = push_back!(list, value, token);
        }

        assert_eq!(list.iter().len(), 4);
        assert_eq!(values(&list), [1, 2, 3, 4]);

        let reversed: std::vec::Vec<u32> = list.iter().rev().copied().collect();
        assert_eq!(reversed, [4, 3, 2, 1]);

        let mut iter = list.iter();
        assert_eq!(iter.next().copied(), Some(1));
        assert_eq!(iter.next_back().copied(), Some(4));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next().copied(), Some(2));
        assert_eq!(iter.next_back().copied(), Some(3));
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());

        token = list.clear(token);
        level.leave(token);
    }

    /// The exclusive iterator writes through to the elements.
    #[test]
    fn iteration_mutates_in_place() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        for value in 1..=3 {
            token = push_back!(list, value, token);
        }

        for value in &mut list {
            *value *= 10;
        }

        assert_eq!(values(&list), [10, 20, 30]);

        token = list.clear(token);
        level.leave(token);
    }

    /// `retain` drops the elements the predicate rejects and keeps the order of
    /// the rest, whether they sit at an end or in the middle.
    #[test]
    fn retain_removes_and_keeps_order() {
        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        for value in 1..=6 {
            token = push_back!(list, value, token);
        }

        // Drops 1 (the head), 3 and 5 (the middle), keeping 6 (the tail).
        token = list.retain(|value| value % 2 == 0, token);

        assert_eq!(values(&list), [2, 4, 6]);
        assert_eq!(list.len(), 3);
        assert_eq!(list.front().copied(), Some(2));
        assert_eq!(list.back().copied(), Some(6));

        // The chain is intact in both directions.
        let reversed: std::vec::Vec<u32> = list.iter().rev().copied().collect();
        assert_eq!(reversed, [6, 4, 2]);

        // Rejecting everything empties the list, tail included.
        token = list.retain(|_| false, token);

        assert!(list.is_empty());
        assert!(list.front().is_none());
        assert!(list.back().is_none());

        token = list.clear(token);
        level.leave(token);
    }

    /// Every element is dropped exactly once, whichever way it leaves the list.
    #[test]
    fn elements_are_dropped_once() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountingDrop;

        impl Drop for CountingDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        let root = unsafe { RootToken::forge() };
        let (level, mut token) = EpilogueLevel::enter(root);
        let mut list = new_list::<CountingDrop>();

        for _ in 0..4 {
            token = push_back!(list, CountingDrop, token);
        }

        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);

        // Popped elements belong to the caller, so nothing is dropped yet.
        let (popped, mut token) = list.pop_front(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 0);
        drop(popped);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 1);

        // `retain` and `clear` drop what they remove.
        token = list.retain(|_| false, token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 4);

        token = list.clear(token);
        assert_eq!(DROPS.load(AtomicOrdering::Relaxed), 4);

        level.leave(token);
    }

    /// Dropping a list that still holds elements cannot free them, so it panics
    /// rather than leaking every node.
    #[test]
    #[should_panic(expected = "A non-empty LinkedList must never be dropped")]
    fn implicit_drop_of_a_non_empty_list_panics() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);
        let mut list = new_list::<u32>();

        let token = push_back!(list, 1, token);

        // The level guard panics when it is dropped as well, which would turn
        // the panic below into a double panic and abort the test process.
        core::mem::forget(level);
        core::mem::forget(token);

        drop(list);
    }
}
