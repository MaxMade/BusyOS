//! A `BTreeMap`-like ordered map backed by a red-black tree.
//!
//! The tree stores nodes individually via the supplied allocator (one
//! allocation per entry). It uses parent pointers and the classic CLRS
//! insert/delete fixup algorithms, so all structural operations are
//! `O(log n)` worst case.

use core::alloc::Layout;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::{mem, ptr};

use crate::kernel::locking::{CanAcquire, LockId, PreviousToken};
use crate::utils::allocator::{Allocator, Error as AllocatorError};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Color {
    Red,
    Black,
}

struct Node<K, V> {
    key: K,
    value: V,
    color: Color,
    parent: Option<NonNull<Node<K, V>>>,
    left: Option<NonNull<Node<K, V>>>,
    right: Option<NonNull<Node<K, V>>>,
}

type Link<K, V> = Option<NonNull<Node<K, V>>>;

/// Returns the colour of node `n`, treating `None` (nil) as [`Color::Black`].
#[inline]
unsafe fn color_of<K, V>(n: Link<K, V>) -> Color {
    match n {
        Some(p) => unsafe { (*p.as_ptr()).color },
        None => Color::Black,
    }
}

/// Sets the colour of node `n`. A `None` argument is silently ignored because
/// the nil sentinel is conceptually always black and cannot be mutated.
#[inline]
unsafe fn set_color<K, V>(n: Link<K, V>, c: Color) {
    if let Some(p) = n {
        unsafe { (*p.as_ptr()).color = c };
    }
}

/// Returns the left-most node in the subtree rooted at `n`.
#[inline]
unsafe fn min_from<K, V>(mut n: NonNull<Node<K, V>>) -> NonNull<Node<K, V>> {
    while let Some(l) = unsafe { (*n.as_ptr()).left } {
        n = l;
    }
    n
}

/// Returns the in-order successor of `n`, or `None` if `n` is the maximum.
#[inline]
unsafe fn successor<K, V>(n: NonNull<Node<K, V>>) -> Link<K, V> {
    // If the node has a right subtree, the successor is its minimum.
    if let Some(r) = unsafe { (*n.as_ptr()).right } {
        return unsafe { Some(min_from(r)) };
    }
    // Otherwise, walk up until we come from a left child.
    let mut cur = n;
    let mut p = unsafe { (*n.as_ptr()).parent };
    while let Some(pp) = p {
        if Some(cur) == unsafe { (*pp.as_ptr()).right } {
            cur = pp;
            p = unsafe { (*pp.as_ptr()).parent };
        } else {
            break;
        }
    }
    p
}

/// An ordered map from `K` to `V`, backed by a red-black tree.
///
/// Insertion, removal, and lookup are all `O(log n)`. Iteration yields entries
/// in ascending key order.
///
/// # Type parameters
///
/// - `K` — key type; must implement [`Ord`] for mutation operations.
/// - `V` — value type.
/// - `ID` — [`LockId`] that identifies the allocator's lock level in the
///   kernel's static ordering. Every mutating method requires a `Token`
///   proving the caller does not hold any lock at or above that level.
/// - `A` — allocator; one `Node`-sized allocation is made per entry.
///
/// # Dropping
///
/// Because releasing nodes requires the lock token and the token cannot be
/// passed to `Drop::drop`, a non-empty `RbTree` **must** be explicitly emptied
/// with [`clear`](RbTree::clear) before it goes out of scope. Dropping a
/// non-empty tree panics.
pub struct RbTree<K, V, ID: LockId, A: Allocator<ID>> {
    root: Link<K, V>,
    len: usize,
    alloc: A,
    phantom: PhantomData<ID>,
}

// The tree owns its nodes; sharing/sending follows the contained types.
unsafe impl<K: Send, V: Send, ID: LockId, A: Allocator<ID> + Send> Send for RbTree<K, V, ID, A> {}
unsafe impl<K: Sync, V: Sync, ID: LockId, A: Allocator<ID> + Sync> Sync for RbTree<K, V, ID, A> {}

// --- Allocator-agnostic helpers ----------------------------------------------

impl<K, V, ID: LockId, A: Allocator<ID>> RbTree<K, V, ID, A> {
    /// Creates an empty tree backed by `alloc`.
    pub const fn new_in(alloc: A) -> Self {
        Self {
            root: None,
            len: 0,
            alloc,
            phantom: PhantomData,
        }
    }

    /// Returns the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrows the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Allocates a new node, initialising it with `node`.
    ///
    /// On success returns `(ptr, token)`. On failure returns `(error, token)`
    /// with the tree unchanged.
    fn alloc_node<Token>(
        &self,
        node: Node<K, V>,
        token: Token,
    ) -> Result<(NonNull<Node<K, V>>, Token), (AllocatorError, Token)>
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        let layout = Layout::new::<Node<K, V>>();

        let (ptr, token) = match self.alloc.allocate(layout, token) {
            Ok((ptr, token)) => (ptr.cast::<Node<K, V>>(), token),
            Err(error) => return Err(error),
        };

        // SAFETY: `ptr` is freshly allocated, correctly sized and aligned for
        // `Node<K, V>`, and exclusively owned by this call.
        unsafe { ptr.as_ptr().write(node) };

        Ok((ptr, token))
    }

    /// Drops the node's contents in place and returns its memory to the
    /// allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live `Node<K, V>` that is no longer reachable
    /// from the tree (all parent/child links to it must already have been
    /// cleared). After this call `ptr` is dangling.
    unsafe fn free_node<Token>(&self, ptr: NonNull<Node<K, V>>, token: Token) -> Token
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        let layout = Layout::new::<Node<K, V>>();
        // SAFETY: caller guarantees `ptr` is live and exclusively owned.
        unsafe {
            ptr::drop_in_place(ptr.as_ptr());
            self.alloc.deallocate(ptr.cast(), layout, token)
        }
    }
}

// --- Ord-dependent methods ---------------------------------------------------

impl<K: Ord, V, ID: LockId, A: Allocator<ID>> RbTree<K, V, ID, A> {
    // --- lookup --------------------------------------------------------------

    /// Walks the tree looking for `key`, returning a raw pointer to the node
    /// if found.
    fn find_node<Q>(&self, key: &Q) -> Link<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut cur = self.root;
        while let Some(n) = cur {
            // SAFETY: `n` is a live node owned by the tree.
            let node = unsafe { n.as_ref() };
            cur = match key.cmp(node.key.borrow()) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => return Some(n),
            };
        }
        None
    }

    /// Returns a shared reference to the value for `key`, or `None` if absent.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_node(key)
            // SAFETY: the node is live and we hold a shared borrow of the tree.
            .map(|n| unsafe { &(*n.as_ptr()).value })
    }

    /// Returns a mutable reference to the value for `key`, or `None` if absent.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_node(key)
            // SAFETY: the node is live and we hold an exclusive borrow of the tree.
            .map(|n| unsafe { &mut (*n.as_ptr()).value })
    }

    /// Returns `true` if `key` is present in the tree.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_node(key).is_some()
    }

    // --- rotations -----------------------------------------------------------
    //
    // Both rotations are standard CLRS operations. They adjust the three
    // pointer pairs affected by the rotation and update the root when the
    // pivot was the root node.

    /// Performs a left rotation around `x`.
    ///
    /// ```text
    ///     x                y
    ///    / \              / \
    ///   A   y    →       x   C
    ///      / \          / \
    ///     B   C        A   B
    /// ```
    ///
    /// # Safety
    ///
    /// `x` must be a live node with a non-nil right child.
    unsafe fn left_rotate(&mut self, x: NonNull<Node<K, V>>) {
        unsafe {
            let y = (*x.as_ptr())
                .right
                .expect("left_rotate needs a right child");

            // B becomes the right child of x.
            (*x.as_ptr()).right = (*y.as_ptr()).left;
            if let Some(yl) = (*y.as_ptr()).left {
                (*yl.as_ptr()).parent = Some(x);
            }

            // y takes x's place in the tree.
            (*y.as_ptr()).parent = (*x.as_ptr()).parent;
            match (*x.as_ptr()).parent {
                None => self.root = Some(y),
                Some(xp) => {
                    if Some(x) == (*xp.as_ptr()).left {
                        (*xp.as_ptr()).left = Some(y);
                    } else {
                        (*xp.as_ptr()).right = Some(y);
                    }
                }
            }

            // x becomes y's left child.
            (*y.as_ptr()).left = Some(x);
            (*x.as_ptr()).parent = Some(y);
        }
    }

    /// Performs a right rotation around `x`.
    ///
    /// ```text
    ///       x              y
    ///      / \            / \
    ///     y   C    →     A   x
    ///    / \                / \
    ///   A   B              B   C
    /// ```
    ///
    /// # Safety
    ///
    /// `x` must be a live node with a non-nil left child.
    unsafe fn right_rotate(&mut self, x: NonNull<Node<K, V>>) {
        unsafe {
            let y = (*x.as_ptr()).left.expect("right_rotate needs a left child");

            // B becomes the left child of x.
            (*x.as_ptr()).left = (*y.as_ptr()).right;
            if let Some(yr) = (*y.as_ptr()).right {
                (*yr.as_ptr()).parent = Some(x);
            }

            // y takes x's place in the tree.
            (*y.as_ptr()).parent = (*x.as_ptr()).parent;
            match (*x.as_ptr()).parent {
                None => self.root = Some(y),
                Some(xp) => {
                    if Some(x) == (*xp.as_ptr()).right {
                        (*xp.as_ptr()).right = Some(y);
                    } else {
                        (*xp.as_ptr()).left = Some(y);
                    }
                }
            }

            // x becomes y's right child.
            (*y.as_ptr()).right = Some(x);
            (*x.as_ptr()).parent = Some(y);
        }
    }

    // --- insert --------------------------------------------------------------

    /// Inserts `key`/`value` into the tree.
    ///
    /// - If `key` was not present, inserts the entry and returns
    ///   `Ok((None, token))`.
    /// - If `key` was already present, replaces the old value **without**
    ///   allocating and returns `Ok((Some(old_value), token))`.
    /// - If a new node could not be allocated, returns
    ///   `Err((AllocatorError, token))` with the tree unchanged.
    pub fn try_insert<Token>(
        &mut self,
        key: K,
        value: V,
        token: Token,
    ) -> Result<(Option<V>, Token), (AllocatorError, Token)>
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        // Standard BST descent, recording the would-be parent.
        let mut parent: Link<K, V> = None;
        let mut cur = self.root;
        while let Some(n) = cur {
            parent = Some(n);
            // SAFETY: `n` is a live node owned by the tree.
            let node = unsafe { &mut *n.as_ptr() };
            cur = match key.cmp(&node.key) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => {
                    // Key already present — replace the value in-place.
                    return Ok((Some(mem::replace(&mut node.value, value)), token));
                }
            };
        }

        // Allocate a red node before touching the tree structure so that an
        // OOM error leaves the tree unchanged.
        let (z, token) = self.alloc_node(
            Node {
                key,
                value,
                color: Color::Red,
                parent,
                left: None,
                right: None,
            },
            token,
        )?;

        // SAFETY: `z` is freshly allocated; `parent` (if any) is a live node.
        unsafe {
            match parent {
                None => self.root = Some(z),
                Some(p) => {
                    if (*z.as_ptr()).key < (*p.as_ptr()).key {
                        (*p.as_ptr()).left = Some(z);
                    } else {
                        (*p.as_ptr()).right = Some(z);
                    }
                }
            }
            self.insert_fixup(z);
        }
        self.len += 1;
        Ok((None, token))
    }

    /// Restores the red-black invariants after inserting node `z`.
    ///
    /// Implements CLRS §13.3 `RB-INSERT-FIXUP`.
    ///
    /// # Safety
    ///
    /// `z` must be a freshly inserted red node that is already linked into the
    /// tree at the correct BST position.
    unsafe fn insert_fixup(&mut self, mut z: NonNull<Node<K, V>>) {
        unsafe {
            while color_of((*z.as_ptr()).parent) == Color::Red {
                // A red parent can never be the root (the root is always black),
                // so the grandparent is guaranteed to exist.
                let zp = (*z.as_ptr()).parent.unwrap();
                let zpp = (*zp.as_ptr()).parent.unwrap();

                if Some(zp) == (*zpp.as_ptr()).left {
                    // Parent is the left child of the grandparent.
                    let uncle = (*zpp.as_ptr()).right;
                    if color_of(uncle) == Color::Red {
                        // Case 1: uncle is red — recolour and move up.
                        set_color(Some(zp), Color::Black);
                        set_color(uncle, Color::Black);
                        set_color(Some(zpp), Color::Red);
                        z = zpp;
                    } else {
                        // Case 2: uncle is black and z is a right child —
                        // left-rotate to convert to case 3.
                        if Some(z) == (*zp.as_ptr()).right {
                            z = zp;
                            self.left_rotate(z);
                        }
                        // Case 3: uncle is black and z is a left child —
                        // recolour and right-rotate.
                        let zp2 = (*z.as_ptr()).parent.unwrap();
                        let zpp2 = (*zp2.as_ptr()).parent.unwrap();
                        set_color(Some(zp2), Color::Black);
                        set_color(Some(zpp2), Color::Red);
                        self.right_rotate(zpp2);
                    }
                } else {
                    // Symmetric: parent is the right child of the grandparent.
                    let uncle = (*zpp.as_ptr()).left;
                    if color_of(uncle) == Color::Red {
                        set_color(Some(zp), Color::Black);
                        set_color(uncle, Color::Black);
                        set_color(Some(zpp), Color::Red);
                        z = zpp;
                    } else {
                        if Some(z) == (*zp.as_ptr()).left {
                            z = zp;
                            self.right_rotate(z);
                        }
                        let zp2 = (*z.as_ptr()).parent.unwrap();
                        let zpp2 = (*zp2.as_ptr()).parent.unwrap();
                        set_color(Some(zp2), Color::Black);
                        set_color(Some(zpp2), Color::Red);
                        self.left_rotate(zpp2);
                    }
                }
            }
            // The root must always be black.
            set_color(self.root, Color::Black);
        }
    }

    // --- remove --------------------------------------------------------------

    /// Replaces the subtree rooted at `u` with the subtree `v`.
    ///
    /// This is the CLRS `RB-TRANSPLANT` subroutine. It does *not* update
    /// `v`'s children; the caller is responsible for the rest of the link
    /// repair.
    ///
    /// # Safety
    ///
    /// `u` must be a live node in the tree. `v` (if `Some`) must also be a
    /// live node, or `None` to splice in a nil sentinel.
    unsafe fn transplant(&mut self, u: NonNull<Node<K, V>>, v: Link<K, V>) {
        unsafe {
            match (*u.as_ptr()).parent {
                None => self.root = v,
                Some(up) => {
                    if Some(u) == (*up.as_ptr()).left {
                        (*up.as_ptr()).left = v;
                    } else {
                        (*up.as_ptr()).right = v;
                    }
                }
            }
            if let Some(vn) = v {
                (*vn.as_ptr()).parent = (*u.as_ptr()).parent;
            }
        }
    }

    /// Removes `key` from the tree, returning its value if present.
    ///
    /// Returns `(None, token)` if `key` was not found.
    pub fn remove<Q, Token>(&mut self, key: &Q, token: Token) -> (Option<V>, Token)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        let z = match self.find_node(key) {
            Some(z) => z,
            None => return (None, token),
        };

        // SAFETY: `z` is a live node found by `find_node`; `remove_node`
        // detaches it from the tree before we read its contents.
        unsafe {
            self.remove_node(z);

            // Move key and value out of the detached node, then free the raw
            // memory. `ptr::read` skips the `Drop` impl for `Node` itself
            // (which has none), so K and V are dropped exactly once: K when
            // `node` goes out of scope, V when it is returned to the caller.
            let node = ptr::read(z.as_ptr());
            let token = self
                .alloc
                .deallocate(z.cast(), Layout::new::<Node<K, V>>(), token);
            self.len -= 1;
            (Some(node.value), token)
        }
    }

    /// Unlinks node `z` from the tree and restores the red-black invariants.
    ///
    /// Implements CLRS §13.4 `RB-DELETE`. After this call `z` is detached but
    /// its memory has **not** been freed; the caller is responsible for that.
    ///
    /// # Safety
    ///
    /// `z` must be a live node in the tree.
    unsafe fn remove_node(&mut self, z: NonNull<Node<K, V>>) {
        unsafe {
            let mut y = z;
            let mut y_original_color = (*y.as_ptr()).color;
            let x: Link<K, V>;
            let x_parent: Link<K, V>;

            if (*z.as_ptr()).left.is_none() {
                // Case 1: no left child — splice z out by replacing it with
                // its right child (possibly nil).
                x = (*z.as_ptr()).right;
                x_parent = (*z.as_ptr()).parent;
                self.transplant(z, x);
            } else if (*z.as_ptr()).right.is_none() {
                // Case 2: no right child — splice z out with its left child.
                x = (*z.as_ptr()).left;
                x_parent = (*z.as_ptr()).parent;
                self.transplant(z, x);
            } else {
                // Case 3: z has two children. Find its in-order successor y
                // (the minimum of the right subtree) and move y into z's
                // position, then delete y's original slot instead. y has at
                // most one child (no left child by definition of minimum).
                y = min_from((*z.as_ptr()).right.unwrap());
                y_original_color = (*y.as_ptr()).color;
                x = (*y.as_ptr()).right;

                if (*y.as_ptr()).parent == Some(z) {
                    // y is z's direct right child.
                    x_parent = Some(y);
                    if let Some(xn) = x {
                        (*xn.as_ptr()).parent = Some(y);
                    }
                } else {
                    x_parent = (*y.as_ptr()).parent;
                    self.transplant(y, (*y.as_ptr()).right);
                    (*y.as_ptr()).right = (*z.as_ptr()).right;
                    (*(*y.as_ptr()).right.unwrap().as_ptr()).parent = Some(y);
                }
                self.transplant(z, Some(y));
                (*y.as_ptr()).left = (*z.as_ptr()).left;
                (*(*y.as_ptr()).left.unwrap().as_ptr()).parent = Some(y);
                (*y.as_ptr()).color = (*z.as_ptr()).color;
            }

            // If the node that was moved or removed was black we may have
            // violated the black-height invariant; fix it up.
            if y_original_color == Color::Black {
                self.delete_fixup(x, x_parent);
            }
        }
    }

    /// Restores the red-black invariants after a deletion that removed a black
    /// node.
    ///
    /// Implements CLRS §13.4 `RB-DELETE-FIXUP`. The `x_parent` argument
    /// carries the parent of `x` explicitly because `x` may be a nil sentinel
    /// which has no parent pointer of its own.
    ///
    /// # Safety
    ///
    /// All nodes reachable via `x` and `x_parent` must be live tree nodes.
    unsafe fn delete_fixup(&mut self, mut x: Link<K, V>, mut x_parent: Link<K, V>) {
        unsafe {
            while x != self.root && color_of(x) == Color::Black {
                // When the loop runs x is "doubly black"; its sibling w is
                // always a real (non-nil) node because the black-heights were
                // balanced before the deletion, so x_parent is always Some.
                let xp = x_parent.unwrap();

                if x == (*xp.as_ptr()).left {
                    let mut w = (*xp.as_ptr()).right;

                    // Case 1: sibling w is red — rotate to make w black.
                    if color_of(w) == Color::Red {
                        set_color(w, Color::Black);
                        set_color(Some(xp), Color::Red);
                        self.left_rotate(xp);
                        w = (*xp.as_ptr()).right;
                    }

                    let wn = w.unwrap();
                    if color_of((*wn.as_ptr()).left) == Color::Black
                        && color_of((*wn.as_ptr()).right) == Color::Black
                    {
                        // Case 2: both of w's children are black — push the
                        // double-black up one level.
                        set_color(w, Color::Red);
                        x = Some(xp);
                        x_parent = (*xp.as_ptr()).parent;
                    } else {
                        // Case 3: w's right child is black — right-rotate w
                        // to convert to case 4.
                        if color_of((*wn.as_ptr()).right) == Color::Black {
                            set_color((*wn.as_ptr()).left, Color::Black);
                            set_color(w, Color::Red);
                            self.right_rotate(wn);
                            w = (*xp.as_ptr()).right;
                        }
                        // Case 4: w's right child is red — left-rotate and
                        // recolour to resolve the double-black.
                        let wn2 = w.unwrap();
                        set_color(w, (*xp.as_ptr()).color);
                        set_color(Some(xp), Color::Black);
                        set_color((*wn2.as_ptr()).right, Color::Black);
                        self.left_rotate(xp);
                        x = self.root;
                        x_parent = None;
                    }
                } else {
                    // Symmetric case: x is a right child.
                    let mut w = (*xp.as_ptr()).left;

                    if color_of(w) == Color::Red {
                        set_color(w, Color::Black);
                        set_color(Some(xp), Color::Red);
                        self.right_rotate(xp);
                        w = (*xp.as_ptr()).left;
                    }

                    let wn = w.unwrap();
                    if color_of((*wn.as_ptr()).right) == Color::Black
                        && color_of((*wn.as_ptr()).left) == Color::Black
                    {
                        set_color(w, Color::Red);
                        x = Some(xp);
                        x_parent = (*xp.as_ptr()).parent;
                    } else {
                        if color_of((*wn.as_ptr()).left) == Color::Black {
                            set_color((*wn.as_ptr()).right, Color::Black);
                            set_color(w, Color::Red);
                            self.left_rotate(wn);
                            w = (*xp.as_ptr()).left;
                        }
                        let wn2 = w.unwrap();
                        set_color(w, (*xp.as_ptr()).color);
                        set_color(Some(xp), Color::Black);
                        set_color((*wn2.as_ptr()).left, Color::Black);
                        self.right_rotate(xp);
                        x = self.root;
                        x_parent = None;
                    }
                }
            }
            // x (or the root) is now singly black.
            set_color(x, Color::Black);
        }
    }

    // --- iteration -----------------------------------------------------------

    /// Returns a shared in-order iterator over `(&K, &V)` pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            // SAFETY: root is None (empty tree) or a live node.
            next: self.root.map(|r| unsafe { min_from(r) }),
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable in-order iterator over `(&K, &mut V)` pairs.
    ///
    /// Keys are immutable through the iterator; mutating them would violate
    /// the BST ordering invariant.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            // SAFETY: same as `iter`.
            next: self.root.map(|r| unsafe { min_from(r) }),
            len: self.len,
            _marker: PhantomData,
        }
    }

    // --- clear ---------------------------------------------------------------

    /// Removes and frees all entries.
    ///
    /// Uses an iterative left-spine rotation to avoid recursion (important in
    /// kernel contexts where stack depth is limited). The token is threaded
    /// through every `deallocate` call and returned once the tree is empty.
    ///
    /// After this call [`is_empty`](RbTree::is_empty) returns `true` and the
    /// tree may be reused or dropped safely.
    pub fn clear<Token>(&mut self, token: Token) -> Token
    where
        Token: CanAcquire<ID::Level> + PreviousToken,
    {
        let mut token = token;
        let mut node = self.root;
        while let Some(n) = node {
            // SAFETY: `n` is a live node we still own.
            unsafe {
                if let Some(l) = (*n.as_ptr()).left {
                    // Right-rotate n down so the left child becomes the new
                    // root of this sub-traversal; avoids deep recursion.
                    (*n.as_ptr()).left = (*l.as_ptr()).right;
                    (*l.as_ptr()).right = Some(n);
                    node = Some(l);
                } else {
                    // n has no left child: free it and continue with the right.
                    node = (*n.as_ptr()).right;
                    token = self.free_node(n, token);
                }
            }
        }
        self.root = None;
        self.len = 0;
        token
    }
}

impl<K, V, ID: LockId, A: Allocator<ID>> Drop for RbTree<K, V, ID, A> {
    /// Panics if the tree is non-empty.
    ///
    /// Memory cannot be released here because `Drop::drop` cannot accept the
    /// lock token required by the allocator. Call [`RbTree::clear`] before the
    /// tree goes out of scope.
    fn drop(&mut self) {
        if !self.is_empty() {
            panic!("A non-empty RbTree must never be dropped. Use RbTree::clear(...) instead!");
        }
    }
}

/// Shared in-order iterator. Produced by [`RbTree::iter`].
pub struct Iter<'a, K, V> {
    next: Link<K, V>,
    len: usize,
    _marker: PhantomData<(&'a K, &'a V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next?;
        // SAFETY: `cur` is a live node; shared refs are bounded by `'a` which
        // is tied to the `&RbTree` that created this iterator.
        unsafe {
            self.next = successor(cur);
            self.len -= 1;
            let node = cur.as_ptr();
            Some((&(*node).key, &(*node).value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}

/// Mutable in-order iterator. Produced by [`RbTree::iter_mut`].
pub struct IterMut<'a, K, V> {
    next: Link<K, V>,
    len: usize,
    _marker: PhantomData<(&'a K, &'a mut V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next?;
        // SAFETY: each node is visited exactly once, so the `&mut V` is
        // unique for the duration of the iteration. The key ref is shared.
        // Both are bounded by `'a`.
        unsafe {
            self.next = successor(cur);
            self.len -= 1;
            let node = cur.as_ptr();
            Some((&(*node).key, &mut (*node).value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {}

impl<'a, K: Ord, V, ID: LockId, A: Allocator<ID>> IntoIterator for &'a RbTree<K, V, ID, A> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Ord, V, ID: LockId, A: Allocator<ID>> IntoIterator for &'a mut RbTree<K, V, ID, A> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;
    use std::vec::Vec;

    use crate::kernel::locking::{EpilogueLevel, RootToken};
    use crate::kernel::locking::{MemoryManagementLevelID, PreviousToken};

    use super::*;

    extern crate std;

    struct TestAllocator;

    impl TestAllocator {
        fn new() -> Self {
            Self
        }
    }

    unsafe impl Allocator<MemoryManagementLevelID> for TestAllocator {
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

    /// Insert `key`/`value`, asserting no allocation error occurs.
    /// Returns `(old_value, token)`.
    macro_rules! insert {
        ($tree:expr, $key:expr, $value:expr, $token:expr) => {
            $tree
                .try_insert($key, $value, $token)
                .unwrap_or_else(|(e, _)| panic!("insert failed: {:?}", e))
        };
    }

    /// Tiny deterministic PRNG — no external crates needed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Recursively verify the red-black invariants and BST order.
    /// Returns the black-height of the subtree.
    unsafe fn verify_node<K: Ord + core::fmt::Debug, V>(
        n: Link<K, V>,
        lower: Option<&K>,
        upper: Option<&K>,
        count: &mut usize,
    ) -> usize {
        match n {
            None => 1, // nil leaf is black
            Some(p) => {
                *count += 1;
                let node = unsafe { p.as_ref() };

                // BST ordering
                if let Some(lo) = lower {
                    assert!(node.key > *lo, "BST order violated (lower bound)");
                }
                if let Some(hi) = upper {
                    assert!(node.key < *hi, "BST order violated (upper bound)");
                }

                // No red node may have a red child
                if node.color == Color::Red {
                    assert_eq!(
                        unsafe { color_of(node.left) },
                        Color::Black,
                        "red-red violation (left)"
                    );
                    assert_eq!(
                        unsafe { color_of(node.right) },
                        Color::Black,
                        "red-red violation (right)"
                    );
                }

                // Parent back-links must be consistent
                if let Some(l) = node.left {
                    assert_eq!(
                        unsafe { (*l.as_ptr()).parent },
                        Some(p),
                        "bad left parent link"
                    );
                }
                if let Some(r) = node.right {
                    assert_eq!(
                        unsafe { (*r.as_ptr()).parent },
                        Some(p),
                        "bad right parent link"
                    );
                }

                let lbh = unsafe { verify_node(node.left, lower, Some(&node.key), count) };
                let rbh = unsafe { verify_node(node.right, Some(&node.key), upper, count) };
                assert_eq!(lbh, rbh, "black-height mismatch at {:?}", node.key);

                lbh + if node.color == Color::Black { 1 } else { 0 }
            }
        }
    }

    fn verify<K: Ord + core::fmt::Debug, V>(
        t: &RbTree<K, V, MemoryManagementLevelID, TestAllocator>,
    ) {
        unsafe {
            assert_eq!(color_of(t.root), Color::Black, "root must be black");
            if let Some(r) = t.root {
                assert_eq!((*r.as_ptr()).parent, None, "root must have no parent");
            }
            let mut count = 0usize;
            verify_node(t.root, None, None, &mut count);
            assert_eq!(count, t.len(), "node count != len");
        }
    }

    #[test]
    fn empty_tree() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.get(&5), None);
        verify(&t);

        // Empty tree: clear is a no-op and returns the token.
        let token = t.clear(token);
        // `t` is now empty and may be dropped safely (Drop checks is_empty).
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn sequential_insert_and_get() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in 0..1000_i32 {
            let (old, t2) = insert!(t, i, i * 10, token);
            assert!(old.is_none(), "expected no previous mapping for {i}");
            token = t2;
            verify(&t);
        }

        assert_eq!(t.len(), 1000);
        for i in 0..1000_i32 {
            assert_eq!(t.get(&i), Some(&(i * 10)));
        }
        assert_eq!(t.get(&1000), None);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn reverse_insert() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in (0..500_i32).rev() {
            (_, token) = insert!(t, i, i, token);
            verify(&t);
        }
        assert_eq!(t.len(), 500);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn duplicate_insert_returns_old() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        let (old, token) = insert!(t, 42, 1, token);
        assert_eq!(old, None);

        let (old, token) = insert!(t, 42, 2, token);
        assert_eq!(old, Some(1));

        let (old, token) = insert!(t, 42, 3, token);
        assert_eq!(old, Some(2));

        assert_eq!(t.len(), 1);
        assert_eq!(t.get(&42), Some(&3));
        verify(&t);

        let token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn iteration_is_sorted() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i64, i64, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());
        let mut model = BTreeMap::new();
        let mut rng = Rng(0x1234_5678_9abc_def0);

        for _ in 0..2000 {
            let k = (rng.next() % 5000) as i64;
            (_, token) = insert!(t, k, k, token);
            model.insert(k, k);
        }

        let got: Vec<_> = t.iter().map(|(k, v)| (*k, *v)).collect();
        let want: Vec<_> = model.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(got, want);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn iter_mut_modifies() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in 0..100_i32 {
            (_, token) = insert!(t, i, i, token);
        }
        for (_, v) in t.iter_mut() {
            *v += 1000;
        }
        for i in 0..100_i32 {
            assert_eq!(t.get(&i), Some(&(i + 1000)));
        }
        verify(&t);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn remove_basic() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in 0..100_i32 {
            (_, token) = insert!(t, i, i * 2, token);
        }

        // Remove even keys.
        for i in (0..100_i32).step_by(2) {
            let (val, t2) = t.remove(&i, token);
            token = t2;
            assert_eq!(val, Some(i * 2), "wrong value removed for key {i}");
            verify(&t);
        }

        assert_eq!(t.len(), 50);

        for i in 0..100_i32 {
            if i % 2 == 0 {
                assert_eq!(t.get(&i), None);
            } else {
                assert_eq!(t.get(&i), Some(&(i * 2)));
            }
        }

        // Removing a key that does not exist returns None.
        let (val, t2) = t.remove(&1000, token);
        token = t2;
        assert_eq!(val, None);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn remove_until_empty() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in 0..300_i32 {
            (_, token) = insert!(t, i, i, token);
        }
        for i in 0..300_i32 {
            let (val, t2) = t.remove(&i, token);
            token = t2;
            assert_eq!(val, Some(i));
            verify(&t);
        }

        assert!(t.is_empty());
        assert_eq!(t.root, None);

        // Explicit clear on an already-empty tree must not panic.
        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn clear_frees_everything() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in 0..500_i32 {
            (_, token) = insert!(t, i, i, token);
        }

        // After clear the tree must be empty and reusable.
        token = t.clear(token);
        assert!(t.is_empty());

        // Reuse after clear.
        (_, token) = insert!(t, 7, 7, token);
        assert_eq!(t.get(&7), Some(&7));
        verify(&t);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn borrowed_key_lookup() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<std::string::String, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        (_, token) = insert!(t, "hello".to_owned(), 1, token);
        (_, token) = insert!(t, "world".to_owned(), 2, token);

        // &str lookup via Borrow<str>
        assert_eq!(t.get("hello"), Some(&1));
        assert_eq!(t.contains_key("world"), true);

        let (val, t2) = t.remove("world", token);
        token = t2;
        assert_eq!(val, Some(2));
        assert_eq!(t.get("world"), None);
        verify(&t);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn values_are_dropped() {
        use std::rc::Rc;

        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let probe = Rc::new(());
        let mut t: RbTree<i32, Rc<()>, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        for i in 0..50_i32 {
            (_, token) = insert!(t, i, Rc::clone(&probe), token);
        }
        assert_eq!(Rc::strong_count(&probe), 51);

        // Remove half explicitly; their Rc clones must be dropped.
        for i in 0..25_i32 {
            let (val, t2) = t.remove(&i, token);
            token = t2;
            drop(val); // explicit drop to make it clear when the Rc is released
        }
        assert_eq!(Rc::strong_count(&probe), 26);

        // clear drops the remaining 25; only `probe` itself remains.
        token = t.clear(token);
        assert_eq!(Rc::strong_count(&probe), 1, "clear must drop all values");

        drop(t);

        epilogue_level.leave(token);
    }

    #[test]
    fn double_remove_returns_none() {
        let root_token = unsafe { RootToken::forge() };
        let (epilogue_level, mut token) = EpilogueLevel::enter(root_token);

        let mut t: RbTree<i32, i32, MemoryManagementLevelID, TestAllocator> =
            RbTree::new_in(TestAllocator::new());

        (_, token) = insert!(t, 99, 99, token);

        let (val, t2) = t.remove(&99, token);
        token = t2;
        assert_eq!(val, Some(99));

        let (val, t2) = t.remove(&99, token);
        token = t2;
        assert_eq!(val, None);

        assert!(t.is_empty());
        verify(&t);

        token = t.clear(token);
        drop(t);

        epilogue_level.leave(token);
    }
}
