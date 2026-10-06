//! Core-local (per-CPU) storage.
//!
//! Variables declared with [`core_local!`](crate::core_local) are placed in
//! the `.percpu` section, which holds the template for a single core's block.
//! A variable is reached as
//!
//! ```text
//! GS base of the current core + (&VAR - _percpu_start)
//! ```
//!
//! so only its offset within the block matters, never its link-time address.
//! The linker script reserves no storage: it only exports what to copy
//! (`_percpu_start`, `_percpu_size`) and how far apart the copies must sit
//! (`_percpu_stride`). At boot, once the real core count is known, the blocks
//! are allocated, the template is copied into each of them, and every core's
//! `GS` is pointed at its own copy — see [`init_block_base`], which fills in
//! the base address a block is reached through.
//!
//! The variable that carries a declaration's name is therefore only the
//! template. It must never be read as a Rust value — [`PerCPU`] keeps its
//! payload private precisely so that the only reachable path goes through
//! the address calculation in [`PerCPU::with`] and friends.

use core::cell::RefCell;
use core::ffi::c_void;

use crate::arch::CPU;
use crate::arch::generic::cpu::{CPU as _, InterruptFlag};
use crate::kernel::bounded_buffer::{BoundedBuffer, Mode};
use crate::kernel::btreemap::BTreeMap;
use crate::kernel::btreeset::BTreeSet;
use crate::kernel::hashmap::{DefaultHashBuilder, HashMap};
use crate::kernel::hashset::HashSet;
use crate::kernel::linked_list::LinkedList;
use crate::kernel::locking::{MemoryManagementLevelID, PreviousToken};
use crate::kernel::mpsc::MPSC;
use crate::kernel::spsc::SPSC;
use crate::kernel::vec::Vec;
use crate::utils::allocator::Allocator;

// Defined by the kernel's linker script only. The bootloader links this crate
// as a library, and the host tests build it for the host, both with neither
// the script nor a `.percpu` section, so in those builds the two accessors
// below panic rather than name the symbols. Naming
// them would fail the bootloader's link as soon as any core-local access ends
// up in it, even one it never runs.
#[cfg(all(not(test), not(feature = "library")))]
unsafe extern "C" {
    /// First byte of the `.percpu` template, defined by the linker script.
    static _percpu_start: u8;

    /// Distance between two core-local blocks, defined by the linker script.
    ///
    /// An absolute symbol: the stride is its *address*, there is nothing to
    /// read at it.
    static _percpu_stride: u8;
}

/// Address of the `.percpu` template, the block core 0 runs on.
#[cfg(all(not(test), not(feature = "library")))]
#[inline]
fn template_start() -> usize {
    (&raw const _percpu_start) as usize
}

/// Distance between two core-local blocks.
#[cfg(all(not(test), not(feature = "library")))]
#[inline]
fn stride() -> usize {
    (&raw const _percpu_stride) as usize
}

/// See the kernel build of this function.
///
/// # Panics
///
/// Always: there is no core-local storage outside the kernel image.
#[cfg(any(test, feature = "library"))]
fn template_start() -> usize {
    panic!("core-local storage only exists in the kernel")
}

/// See the kernel build of this function.
///
/// # Panics
///
/// Always: there is no core-local storage outside the kernel image.
#[cfg(any(test, feature = "library"))]
fn stride() -> usize {
    panic!("core-local storage only exists in the kernel")
}

/// Base address of `cpu_id`'s core-local block.
///
/// The blocks sit one `_percpu_stride` apart, starting at the template
/// itself, so core 0 runs on the template and core `n` on the `n`-th copy the
/// bootloader has made.
///
/// `cpu_id` is not validated — the core count is not known here, and an id
/// beyond it simply yields an address past the last block.
#[inline]
pub fn block_base(cpu_id: usize) -> *const c_void {
    let addr = template_start() + cpu_id * stride();
    addr as _
}

/// Publishes a block's own base address in its first quadword and returns
/// that address.
///
/// This is the one write that makes a block usable: [`PerCPU`] resolves every
/// access as `gs:0 + offset`, so the slot the linker script reserves at offset
/// 0 has to name the block before the first core-local access on that core.
/// Pointing `GS` at the block is left to the caller, since how a core carries
/// its base is architecture-specific.
///
/// # Safety
///
/// `cpu_id` must be below the number of cores the bootloader has replicated
/// the template for; otherwise this writes past the last block, into memory
/// belonging to something else.
///
/// Nothing on the calling core may have touched a core-local variable yet,
/// and no other core may be running on this block.
#[inline]
pub unsafe fn init_block_base(cpu_id: usize) -> *const c_void {
    let base = block_base(cpu_id);

    // SAFETY: per the contract above, `base` names this core's block, whose
    // first quadword is reserved for exactly this by the linker script.
    unsafe { core::ptr::with_exposed_provenance_mut::<usize>(base as _).write(base as _) };

    base
}

/// Declares one or more core-local variables.
///
/// ```ignore
/// core_local! {
///     (pub) static FOO: *const c_void;   // starts out unset
///     static BAR: usize = 0;             // every core starts at 0
/// }
/// ```
///
/// A variable declared without an initialiser starts out *unset* on every core
/// and [`PerCPU::with`] panics until something calls [`PerCPU::set`]. With
/// `= <const expr>` the value becomes part of the template, so every core
/// starts out holding it.
///
/// An initialiser requires [`T: TemplateValue`](TemplateValue), because the
/// template is bit-copied into every core's block at boot; see
/// [`PerCPU::with_value`]. Every `Copy` type qualifies, and so do the kernel's
/// collections, whose only const-constructible value is the empty one:
///
/// ```ignore
/// core_local! {
///     static WORK: LinkedList<Work> = LinkedList::new();
/// }
/// ```
///
/// `Default` cannot be used here — it is not a `const fn`, and the macro has
/// no way to know whether a type implements it. Use
/// [`PerCPU::with_or_default`] instead, which installs `T::default()` on
/// first use.
///
/// This macro is the only supported way to create a [`PerCPU`]: its
/// constructors are `unsafe` and require the item to live in `.percpu`.
///
/// Any visibility is accepted — `pub`, `pub(crate)`, `pub(super)`,
/// `pub(in some::path)`, or none for a private item — written either bare
/// (`pub(crate) static ...`) or parenthesised (`(pub(crate)) static ...`).
/// Both spellings expand identically, and `()` is an explicit way to write
/// "private".
///
/// A declaration may additionally ask for an exported accessor for its
/// offset, which is how assembly and other non-Rust callers reach a
/// core-local variable:
///
/// ```ignore
/// core_local! {
///     #[export_offset(__boot_stack_offset)]
///     /// Boot stack of a core.
///     pub static BOOT_STACK: Stack = Stack([0; SIZE]);
/// }
/// ```
///
/// This emits, next to the item itself,
///
/// ```ignore
/// #[unsafe(no_mangle)]
/// pub extern "C" fn __boot_stack_offset(cpu_id: usize) -> usize
/// ```
///
/// which returns [`PerCPU::offset`] for `cpu_id`, i.e. the offset of that
/// core's copy relative to `_percpu_start`. The symbol name has to be spelled
/// out, since a `macro_rules!` macro cannot build an identifier from the name
/// of the item.
///
/// `#[export_offset(...)]` must be the *first* attribute of the declaration:
/// the macro matches it literally, and a preceding `#[...]` or doc comment
/// would be swallowed by the generic attribute list instead.
///
/// The items are deliberately *not* `static mut`: [`PerCPU`] carries its own
/// [`UnsafeCell`], so mutation needs no `mut` on the item, and a plain
/// `static` can be used directly as `FOO.with(...)` — a `static mut` would
/// force every call site through `&raw const` to satisfy edition 2024.
#[macro_export]
macro_rules! core_local {
    () => {};

    // ---- parenthesised visibility: strip the parentheses and re-enter ----
    //
    // Everything after `static` is forwarded verbatim, so these two rules
    // cover the initialised and uninitialised forms alike.

    // `()` — explicitly private. Needs its own rule: `$vis:vis` does not
    // match the empty visibility when the very next token is `)`.
    (
        $(#[$attr:meta])*
        () static $($tail:tt)*
    ) => {
        $crate::core_local! {
            $(#[$attr])*
            static $($tail)*
        }
    };

    (
        $(#[$attr:meta])*
        ($vis:vis) static $($tail:tt)*
    ) => {
        $crate::core_local! {
            $(#[$attr])*
            $vis static $($tail)*
        }
    };

    // ---- exported offset accessor ----------------------------------------
    //
    // Internal rule emitting the accessor itself, shared by the two forms
    // below.
    (@offset $offset:ident, $name:ident) => {
        #[doc = concat!(
            "Offset of `", stringify!($name), "` in the core-local block of ",
            "`cpu_id`, relative to `_percpu_start`."
        )]
        #[unsafe(no_mangle)]
        pub extern "C" fn $offset(cpu_id: usize) -> usize {
            $name.offset(cpu_id)
        }
    };

    // The marker is matched literally, so it has to come first: the generic
    // attribute list of the rules below would otherwise consume it as an
    // ordinary `#[...]`. The parenthesised-visibility rules above pass it
    // through as one of `$attr`, so both spellings of the visibility reach
    // these rules.
    //
    // The item itself is emitted by re-entering the macro, which keeps the
    // initialised and uninitialised forms in one place.

    (
        #[export_offset($offset:ident)]
        $(#[$attr:meta])*
        $vis:vis static $name:ident : $ty:ty = $init:expr;
        $($rest:tt)*
    ) => {
        $crate::core_local! {
            $(#[$attr])*
            $vis static $name: $ty = $init;
        }

        $crate::core_local! { @offset $offset, $name }

        $crate::core_local! { $($rest)* }
    };

    (
        #[export_offset($offset:ident)]
        $(#[$attr:meta])*
        $vis:vis static $name:ident : $ty:ty;
        $($rest:tt)*
    ) => {
        $crate::core_local! {
            $(#[$attr])*
            $vis static $name: $ty;
        }

        $crate::core_local! { @offset $offset, $name }

        $crate::core_local! { $($rest)* }
    };

    // ---- emit, with an explicit initialiser -------------------------------
    (
        $(#[$attr:meta])*
        $vis:vis static $name:ident : $ty:ty = $init:expr;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        #[unsafe(link_section = ".percpu")]
        // SAFETY: `link_section` above places the item in `.percpu`, which is
        // what `PerCPU`'s constructors require.
        $vis static $name: $crate::kernel::core_local::PerCPU<$ty> =
            unsafe { $crate::kernel::core_local::PerCPU::with_value($init) };

        $crate::core_local! { $($rest)* }
    };

    // ---- emit, starting out empty ----------------------------------------
    (
        $(#[$attr:meta])*
        $vis:vis static $name:ident : $ty:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        #[unsafe(link_section = ".percpu")]
        // SAFETY: `link_section` above places the item in `.percpu`, which is
        // what `PerCPU`'s constructors require.
        $vis static $name: $crate::kernel::core_local::PerCPU<$ty> =
            unsafe { $crate::kernel::core_local::PerCPU::new() };

        $crate::core_local! { $($rest)* }
    };
}

/// A value that may stand in the `.percpu` template.
///
/// The template is *bit-copied* into every core's block at boot, so an
/// initialiser given to [`core_local!`](crate::core_local) is duplicated once
/// per core without any constructor or `Clone` running. This trait is what
/// says a value survives that.
///
/// # Safety
///
/// Every copy has to be an independent, valid value of the type, which means
/// the value must own nothing — no allocation, no handle, no reference count —
/// and must not point into itself. A value that owns something would end up
/// with one owner per core, and dropping any two of them would be a double
/// free.
///
/// The bound is on the *type*, while what is really required holds of the
/// particular value in the template, so an implementation for a type that is
/// not [`Copy`] is only sound if no const expression can build a value of it
/// that owns something. That is the case for the kernel's collections, and it
/// is the reason each of them is listed below by hand.
pub unsafe trait TemplateValue {}

// SAFETY: a `Copy` type owns nothing by definition — `Copy` and `Drop` are
// mutually exclusive — and copying one is already a bit-copy.
unsafe impl<T: Copy> TemplateValue for T {}

// The collections below are not `Copy`, and must never become `Copy`: each of
// them owns whatever it holds, and each panics in `Drop` rather than free it
// without a token. They are listed here anyway because the *only* value of
// any of them a const expression can produce is the empty one — putting
// anything in a collection allocates, and every operation that allocates
// takes a token and is therefore not `const`. An empty collection holds no
// block and no node, so bit-copying it hands every core its own, equally
// empty, collection.
//
// Each impl requires the allocator handle to be `Copy`, for the reason the
// blanket impl above gives: it is duplicated along with the collection. The
// kernel `Heap`, which is the default, is a unit struct and qualifies.
//
// Adding a `Copy` impl to any of these types would make the impl here overlap
// the blanket one and fail to compile, which is the outcome to want: such a
// type could be copied while it holds something.

// SAFETY: as argued above.
unsafe impl<T, A> TemplateValue for LinkedList<T, A> where
    A: Allocator<MemoryManagementLevelID> + Copy
{
}

// SAFETY: as above.
unsafe impl<T, A> TemplateValue for Vec<T, A> where A: Allocator<MemoryManagementLevelID> + Copy {}

// SAFETY: as above.
unsafe impl<K, V, A> TemplateValue for BTreeMap<K, V, A> where
    A: Allocator<MemoryManagementLevelID> + Copy
{
}

// SAFETY: as above.
unsafe impl<T, A> TemplateValue for BTreeSet<T, A> where A: Allocator<MemoryManagementLevelID> + Copy
{}

// SAFETY: as argued above.
unsafe impl<T, const N: usize> TemplateValue for MPSC<T, N> {}

// SAFETY: as argued above.
unsafe impl<T, const N: usize> TemplateValue for SPSC<T, N> {}

// SAFETY: as argued above.
unsafe impl<T, const N: usize, M: Mode> TemplateValue for BoundedBuffer<T, N, M> {}

// The two hashed collections are pinned to `DefaultHashBuilder`, the hasher
// they use unless another one is named, rather than taking any `S: Copy`: a
// map holds its builder by value, so the builder is duplicated along with the
// map, and `BuildHasherDefault` is not `Copy` — it owns nothing either, being
// a `PhantomData`, but std says so nowhere the compiler can use. A map built
// on some other hasher needs its own impl here, and is sound to add exactly
// when that hasher owns nothing.

// SAFETY: as argued above, and `DefaultHashBuilder` is a zero-sized
// `PhantomData` marker that owns nothing.
unsafe impl<K, V, A> TemplateValue for HashMap<K, V, DefaultHashBuilder, A> where
    A: Allocator<MemoryManagementLevelID> + Copy
{
}

// SAFETY: as above.
unsafe impl<T, A> TemplateValue for HashSet<T, DefaultHashBuilder, A> where
    A: Allocator<MemoryManagementLevelID> + Copy
{
}

/// Storage for one core-local variable.
///
/// Starts out unset on every core; [`set`](PerCPU::set) installs a value, and
/// an initialiser given to [`core_local!`](crate::core_local) puts one in the
/// template.
#[repr(transparent)]
pub struct PerCPU<T>(RefCell<Option<T>>);

// SAFETY: every core reaches a distinct copy through its own GS base, so the
// value is never shared between cores, and every accessor masks interrupts
// for the whole of its access, so a prologue on this core cannot observe a
// half-finished one. The template itself is unreachable: the payload is
// private and every accessor redirects through `local`.
unsafe impl<T> Sync for PerCPU<T> {}

/// Masks interrupts for as long as it is alive, restoring the previous state
/// on drop.
///
/// Core-local access has to be atomic with respect to this core: a prologue
/// interrupting the middle of a `with_mut` would alias the `&mut` handed to
/// the closure. Masking also pins the thread to this core, so a preemption
/// cannot carry a borrow of *this* core's block onto another one.
struct InterruptGuard(InterruptFlag);

impl InterruptGuard {
    #[inline]
    fn new() -> Self {
        let flag = CPU::interrupt_flag();

        // SAFETY: the previous state is captured above and restored on drop.
        unsafe { CPU::raw_disable_interrupts() };

        Self(flag)
    }
}

impl Drop for InterruptGuard {
    #[inline]
    fn drop(&mut self) {
        if self.0 == InterruptFlag::Enabled {
            // SAFETY: interrupts were enabled when this guard was created.
            unsafe { CPU::raw_enable_interrupts() };
        }
    }
}

impl<T> PerCPU<T> {
    /// Creates an unset slot.
    ///
    /// # Safety
    ///
    /// The result must end up in the `.percpu` section. Every accessor
    /// locates this core's copy as `GS base + (self - _percpu_start)`, which
    /// is a meaningless address for a `PerCPU` living anywhere else — on the
    /// stack, in `.data`, or inside another struct. Use
    /// [`core_local!`](crate::core_local), which places the item correctly.
    pub const unsafe fn new() -> Self {
        Self(RefCell::new(None))
    }

    /// Creates a slot that every core starts out holding.
    ///
    /// `value` has to be a const expression, since it becomes part of the
    /// template.
    ///
    /// [`T: TemplateValue`](TemplateValue) is required because the template is
    /// *bit-copied* into every core's block at boot: every `Copy` type, and
    /// the kernel's collections, whose only const-constructible value is the
    /// empty one. See the trait for what a type has to satisfy.
    ///
    /// # Safety
    ///
    /// As for [`new`](PerCPU::new): the result must end up in `.percpu`.
    pub const unsafe fn with_value(value: T) -> Self
    where
        T: TemplateValue,
    {
        Self(RefCell::new(Some(value)))
    }

    /// Offset of this variable within a single block.
    ///
    /// Taken as a difference rather than as a bare address: `.percpu` is
    /// linked at a normal virtual address, and the subtraction also survives
    /// PIE relocation, since both operands are shifted by the same base.
    #[inline]
    fn block_offset(&self) -> usize {
        self as *const Self as usize - template_start()
    }

    /// Offset of `cpu_id`'s copy of this variable, relative to
    /// `_percpu_start`.
    ///
    /// The blocks sit one `_percpu_stride` apart and start at the template
    /// itself, so `cpu_id`'s copy lives at
    ///
    /// ```text
    /// _percpu_start + VAR.offset(cpu_id)
    /// ```
    ///
    /// This is the way to reach a variable *without* going through `GS`:
    /// early boot, where no core has a base yet, and code preparing another
    /// core's block. A core reaching its own variable uses
    /// [`with`](PerCPU::with) and friends instead, which resolve the block
    /// through `GS` and need no core id.
    ///
    /// `cpu_id` is not validated — the core count is not known here, and an
    /// id beyond it simply yields an offset past the last block.
    #[inline]
    pub fn offset(&self, cpu_id: usize) -> usize {
        self.block_offset() + cpu_id * stride()
    }

    /// This core's copy of the variable.
    #[inline]
    fn local(&self) -> &RefCell<Option<T>> {
        let offset = self.block_offset();

        // Offset 0 of every block holds that block's own base address.
        let base: usize;
        unsafe {
            core::arch::asm!(
                "mov {}, gs:0",
                out(reg) base,
                options(nomem, nostack, preserves_flags)
            );
        }

        // SAFETY: the block belongs to this core and lives for as long as the
        // kernel does, so handing out a reference borrowed from the template
        // is sound.
        unsafe {
            &*core::ptr::with_exposed_provenance::<Self>(base + offset).cast::<RefCell<Option<T>>>()
        }
    }

    /// Runs `f` on this core's value.
    ///
    /// Interrupts are masked for the duration, so a prologue on this core
    /// cannot observe or disturb the borrow.
    ///
    /// # Panics
    ///
    /// If no value has been installed on this core yet, or if this variable
    /// is already borrowed mutably.
    pub fn with<R, F: FnOnce(&T) -> R>(&self, f: F) -> R {
        let _interrupts = InterruptGuard::new();

        match &*self.local().borrow() {
            Some(value) => f(value),
            None => panic!("core-local variable read before it was set"),
        }
    }

    /// Runs `f` on this core's value, mutably, and hands the token back.
    ///
    /// Handing out `&mut T` while arbitrary code runs is the one access that
    /// really needs the interrupt discipline, so it follows the same protocol
    /// as the locking hierarchy: the caller's token is consumed for the
    /// duration and returned alongside `f`'s result. While the closure runs
    /// the caller holds no token and can therefore acquire nothing else.
    ///
    /// # Panics
    ///
    /// If no value has been installed on this core yet, or if this variable
    /// is already borrowed.
    pub fn with_mut<R, F, Token>(&self, token: Token, f: F) -> (R, Token)
    where
        F: FnOnce(&mut T) -> R,
        Token: PreviousToken,
    {
        // Interrupts off first, exactly as when taking a Prologue-level lock.
        let state = CPU::disable_interrupts(token);

        let result = match &mut *self.local().borrow_mut() {
            Some(value) => f(value),
            None => panic!("core-local variable read before it was set"),
        };

        (result, CPU::restore_interrupts(state))
    }

    /// Installs `value` on this core, returning the previous one and the
    /// token.
    ///
    /// Takes a token for the same reason [`with_mut`](PerCPU::with_mut) does:
    /// it mutates this core's slot.
    pub fn set<Token>(&self, token: Token, value: T) -> (Option<T>, Token)
    where
        Token: PreviousToken,
    {
        let state = CPU::disable_interrupts(token);

        let previous = self.local().borrow_mut().replace(value);

        (previous, CPU::restore_interrupts(state))
    }

    /// Removes this core's value, returning it and the token.
    pub fn take<Token>(&self, token: Token) -> (Option<T>, Token)
    where
        Token: PreviousToken,
    {
        let state = CPU::disable_interrupts(token);

        let previous = self.local().borrow_mut().take();

        (previous, CPU::restore_interrupts(state))
    }

    /// Whether a value is installed on this core.
    pub fn is_set(&self) -> bool {
        let _interrupts = InterruptGuard::new();

        self.local().borrow().is_some()
    }
}

impl<T: Default> PerCPU<T> {
    /// Runs `f` on this core's value, installing `T::default()` first if the
    /// slot is still empty.
    ///
    /// This is the runtime counterpart to an explicit initialiser: `Default`
    /// cannot be used in the template, because `Default::default` is not a
    /// `const fn`.
    ///
    /// Takes a token even though it hands out `&T`, because installing the
    /// default mutates this core's slot.
    pub fn with_or_default<R, F, Token>(&self, token: Token, f: F) -> (R, Token)
    where
        F: FnOnce(&T) -> R,
        Token: PreviousToken,
    {
        let state = CPU::disable_interrupts(token);

        let result = f(self.local().borrow_mut().get_or_insert_with(T::default));

        (result, CPU::restore_interrupts(state))
    }

    /// Runs `f` on this core's value, mutably, installing `T::default()`
    /// first if the slot is still empty, and hands the token back.
    ///
    /// See [`with_mut`](PerCPU::with_mut) for why this one takes a token.
    pub fn with_mut_or_default<R, F, Token>(&self, token: Token, f: F) -> (R, Token)
    where
        F: FnOnce(&mut T) -> R,
        Token: PreviousToken,
    {
        let state = CPU::disable_interrupts(token);

        let result = f(self.local().borrow_mut().get_or_insert_with(T::default));

        (result, CPU::restore_interrupts(state))
    }
}

// No `Default` impl on purpose: it would be a *safe* constructor, and so
// would reopen the hole that `new` being `unsafe` closes.
