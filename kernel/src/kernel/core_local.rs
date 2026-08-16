//! Core-local (per-CPU) storage.
//!
//! Variables declared with [`core_local!`] are placed in the `.percpu`
//! section, which holds the template for a single core's block. A variable
//! is reached as
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
//! `GS` is pointed at its own copy.
//!
//! The variable that carries a declaration's name is therefore only the
//! template. It must never be read as a Rust value — [`PerCPU`] keeps its
//! payload private precisely so that the only reachable path goes through
//! the address calculation in [`PerCPU::with`] and friends.

use core::cell::RefCell;

use crate::arch::CPU;
use crate::arch::generic::cpu::{CPU as _, InterruptFlag};
use crate::kernel::locking::PreviousToken;

unsafe extern "C" {
    /// First byte of the `.percpu` template, defined by the linker script.
    static _percpu_start: u8;
}

/// Declares one or more core-local variables.
///
/// ```ignore
/// core_local! {
///     (pub) static FOO: *const c_void;   // starts out empty
///     static BAR: usize = 0;             // every core starts at 0
/// }
/// ```
///
/// A variable declared without an initialiser starts out empty on every core
/// and [`PerCPU::with`] panics until something calls [`PerCPU::set`]. With
/// `= <const expr>` the value becomes part of the template, so every core
/// starts out holding it.
///
/// An initialiser requires `T: Copy`, because the template is bit-copied into
/// every core's block at boot; see [`PerCPU::with_value`].
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

/// Storage for one core-local variable.
///
/// Starts out empty on every core; [`set`](PerCPU::set) installs a value.
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
    /// Creates an empty slot.
    ///
    /// # Safety
    ///
    /// The result must end up in the `.percpu` section. Every accessor
    /// locates this core's copy as `GS base + (self - _percpu_start)`, which
    /// is a meaningless address for a `PerCPU` living anywhere else — on the
    /// stack, in `.data`, or inside another struct. Use [`core_local!`],
    /// which places the item correctly.
    pub const unsafe fn new() -> Self {
        Self(RefCell::new(None))
    }

    /// Creates a slot that every core starts out holding.
    ///
    /// `value` has to be a const expression, since it becomes part of the
    /// template.
    ///
    /// `T: Copy` is required because the template is *bit-copied* into every
    /// core's block at boot. A type that owns something would end up with one
    /// owner per core, and dropping any two of them would be a double free.
    ///
    /// # Safety
    ///
    /// As for [`new`](PerCPU::new): the result must end up in `.percpu`.
    pub const unsafe fn with_value(value: T) -> Self
    where
        T: Copy,
    {
        Self(RefCell::new(Some(value)))
    }

    /// This core's copy of the variable.
    #[inline]
    fn local(&self) -> &RefCell<Option<T>> {
        // Offset of this variable within a block. Taken as a difference
        // rather than as a bare address: `.percpu` is linked at a normal
        // virtual address, and the subtraction also survives PIE relocation,
        // since both operands are shifted by the same base.
        let offset = self as *const Self as usize - (&raw const _percpu_start) as usize;

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
        unsafe { &*core::ptr::with_exposed_provenance::<Self>(base + offset).cast::<RefCell<Option<T>>>() }
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
