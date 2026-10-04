//! Architecture-generic memory management primitives.
//!
//! This module defines the core types and traits for virtual memory management
//! in a portable, architecture-independent way. Concrete implementations are
//! provided by each architecture module (e.g. `arch::x86_64::paging`).
//!
//! # Overview
//!
//! - [`VirtualAddress`] and [`PhysicalAddress`] are distinct pointer wrappers,
//!   preventing accidental mixing of address spaces at the type level.
//! - [`AccessRights`] encodes read/write/execute permissions as a compact
//!   bitmask.
//! - [`PageSize`] describes the page size used for a mapping.
//! - [`CachingMode`] describes the memory type of a mapping.
//! - [`PageFrameAllocator`] abstracts physical memory allocation behind
//!   the locking hierarchy.
//! - [`Paging`] is the architecture-generic interface for manipulating
//!   hardware page tables.

use core::ffi::c_void;
use core::fmt::Display;

use arch_generic_derive::Address;

use crate::{
    kernel::locking::{CanAcquire, PreviousToken, level},
    user::errno::{Errno, ToErrno},
};

/// Represents the privilege level at which a memory region is accessible.
///
/// Maps directly to the hardware concept of ring levels — on x86_64 this
/// corresponds to the `U/S` (User/Supervisor) bit in page table entries.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrivilegeLevel {
    /// Accessible from userspace (ring 3 on x86_64).
    User,
    /// Accessible from kernel only (ring 0 on x86_64).
    Kernel,
}

impl Display for PrivilegeLevel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PrivilegeLevel::User => write!(f, "user"),
            PrivilegeLevel::Kernel => write!(f, "kernel"),
        }
    }
}

/// A single memory access permission.
///
/// The numeric value of each variant corresponds to its bit position within
/// an [`AccessRights`] bitmask, allowing efficient bitwise composition.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AccessRight {
    /// Page may be read.
    Readable = 0,
    /// Page may be written.
    Writable = 1,
    /// Page may be executed.
    ///
    /// On x86_64 this corresponds to the *absence* of the `NX`
    /// (No-Execute) bit. Requires `EFER.NXE = 1` for NX support to be
    /// active.
    Executable = 2,
}

impl Display for AccessRight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AccessRight::Readable => write!(f, "readable"),
            AccessRight::Writable => write!(f, "writable"),
            AccessRight::Executable => write!(f, "executable"),
        }
    }
}

/// A compact bitmask encoding a combined set of [`AccessRight`]s.
///
/// Internally stores one bit per [`AccessRight`] variant at the bit position
/// given by its discriminant:
///
/// ```text
/// bit 0 — Readable
/// bit 1 — Writable
/// bit 2 — Executable
/// ```
///
/// # Display
///
/// Formats as a Unix-style permission string, e.g. `[rwx]`, `[r--]`, `[-w-]`.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AccessRights(u8);

impl Display for AccessRights {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut buf = ['[' as u8, '-' as u8, '-' as u8, '-' as u8, ']' as u8];

        if self.0 & (1 << AccessRight::Readable as u8) != 0 {
            buf[1] = 'r' as u8;
        }

        if self.0 & (1 << AccessRight::Writable as u8) != 0 {
            buf[2] = 'w' as u8;
        }

        if self.0 & (1 << AccessRight::Executable as u8) != 0 {
            buf[3] = 'x' as u8;
        }

        let str = str::from_utf8(&buf).unwrap();
        write!(f, "{}", str)
    }
}

impl AccessRights {
    /// Returns an [`AccessRights`] with no permissions set (`[---]`).
    pub const fn none() -> Self {
        Self(0)
    }

    /// Returns an [`AccessRights`] with custom permissions set (`[rxx]`).
    pub const fn custom(r: bool, w: bool, x: bool) -> Self {
        let mut result = Self::none();

        if r {
            result.set_readable(true);
        }

        if w {
            result.set_writable(true);
        }

        if x {
            result.set_executable(true);
        }

        result
    }

    /// Returns an [`AccessRights`] with all permissions set (`[rwx]`).
    pub const fn full() -> Self {
        Self(
            1 << AccessRight::Readable as u8
                | 1 << AccessRight::Writable as u8
                | 1 << AccessRight::Executable as u8,
        )
    }

    /// Constructs an [`AccessRights`] from a slice of [`AccessRight`] values.
    ///
    /// Duplicate entries are ignored — the result is a bitmask union of all
    /// provided rights.
    ///
    /// # Example
    ///
    /// ```text
    /// let rights = AccessRights::new(&[AccessRight::Readable, AccessRight::Executable]);
    /// assert!(rights.is_readable());
    /// assert!(!rights.is_writable());
    /// assert!(rights.is_executable());
    /// ```
    pub const fn new(access_rights: &[AccessRight]) -> Self {
        let mut result = Self(0);

        let mut i = 0;
        while i < access_rights.len() {
            result.0 |= 1 << access_rights[i] as u8;
            i += 1;
        }

        result
    }

    /// Sets or clears the [`AccessRight::Readable`] bit.
    pub const fn set_readable(&mut self, readable: bool) {
        match readable {
            true => self.0 |= 1 << AccessRight::Readable as u8,
            false => self.0 &= !(1 << AccessRight::Readable as u8),
        };
    }

    /// Sets or clears the [`AccessRight::Writable`] bit.
    pub const fn set_writable(&mut self, writable: bool) {
        match writable {
            true => self.0 |= 1 << AccessRight::Writable as u8,
            false => self.0 &= !(1 << AccessRight::Writable as u8),
        };
    }

    /// Sets or clears the [`AccessRight::Executable`] bit.
    pub const fn set_executable(&mut self, executable: bool) {
        match executable {
            true => self.0 |= 1 << AccessRight::Executable as u8,
            false => self.0 &= !(1 << AccessRight::Executable as u8),
        };
    }

    /// Returns `true` if the [`AccessRight::Readable`] bit is set.
    pub const fn is_readable(&self) -> bool {
        self.0 & (1 << AccessRight::Readable as u8) != 0
    }

    /// Returns `true` if the [`AccessRight::Writable`] bit is set.
    pub const fn is_writable(&self) -> bool {
        self.0 & (1 << AccessRight::Writable as u8) != 0
    }

    /// Returns `true` if the [`AccessRight::Executable`] bit is set.
    pub const fn is_executable(&self) -> bool {
        self.0 & (1 << AccessRight::Executable as u8) != 0
    }
}

/// A virtual memory address typed by its pointee `T`.
///
/// Wraps a `*mut T` with a full pointer API generated by `#[derive(Address)]`.
/// Distinct from [`PhysicalAddress`] at the type level — passing one where
/// the other is expected is a compile error, preventing address-space
/// confusion.
///
/// Virtual addresses are valid only in the context of a specific page table;
/// dereferencing one without an active mapping is undefined behaviour.
#[derive(Address)]
#[repr(transparent)]
pub struct VirtualAddress<T>(*mut T);

unsafe impl<T> Sync for VirtualAddress<T> {}

/// A physical memory address typed by its pointee `T`.
///
/// Wraps a `*mut T` with a full pointer API generated by `#[derive(Address)]`.
/// Distinct from [`VirtualAddress`] at the type level — passing one where
/// the other is expected is a compile error.
///
/// Physical addresses refer to locations in the physical memory bus and are
/// only meaningful to hardware (MMU, DMA controllers, page table entries).
/// They must be mapped into a virtual address space before the CPU can access
/// the memory they describe.
#[derive(Address)]
#[repr(transparent)]
pub struct PhysicalAddress<T>(*mut T);

unsafe impl<T> Sync for PhysicalAddress<T> {}

/// Errors that can occur during paging operations.
#[derive(Debug)]
pub enum Error {
    /// The physical memory allocator has no free frames of the requested
    /// [`PageSize`].
    OutOfMemory,
    /// For the requested virtual address exists no mapping.
    NotMapped,
    /// For the requested virtual address a conflict exists, e.g., trying to map a huge page which would overwrite a page table.
    Conflict,
    /// For invalid address format, e.g., non-canonical addresses
    InvalidAddress,
}

impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::OutOfMemory => write!(f, "out of memory"),
            Error::NotMapped => write!(f, "no mapping"),
            Error::Conflict => write!(f, "conflict"),
            Error::InvalidAddress => write!(f, "invalid address"),
        }
    }
}

impl ToErrno for Error {
    fn to_errno(&self) -> Errno {
        match self {
            Error::OutOfMemory => Errno::ENOMEM,
            Error::NotMapped => Errno::EFAULT,
            Error::Conflict => Errno::EEXISTS,
            Error::InvalidAddress => Errno::EINVAL,
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

/// The size of a mapped page.
///
/// Larger pages reduce TLB pressure and page-walk overhead at the cost of
/// internal fragmentation. Hardware support for
/// [`Gigantic`](PageSize::Gigantic) pages must be verified at runtime
/// (e.g. `pdpe1gb` CPUID flag on x86_64).
///
/// Maps to hardware page table levels on x86_64:
///
/// | Variant    | Size  | Page table level |
/// |------------|-------|-----------------|
/// | `Regular`  | 4 KiB | PT (PTE)        |
/// | `Huge`     | 2 MiB | PMD entry       |
/// | `Gigantic` | 1 GiB | PUD entry       |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageSize {
    /// 4 KiB page — standard page size, mapped at the PTE level.
    Regular,
    /// 2 MiB huge page — mapped directly at the PMD level, skipping the PT.
    Huge,
    /// 1 GiB gigantic page — mapped directly at the PUD level, skipping
    /// PMD and PT.
    ///
    /// Requires hardware support. On x86_64, verify with:
    /// ```bash
    /// grep pdpe1gb /proc/cpuinfo
    /// ```
    Gigantic,
}

impl Display for PageSize {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PageSize::Regular => write!(f, "regular"),
            PageSize::Huge => write!(f, "huge"),
            PageSize::Gigantic => write!(f, "gigantic"),
        }
    }
}

/// The memory type of a mapping — how far the CPU may go in caching accesses
/// made through it.
///
/// Ordinary RAM wants [`Normal`](CachingMode::Normal), which is what
/// [`Default`] gives and what leaves the choice where it belongs, with the
/// architecture. The named modes exist for addresses that are not RAM, where
/// a cache's freedom to delay, merge and reorder accesses is visible to
/// whatever sits behind them — a device register that counts every read, a
/// framebuffer that only wants the writes to arrive eventually.
///
/// The mode governs a mapping, not a frame: the same physical memory reached
/// through two mappings of different modes is cached two different ways, and
/// on x86_64 that is an aliasing rule violation the architecture leaves
/// undefined. A frame should therefore be mapped with one mode at a time,
/// which includes the mode it carries in the kernel's direct map.
///
/// On x86_64 the mode is encoded as an index into the `IA32_PAT` MSR, built
/// from the `PWT`, `PCD` and `PAT` bits of the leaf entry; see
/// `arch::x86_64::paging::PAT_LAYOUT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CachingMode {
    /// Whatever the architecture caches ordinary memory as — write-back,
    /// wherever the choice exists.
    ///
    /// The mode for RAM, and the one to ask for when the memory behind a
    /// mapping has no demands of its own. It names no memory type, so a
    /// mapping made with it comes back out of [`Paging::resolve`] as the
    /// concrete mode the architecture picked, not as this.
    #[default]
    Normal,
    /// Write-back (`WB`) — reads and writes are cached, writes reach memory
    /// only when the line is evicted.
    ///
    /// What [`Normal`](CachingMode::Normal) amounts to on the architectures
    /// that offer it; worth naming outright only where a mapping has to be
    /// write-back for a reason of its own.
    WriteBack,
    /// Uncached (`UC`) — every access goes to the bus, in program order,
    /// exactly as many times as the program makes it.
    ///
    /// The mode for device registers, where a read can have a side effect and
    /// a dropped write is a lost command.
    Uncached,
    /// Write-combined (`WC`) — reads are uncached, writes are gathered in a
    /// buffer and released as bursts, in no particular order.
    ///
    /// For memory written in bulk and never read back, a framebuffer being
    /// the usual one. Anything that cares about the order its writes land in
    /// needs a fence, or another mode.
    WriteCombined,
    /// Write-through (`WT`) — reads are cached, writes update the cache and
    /// memory both.
    ///
    /// For memory a second party reads without going through this CPU's
    /// caches, while the CPU itself still benefits from caching its reads.
    WriteThrough,
}

impl Display for CachingMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CachingMode::Normal => write!(f, "normal"),
            CachingMode::WriteBack => write!(f, "write-back"),
            CachingMode::Uncached => write!(f, "uncached"),
            CachingMode::WriteCombined => write!(f, "write-combined"),
            CachingMode::WriteThrough => write!(f, "write-through"),
        }
    }
}

/// Allocates and frees physical page frames.
///
/// Implementors manage a pool of physical memory and hand out frames on demand,
/// one regular page at a time. A [`PageSize`] is a property of a *mapping* —
/// how much of the address space one page table entry covers — not of the
/// memory handed out here, and the only frames this trait is asked for are the
/// page tables themselves, which are one regular page each. A huge or gigantic
/// mapping is made from a frame its caller already holds.
/// All operations require a token at or above the `Memory` lock level, enforcing
/// that physical memory allocation always occurs under the correct lock
/// hierarchy.
///
/// `Memory` rather than `MemoryManagement`: handing out frames *is* the raw
/// memory this hierarchy names, and everything built on top of it — page
/// mappings, the heap — sits at `MemoryManagement` and has to be able to reach
/// down to it. A heap that runs dry refilling itself from the frame allocator
/// is exactly that descent.
///
/// The token is threaded through each call (consumed and returned) rather than
/// held by the allocator, keeping it compatible with the hierarchy's linear
/// token model.
pub trait PageFrameAllocator {
    /// Allocates a single physical page frame of
    /// [`REGULAR_PAGE_SIZE`](Paging::REGULAR_PAGE_SIZE) bytes.
    ///
    /// Returns the physical address of the allocated frame together with the
    /// token on success, or the error together with the token on failure so
    /// the caller can continue using the hierarchy.
    ///
    /// The frame is aligned to its own size, which is what a page table entry
    /// requires of it.
    fn allocate<Token>(token: Token) -> Result<(PhysicalAddress<c_void>, Token), (Error, Token)>
    where
        Token: CanAcquire<level::Memory> + PreviousToken;

    /// Frees a previously allocated physical page frame.
    ///
    /// # Safety
    ///
    /// - `phys_addr` must have been obtained from a previous call to
    ///   [`allocate`](PageFrameAllocator::allocate).
    /// - The frame must not be referenced by any active page table entry.
    /// - Calling this twice for the same frame is undefined behaviour.
    unsafe fn deallocate<Token>(phys_addr: PhysicalAddress<c_void>, token: Token) -> Token
    where
        Token: CanAcquire<level::Memory> + PreviousToken;
}

/// Architecture-generic interface for managing a hardware page table.
///
/// Implementors provide the concrete page table manipulation for a specific
/// architecture (e.g. `arch::x86_64::paging`). Generic kernel code
/// programs all memory mappings exclusively through this trait, keeping
/// architecture-specific page table structures behind the abstraction
/// boundary.
///
/// # Activation
///
/// A newly created page table is inactive — it has no effect on address
/// translation until [`activate`](Paging::activate) installs it in the hardware
/// register (CR3 on x86_64, `TTBR0`/`TTBR1` on AArch64).
///
/// # Safety
///
/// Several methods are `unsafe` because incorrect use can violate memory
/// safety globally: a bad mapping can corrupt kernel or user memory, cause
/// undefined behaviour on the next memory access, or crash the system.
pub trait Paging<PFA: PageFrameAllocator> {
    /// Size of a [`Regular`](PageSize::Regular) page in bytes.
    const REGULAR_PAGE_SIZE: usize;

    /// Size of a [`Huge`](PageSize::Huge) page in bytes, or `None` if the
    /// architecture has no such page size.
    const HUGE_PAGE_SIZE: Option<usize>;

    /// Size of a [`Gigantic`](PageSize::Gigantic) page in bytes, or `None` if
    /// the architecture has no such page size.
    const GIGANTIC_PAGE_SIZE: Option<usize>;

    /// Base-two logarithm of [`REGULAR_PAGE_SIZE`](Self::REGULAR_PAGE_SIZE).
    ///
    /// The width of a regular page's offset field, which is what a
    /// power-of-two allocator is parameterized by rather than the size itself.
    const REGULAR_PAGE_SHIFT: usize;

    /// Base-two logarithm of [`HUGE_PAGE_SIZE`](Self::HUGE_PAGE_SIZE), `None`
    /// exactly when that one is.
    const HUGE_PAGE_SHIFT: Option<usize>;

    /// Base-two logarithm of
    /// [`GIGANTIC_PAGE_SIZE`](Self::GIGANTIC_PAGE_SIZE), `None` exactly when
    /// that one is.
    const GIGANTIC_PAGE_SHIFT: Option<usize>;

    /// Gets the actual page size for [`PageSize`].
    ///
    /// If the target architecture supports the given page size, its size in
    /// bytes is returned. Otherwise, `None` is returned.
    fn page_size(page_size: PageSize) -> Option<usize>;

    /// Destroys the page tables and frees all associated page table frames.
    ///
    /// This must be called instead of letting [`Paging`] drop, since
    /// dropping without freeing the frames would leak memory. The method
    /// consumes `self` and returns the lock token once cleanup is complete.
    ///
    /// For each non-table page frame, `cb` will be invoked.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the page tables are no longer active before
    /// calling this function.
    ///
    /// Destroying active page tables causes all subsequent memory accesses on
    /// that CPU to fault.
    unsafe fn destroy<Token, CB>(self, cb: CB, token: Token) -> Token
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken,
        CB: FnMut(PhysicalAddress<c_void>, PageSize, Token) -> Token;

    /// Maps `virt_addr` to `phys_addr` with the given attributes.
    ///
    /// If a mapping already existed at `virt_addr`, it is replaced and the
    /// old [`PhysicalAddress`] is returned as `Ok(Some(...))`. If no prior
    /// mapping existed, returns `Ok(None)`.
    ///
    /// # Arguments
    ///
    /// - `virt_addr` — the virtual address to map.
    /// - `phys_addr` — the physical address to map it to.
    /// - `privilege_level` — whether the mapping is accessible from userspace.
    /// - `access_rights` — read/write/execute permissions.
    /// - `caching_mode` — the memory type accesses through this mapping get.
    /// - `size` — page size (4 KiB / 2 MiB / 1 GiB).
    ///
    /// # Safety
    ///
    /// - `phys_addr` must refer to valid physical memory for the entire
    ///   lifetime of the mapping.
    /// - Mapping the wrong physical address can silently corrupt memory.
    /// - The caller must ensure no conflicting aliases are created (e.g.
    ///   mapping the same frame as both writable and executable in a
    ///   W^X policy, or under two different [`CachingMode`]s).
    unsafe fn map<T, Token>(
        &mut self,
        virt_addr: VirtualAddress<T>,
        phys_addr: PhysicalAddress<T>,
        privilege_level: PrivilegeLevel,
        access_rights: AccessRights,
        caching_mode: CachingMode,
        size: PageSize,
        token: Token,
    ) -> Result<(Option<(PhysicalAddress<T>, PageSize)>, Token), (Error, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken;

    /// Removes the mapping for `virt_addr`, returning the physical address
    /// it previously pointed to.
    ///
    /// The caller is responsible for invalidating the TLB entry afterwards
    /// via [`invalidate`](Paging::invalidate) or
    /// [`invalidate_all`](Paging::invalidate_all). Failing to do so may
    /// allow stale translations to be used until the next context switch.
    ///
    /// # Safety
    ///
    /// - Unmapping a page that is still in use will cause a page fault on
    ///   the next access through that virtual address.
    /// - The caller must ensure no live references into the mapped region
    ///   exist at the time of unmapping.
    unsafe fn unmap<T, Token>(
        &mut self,
        virt_addr: VirtualAddress<T>,
        token: Token,
    ) -> Result<(PhysicalAddress<T>, PageSize, Token), (Error, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken;

    /// Walks the page table to resolve `virt_addr` to its physical address
    /// and mapping attributes.
    ///
    /// Returns `(physical_address, privilege_level, access_rights,
    /// caching_mode, page_size)` for the mapping covering `virt_addr`, or an
    /// error if no mapping exists.
    fn resolve<T>(
        &self,
        virt_addr: VirtualAddress<T>,
    ) -> Result<
        (
            PhysicalAddress<T>,
            PrivilegeLevel,
            AccessRights,
            CachingMode,
            PageSize,
        ),
        Error,
    >;

    /// Invalidates the TLB entry for a single virtual address on the local CPU.
    ///
    /// Must be called after [`unmap`](Paging::unmap) or any permission change
    /// to prevent the CPU from using a stale cached translation. On x86_64
    /// this emits an `invlpg` instruction.
    ///
    /// For bulk invalidations prefer [`invalidate_all`](Paging::invalidate_all).
    fn invalidate<T>(virt_addr: VirtualAddress<T>);

    /// Flushes all TLB entries on the local CPU.
    ///
    /// More expensive than [`invalidate`](Paging::invalidate) — prefer
    /// single-page invalidation when only one mapping has changed. On x86_64
    /// this is typically achieved by reloading CR3.
    fn invalidate_all();

    /// Installs this page table as the active one on the current CPU.
    ///
    /// On x86_64 this writes the physical address of the root PGD into CR3,
    /// immediately affecting all subsequent virtual address translations on
    /// this CPU.
    ///
    /// # Safety
    ///
    /// - The page table must contain valid mappings for all kernel addresses
    ///   that may be accessed after this call, including the current
    ///   instruction pointer, stack, and any interrupt handlers.
    /// - Activating an incomplete or malformed page table will immediately
    ///   cause a fault or silent memory corruption.
    unsafe fn activate(&self);
}

/// Translation both ways through an architecture's direct map of physical
/// memory.
///
/// [`Paging`] maps a virtual address to a physical one by walking page tables,
/// which needs the tables themselves to be reachable. The kernel solves that
/// the usual way, with a window in which all of physical memory appears at a
/// fixed offset, and this trait is that window: translation by arithmetic
/// alone, no page walk and no lock.
///
/// The trade is that it only covers addresses *in* the window. Nothing here
/// says whether a page is mapped, what rights it carries, or which address
/// space it belongs to — for that, go through [`Paging`].
///
/// Implemented per architecture because both the window's base and its extent
/// are part of the address space layout the architecture's page tables are
/// built with; see `arch::x86_64::paging::Paging::kernel_mapping` for that
/// layout on x86_64.
pub trait ReversePaging<PFA: PageFrameAllocator> {
    /// Physical address of the byte that `virt_addr` names in the direct map.
    ///
    /// # Panics
    ///
    /// If `virt_addr` lies outside the direct map.
    ///
    /// # Safety
    ///
    /// The result describes a location on the physical memory bus, so it is
    /// only meaningful for as long as the memory behind `virt_addr` is not
    /// handed to someone else. The caller must own that memory, or otherwise
    /// know it stays put.
    unsafe fn virt_to_phys<T>(virt_addr: VirtualAddress<T>) -> PhysicalAddress<T>;

    /// Virtual address at which `phys_addr` appears in the direct map.
    ///
    /// # Panics
    ///
    /// If `phys_addr` lies beyond the physical memory the direct map covers.
    ///
    /// # Safety
    ///
    /// The result is only dereferenceable while page tables carrying the
    /// direct map are active — the kernel's own, as built by
    /// `Paging::kernel_mapping`. Called on any others, or before they go live,
    /// it hands back a dangling pointer.
    ///
    /// The direct map spans all of physical memory, not just its usable parts,
    /// so a translated address being inside it says nothing about there being
    /// RAM behind it.
    unsafe fn phys_to_virt<T>(phys_addr: PhysicalAddress<T>) -> VirtualAddress<T>;
}

#[cfg(test)]
mod test {
    use super::*;

    extern crate std;

    /// A bitmask union of the rights given, and nothing else — the empty slice
    /// included.
    #[test]
    fn rights_are_the_union_of_what_is_asked_for() {
        let rights = AccessRights::new(&[AccessRight::Readable, AccessRight::Executable]);

        assert!(rights.is_readable());
        assert!(!rights.is_writable());
        assert!(rights.is_executable());

        assert_eq!(AccessRights::new(&[]), AccessRights::none());
    }

    /// Duplicates are ignored: setting a bit twice is setting it once.
    #[test]
    fn duplicate_rights_are_ignored() {
        let once = AccessRights::new(&[AccessRight::Writable]);
        let twice = AccessRights::new(&[AccessRight::Writable, AccessRight::Writable]);

        assert_eq!(once, twice);
    }

    /// Every right named individually is the same as [`AccessRights::full`].
    #[test]
    fn naming_every_right_is_full() {
        let rights = AccessRights::new(&[
            AccessRight::Readable,
            AccessRight::Writable,
            AccessRight::Executable,
        ]);

        assert_eq!(rights, AccessRights::full());
    }

    #[test]
    fn none_has_nothing_and_full_has_everything() {
        let none = AccessRights::none();

        assert!(!none.is_readable());
        assert!(!none.is_writable());
        assert!(!none.is_executable());

        let full = AccessRights::full();

        assert!(full.is_readable());
        assert!(full.is_writable());
        assert!(full.is_executable());
    }

    #[test]
    fn custom_sets_exactly_the_rights_it_is_given() {
        assert_eq!(
            AccessRights::custom(false, false, false),
            AccessRights::none()
        );
        assert_eq!(AccessRights::custom(true, true, true), AccessRights::full());

        let rw = AccessRights::custom(true, true, false);

        assert!(rw.is_readable());
        assert!(rw.is_writable());
        assert!(!rw.is_executable());
    }

    /// The setters clear as well as set, so a right can be taken back.
    #[test]
    fn a_right_can_be_set_and_cleared_again() {
        let mut rights = AccessRights::none();

        rights.set_writable(true);
        assert!(rights.is_writable());

        rights.set_writable(false);
        assert!(!rights.is_writable());
        assert_eq!(rights, AccessRights::none());
    }

    /// A caller with no reason to care gets the architecture's own answer,
    /// rather than a memory type this layer picked for it.
    #[test]
    fn the_default_caching_mode_is_the_architecture_s() {
        assert_eq!(CachingMode::default(), CachingMode::Normal);
    }

    #[test]
    fn caching_modes_format_as_their_names() {
        assert_eq!(std::format!("{}", CachingMode::Normal), "normal");
        assert_eq!(std::format!("{}", CachingMode::WriteBack), "write-back");
        assert_eq!(std::format!("{}", CachingMode::Uncached), "uncached");
        assert_eq!(
            std::format!("{}", CachingMode::WriteCombined),
            "write-combined"
        );
        assert_eq!(
            std::format!("{}", CachingMode::WriteThrough),
            "write-through"
        );
    }

    /// Formatted Unix-style, which is what the paging code prints.
    #[test]
    fn rights_format_as_a_permission_string() {
        assert_eq!(std::format!("{}", AccessRights::none()), "[---]");
        assert_eq!(std::format!("{}", AccessRights::full()), "[rwx]");
        assert_eq!(
            std::format!("{}", AccessRights::custom(true, false, true)),
            "[r-x]"
        );
        assert_eq!(
            std::format!("{}", AccessRights::custom(false, true, false)),
            "[-w-]"
        );
    }
}
