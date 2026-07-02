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
    /// ```rust
    /// let rights = AccessRights::new(&[AccessRight::Readable, AccessRight::Executable]);
    /// assert!(rights.is_readable());
    /// assert!(!rights.is_writable());
    /// assert!(rights.executable());
    /// ```
    pub const fn new(access_rights: &[AccessRight]) -> Self {
        let mut result = Self(0);

        let mut i = 0;
        loop {
            if i > access_rights.len() {
                break;
            }
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
pub struct VirtualAddress<T>(*mut T);

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
pub struct PhysicalAddress<T>(*mut T);

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

/// Allocates and frees physical page frames.
///
/// Implementors manage a pool of physical memory and hand out frames on demand.
/// All operations require a token at or above the `MemoryManagement` lock level,
/// enforcing that physical memory allocation always occurs under the correct
/// lock hierarchy.
///
/// The token is threaded through each call (consumed and returned) rather than
/// held by the allocator, keeping it compatible with the hierarchy's linear
/// token model.
pub trait PageFrameAllocator {
    /// Allocates a single physical page frame of the requested [`PageSize`].
    ///
    /// Returns the physical address of the allocated frame together with the
    /// token on success, or the error together with the token on failure so
    /// the caller can continue using the hierarchy.
    fn allocate<Token>(
        page_size: PageSize,
        token: Token,
    ) -> Result<(PhysicalAddress<c_void>, Token), (Error, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken;

    /// Frees a previously allocated physical page frame.
    ///
    /// # Safety
    ///
    /// - `phys_addr` must have been obtained from a previous call to
    ///   [`allocate`](PageFrameAllocator::allocate) with the same `page_size`.
    /// - The frame must not be referenced by any active page table entry.
    /// - Calling this twice for the same frame is undefined behaviour.
    unsafe fn deallocate<Token>(
        phys_addr: PhysicalAddress<c_void>,
        page_size: PageSize,
        token: Token,
    ) -> Token
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken;
}

/// Architecture-generic interface for managing a hardware page table.
///
/// Implementors provide the concrete page table manipulation for a specific
/// architecture (e.g. `arch::amd64::paging`). Generic kernel code
/// programs all memory mappings exclusively through this trait, keeping
/// architecture-specific page table structures behind the abstraction
/// boundary.
///
/// # Activation
///
/// A newly created page table is inactive — it has no effect on address
/// translation until [`active`](Paging::active) installs it in the hardware
/// register (CR3 on x86_64, `TTBR0`/`TTBR1` on AArch64).
///
/// # Safety
///
/// Several methods are `unsafe` because incorrect use can violate memory
/// safety globally: a bad mapping can corrupt kernel or user memory, cause
/// undefined behaviour on the next memory access, or crash the system.
pub trait Paging<PFA: PageFrameAllocator> {
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
    /// - `size` — page size (4 KiB / 2 MiB / 1 GiB).
    ///
    /// # Safety
    ///
    /// - `phys_addr` must refer to valid physical memory for the entire
    ///   lifetime of the mapping.
    /// - Mapping the wrong physical address can silently corrupt memory.
    /// - The caller must ensure no conflicting aliases are created (e.g.
    ///   mapping the same frame as both writable and executable in a
    ///   W^X policy).
    unsafe fn map<T, Token>(
        &mut self,
        virt_addr: VirtualAddress<T>,
        phys_addr: PhysicalAddress<T>,
        privilege_level: PrivilegeLevel,
        access_rights: AccessRights,
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
    /// Returns `(physical_address, privilege_level, access_rights, page_size)`
    /// for the mapping covering `virt_addr`, or an error if no mapping exists.
    fn resolve<T>(
        &self,
        virt_addr: VirtualAddress<T>,
    ) -> Result<(PhysicalAddress<T>, PrivilegeLevel, AccessRights, PageSize), Error>;

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
    unsafe fn active(&self);
}
