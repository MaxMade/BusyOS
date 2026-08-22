use core::ffi::c_void;
use core::fmt::Display;
use core::marker::PhantomData;

use bitfield_struct::bitfield;

use crate::{
    arch::generic::paging::{
        AccessRights, Error as PagingError, PageFrameAllocator, PageSize, Paging as _,
        PhysicalAddress, PrivilegeLevel, VirtualAddress,
    },
    kernel::{
        bootinfo::Bootinfo,
        locking::{CanAcquire, PreviousToken, level},
    },
};

/// Number of bits to shift a page frame number to obtain a physical address
/// for a regular (4 KiB) page.
pub const REGULAR_PAGE_SHIFT: usize = 12;

/// Number of bits to shift a page frame number to obtain a physical address
/// for a huge (2 MiB) page.
pub const HUGE_PAGE_SHIFT: usize = 9 + 12;

/// Number of bits to shift a page frame number to obtain a physical address
/// for a gigantic (1 GiB) page.
pub const GIGANTIC_PAGE_SHIFT: usize = 9 + 9 + 12;

/// Regular page size (4 KiB).
pub const REGULAR_PAGE_SIZE: usize = 4096;

/// Huge page size (2 MiB).
pub const HUGE_PAGE_SIZE: usize = ENTRIES_PER_TABLE * REGULAR_PAGE_SIZE;

/// Gigantic page size (1 GiB).
pub const GIGANTIC_PAGE_SIZE: usize = ENTRIES_PER_TABLE * ENTRIES_PER_TABLE * REGULAR_PAGE_SIZE;

/// Number of entries per page table.
const ENTRIES_PER_TABLE: usize = 512;

/// x86_64 `cr2` register.
///
/// Holds the linear address that caused the most recent page fault.
/// Read in a page-fault handler to determine which address was accessed.
#[derive(Debug)]
pub struct CR2(u64);

impl Display for CR2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x{:016x}", self.0)
    }
}

impl CR2 {
    /// Reads the current value of the `cr2` register.
    ///
    /// Returns the faulting linear address as a [`VirtualAddress`].
    pub fn read<T>() -> VirtualAddress<T> {
        let val: u64;
        unsafe {
            core::arch::asm!(
                "mov {}, cr2",
                out(reg) val,
                options(nomem, nostack, preserves_flags)
            );
        }
        VirtualAddress::new(val as _)
    }
}

/// x86_64 `cr3` register.
///
/// Points to the root of the active page table (PML4). Writing this register
/// implicitly flushes all non-global TLB entries.
#[bitfield(u64)]
pub struct CR3 {
    /// Ignored on most configurations (bits [2:0]).
    #[bits(3)]
    ignored_0: u8,

    /// Page-level Write-Through (PWT).
    ///
    /// Controls the caching write policy for the root page table (PML4) page.
    #[bits(1)]
    pub pwt: bool,

    /// Page-level Cache Disable (PCD).
    ///
    /// If 1, the root page table (PML4) page is not cached.
    #[bits(1)]
    pub pcd: bool,

    /// Ignored (bits [11:5]).
    #[bits(7)]
    ignored_1: u8,

    /// Physical address of the PML4 table (bits [51:12]).
    ///
    /// The address must be 4 KiB aligned; the lower 12 bits are always zero.
    #[bits(40)]
    pub addr: u64,

    /// Reserved (bits [63:52]). Must be zero.
    #[bits(12)]
    reserved: u16,
}

impl Display for CR3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "0x{:016x} (address: 0x{:016x})",
            self.0,
            self.addr() << REGULAR_PAGE_SHIFT as u64,
        )
    }
}

impl CR3 {
    /// Reads the current value of the `cr3` register.
    pub fn read() -> Self {
        let val: u64;
        unsafe {
            core::arch::asm!(
                "mov {}, cr3",
                out(reg) val,
                options(nomem, nostack, preserves_flags)
            );
        }
        Self(val)
    }

    /// Writes `self` to the `cr3` register, activating the page table.
    ///
    /// This operation implicitly flushes all non-global TLB entries.
    ///
    /// # Safety
    ///
    /// Activating an invalid or incomplete page table can violate memory safety
    /// globally: a bad mapping can corrupt kernel or user memory, cause undefined
    /// behaviour on the next memory access, or crash the system. The caller must
    /// ensure the page table covers all currently needed kernel addresses,
    /// including the instruction pointer and stack.
    pub unsafe fn write(self) {
        unsafe {
            core::arch::asm!(
                "mov cr3, {}",
                in(reg) self.0,
                options(nomem, nostack, preserves_flags)
            );
        }
    }

    /// Returns the physical address of the PML4 table stored in `cr3`.
    fn pml4(&self) -> PhysicalAddress<PML4> {
        let addr = self.addr() << REGULAR_PAGE_SHIFT;
        PhysicalAddress::new(addr as _)
    }

    /// Sets the physical address of the PML4 table in `cr3`.
    ///
    /// # Panics
    ///
    /// Panics if `addr` is not 4 KiB aligned.
    fn set_pml4(&mut self, addr: PhysicalAddress<PML4>) {
        assert!(
            addr.addr() % REGULAR_PAGE_SIZE == 0,
            "PML4 address must be 4 KiB aligned"
        );
        self.set_addr((addr.addr() >> REGULAR_PAGE_SHIFT) as _);
    }
}

/// The target of a page table entry — either the physical address of the
/// next-level table or the physical address of a leaf page.
enum PageTableTarget<PT> {
    /// Points to the next-level page table.
    PageTable(PhysicalAddress<PT>),
    /// Maps a leaf page directly (huge or gigantic).
    Page(PhysicalAddress<c_void>),
}

/// Common interface for all x86_64 page table entry types.
#[allow(unused)]
trait PageTableEntry {
    /// The type returned by [`target`](PageTableEntry::target): either a
    /// physical address of the next-level table or of a leaf page.
    type Target;

    /// Resets the entry to zero (not present).
    fn reset(&mut self);

    /// Returns `true` if the entry is present.
    fn is_present(&self) -> bool;

    /// Sets the present bit of the entry.
    fn set_present(&mut self, present: bool);

    /// Returns `true` if the region governed by this entry is writable.
    fn is_writable(&self) -> bool;

    /// Sets the writable bit of the entry.
    fn set_writable(&mut self, writable: bool);

    /// Returns `true` if userspace (ring 3) may access the region governed
    /// by this entry.
    fn is_user_accessible(&self) -> bool;

    /// Sets the user/supervisor bit of the entry.
    fn set_user_accessible(&mut self, user_accessible: bool);

    /// Returns `true` if code execution is forbidden from the region governed
    /// by this entry (NX bit set).
    fn is_no_execute(&self) -> bool;

    /// Sets the no-execute (NX) bit of the entry.
    fn set_no_execute(&mut self, no_execute: bool);

    /// Returns the address this entry points to, together with its kind
    /// (next-level table or leaf page).
    fn target(&self) -> Self::Target;

    /// Sets the target address (and kind) of this entry.
    fn set_target(&mut self, target: Self::Target);

    /// Returns the effective access rights encoded in this entry.
    ///
    /// On x86_64 a present entry is always readable, so the readable bit is
    /// always set. The writable and executable bits are derived from the
    /// writable and no-execute hardware bits respectively.
    fn access_rights(&self) -> AccessRights {
        let mut access_rights = AccessRights::none();
        access_rights.set_readable(true);

        if self.is_writable() {
            access_rights.set_writable(true);
        }

        if !self.is_no_execute() {
            access_rights.set_executable(true);
        }

        access_rights
    }

    /// Applies the given access rights to this entry.
    ///
    /// # Panics
    ///
    /// Panics if `access_right` is not readable: x86_64 paging cannot express
    /// a present-but-non-readable mapping.
    fn set_access_rights(&mut self, access_right: AccessRights) {
        if access_right.is_readable() {
            // Nothing to do — present implies readable on x86_64.
        } else {
            panic!("x86_64 paging does not support non-readable memory!");
        }

        self.set_writable(access_right.is_writable());
        self.set_no_execute(!access_right.is_executable());
    }

    /// Returns the privilege level required to access the region governed by
    /// this entry.
    fn privilege_level(&self) -> PrivilegeLevel {
        match self.is_user_accessible() {
            true => PrivilegeLevel::User,
            false => PrivilegeLevel::Kernel,
        }
    }

    /// Sets the privilege level required to access the region governed by
    /// this entry.
    fn set_privilege_level(&mut self, privilege_level: PrivilegeLevel) {
        match privilege_level {
            PrivilegeLevel::User => self.set_user_accessible(true),
            PrivilegeLevel::Kernel => self.set_user_accessible(false),
        }
    }
}

/// A 4 KiB-aligned array of 512 page table entries.
#[allow(unused)]
trait PageTable
where
    Self::PTE: PageTableEntry,
{
    /// The concrete page table entry type stored in this table.
    type PTE;

    /// Number of bits to shift a virtual address right to obtain the index
    /// into this table.
    const ADDRESS_SHIFT: usize;

    /// Returns a shared reference to all entries in the table.
    fn entries(&self) -> &[Self::PTE; ENTRIES_PER_TABLE];

    /// Returns a mutable reference to all entries in the table.
    fn entries_mut(&mut self) -> &mut [Self::PTE; ENTRIES_PER_TABLE];

    /// Resets every entry in the table to zero (not present).
    fn reset(&mut self) {
        for entry in self.entries_mut().iter_mut() {
            entry.reset();
        }
    }

    /// Returns `true` if every entry in the table is not present.
    ///
    /// Used to determine whether an intermediate table can be freed after a
    /// mapping is removed.
    fn is_unused(&self) -> bool {
        for entry in self.entries().iter() {
            if entry.is_present() {
                return false;
            }
        }

        true
    }

    /// Returns a shared reference to the entry that covers `virt_addr`.
    fn entry_for<T>(&self, virt_addr: VirtualAddress<T>) -> &Self::PTE {
        let index = (virt_addr.addr() >> Self::ADDRESS_SHIFT) % ENTRIES_PER_TABLE;
        &self.entries()[index]
    }

    /// Returns a mutable reference to the entry that covers `virt_addr`.
    fn entry_for_mut<T>(&mut self, virt_addr: VirtualAddress<T>) -> &mut Self::PTE {
        let index = (virt_addr.addr() >> Self::ADDRESS_SHIFT) % ENTRIES_PER_TABLE;
        &mut self.entries_mut()[index]
    }
}

/// An entry in the *P*age *M*ap *L*evel *4* (PML4) table.
///
/// Each entry covers 512 GiB of virtual address space and points to a
/// Page Directory Pointer (PDP) table.
#[bitfield(u64)]
pub struct PML4Entry {
    /// Present.
    ///
    /// If 0, all other bits are ignored and any access through this entry
    /// raises a page fault.
    #[bits(1)]
    pub present: bool,

    /// Read/Write.
    ///
    /// If 0, the entire 512 GiB region covered by this entry is read-only
    /// from all levels below.
    #[bits(1)]
    pub writable: bool,

    /// User/Supervisor.
    ///
    /// If 1, userspace (ring 3) may access this region.
    /// If 0, only the kernel (ring 0) may access it.
    #[bits(1)]
    pub user_accessible: bool,

    /// Page-level Write-Through (PWT).
    ///
    /// Controls the caching write policy for the pointed-to PDP page.
    #[bits(1)]
    pub pwt: bool,

    /// Page-level Cache Disable (PCD).
    ///
    /// If 1, the pointed-to PDP page is not cached.
    #[bits(1)]
    pub pcd: bool,

    /// Accessed.
    ///
    /// Set by the CPU when this entry is used during address translation.
    /// Never cleared by hardware; software must clear it manually.
    #[bits(1)]
    pub accessed: bool,

    /// Ignored by hardware (bit 6).
    #[bits(1)]
    ignored_0: u8,

    /// Must be 0 (bit 7).
    #[bits(1)]
    zero: u8,

    /// Available for OS use (bits [11:8]).
    #[bits(4)]
    pub available_0: u8,

    /// Physical address of the PDP table (bits [51:12]).
    ///
    /// The address must be 4 KiB aligned.
    #[bits(40)]
    pub addr: u64,

    /// Available for OS use (bits [62:52]).
    #[bits(11)]
    pub available_1: u16,

    /// No-Execute (NX).
    ///
    /// If 1, code cannot be executed from any page reachable through this
    /// entry. Requires `EFER.NXE = 1`.
    #[bits(1)]
    pub no_execute: bool,
}

impl Display for PML4Entry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.present() {
            true => write!(
                f,
                "0x{:016x} (present: {}, writable: {}, user-accessible: {}, \
                 accessed: {}, available: 0b{:04b}, address: 0x{:016x}, \
                 available: 0b{:011b}, non-executable: {})",
                self.0,
                true,
                self.writable(),
                self.user_accessible(),
                self.accessed(),
                self.available_0(),
                self.addr() << REGULAR_PAGE_SHIFT as u64,
                self.available_1(),
                self.no_execute(),
            ),
            false => write!(f, "0x{:016x} (present: {})", self.0, false),
        }
    }
}

impl PageTableEntry for PML4Entry {
    /// PML4 entries always point to a PDP table — no leaf pages at this level.
    type Target = PhysicalAddress<PDP>;

    fn reset(&mut self) {
        self.0 = 0;
    }

    fn is_present(&self) -> bool {
        self.present()
    }
    fn is_writable(&self) -> bool {
        self.writable()
    }
    fn is_user_accessible(&self) -> bool {
        self.user_accessible()
    }
    fn is_no_execute(&self) -> bool {
        self.no_execute()
    }

    fn set_present(&mut self, val: bool) {
        PML4Entry::set_present(self, val);
    }

    fn set_writable(&mut self, val: bool) {
        PML4Entry::set_writable(self, val);
    }

    fn set_user_accessible(&mut self, val: bool) {
        PML4Entry::set_user_accessible(self, val);
    }

    fn set_no_execute(&mut self, val: bool) {
        PML4Entry::set_no_execute(self, val);
    }

    fn target(&self) -> PhysicalAddress<PDP> {
        let addr = self.addr() << REGULAR_PAGE_SHIFT;
        PhysicalAddress::new(addr as _)
    }

    fn set_target(&mut self, target: PhysicalAddress<PDP>) {
        assert!(
            target.addr() % REGULAR_PAGE_SIZE == 0,
            "PDP address must be 4 KiB aligned"
        );
        PML4Entry::set_addr(self, (target.addr() >> REGULAR_PAGE_SHIFT) as _);
    }
}

/// The Page Map Level 4 (PML4) table — the root of the four-level hierarchy.
#[derive(Debug)]
#[repr(C, align(4096))]
struct PML4 {
    entries: [PML4Entry; ENTRIES_PER_TABLE],
}

impl PageTable for PML4 {
    type PTE = PML4Entry;

    const ADDRESS_SHIFT: usize = 12 + 9 + 9 + 9;

    fn entries(&self) -> &[Self::PTE; ENTRIES_PER_TABLE] {
        &self.entries
    }

    fn entries_mut(&mut self) -> &mut [Self::PTE; ENTRIES_PER_TABLE] {
        &mut self.entries
    }
}

/// An entry in the *P*age *D*irectory *P*ointer (PDP) table.
///
/// Each entry covers 1 GiB of virtual address space. When
/// [`gigantic_page`](PDPEntry::gigantic_page) is set, the entry maps a 1 GiB
/// page directly; otherwise it points to a Page Directory (PD) table.
#[bitfield(u64)]
pub struct PDPEntry {
    /// Present.
    ///
    /// If 0, all other bits are ignored and any access through this entry
    /// raises a page fault.
    #[bits(1)]
    pub present: bool,

    /// Read/Write.
    ///
    /// If 0, the entire 1 GiB region is read-only from all levels below.
    #[bits(1)]
    pub writable: bool,

    /// User/Supervisor.
    ///
    /// If 1, userspace (ring 3) may access this region.
    /// If 0, only the kernel (ring 0) may access it.
    #[bits(1)]
    pub user_accessible: bool,

    /// Page-level Write-Through (PWT).
    ///
    /// Controls the caching write policy for the pointed-to PD page or
    /// 1 GiB page. See [`gigantic_page`](PDPEntry::gigantic_page).
    #[bits(1)]
    pub pwt: bool,

    /// Page-level Cache Disable (PCD).
    ///
    /// If 1, the pointed-to PD page or 1 GiB page is not cached.
    /// See [`gigantic_page`](PDPEntry::gigantic_page).
    #[bits(1)]
    pub pcd: bool,

    /// Accessed.
    ///
    /// Set by the CPU when this entry is used during address translation.
    /// Never cleared by hardware; software must clear it manually.
    #[bits(1)]
    pub accessed: bool,

    /// Dirty.
    ///
    /// Valid only when [`gigantic_page`](PDPEntry::gigantic_page) = 1.
    /// Set by the CPU on the first write to the 1 GiB page.
    #[bits(1)]
    pub dirty: bool,

    /// Gigantic page.
    ///
    /// If 0, this entry points to a PD table.
    /// If 1, this entry directly maps a 1 GiB page.
    ///
    /// Requires `pdpe1gb` CPU support — verify with:
    /// ```bash
    /// grep pdpe1gb /proc/cpuinfo
    /// ```
    #[bits(1)]
    pub gigantic_page: bool,

    /// Global.
    ///
    /// Valid only when [`gigantic_page`](PDPEntry::gigantic_page) = 1.
    /// If 1, the TLB entry is not flushed on [`CR3`] reload.
    /// Requires `CR4.PGE = 1`.
    #[bits(1)]
    pub global: bool,

    /// Available for OS use (bits [11:9]).
    #[bits(3)]
    pub available_0: u8,

    /// Physical address field (bits [51:12]).
    ///
    /// - [`gigantic_page`](PDPEntry::gigantic_page) = 0: bits [51:12] of the
    ///   PD table physical address (4 KiB aligned).
    /// - [`gigantic_page`](PDPEntry::gigantic_page) = 1: bit 12 = PAT;
    ///   bits [20:13] = MBZ; bits [51:30] = 1 GiB page address (1 GiB aligned).
    #[bits(40)]
    pub addr: u64,

    /// Available for OS use (bits [62:52]).
    #[bits(11)]
    pub available_1: u16,

    /// No-Execute (NX).
    ///
    /// If 1, code cannot be executed from any page reachable through this
    /// entry. Requires `EFER.NXE = 1`.
    #[bits(1)]
    pub no_execute: bool,
}

impl Display for PDPEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.present() {
            true => match self.gigantic_page() {
                true => write!(
                    f,
                    "0x{:016x} (present: {}, writable: {}, user-accessible: {}, \
                     accessed: {}, dirty: {}, global: {}, available: 0b{:03b}, \
                     address: 0x{:016x}, available: 0b{:011b}, non-executable: {})",
                    self.0,
                    true,
                    self.writable(),
                    self.user_accessible(),
                    self.accessed(),
                    self.dirty(),
                    self.global(),
                    self.available_0(),
                    self.addr() << GIGANTIC_PAGE_SHIFT as u64,
                    self.available_1(),
                    self.no_execute(),
                ),
                false => write!(
                    f,
                    "0x{:016x} (present: {}, writable: {}, user-accessible: {}, \
                     accessed: {}, available: 0b{:03b}, address: 0x{:016x}, \
                     available: 0b{:011b}, non-executable: {})",
                    self.0,
                    true,
                    self.writable(),
                    self.user_accessible(),
                    self.accessed(),
                    self.available_0(),
                    self.addr() << REGULAR_PAGE_SHIFT as u64,
                    self.available_1(),
                    self.no_execute(),
                ),
            },
            false => write!(f, "0x{:016x} (present: {})", self.0, false),
        }
    }
}

impl PageTableEntry for PDPEntry {
    type Target = PageTableTarget<PD>;

    fn reset(&mut self) {
        self.0 = 0;
    }

    fn is_present(&self) -> bool {
        self.present()
    }
    fn is_writable(&self) -> bool {
        self.writable()
    }
    fn is_user_accessible(&self) -> bool {
        self.user_accessible()
    }
    fn is_no_execute(&self) -> bool {
        self.no_execute()
    }

    fn set_present(&mut self, val: bool) {
        PDPEntry::set_present(self, val);
    }

    fn set_writable(&mut self, val: bool) {
        PDPEntry::set_writable(self, val);
    }

    fn set_user_accessible(&mut self, val: bool) {
        PDPEntry::set_user_accessible(self, val);
    }

    fn set_no_execute(&mut self, val: bool) {
        PDPEntry::set_no_execute(self, val);
    }

    fn target(&self) -> PageTableTarget<PD> {
        let addr = (self.addr() << REGULAR_PAGE_SHIFT) as u64;

        match self.gigantic_page() {
            false => PageTableTarget::PageTable(PhysicalAddress::new(addr as _)),
            true => {
                let aligned = addr & !(GIGANTIC_PAGE_SIZE as u64 - 1);
                PageTableTarget::Page(PhysicalAddress::new(aligned as _))
            }
        }
    }

    fn set_target(&mut self, target: PageTableTarget<PD>) {
        match target {
            PageTableTarget::PageTable(phys_addr) => {
                assert!(
                    phys_addr.addr() % REGULAR_PAGE_SIZE == 0,
                    "PD address must be 4 KiB aligned"
                );
                PDPEntry::set_gigantic_page(self, false);
                PDPEntry::set_addr(self, (phys_addr.addr() >> REGULAR_PAGE_SHIFT) as _);
            }
            PageTableTarget::Page(phys_addr) => {
                assert!(
                    phys_addr.addr() % GIGANTIC_PAGE_SIZE == 0,
                    "gigantic page address must be 1 GiB aligned"
                );
                PDPEntry::set_gigantic_page(self, true);
                // TODO(@MaxMade): PAT for gigantic pages not yet handled.
                PDPEntry::set_addr(self, (phys_addr.addr() >> REGULAR_PAGE_SHIFT) as _);
            }
        }
    }
}

/// The Page Directory Pointer (PDP) table.
#[derive(Debug)]
#[repr(C, align(4096))]
struct PDP {
    entries: [PDPEntry; ENTRIES_PER_TABLE],
}

impl PageTable for PDP {
    type PTE = PDPEntry;

    const ADDRESS_SHIFT: usize = 12 + 9 + 9;

    fn entries(&self) -> &[PDPEntry; ENTRIES_PER_TABLE] {
        &self.entries
    }

    fn entries_mut(&mut self) -> &mut [PDPEntry; ENTRIES_PER_TABLE] {
        &mut self.entries
    }
}

/// An entry in the *P*age *D*irectory (PD) table.
///
/// Each entry covers 2 MiB of virtual address space. When
/// [`huge_page`](PDEntry::huge_page) is set, the entry maps a 2 MiB page
/// directly; otherwise it points to a Page Table (PT).
#[bitfield(u64)]
pub struct PDEntry {
    /// Present.
    ///
    /// If 0, all other bits are ignored and any access through this entry
    /// raises a page fault.
    #[bits(1)]
    pub present: bool,

    /// Read/Write.
    ///
    /// If 0, the entire 2 MiB region is read-only from all levels below.
    #[bits(1)]
    pub writable: bool,

    /// User/Supervisor.
    ///
    /// If 1, userspace (ring 3) may access this region.
    /// If 0, only the kernel (ring 0) may access it.
    #[bits(1)]
    pub user_accessible: bool,

    /// Page-level Write-Through (PWT).
    ///
    /// Controls the caching write policy for the pointed-to PT page or
    /// 2 MiB page. See [`huge_page`](PDEntry::huge_page).
    #[bits(1)]
    pub pwt: bool,

    /// Page-level Cache Disable (PCD).
    ///
    /// If 1, the pointed-to PT page or 2 MiB page is not cached.
    /// See [`huge_page`](PDEntry::huge_page).
    #[bits(1)]
    pub pcd: bool,

    /// Accessed.
    ///
    /// Set by the CPU when this entry is used during address translation.
    /// Never cleared by hardware; software must clear it manually.
    #[bits(1)]
    pub accessed: bool,

    /// Dirty.
    ///
    /// Valid only when [`huge_page`](PDEntry::huge_page) = 1.
    /// Set by the CPU on the first write to the 2 MiB page.
    #[bits(1)]
    pub dirty: bool,

    /// Huge page.
    ///
    /// If 0, this entry points to a PT table.
    /// If 1, this entry directly maps a 2 MiB page.
    #[bits(1)]
    pub huge_page: bool,

    /// Global.
    ///
    /// Valid only when [`huge_page`](PDEntry::huge_page) = 1.
    /// If 1, the TLB entry is not flushed on [`CR3`] reload.
    /// Requires `CR4.PGE = 1`.
    #[bits(1)]
    pub global: bool,

    /// Available for OS use (bits [11:9]).
    #[bits(3)]
    pub available_0: u8,

    /// Physical address field (bits [51:12]).
    ///
    /// - [`huge_page`](PDEntry::huge_page) = 0: bits [51:12] of the PT
    ///   physical address (4 KiB aligned).
    /// - [`huge_page`](PDEntry::huge_page) = 1: bit 12 = PAT;
    ///   bits [20:13] = MBZ; bits [51:21] = 2 MiB page address (2 MiB aligned).
    #[bits(40)]
    pub addr: u64,

    /// Available for OS use (bits [62:52]).
    #[bits(11)]
    pub available_1: u16,

    /// No-Execute (NX).
    ///
    /// If 1, code cannot be executed from any page reachable through this
    /// entry. Requires `EFER.NXE = 1`.
    #[bits(1)]
    pub no_execute: bool,
}

impl Display for PDEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.present() {
            true => match self.huge_page() {
                true => write!(
                    f,
                    "0x{:016x} (present: {}, writable: {}, user-accessible: {}, \
                     accessed: {}, dirty: {}, global: {}, available: 0b{:03b}, \
                     address: 0x{:016x}, available: 0b{:011b}, non-executable: {})",
                    self.0,
                    true,
                    self.writable(),
                    self.user_accessible(),
                    self.accessed(),
                    self.dirty(),
                    self.global(),
                    self.available_0(),
                    self.addr() << HUGE_PAGE_SHIFT as u64,
                    self.available_1(),
                    self.no_execute(),
                ),
                false => write!(
                    f,
                    "0x{:016x} (present: {}, writable: {}, user-accessible: {}, \
                     accessed: {}, available: 0b{:03b}, address: 0x{:016x}, \
                     available: 0b{:011b}, non-executable: {})",
                    self.0,
                    true,
                    self.writable(),
                    self.user_accessible(),
                    self.accessed(),
                    self.available_0(),
                    self.addr() << REGULAR_PAGE_SHIFT as u64,
                    self.available_1(),
                    self.no_execute(),
                ),
            },
            false => write!(f, "0x{:016x} (present: {})", self.0, false),
        }
    }
}

impl PageTableEntry for PDEntry {
    type Target = PageTableTarget<PT>;

    fn reset(&mut self) {
        self.0 = 0;
    }

    fn is_present(&self) -> bool {
        self.present()
    }
    fn is_writable(&self) -> bool {
        self.writable()
    }
    fn is_user_accessible(&self) -> bool {
        self.user_accessible()
    }
    fn is_no_execute(&self) -> bool {
        self.no_execute()
    }

    fn set_present(&mut self, val: bool) {
        PDEntry::set_present(self, val);
    }

    fn set_writable(&mut self, val: bool) {
        PDEntry::set_writable(self, val);
    }

    fn set_user_accessible(&mut self, val: bool) {
        PDEntry::set_user_accessible(self, val);
    }

    fn set_no_execute(&mut self, val: bool) {
        PDEntry::set_no_execute(self, val);
    }

    fn target(&self) -> PageTableTarget<PT> {
        let addr = (self.addr() << REGULAR_PAGE_SHIFT) as u64;

        match self.huge_page() {
            false => PageTableTarget::PageTable(PhysicalAddress::new(addr as _)),
            true => {
                let aligned = addr & !(HUGE_PAGE_SIZE as u64 - 1);
                PageTableTarget::Page(PhysicalAddress::new(aligned as _))
            }
        }
    }

    fn set_target(&mut self, target: PageTableTarget<PT>) {
        match target {
            PageTableTarget::PageTable(phys_addr) => {
                assert!(
                    phys_addr.addr() % REGULAR_PAGE_SIZE == 0,
                    "PT address must be 4 KiB aligned"
                );
                PDEntry::set_huge_page(self, false);
                PDEntry::set_addr(self, (phys_addr.addr() >> REGULAR_PAGE_SHIFT) as _);
            }
            PageTableTarget::Page(phys_addr) => {
                assert!(
                    phys_addr.addr() % HUGE_PAGE_SIZE == 0,
                    "huge page address must be 2 MiB aligned"
                );
                PDEntry::set_huge_page(self, true);
                // TODO(@MaxMade): PAT for huge pages not yet handled.
                PDEntry::set_addr(self, (phys_addr.addr() >> REGULAR_PAGE_SHIFT) as _);
            }
        }
    }
}

/// The Page Directory (PD) table.
#[derive(Debug)]
#[repr(C, align(4096))]
struct PD {
    entries: [PDEntry; ENTRIES_PER_TABLE],
}

impl PageTable for PD {
    type PTE = PDEntry;

    const ADDRESS_SHIFT: usize = 12 + 9;

    fn entries(&self) -> &[PDEntry; ENTRIES_PER_TABLE] {
        &self.entries
    }

    fn entries_mut(&mut self) -> &mut [PDEntry; ENTRIES_PER_TABLE] {
        &mut self.entries
    }
}

/// An entry in the *P*age *T*able (PT).
///
/// Each entry maps one 4 KiB physical page.
#[bitfield(u64)]
pub struct PTEntry {
    /// Present.
    ///
    /// If 0, all other bits are ignored and any access raises a page fault.
    #[bits(1)]
    pub present: bool,

    /// Read/Write.
    ///
    /// If 0, the 4 KiB page is read-only.
    #[bits(1)]
    pub writable: bool,

    /// User/Supervisor.
    ///
    /// If 1, userspace (ring 3) may access this page.
    /// If 0, only the kernel (ring 0) may access it.
    #[bits(1)]
    pub user_accessible: bool,

    /// Page-level Write-Through (PWT).
    ///
    /// Controls the caching write policy for this 4 KiB page.
    #[bits(1)]
    pub pwt: bool,

    /// Page-level Cache Disable (PCD).
    ///
    /// If 1, this 4 KiB page is not cached.
    #[bits(1)]
    pub pcd: bool,

    /// Accessed.
    ///
    /// Set by the CPU on the first read or write to the page.
    /// Never cleared by hardware; used by the OS for working-set tracking.
    #[bits(1)]
    pub accessed: bool,

    /// Dirty.
    ///
    /// Set by the CPU on the first write to the page.
    /// Never cleared by hardware; used by the OS to detect modified pages
    /// that need writeback before reclaim.
    #[bits(1)]
    pub dirty: bool,

    /// Page Attribute Table index bit 2 (PAT, bit 7).
    ///
    /// Together with [`pwt`](PTEntry::pwt) (PAT bit 0) and
    /// [`pcd`](PTEntry::pcd) (PAT bit 1), selects one of eight memory-type
    /// entries from the `IA32_PAT` MSR.
    #[bits(1)]
    pub pat: bool,

    /// Global.
    ///
    /// If 1, the TLB entry for this page is not flushed on [`CR3`] reload.
    /// Typically set for kernel pages mapped in all address spaces.
    /// Requires `CR4.PGE = 1`.
    #[bits(1)]
    pub global: bool,

    /// Available for OS use (bits [11:9]).
    #[bits(3)]
    pub available_0: u8,

    /// Physical address of the 4 KiB page (bits [51:12]).
    ///
    /// The page must be 4 KiB aligned; bits [11:0] are not stored.
    #[bits(40)]
    pub addr: u64,

    /// Available for OS use (bits [58:52]).
    #[bits(7)]
    pub available_1: u8,

    /// Protection Key (bits [62:59]).
    ///
    /// Selects one of 16 protection-key domains when `CR4.PKE = 1`.
    /// Ignored when `CR4.PKE = 0`.
    #[bits(4)]
    pub protection_key: u8,

    /// No-Execute (NX).
    ///
    /// If 1, code cannot be executed from this page.
    /// Requires `EFER.NXE = 1`.
    #[bits(1)]
    pub no_execute: bool,
}

impl Display for PTEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.present() {
            true => write!(
                f,
                "0x{:016x} (present: {}, writable: {}, user-accessible: {}, \
                 accessed: {}, dirty: {}, global: {}, available: 0b{:03b}, \
                 address: 0x{:016x}, available: 0b{:07b}, non-executable: {})",
                self.0,
                true,
                self.writable(),
                self.user_accessible(),
                self.accessed(),
                self.dirty(),
                self.global(),
                self.available_0(),
                self.addr() << REGULAR_PAGE_SHIFT as u64,
                self.available_1(),
                self.no_execute(),
            ),
            false => write!(f, "0x{:016x} (present: {})", self.0, false),
        }
    }
}

impl PageTableEntry for PTEntry {
    /// PT entries always map a 4 KiB leaf page.
    type Target = PhysicalAddress<c_void>;

    fn reset(&mut self) {
        self.0 = 0;
    }

    fn is_present(&self) -> bool {
        self.present()
    }
    fn is_writable(&self) -> bool {
        self.writable()
    }
    fn is_user_accessible(&self) -> bool {
        self.user_accessible()
    }
    fn is_no_execute(&self) -> bool {
        self.no_execute()
    }

    fn set_present(&mut self, val: bool) {
        PTEntry::set_present(self, val);
    }

    fn set_writable(&mut self, val: bool) {
        PTEntry::set_writable(self, val);
    }

    fn set_user_accessible(&mut self, val: bool) {
        PTEntry::set_user_accessible(self, val);
    }

    fn set_no_execute(&mut self, val: bool) {
        PTEntry::set_no_execute(self, val);
    }

    fn target(&self) -> PhysicalAddress<c_void> {
        let addr = self.addr() << REGULAR_PAGE_SHIFT;
        PhysicalAddress::new(addr as _)
    }

    fn set_target(&mut self, target: PhysicalAddress<c_void>) {
        assert!(
            target.addr() % REGULAR_PAGE_SIZE == 0,
            "page address must be 4 KiB aligned"
        );
        PTEntry::set_addr(self, (target.addr() >> REGULAR_PAGE_SHIFT) as _);
    }
}

/// The Page Table (PT) — the leaf level of the four-level hierarchy.
#[derive(Debug)]
#[repr(C, align(4096))]
struct PT {
    entries: [PTEntry; ENTRIES_PER_TABLE],
}

impl PageTable for PT {
    type PTE = PTEntry;

    const ADDRESS_SHIFT: usize = 12;

    fn entries(&self) -> &[PTEntry; ENTRIES_PER_TABLE] {
        &self.entries
    }

    fn entries_mut(&mut self) -> &mut [PTEntry; ENTRIES_PER_TABLE] {
        &mut self.entries
    }
}

/// x86_64 four-level page table implementation.
///
/// Physical frame 0 is reserved as the "no PML4 yet" sentinel: a zeroed
/// `cr3` is taken to mean that no root table has been allocated yet. The
/// frame allocator must therefore never return physical frame 0.
#[derive(Debug)]
pub struct Paging<PFA: PageFrameAllocator> {
    /// The `cr3` register value describing the root (PML4) of this hierarchy.
    cr3: CR3,
    /// Binds the frame allocator type without storing a value.
    phantom: PhantomData<PFA>,
    /// Offset added to a page table's physical address to obtain the
    /// corresponding kernel virtual address (physical-to-virtual offset).
    page_table_shift: usize,
}

impl<PFA: PageFrameAllocator> Paging<PFA> {
    /// Creates temporary kernel page tables for jumping to upper half of
    /// address space.
    ///
    /// Identity-maps the first 512 GiB of physical memory, and maps the 511
    /// GiB starting at 1 GiB (`0x4000_0000`..`0x80_0000_0000`) to the
    /// higher-half base at `0xFFFF_8000_0000_0000`, using 1 GiB gigantic
    /// pages throughout.
    ///
    /// The higher-half window starts at 1 GiB because the bootloader pins the
    /// kernel image there, so `KERNEL_VMA_START` resolves to the image's
    /// physical location and a section's link address is also the address it
    /// runs at. Intended only as a trampoline for the jump to the higher
    /// half.
    ///
    /// # Safety
    ///
    /// The caller must ensure that every allocated page frame using
    /// `PFA::allocate` is currently identify mapped.
    pub unsafe fn temporary_upper_half<Token>(
        token: Token,
    ) -> Result<(Self, Token), (PagingError, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken,
    {
        let mut token = token;

        let mut page_tables = Paging {
            cr3: CR3::new(),
            phantom: PhantomData,
            page_table_shift: 0x0,
        };

        for i in 0..ENTRIES_PER_TABLE as u64 {
            let step = GIGANTIC_PAGE_SIZE as u64;
            let phys_addr = PhysicalAddress::new((i * step) as *mut c_void);
            let virt_addr_lower = VirtualAddress::new((i * step) as *mut c_void);

            token = match unsafe {
                page_tables.map(
                    virt_addr_lower,
                    phys_addr,
                    PrivilegeLevel::Kernel,
                    AccessRights::full(),
                    PageSize::Gigantic,
                    token,
                )
            } {
                Ok((previous, token)) => {
                    assert!(previous.is_none());
                    token
                }
                Err((error, mut token)) => {
                    token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                    return Err((error, token));
                }
            };
        }

        for i in 0..ENTRIES_PER_TABLE as u64 - 1 {
            let step = GIGANTIC_PAGE_SIZE as u64;
            let phys_addr = PhysicalAddress::new((1024 * 1024 * 1024 + i * step) as *mut c_void);
            let virt_addr_upper =
                VirtualAddress::new((0xffff_8000_0000_0000u64 + i * step) as *mut c_void);

            token = match unsafe {
                page_tables.map(
                    virt_addr_upper,
                    phys_addr,
                    PrivilegeLevel::Kernel,
                    AccessRights::full(),
                    PageSize::Gigantic,
                    token,
                )
            } {
                Ok((previous, token)) => {
                    assert!(previous.is_none());
                    token
                }
                Err((error, mut token)) => {
                    token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                    return Err((error, token));
                }
            };
        }

        Ok((page_tables, token))
    }

    /// Creates the initial kernel page tables.
    ///
    /// # Address Space Layout
    ///
    /// ```text
    /// ┌─────────────────────────────────────────────────────────┐
    /// │ 0x0000_0000_0000_0000                                   │
    /// │                                                         │
    /// │                                     Userspace (128 TiB) │
    /// │                               64 TiB canonical / usable │
    /// │                                                         │
    /// │ 0x0000_7FFF_FFFF_FFFF                                   │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    /// │ 0x0000_8000_0000_0000                                   │
    /// │                                                         │
    /// │                            Non-canonical hole (~16 EiB) │
    /// │                    bits [63:48] must sign-extend bit 47 │
    /// │                                                         │
    /// │ 0xFFFF_7FFF_FFFF_FFFF                                   │
    /// ├─────────────────────────────────────────────────────────┤
    /// │ 0xFFFF_8000_0000_0000       ┐                           │
    /// │                             │                           │
    /// │  ┌───────────────────────┐  │                           │
    /// │  │ 0xFFFF_8000_0000_0000 │  │                           │
    /// │  │                       │  │     Kernelspace (128 TiB) │
    /// │  │   Kernel image        │  │                           │
    /// │  │   .text / .rodata     │  │                           │
    /// │  │   .data  / .bss       │  │                           │
    /// │  │   stack  / heap       │  │                           │
    /// │  │                       │  │                           │
    /// │  │        64 TiB         │  │                           │
    /// │  │                       │  │                           │
    /// │  │ 0xFFFF_BFFF_FFFF_FFFF │  │                           │
    /// │  ├───────────────────────┤  │                           │
    /// │  │ 0xFFFF_C000_0000_0000 │  │                           │
    /// │  │                       │  │                           │
    /// │  │  Physical memory map  │  │                           │
    /// │  │  (direct map of all   │  │                           │
    /// │  │   physical RAM)       │  │                           │
    /// │  │                       │  │                           │
    /// │  │        64 TiB         │  │                           │
    /// │  │                       │  │                           │
    /// │  │ 0xFFFF_FFFF_FFFF_FFFF │  │                           │
    /// │  └───────────────────────┘  │                           │
    /// │                             │                           │
    /// │ 0xFFFF_FFFF_FFFF_FFFF       ┘                           │
    /// └─────────────────────────────────────────────────────────┘
    /// ```
    pub fn kernel_mapping<Token>(
        bootinfo: &Bootinfo,
        token: Token,
    ) -> Result<(Self, Token), (PagingError, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken,
    {
        // Create empty page tables
        let mut page_tables = Self {
            cr3: CR3::new(),
            phantom: PhantomData,
            page_table_shift: 0xffff_c000_0000_0000,
        };

        // Declare helper function
        let try_map = |page_tables: &mut Self,
                       virt_addr: VirtualAddress<c_void>,
                       phys_addr: PhysicalAddress<c_void>,
                       len: usize,
                       access_rights: AccessRights,
                       mut token: Token|
         -> Result<Token, (PagingError, Token)> {
            // Sanity check:
            if (virt_addr.addr() % REGULAR_PAGE_SIZE) != 0 {
                panic!("Input virtual address must be aligned!");
            }
            if (phys_addr.addr() % REGULAR_PAGE_SIZE) != 0 {
                panic!("Input physical address must be aligned!");
            }
            if (len % REGULAR_PAGE_SIZE) != 0 {
                panic!("Mapped range must be a multiple of the page size!");
            }

            let mut offset = 0;
            'outer: while offset < len {
                let remaining = len - offset;

                // Try different page size
                for (size, page_size) in [
                    (GIGANTIC_PAGE_SIZE, PageSize::Gigantic),
                    (HUGE_PAGE_SIZE, PageSize::Huge),
                    (REGULAR_PAGE_SIZE, PageSize::Regular),
                ] {
                    if (virt_addr.addr() + offset) % size == 0
                        && (phys_addr.addr() + offset) % size == 0
                        && remaining >= size
                    {
                        let previous;
                        (previous, token) = unsafe {
                            page_tables.map(
                                virt_addr.byte_add(offset),
                                phys_addr.byte_add(offset),
                                PrivilegeLevel::Kernel,
                                access_rights,
                                page_size,
                                token,
                            )?
                        };
                        assert!(previous.is_none());

                        offset += size;
                        continue 'outer;
                    }
                }
                unreachable!();
            }

            Ok(token)
        };

        // Create physical memory map
        let token = match try_map(
            &mut page_tables,
            VirtualAddress::new(0xFFFF_C000_0000_0000u64 as _),
            PhysicalAddress::new(0x0000_0000_0000_0000u64 as _),
            64usize * 1024 * 1024 * 1024 * 1024,
            AccessRights::custom(true, true, false),
            token,
        ) {
            Ok(token) => token,
            Err((error, mut token)) => {
                // Free page tables
                token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                return Err((error, token));
            }
        };

        // Map .text segment
        let token = match try_map(
            &mut page_tables,
            VirtualAddress::new(
                (bootinfo.kernel_text_start.addr() + bootinfo.kernel_virt_phys_offset) as _,
            ),
            bootinfo.kernel_text_start,
            bootinfo.kernel_text_size,
            AccessRights::custom(true, false, true),
            token,
        ) {
            Ok(token) => token,
            Err((error, mut token)) => {
                // Free page tables
                token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                return Err((error, token));
            }
        };

        // Map .rodata segment
        let token = match try_map(
            &mut page_tables,
            VirtualAddress::new(
                (bootinfo.kernel_rodata_start.addr() + bootinfo.kernel_virt_phys_offset) as _,
            ),
            bootinfo.kernel_rodata_start,
            bootinfo.kernel_rodata_size,
            AccessRights::custom(true, false, false),
            token,
        ) {
            Ok(token) => token,
            Err((error, mut token)) => {
                // Free page tables
                token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                return Err((error, token));
            }
        };

        // Map .data segment
        let token = match try_map(
            &mut page_tables,
            VirtualAddress::new(
                (bootinfo.kernel_data_start.addr() + bootinfo.kernel_virt_phys_offset) as _,
            ),
            bootinfo.kernel_data_start,
            bootinfo.kernel_data_size,
            AccessRights::custom(true, true, false),
            token,
        ) {
            Ok(token) => token,
            Err((error, mut token)) => {
                // Free page tables
                token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                return Err((error, token));
            }
        };

        // Map .bss segment
        let token = match try_map(
            &mut page_tables,
            VirtualAddress::new(
                (bootinfo.kernel_bss_start.addr() + bootinfo.kernel_virt_phys_offset) as _,
            ),
            bootinfo.kernel_bss_start,
            bootinfo.kernel_bss_size,
            AccessRights::custom(true, true, false),
            token,
        ) {
            Ok(token) => token,
            Err((error, mut token)) => {
                // Free page tables
                token = unsafe { page_tables.destroy(|_, _, token| token, token) };
                return Err((error, token));
            }
        };

        Ok((page_tables, token))
    }

    /// Checkes if a virtual address is canonical.
    ///
    /// For x86_64, the most significant 16 bits of any virtual address, bits 48
    /// through 63, must be copies of bit 47. Otherwise, any access will raise
    /// an exception.
    #[inline]
    pub fn is_canonical<T>(virt_addr: VirtualAddress<T>) -> bool {
        if virt_addr.addr() <= 0x00007fffffffffff {
            return true;
        }

        if virt_addr.addr() >= 0xffff800000000000 {
            return true;
        }

        false
    }

    /// Creates an uninitialized set of page tables.
    ///
    /// The page table is inactive until a PML4 is allocated (on the first
    /// call to [`map`](Paging::map)) and activated via [`Paging::active`].
    pub const fn new(page_table_shift: usize) -> Self {
        Self {
            cr3: CR3::new(),
            phantom: PhantomData,
            page_table_shift,
        }
    }

    /// Converts the physical address of a page table to a virtual address by
    /// applying the stored physical-to-virtual offset.
    ///
    /// # Panics
    ///
    /// Panics if `phys_addr` is not 4 KiB aligned, or if adding the offset
    /// overflows.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `phys_addr` refers to a valid, allocated
    /// page table frame whose virtual alias is correctly established by the
    /// kernel's physical memory mapping. Passing an arbitrary address will
    /// produce a dangling virtual pointer and any subsequent dereference is
    /// undefined behaviour.
    unsafe fn phys_to_virt<PT: PageTable>(
        &self,
        phys_addr: PhysicalAddress<PT>,
    ) -> VirtualAddress<PT> {
        assert!(phys_addr.align_offset(REGULAR_PAGE_SIZE) == 0);

        let addr = phys_addr.addr().checked_add(self.page_table_shift).unwrap();

        VirtualAddress::new(addr as _)
    }
}

impl<PFA: PageFrameAllocator> crate::arch::generic::paging::Paging<PFA> for Paging<PFA> {
    fn page_size(page_size: PageSize) -> Option<usize> {
        match page_size {
            PageSize::Regular => Some(REGULAR_PAGE_SIZE),
            PageSize::Huge => Some(HUGE_PAGE_SIZE),
            PageSize::Gigantic => Some(GIGANTIC_PAGE_SIZE),
        }
    }

    /// Destroys the page tables and frees all associated page table frames.
    ///
    /// This must be called instead of letting [`PageTables`] drop, since
    /// dropping without freeing the frames would leak memory. The method
    /// consumes `self` and returns the lock token once cleanup is complete.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the page tables are no longer active (i.e.
    /// not referenced by any CPU's `cr3`) before calling this function.
    /// Destroying active page tables causes all subsequent memory accesses on
    /// that CPU to fault.
    unsafe fn destroy<Token, CB>(self, cb: CB, token: Token) -> Token
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken,
        CB: FnMut(PhysicalAddress<c_void>, PageSize, Token) -> Token,
    {
        let mut cb = cb;
        let mut token = token;

        //
        // Resolve the PML4 root. If cr3 is null the hierarchy is empty.
        //
        let pml4_phys = self.cr3.pml4();
        let pml4 = match pml4_phys.is_null() {
            true => {
                core::mem::forget(self);
                return token;
            }
            false => {
                // Safety: the frame was placed in cr3 by a prior map call and
                // must still be valid.
                unsafe { self.phys_to_virt(pml4_phys).as_mut() }
            }
        };

        //
        // Walk the PML4.
        //
        for pml4_entry in pml4.entries_mut() {
            if !pml4_entry.is_present() {
                continue;
            }
            let pdp_phys = pml4_entry.target();

            // Safety: address written by a prior map call; refers to a valid frame.
            let pdp = unsafe { self.phys_to_virt(pdp_phys).as_mut() };

            //
            // Walk the PDP.
            //
            for pdp_entry in pdp.entries_mut() {
                if !pdp_entry.is_present() {
                    continue;
                }

                let pd_phys = match pdp_entry.target() {
                    PageTableTarget::Page(phys_addr) => {
                        // Gigantic leaf — clear the entry and hand the frame
                        // to the callback.
                        pdp_entry.set_present(false);
                        pdp_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));
                        token = cb(phys_addr, PageSize::Gigantic, token);
                        continue;
                    }
                    PageTableTarget::PageTable(phys_addr) => phys_addr,
                };

                // Safety: address written by a prior map call; refers to a valid frame.
                let pd = unsafe { self.phys_to_virt(pd_phys).as_mut() };

                //
                // Walk the PD.
                //
                for pd_entry in pd.entries_mut() {
                    if !pd_entry.is_present() {
                        continue;
                    }

                    let pt_phys = match pd_entry.target() {
                        PageTableTarget::Page(phys_addr) => {
                            // Huge leaf — clear the entry and hand the frame
                            // to the callback.
                            pd_entry.set_present(false);
                            pd_entry
                                .set_target(PageTableTarget::PageTable(PhysicalAddress::null()));
                            token = cb(phys_addr, PageSize::Huge, token);
                            continue;
                        }
                        PageTableTarget::PageTable(phys_addr) => phys_addr,
                    };

                    // Safety: address written by a prior map call; refers to a valid frame.
                    let pt = unsafe { self.phys_to_virt(pt_phys).as_mut() };

                    //
                    // Walk the PT.
                    //
                    for pt_entry in pt.entries_mut() {
                        if !pt_entry.is_present() {
                            continue;
                        }

                        let phys_addr = pt_entry.target();

                        // Regular leaf — clear the entry and hand the frame
                        // to the callback.
                        pt_entry.set_present(false);
                        pt_entry.set_target(PhysicalAddress::null());
                        token = cb(phys_addr, PageSize::Regular, token);
                    }

                    // All PT entries have been cleared; free the PT frame.
                    pd_entry.set_present(false);
                    pd_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));
                    // Safety: pt_phys was allocated by map and is no longer
                    // referenced by any entry.
                    token = unsafe { PFA::deallocate(pt_phys.cast(), token) };
                }

                // All PD entries have been cleared; free the PD frame.
                pdp_entry.set_present(false);
                pdp_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));
                // Safety: pd_phys was allocated by map and is no longer
                // referenced by any entry.
                token = unsafe { PFA::deallocate(pd_phys.cast(), token) };
            }

            // All PDP entries have been cleared; free the PDP frame.
            pml4_entry.set_present(false);
            pml4_entry.set_target(PhysicalAddress::null());
            // Safety: pdp_phys was allocated by map and is no longer
            // referenced by any entry.
            token = unsafe { PFA::deallocate(pdp_phys.cast(), token) };
        }

        token = unsafe { PFA::deallocate(pml4_phys.cast(), token) };

        core::mem::forget(self);

        token
    }

    /// Maps `virtual_address` to `physical_address` at the requested
    /// page size, allocating intermediate page tables as needed.
    ///
    /// On success, returns the previously mapped frame (if the entry was
    /// already present) together with its page size. When `Ok(Some(_))` is
    /// returned, a live mapping was replaced and the caller **must** call
    /// [`invalidate`](Paging::invalidate) on `virtual_address` before relying
    /// on the new translation.
    ///
    /// Intermediate (non-leaf) entries are created with the most permissive
    /// flags (user-accessible, writable, executable); the effective
    /// permissions of a mapping are governed solely by its leaf entry.
    ///
    /// # Errors
    ///
    /// - [`PagingError::OutOfMemory`] — a page table frame could not be
    ///   allocated. Any frames allocated earlier in the same call are freed
    ///   before returning.
    /// - [`PagingError::Conflict`] — the requested page size conflicts with
    ///   an existing mapping (e.g. a 1 GiB page requested where a populated
    ///   PD already exists, or vice versa).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `physical_address` points to a valid, exclusively owned frame.
    /// - `privilege_level` and `access_rights` are correct for the intended
    ///   use; incorrect values can expose kernel memory to userspace or allow
    ///   unintended writes.
    /// - `virtual_address` is canonical on x86_64.
    unsafe fn map<T, Token>(
        &mut self,
        virtual_address: VirtualAddress<T>,
        physical_address: PhysicalAddress<T>,
        privilege_level: PrivilegeLevel,
        access_rights: AccessRights,
        size: PageSize,
        token: Token,
    ) -> Result<(Option<(PhysicalAddress<T>, PageSize)>, Token), (PagingError, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken,
    {
        let mut token = token;

        // Check if virtual address is canonical
        if !Self::is_canonical(virtual_address) {
            return Err((PagingError::InvalidAddress, token));
        }

        // Frames allocated during this call, tracked for rollback on failure.
        let mut pml4_phys_allocated: Option<PhysicalAddress<PML4>> = None;
        let mut pdp_phys_allocated: Option<PhysicalAddress<PDP>> = None;
        let mut pd_phys_allocated: Option<PhysicalAddress<PD>> = None;

        //
        // Resolve the PML4 (root), allocating it if `cr3` is empty.
        //
        let mut pml4_phys = self.cr3.pml4();
        let pml4 = match pml4_phys.is_null() {
            true => {
                pml4_phys = match PFA::allocate(token) {
                    Ok((phys, t)) => {
                        token = t;
                        phys.cast()
                    }
                    Err(result) => return Err(result),
                };
                pml4_phys_allocated = Some(pml4_phys);

                // Safety: a freshly allocated, 4 KiB-aligned frame is always
                // convertible via the physical-to-virtual offset.
                let mut pml4_virt = unsafe { self.phys_to_virt(pml4_phys) };
                let pml4 = unsafe { pml4_virt.as_mut() };
                pml4.reset();

                self.cr3.set_pml4(pml4_phys);
                pml4
            }
            false => {
                // Safety: the frame was placed in cr3 by a prior call to this
                // function or by `active`; it must therefore still be valid.
                unsafe { self.phys_to_virt(pml4_phys).as_mut() }
            }
        };
        let pml4_entry = pml4.entry_for_mut(virtual_address);

        //
        // Resolve the PDP, allocating it if the PML4 entry is absent.
        //
        let pdp = match pml4_entry.present() {
            true => {
                let pdp_phys = pml4_entry.target();
                // Safety: the address was written by a prior map call and
                // refers to a valid, allocated page table frame.
                unsafe { self.phys_to_virt(pdp_phys).as_mut() }
            }
            false => {
                let pdp_phys: PhysicalAddress<PDP> = match PFA::allocate(token) {
                    Ok((phys, t)) => {
                        token = t;
                        phys.cast()
                    }
                    Err((error, mut token)) => {
                        // Roll back the freshly allocated PML4 (if any).
                        if let Some(p) = pml4_phys_allocated {
                            self.cr3.set_pml4(PhysicalAddress::null());
                            // Safety: `p` was allocated by this call and
                            // has not yet been exposed to the hardware.
                            token = unsafe { PFA::deallocate(p.cast(), token) };
                        }
                        return Err((error, token));
                    }
                };
                pdp_phys_allocated = Some(pdp_phys);

                // Safety: freshly allocated frame, convertible via the offset.
                let mut pdp_virt = unsafe { self.phys_to_virt(pdp_phys) };
                let pdp = unsafe { pdp_virt.as_mut() };
                pdp.reset();

                // Intermediate entries are maximally permissive; the leaf
                // entry governs the effective permissions of the mapping.
                pml4_entry.set_privilege_level(PrivilegeLevel::User);
                pml4_entry.set_access_rights(AccessRights::full());
                pml4_entry.set_target(pdp_phys);
                pml4_entry.set_present(true);
                pdp
            }
        };
        let pdp_entry = pdp.entry_for_mut(virtual_address);

        //
        // Leaf at PDP level: 1 GiB gigantic page.
        //
        if size == PageSize::Gigantic {
            let old = match pdp_entry.present() {
                true => match pdp_entry.target() {
                    // Overwriting a live PD pointer with a leaf is a conflict.
                    PageTableTarget::PageTable(_) => return Err((PagingError::Conflict, token)),
                    PageTableTarget::Page(phys_addr) => {
                        Some((phys_addr.cast(), PageSize::Gigantic))
                    }
                },
                false => None,
            };

            pdp_entry.set_target(PageTableTarget::Page(physical_address.cast()));
            pdp_entry.set_privilege_level(privilege_level);
            pdp_entry.set_access_rights(access_rights);
            pdp_entry.set_accessed(false);
            pdp_entry.set_dirty(false);
            pdp_entry.set_present(true);

            return Ok((old, token));
        }

        //
        // Resolve the PD, allocating it if the PDP entry is absent.
        //
        let pd = match pdp_entry.present() {
            true => match pdp_entry.target() {
                PageTableTarget::PageTable(phys_addr) => {
                    // Safety: address written by a prior map call; valid frame.
                    unsafe { self.phys_to_virt(phys_addr.cast()).as_mut() }
                }
                // A gigantic leaf blocks finer-grained mapping at this slot.
                PageTableTarget::Page(_) => return Err((PagingError::Conflict, token)),
            },
            false => {
                let pd_phys: PhysicalAddress<PD> = match PFA::allocate(token) {
                    Ok((phys, t)) => {
                        token = t;
                        phys.cast()
                    }
                    Err((error, mut token)) => {
                        // Roll back PDP and PML4 if freshly allocated.
                        if let Some(p) = pdp_phys_allocated {
                            pml4_entry.set_present(false);
                            pml4_entry.set_target(PhysicalAddress::null());

                            // Safety: `p` was allocated this call; not yet
                            // visible to hardware.
                            token = unsafe { PFA::deallocate(p.cast(), token) };
                        }
                        if let Some(p) = pml4_phys_allocated {
                            self.cr3.set_pml4(PhysicalAddress::null());
                            // Safety: same guarantee as above.
                            token = unsafe { PFA::deallocate(p.cast(), token) };
                        }
                        return Err((error, token));
                    }
                };
                pd_phys_allocated = Some(pd_phys);

                // Safety: freshly allocated frame, convertible via the offset.
                let mut pd_virt = unsafe { self.phys_to_virt(pd_phys) };
                let pd = unsafe { pd_virt.as_mut() };
                pd.reset();

                // Update the PDP entry to point to the new PD.
                pdp_entry.set_privilege_level(PrivilegeLevel::User);
                pdp_entry.set_access_rights(AccessRights::full());
                pdp_entry.set_target(PageTableTarget::PageTable(pd_phys));
                pdp_entry.set_present(true);
                pd
            }
        };
        let pd_entry = pd.entry_for_mut(virtual_address);

        //
        // Leaf at PD level: 2 MiB huge page.
        //
        if size == PageSize::Huge {
            let old = match pd_entry.present() {
                true => match pd_entry.target() {
                    // Overwriting a live PT pointer with a leaf is a conflict.
                    PageTableTarget::PageTable(_) => return Err((PagingError::Conflict, token)),
                    PageTableTarget::Page(phys_addr) => Some((phys_addr.cast(), PageSize::Huge)),
                },
                false => None,
            };

            pd_entry.set_target(PageTableTarget::Page(physical_address.cast()));
            pd_entry.set_privilege_level(privilege_level);
            pd_entry.set_access_rights(access_rights);
            pd_entry.set_accessed(false);
            pd_entry.set_dirty(false);
            pd_entry.set_present(true);

            return Ok((old, token));
        }

        //
        // Resolve the PT, allocating it if the PD entry is absent.
        //
        let pt = match pd_entry.present() {
            true => match pd_entry.target() {
                PageTableTarget::PageTable(phys_addr) => {
                    // Safety: address written by a prior map call; valid frame.
                    unsafe { self.phys_to_virt(phys_addr.cast()).as_mut() }
                }
                // A huge leaf blocks finer-grained mapping at this slot.
                PageTableTarget::Page(_) => return Err((PagingError::Conflict, token)),
            },
            false => {
                let pt_phys: PhysicalAddress<PT> = match PFA::allocate(token) {
                    Ok((phys, t)) => {
                        token = t;
                        phys.cast()
                    }
                    Err((error, mut token)) => {
                        // PT is the last fallible step; roll back PD, PDP, and
                        // PML4 in reverse order of allocation.
                        if let Some(p) = pd_phys_allocated {
                            pdp_entry.set_present(false);
                            pdp_entry
                                .set_target(PageTableTarget::PageTable(PhysicalAddress::null()));

                            // Safety: `p` was allocated this call; not yet
                            // visible to hardware.
                            token = unsafe { PFA::deallocate(p.cast(), token) };
                        }
                        if let Some(p) = pdp_phys_allocated {
                            pml4_entry.set_present(false);
                            pml4_entry.set_target(PhysicalAddress::null());

                            // Safety: same guarantee as above.
                            token = unsafe { PFA::deallocate(p.cast(), token) };
                        }
                        if let Some(p) = pml4_phys_allocated {
                            self.cr3.set_pml4(PhysicalAddress::null());
                            // Safety: same guarantee as above.
                            token = unsafe { PFA::deallocate(p.cast(), token) };
                        }
                        return Err((error, token));
                    }
                };

                // Safety: freshly allocated frame, convertible via the offset.
                let mut pt_virt = unsafe { self.phys_to_virt(pt_phys) };
                let pt = unsafe { pt_virt.as_mut() };
                pt.reset();

                // Update the PD entry to point to the new PT.
                pd_entry.set_privilege_level(PrivilegeLevel::User);
                pd_entry.set_access_rights(AccessRights::full());
                pd_entry.set_target(PageTableTarget::PageTable(pt_phys));
                pd_entry.set_present(true);
                pt
            }
        };
        let pt_entry = pt.entry_for_mut(virtual_address);

        //
        // Leaf at PT level: 4 KiB regular page.
        //
        let old = match pt_entry.present() {
            true => Some((pt_entry.target().cast(), PageSize::Regular)),
            false => None,
        };

        pt_entry.set_target(physical_address.cast());
        pt_entry.set_privilege_level(privilege_level);
        pt_entry.set_access_rights(access_rights);
        pt_entry.set_accessed(false);
        pt_entry.set_dirty(false);
        pt_entry.set_present(true);

        Ok((old, token))
    }

    /// Removes the mapping for `virtual_address`, freeing any intermediate
    /// page table frames that become empty as a result.
    ///
    /// On success, returns the physical address that was mapped together with
    /// its page_size and the lock token.
    ///
    /// # Errors
    ///
    /// Returns [`PagingError::NotMapped`] if there is no mapping for
    /// `virtual_address`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `virtual_address` is no longer in use by any code or data after this
    ///   call returns.
    /// - [`invalidate`](Paging::invalidate) is called for `virtual_address`
    ///   before any CPU accesses the corresponding virtual range again; a
    ///   stale TLB entry would otherwise point to a potentially freed frame.
    unsafe fn unmap<T, Token>(
        &mut self,
        virtual_address: VirtualAddress<T>,
        token: Token,
    ) -> Result<(PhysicalAddress<T>, PageSize, Token), (PagingError, Token)>
    where
        Token: CanAcquire<level::MemoryManagement> + PreviousToken,
    {
        let mut token = token;

        // Check if virtual address is canonical
        if !Self::is_canonical(virtual_address) {
            return Err((PagingError::InvalidAddress, token));
        }

        //
        // Resolve the PML4 root.
        //
        let pml4_phys = self.cr3.pml4();
        let pml4 = match pml4_phys.is_null() {
            true => return Err((PagingError::NotMapped, token)),
            false => {
                // Safety: the frame was placed in cr3 by a prior map call and
                // must still be valid.
                unsafe { self.phys_to_virt(pml4_phys).as_mut() }
            }
        };
        let pml4_entry = pml4.entry_for_mut(virtual_address);

        //
        // Resolve the PDP.
        //
        let pdp_phys = match pml4_entry.present() {
            true => pml4_entry.target(),
            false => return Err((PagingError::NotMapped, token)),
        };
        // Safety: address written by a prior map call; refers to a valid frame.
        let pdp = unsafe { self.phys_to_virt(pdp_phys).as_mut() };
        let pdp_entry = pdp.entry_for_mut(virtual_address);

        //
        // Resolve the PD — or handle a 1 GiB gigantic page leaf.
        //
        let pd_phys = match pdp_entry.present() {
            true => match pdp_entry.target() {
                PageTableTarget::PageTable(phys_addr) => phys_addr,
                PageTableTarget::Page(phys_addr) => {
                    // Remove the gigantic-page mapping.
                    pdp_entry.set_present(false);
                    pdp_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));

                    // Free the PDP if it is now empty.
                    if pdp.is_unused() {
                        pml4_entry.set_present(false);
                        pml4_entry.set_target(PhysicalAddress::null());

                        // Safety: pdp_phys was allocated by map and is no
                        // longer referenced by any entry.
                        token = unsafe { PFA::deallocate(pdp_phys.cast(), token) };
                    }

                    return Ok((phys_addr.cast(), PageSize::Gigantic, token));
                }
            },
            false => return Err((PagingError::NotMapped, token)),
        };
        // Safety: address written by a prior map call; refers to a valid frame.
        let pd = unsafe { self.phys_to_virt(pd_phys).as_mut() };
        let pd_entry = pd.entry_for_mut(virtual_address);

        //
        // Resolve the PT — or handle a 2 MiB huge page leaf.
        //
        let pt_phys = match pd_entry.present() {
            true => match pd_entry.target() {
                PageTableTarget::PageTable(phys_addr) => phys_addr,
                PageTableTarget::Page(phys_addr) => {
                    // Remove the huge-page mapping.
                    pd_entry.set_present(false);
                    pd_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));

                    // Free the PD if it is now empty.
                    if pd.is_unused() {
                        pdp_entry.set_present(false);
                        pdp_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));

                        // Safety: pd_phys was allocated by map and is no
                        // longer referenced by any entry.
                        token = unsafe { PFA::deallocate(pd_phys.cast(), token) };
                    }

                    // Free the PDP if it is now empty.
                    if pdp.is_unused() {
                        pml4_entry.set_present(false);
                        pml4_entry.set_target(PhysicalAddress::null());

                        // Safety: pdp_phys was allocated by map and is no
                        // longer referenced by any entry.
                        token = unsafe { PFA::deallocate(pdp_phys.cast(), token) };
                    }

                    return Ok((phys_addr.cast(), PageSize::Huge, token));
                }
            },
            false => return Err((PagingError::NotMapped, token)),
        };
        // Safety: address written by a prior map call; refers to a valid frame.
        let pt = unsafe { self.phys_to_virt(pt_phys).as_mut() };
        let pt_entry = pt.entry_for_mut(virtual_address);

        //
        // Handle the 4 KiB regular page leaf.
        //
        match pt_entry.present() {
            true => {
                let phys_addr = pt_entry.target();

                // Remove the regular-page mapping.
                pt_entry.set_present(false);
                pt_entry.set_target(PhysicalAddress::null());

                // Free the PT if it is now empty.
                if pt.is_unused() {
                    pd_entry.set_present(false);
                    pd_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));

                    // Safety: pt_phys was allocated by map and is no longer
                    // referenced by any entry.
                    token = unsafe { PFA::deallocate(pt_phys.cast(), token) };
                }

                // Free the PD if it is now empty.
                if pd.is_unused() {
                    pdp_entry.set_present(false);
                    pdp_entry.set_target(PageTableTarget::PageTable(PhysicalAddress::null()));

                    // Safety: pd_phys was allocated by map and is no longer
                    // referenced by any entry.
                    token = unsafe { PFA::deallocate(pd_phys.cast(), token) };
                }

                // Free the PDP if it is now empty.
                if pdp.is_unused() {
                    pml4_entry.set_present(false);
                    pml4_entry.set_target(PhysicalAddress::null());

                    // Safety: pdp_phys was allocated by map and is no longer
                    // referenced by any entry.
                    token = unsafe { PFA::deallocate(pdp_phys.cast(), token) };
                }

                Ok((phys_addr.cast(), PageSize::Regular, token))
            }
            false => Err((PagingError::NotMapped, token)),
        }
    }

    /// Resolves `virtual_address` by walking the page table and returns the
    /// mapped physical address, privilege level, effective access rights, and
    /// page size.
    ///
    /// The effective access rights are computed by AND-ing the writable and
    /// no-execute bits across every walked level; a present entry is always
    /// readable on x86_64. The effective privilege level is `User` only if
    /// every level has `U/S = 1`.
    ///
    /// # Errors
    ///
    /// Returns [`PagingError::NotMapped`] if any level of the walk yields a
    /// not-present entry, or if no PML4 has been allocated yet.
    fn resolve<T>(
        &self,
        virtual_address: VirtualAddress<T>,
    ) -> Result<(PhysicalAddress<T>, PrivilegeLevel, AccessRights, PageSize), PagingError> {
        // Check if virtual address is canonical
        if !Self::is_canonical(virtual_address) {
            return Err(PagingError::InvalidAddress);
        }

        //
        // Resolve the PML4 root.
        //
        let pml4_phys = self.cr3.pml4();
        if pml4_phys.is_null() {
            return Err(PagingError::NotMapped);
        }
        // Safety: the frame was placed in cr3 by a prior map call; valid.
        let pml4 = unsafe { self.phys_to_virt(pml4_phys).as_ref() };
        let pml4_entry = pml4.entry_for(virtual_address);

        if !pml4_entry.is_present() {
            return Err(PagingError::NotMapped);
        }
        let pml4_w = pml4_entry.is_writable();
        let pml4_x = !pml4_entry.is_no_execute();
        let pml4_u = pml4_entry.is_user_accessible();

        //
        // Resolve the PDP.
        //
        // Safety: address written by a prior map call; refers to a valid frame.
        let pdp = unsafe { self.phys_to_virt(pml4_entry.target()).as_ref() };
        let pdp_entry = pdp.entry_for(virtual_address);

        if !pdp_entry.is_present() {
            return Err(PagingError::NotMapped);
        }
        let pdp_w = pdp_entry.is_writable();
        let pdp_x = !pdp_entry.is_no_execute();
        let pdp_u = pdp_entry.is_user_accessible();

        //
        // Resolve the PD — or return a 1 GiB gigantic page result.
        //
        let pd_phys = match pdp_entry.target() {
            PageTableTarget::Page(phys_addr) => {
                let access_rights = AccessRights::custom(true, pml4_w & pdp_w, pml4_x & pdp_x);
                let privilege_level = match pml4_u & pdp_u {
                    true => PrivilegeLevel::User,
                    false => PrivilegeLevel::Kernel,
                };
                return Ok((
                    phys_addr.cast(),
                    privilege_level,
                    access_rights,
                    PageSize::Gigantic,
                ));
            }
            PageTableTarget::PageTable(phys_addr) => phys_addr,
        };
        // Safety: address written by a prior map call; refers to a valid frame.
        let pd = unsafe { self.phys_to_virt(pd_phys).as_ref() };
        let pd_entry = pd.entry_for(virtual_address);

        if !pd_entry.is_present() {
            return Err(PagingError::NotMapped);
        }
        let pd_w = pd_entry.is_writable();
        let pd_x = !pd_entry.is_no_execute();
        let pd_u = pd_entry.is_user_accessible();

        //
        // Resolve the PT — or return a 2 MiB huge page result.
        //
        let pt_phys = match pd_entry.target() {
            PageTableTarget::Page(phys_addr) => {
                let access_rights =
                    AccessRights::custom(true, pml4_w & pdp_w & pd_w, pml4_x & pdp_x & pd_x);
                let privilege_level = match pml4_u & pdp_u & pd_u {
                    true => PrivilegeLevel::User,
                    false => PrivilegeLevel::Kernel,
                };
                return Ok((
                    phys_addr.cast(),
                    privilege_level,
                    access_rights,
                    PageSize::Huge,
                ));
            }
            PageTableTarget::PageTable(phys_addr) => phys_addr,
        };
        // Safety: address written by a prior map call; refers to a valid frame.
        let pt = unsafe { self.phys_to_virt(pt_phys).as_ref() };
        let pt_entry = pt.entry_for(virtual_address);

        if !pt_entry.is_present() {
            return Err(PagingError::NotMapped);
        }
        let pt_w = pt_entry.is_writable();
        let pt_x = !pt_entry.is_no_execute();
        let pt_u = pt_entry.is_user_accessible();

        let access_rights = AccessRights::custom(
            true,
            pml4_w & pdp_w & pd_w & pt_w,
            pml4_x & pdp_x & pd_x & pt_x,
        );
        let privilege_level = match pml4_u & pdp_u & pd_u & pt_u {
            true => PrivilegeLevel::User,
            false => PrivilegeLevel::Kernel,
        };

        Ok((
            pt_entry.target().cast(),
            privilege_level,
            access_rights,
            PageSize::Regular,
        ))
    }

    fn invalidate<T>(virt_addr: VirtualAddress<T>) {
        unsafe {
            core::arch::asm!(
                "invlpg [{}]",
                in(reg) virt_addr.as_ptr(),
                options(nostack, preserves_flags)
            );
        }
    }

    fn invalidate_all() {
        // Reloading `cr3` with its current value flushes all non-global TLB
        // entries without changing the active page table.
        //
        // Safety: the current `cr3` is already active and valid; reloading it
        // does not change the memory mapping.
        let cr3 = CR3::read();
        unsafe { cr3.write() };
    }

    unsafe fn active(&self) {
        // Safety: the caller is responsible for ensuring the page table covers
        // all addresses currently in use, including the instruction pointer and
        // stack. See [`CR3::write`] for the full contract.
        unsafe { self.cr3.write() };
    }

    const REGULAR_PAGE_SIZE: usize = REGULAR_PAGE_SIZE;

    const HUGE_PAGE_SIZE: Option<usize> = Some(HUGE_PAGE_SIZE);

    const GIGANTIC_PAGE_SIZE: Option<usize> = Some(GIGANTIC_PAGE_SIZE);
}

impl<PFA: PageFrameAllocator> Drop for Paging<PFA> {
    fn drop(&mut self) {
        // PageTables must be explicitly destroyed via `destroy()` to ensure
        // all page table frames are freed. Automatic dropping is a bug.
        panic!(
            "PageTables must never be dropped automatically; \
             call PageTables::destroy(...) instead"
        )
    }
}

#[cfg(test)]
mod test {
    use std::{
        cell::RefCell,
        collections::{HashMap, HashSet},
        num::NonZero,
        sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    use nix::sys::mman::{MapFlags, ProtFlags};

    use crate::kernel::locking::{RootToken, SyscallLevel};

    use super::*;

    extern crate std;

    const VIRT_PHYS_SHIFT: usize = 32 * 1024 * 1024 * 1024;

    std::thread_local! {
        static PAGE_FRAMES: RefCell<HashSet<PhysicalAddress<c_void>>> = RefCell::new(HashSet::new());
    }
    static PAGE_FRAME_ADDR: AtomicUsize = AtomicUsize::new(VIRT_PHYS_SHIFT + 16 * 1024 * 1024);

    struct TestPageFrameAllocator;

    impl TestPageFrameAllocator {
        fn virt_to_phys<T>(virt_addr: VirtualAddress<T>) -> PhysicalAddress<T> {
            PhysicalAddress::new((virt_addr.addr() - VIRT_PHYS_SHIFT) as _)
        }

        fn leaked() -> bool {
            PAGE_FRAMES.with_borrow(|page_frames| !page_frames.is_empty())
        }

        fn next_virtual_addr(page_size: PageSize) -> VirtualAddress<c_void> {
            let size = match page_size {
                PageSize::Regular => REGULAR_PAGE_SIZE,
                PageSize::Huge => HUGE_PAGE_SIZE,
                PageSize::Gigantic => GIGANTIC_PAGE_SIZE,
            };

            let mut prev = PAGE_FRAME_ADDR.load(AtomicOrdering::Relaxed);
            let ptr = loop {
                let offset = prev % size;
                let new = prev + size - offset + size;
                match PAGE_FRAME_ADDR.compare_exchange_weak(
                    prev,
                    new,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => break prev + size - offset,
                    Err(next_prev) => prev = next_prev,
                };
            };

            assert!(ptr % size == 0);

            VirtualAddress::new(ptr as _)
        }
    }

    impl PageFrameAllocator for TestPageFrameAllocator {
        fn allocate<Token>(
            token: Token,
        ) -> Result<(PhysicalAddress<c_void>, Token), (PagingError, Token)>
        where
            Token: CanAcquire<level::Memory> + PreviousToken,
        {
            // Calculate next address
            let virt_addr = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);

            // Allocate page frame
            let virt_addr = match unsafe {
                nix::sys::mman::mmap_anonymous(
                    Some(NonZero::new(virt_addr.addr() as usize).unwrap()),
                    NonZero::new(REGULAR_PAGE_SIZE).unwrap(),
                    ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                    MapFlags::MAP_PRIVATE,
                )
            } {
                Ok(phys_addr) => VirtualAddress::new(phys_addr.as_ptr()),
                Err(error) => panic!(
                    "Unable to allocate virtual address {:p}: {}",
                    virt_addr, error
                ),
            };
            let phys_addr = Self::virt_to_phys(virt_addr);

            // Update allocator statistics
            let known = PAGE_FRAMES.with_borrow_mut(|page_frames| page_frames.insert(phys_addr));
            if !known {
                panic!("Found duplicate page frame {:p}", phys_addr);
            }

            Ok((phys_addr, token))
        }

        unsafe fn deallocate<Token>(phys_addr: PhysicalAddress<c_void>, token: Token) -> Token
        where
            Token: CanAcquire<level::Memory> + PreviousToken,
        {
            // Update allocator statistics
            let known = PAGE_FRAMES.with_borrow_mut(|page_frames| page_frames.remove(&phys_addr));
            if !known {
                panic!(
                    "Detected unknown physical address for deallocate! {:p}",
                    phys_addr
                );
            }

            token
        }
    }

    #[test]
    fn map_resolve_4k() {
        // Enter syscall level
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        // Track page frames
        let mut page_frames: HashMap<PhysicalAddress<c_void>, PageSize> = HashMap::new();

        // Map 4k page
        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let src_virt_addr = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let dst_phys_addr = TestPageFrameAllocator::virt_to_phys(src_virt_addr);

        let token = match unsafe {
            paging.map(
                src_virt_addr,
                dst_phys_addr,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((prev, token)) => {
                assert!(prev.is_none());
                token
            }
            Err((error, _token)) => panic!("Unexpected error during mapping: {}", error),
        };
        page_frames.insert(dst_phys_addr, PageSize::Regular);

        // Resolve mapping
        match paging.resolve(src_virt_addr) {
            Ok((phys_addr, priv_level, access_rights, page_size)) => {
                assert!(
                    phys_addr == dst_phys_addr,
                    "Expected: {:?}, got: {:?}",
                    dst_phys_addr,
                    phys_addr
                );
                assert!(
                    priv_level == PrivilegeLevel::User,
                    "Expected: {}, got: {}",
                    PrivilegeLevel::User,
                    priv_level
                );
                assert!(
                    access_rights == AccessRights::full(),
                    "Expected: {}, got: {}",
                    AccessRights::full(),
                    access_rights
                );
                assert!(
                    page_size == PageSize::Regular,
                    "Expected: {}, got: {}",
                    PageSize::Regular,
                    page_size
                );
            }
            Err(error) => panic!("Unexpected error during resolving: {}", error),
        };

        // Perform clean up
        let pf_cb = |phys_addr, page_size, token| {
            let prev = page_frames.remove(&phys_addr);
            assert!(prev == Some(page_size));
            token
        };
        let token = unsafe { paging.destroy(pf_cb, token) };

        // Leave syscall level
        syscall_level.leave(token);

        // Check if no page frames were leaked
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn map_resolve_2m() {
        // Enter syscall level
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        // Track page frames
        let mut page_frames: HashMap<PhysicalAddress<c_void>, PageSize> = HashMap::new();

        // Map 2M page
        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let src_virt_addr = TestPageFrameAllocator::next_virtual_addr(PageSize::Huge);
        let dst_phys_addr = TestPageFrameAllocator::virt_to_phys(src_virt_addr);

        let token = match unsafe {
            paging.map(
                src_virt_addr,
                dst_phys_addr,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Huge,
                token,
            )
        } {
            Ok((prev, token)) => {
                assert!(prev.is_none());
                token
            }
            Err((error, _token)) => panic!("Unexpected error during mapping: {}", error),
        };
        page_frames.insert(dst_phys_addr, PageSize::Huge);

        // Resolve mapping
        match paging.resolve(src_virt_addr) {
            Ok((phys_addr, priv_level, access_rights, page_size)) => {
                assert!(
                    phys_addr == dst_phys_addr,
                    "Expected: {:?}, got: {:?}",
                    dst_phys_addr,
                    phys_addr
                );
                assert!(
                    priv_level == PrivilegeLevel::User,
                    "Expected: {}, got: {}",
                    PrivilegeLevel::User,
                    priv_level
                );
                assert!(
                    access_rights == AccessRights::full(),
                    "Expected: {}, got: {}",
                    AccessRights::full(),
                    access_rights
                );
                assert!(
                    page_size == PageSize::Huge,
                    "Expected: {}, got: {}",
                    PageSize::Huge,
                    page_size
                );
            }
            Err(error) => panic!("Unexpected error during resolving: {}", error),
        };

        // Perform clean up
        let pf_cb = |phys_addr, page_size, token| {
            let prev = page_frames.remove(&phys_addr);
            assert!(
                prev == Some(page_size),
                "Expected: {:?}, got: {:?}",
                prev,
                Some(page_size)
            );
            token
        };
        let token = unsafe { paging.destroy(pf_cb, token) };

        // Leave syscall level
        syscall_level.leave(token);

        // Check if no page frames were leaked
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn map_resolve_1g() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut page_frames: HashMap<PhysicalAddress<c_void>, PageSize> = HashMap::new();
        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let src_virt_addr = TestPageFrameAllocator::next_virtual_addr(PageSize::Gigantic);
        let dst_phys_addr = TestPageFrameAllocator::virt_to_phys(src_virt_addr);

        let token = match unsafe {
            paging.map(
                src_virt_addr,
                dst_phys_addr,
                PrivilegeLevel::Kernel,
                AccessRights::full(),
                PageSize::Gigantic,
                token,
            )
        } {
            Ok((prev, token)) => {
                assert!(prev.is_none());
                token
            }
            Err((error, _token)) => panic!("Unexpected error during mapping: {}", error),
        };
        page_frames.insert(dst_phys_addr, PageSize::Gigantic);

        match paging.resolve(src_virt_addr) {
            Ok((phys_addr, priv_level, access_rights, page_size)) => {
                assert_eq!(phys_addr, dst_phys_addr);
                assert_eq!(priv_level, PrivilegeLevel::Kernel);
                assert_eq!(access_rights, AccessRights::full());
                assert_eq!(page_size, PageSize::Gigantic);
            }
            Err(error) => panic!("Unexpected error during resolving: {}", error),
        }

        let pf_cb = |phys_addr, page_size, token| {
            let prev = page_frames.remove(&phys_addr);
            assert_eq!(prev, Some(page_size));
            token
        };
        let token = unsafe { paging.destroy(pf_cb, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn resolve_empty() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let addr = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);

        match paging.resolve(addr) {
            Err(PagingError::NotMapped) => {}
            other => panic!("Expected NotMapped, got {:?}", other),
        }

        // destroy an empty hierarchy (no PML4 allocated yet)
        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn resolve_unmapped_address() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        // Map one page so the PML4 is populated.
        let mapped_virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let mapped_phys = TestPageFrameAllocator::virt_to_phys(mapped_virt);
        let token = match unsafe {
            paging.map(
                mapped_virt,
                mapped_phys,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        // A completely different, never-mapped virtual address.
        let unmapped_virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        match paging.resolve(unmapped_virt) {
            Err(PagingError::NotMapped) => {}
            other => panic!("Expected NotMapped, got {:?}", other),
        }

        let token = unsafe {
            paging.destroy(
                |_, _, t| t, // leaf frames were never tracked; ignore them for this check
                token,
            )
        };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn map_unmap_4k() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        // unmap
        let token = match unsafe { paging.unmap(virt, token) } {
            Ok((returned_phys, page_size, t)) => {
                assert_eq!(returned_phys, phys);
                assert_eq!(page_size, PageSize::Regular);
                t
            }
            Err((e, _)) => panic!("Unexpected error during unmap: {}", e),
        };

        // should no longer resolve
        match paging.resolve(virt) {
            Err(PagingError::NotMapped) => {}
            other => panic!("Expected NotMapped after unmap, got {:?}", other),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn map_unmap_2m() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Huge);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Huge,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        let token = match unsafe { paging.unmap(virt, token) } {
            Ok((returned_phys, page_size, t)) => {
                assert_eq!(returned_phys, phys);
                assert_eq!(page_size, PageSize::Huge);
                t
            }
            Err((e, _)) => panic!("Unexpected error during unmap: {}", e),
        };

        match paging.resolve(virt) {
            Err(PagingError::NotMapped) => {}
            other => panic!("Expected NotMapped after unmap, got {:?}", other),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn map_unmap_1g() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Gigantic);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::Kernel,
                AccessRights::full(),
                PageSize::Gigantic,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        let token = match unsafe { paging.unmap(virt, token) } {
            Ok((returned_phys, page_size, t)) => {
                assert_eq!(returned_phys, phys);
                assert_eq!(page_size, PageSize::Gigantic);
                t
            }
            Err((e, _)) => panic!("Unexpected error during unmap: {}", e),
        };

        match paging.resolve(virt) {
            Err(PagingError::NotMapped) => {}
            other => panic!("Expected NotMapped after unmap, got {:?}", other),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn unmap_not_mapped() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        // Map one page so there is a PML4, but try to unmap a different address.
        let virt_a = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys_a = TestPageFrameAllocator::virt_to_phys(virt_a);
        let token = match unsafe {
            paging.map(
                virt_a,
                phys_a,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        let virt_b = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        match unsafe { paging.unmap(virt_b, token) } {
            Err((PagingError::NotMapped, t)) => {
                let token = unsafe { paging.destroy(|_, _, t| t, t) };
                syscall_level.leave(token);
            }
            Ok(_) => panic!("Expected NotMapped, got Ok"),
            Err((e, _)) => panic!("Expected NotMapped, got {}", e),
        }

        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn conflict_regular_then_gigantic() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        // Map a 4 KiB page; this allocates PML4 → PDP → PD → PT.
        let virt_4k = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys_4k = TestPageFrameAllocator::virt_to_phys(virt_4k);
        let token = match unsafe {
            paging.map(
                virt_4k,
                phys_4k,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        // Now request a gigantic page that aliases the same PDP entry.
        // The PDP entry already points to a PD table, so this must conflict.
        let phys_1g = TestPageFrameAllocator::virt_to_phys(virt_4k); // reuse address; alignment doesn't matter here
        match unsafe {
            paging.map(
                virt_4k,
                phys_1g,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Gigantic,
                token,
            )
        } {
            Err((PagingError::Conflict, t)) => {
                let token = unsafe { paging.destroy(|_, _, t| t, t) };
                syscall_level.leave(token);
            }
            Ok(_) => panic!("Expected Conflict, got Ok"),
            Err((e, _)) => panic!("Expected Conflict, got {}", e),
        }

        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn conflict_huge_then_regular() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let virt_2m = TestPageFrameAllocator::next_virtual_addr(PageSize::Huge);
        let phys_2m = TestPageFrameAllocator::virt_to_phys(virt_2m);
        let token = match unsafe {
            paging.map(
                virt_2m,
                phys_2m,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Huge,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        // Try to place a 4 KiB page inside the same 2 MiB slot.
        let phys_4k = phys_2m;
        match unsafe {
            paging.map(
                virt_2m,
                phys_4k,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Err((PagingError::Conflict, t)) => {
                let token = unsafe { paging.destroy(|_, _, t| t, t) };
                syscall_level.leave(token);
            }
            Ok(_) => panic!("Expected Conflict, got Ok"),
            Err((e, _)) => panic!("Expected Conflict, got {}", e),
        }

        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn remap_4k_returns_old_frame() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys_a = TestPageFrameAllocator::virt_to_phys(virt);
        let phys_b = TestPageFrameAllocator::virt_to_phys(
            TestPageFrameAllocator::next_virtual_addr(PageSize::Regular),
        );

        // First mapping.
        let token = match unsafe {
            paging.map(
                virt,
                phys_a,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((prev, t)) => {
                assert!(prev.is_none(), "Expected no previous mapping");
                t
            }
            Err((e, _)) => panic!("{}", e),
        };

        // Second mapping at the same virtual address.
        let token = match unsafe {
            paging.map(
                virt,
                phys_b,
                PrivilegeLevel::Kernel,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((prev, t)) => {
                let (old_phys, old_size) = prev.expect("Expected previous mapping to be returned");
                assert_eq!(old_phys, phys_a);
                assert_eq!(old_size, PageSize::Regular);
                t
            }
            Err((e, _)) => panic!("{}", e),
        };

        // The new mapping resolves to phys_b with Kernel privilege.
        match paging.resolve(virt) {
            Ok((phys, priv_level, _, size)) => {
                assert_eq!(phys, phys_b);
                assert_eq!(priv_level, PrivilegeLevel::Kernel);
                assert_eq!(size, PageSize::Regular);
            }
            Err(e) => panic!("{}", e),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn remap_2m_returns_old_frame() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Huge);
        let phys_a = TestPageFrameAllocator::virt_to_phys(virt);
        let phys_b = TestPageFrameAllocator::virt_to_phys(
            TestPageFrameAllocator::next_virtual_addr(PageSize::Huge),
        );

        let token = match unsafe {
            paging.map(
                virt,
                phys_a,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Huge,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        let token = match unsafe {
            paging.map(
                virt,
                phys_b,
                PrivilegeLevel::Kernel,
                AccessRights::full(),
                PageSize::Huge,
                token,
            )
        } {
            Ok((prev, t)) => {
                let (old_phys, old_size) = prev.expect("Expected previous mapping");
                assert_eq!(old_phys, phys_a);
                assert_eq!(old_size, PageSize::Huge);
                t
            }
            Err((e, _)) => panic!("{}", e),
        };

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn access_rights_read_only_kernel() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let ro = AccessRights::custom(true, false, false); // readable, not writable, not executable
        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::Kernel,
                ro,
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        match paging.resolve(virt) {
            Ok((_, priv_level, access_rights, _)) => {
                assert_eq!(priv_level, PrivilegeLevel::Kernel);
                assert!(access_rights.is_readable());
                assert!(!access_rights.is_writable());
                assert!(!access_rights.is_executable());
            }
            Err(e) => panic!("{}", e),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn access_rights_executable_user() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let rx = AccessRights::custom(true, false, true); // readable + executable
        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::User,
                rx,
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        match paging.resolve(virt) {
            Ok((_, priv_level, access_rights, _)) => {
                assert_eq!(priv_level, PrivilegeLevel::User);
                assert!(access_rights.is_readable());
                assert!(!access_rights.is_writable());
                assert!(access_rights.is_executable());
            }
            Err(e) => panic!("{}", e),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn multiple_4k_mappings() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let mut token = token;

        const N: usize = 8;
        let mut virts = Vec::with_capacity(N);
        let mut physs = Vec::with_capacity(N);

        // Map N pages.
        for _ in 0..N {
            let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
            let phys = TestPageFrameAllocator::virt_to_phys(virt);
            token = match unsafe {
                paging.map(
                    virt,
                    phys,
                    PrivilegeLevel::User,
                    AccessRights::full(),
                    PageSize::Regular,
                    token,
                )
            } {
                Ok((_, t)) => t,
                Err((e, _)) => panic!("{}", e),
            };
            virts.push(virt);
            physs.push(phys);
        }

        // Resolve each.
        for i in 0..N {
            match paging.resolve(virts[i]) {
                Ok((phys, _, _, size)) => {
                    assert_eq!(phys, physs[i]);
                    assert_eq!(size, PageSize::Regular);
                }
                Err(e) => panic!("resolve failed for mapping {}: {}", i, e),
            }
        }

        // Unmap each and verify the others are still present.
        for i in 0..N {
            token = match unsafe { paging.unmap(virts[i], token) } {
                Ok((phys, size, t)) => {
                    assert_eq!(phys, physs[i]);
                    assert_eq!(size, PageSize::Regular);
                    t
                }
                Err((e, _)) => panic!("unmap failed for mapping {}: {}", i, e),
            };

            // The unmapped address must no longer resolve.
            match paging.resolve(virts[i]) {
                Err(PagingError::NotMapped) => {}
                other => panic!("Expected NotMapped after unmap {}, got {:?}", i, other),
            }

            // All subsequent addresses must still resolve.
            for j in (i + 1)..N {
                match paging.resolve(virts[j]) {
                    Ok((phys, _, _, _)) => assert_eq!(phys, physs[j]),
                    Err(e) => panic!(
                        "resolve unexpectedly failed for mapping {} after unmap {}: {}",
                        j, i, e
                    ),
                }
            }
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn mixed_page_sizes() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut page_frames: HashMap<PhysicalAddress<c_void>, PageSize> = HashMap::new();
        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let virt_1g = TestPageFrameAllocator::next_virtual_addr(PageSize::Gigantic);
        let phys_1g = TestPageFrameAllocator::virt_to_phys(virt_1g);

        let virt_2m = TestPageFrameAllocator::next_virtual_addr(PageSize::Huge);
        let phys_2m = TestPageFrameAllocator::virt_to_phys(virt_2m);

        let virt_4k = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys_4k = TestPageFrameAllocator::virt_to_phys(virt_4k);

        let mut token = token;

        for (v, p, size) in [
            (virt_1g, phys_1g, PageSize::Gigantic),
            (virt_2m, phys_2m, PageSize::Huge),
            (virt_4k, phys_4k, PageSize::Regular),
        ] {
            token = match unsafe {
                paging.map(
                    v,
                    p,
                    PrivilegeLevel::Kernel,
                    AccessRights::full(),
                    size,
                    token,
                )
            } {
                Ok((_, t)) => t,
                Err((e, _)) => panic!("map failed ({:?}): {}", size, e),
            };
            page_frames.insert(p, size);
        }

        // Verify each resolves correctly.
        assert_eq!(paging.resolve(virt_1g).unwrap().3, PageSize::Gigantic);
        assert_eq!(paging.resolve(virt_2m).unwrap().3, PageSize::Huge);
        assert_eq!(paging.resolve(virt_4k).unwrap().3, PageSize::Regular);

        let pf_cb = |phys_addr, page_size, token| {
            let prev = page_frames.remove(&phys_addr);
            assert_eq!(
                prev,
                Some(page_size),
                "unexpected frame in destroy callback"
            );
            token
        };
        let token = unsafe { paging.destroy(pf_cb, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn double_unmap_returns_not_mapped() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::User,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        // First unmap succeeds.
        let token = match unsafe { paging.unmap(virt, token) } {
            Ok((_, _, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        // Second unmap must return NotMapped.
        match unsafe { paging.unmap(virt, token) } {
            Err((PagingError::NotMapped, t)) => {
                let token = unsafe { paging.destroy(|_, _, t| t, t) };
                syscall_level.leave(token);
            }
            Ok(_) => panic!("Expected NotMapped on second unmap, got Ok"),
            Err((e, _)) => panic!("Expected NotMapped, got {}", e),
        }

        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn destroy_invokes_callback_for_each_leaf() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let mut token = token;

        const N: usize = 4;
        let mut expected: HashMap<PhysicalAddress<c_void>, PageSize> = HashMap::new();

        for _ in 0..N {
            let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
            let phys = TestPageFrameAllocator::virt_to_phys(virt);
            token = match unsafe {
                paging.map(
                    virt,
                    phys,
                    PrivilegeLevel::User,
                    AccessRights::full(),
                    PageSize::Regular,
                    token,
                )
            } {
                Ok((_, t)) => t,
                Err((e, _)) => panic!("{}", e),
            };
            expected.insert(phys, PageSize::Regular);
        }

        // destroy must call back exactly once per leaf frame.
        let mut seen: HashMap<PhysicalAddress<c_void>, PageSize> = HashMap::new();
        let pf_cb = |phys_addr: PhysicalAddress<c_void>, page_size, token| {
            let prev = seen.insert(phys_addr, page_size);
            assert!(
                prev.is_none(),
                "destroy callback invoked twice for {:p}",
                phys_addr
            );
            token
        };
        let token = unsafe { paging.destroy(pf_cb, token) };

        assert_eq!(
            seen, expected,
            "destroy callback set differs from mapped frames"
        );

        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn privilege_level_kernel() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);
        let virt = TestPageFrameAllocator::next_virtual_addr(PageSize::Regular);
        let phys = TestPageFrameAllocator::virt_to_phys(virt);

        let token = match unsafe {
            paging.map(
                virt,
                phys,
                PrivilegeLevel::Kernel,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Ok((_, t)) => t,
            Err((e, _)) => panic!("{}", e),
        };

        match paging.resolve(virt) {
            Ok((_, priv_level, _, _)) => assert_eq!(priv_level, PrivilegeLevel::Kernel),
            Err(e) => panic!("{}", e),
        }

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn map_non_canonical() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let token = match unsafe {
            paging.map(
                VirtualAddress::<usize>::new(0x01007ffffffff000usize as _),
                PhysicalAddress::<usize>::new(0x00007ffffffff000usize as _),
                PrivilegeLevel::Kernel,
                AccessRights::full(),
                PageSize::Regular,
                token,
            )
        } {
            Err((PagingError::InvalidAddress, token)) => {
                /* Expected behaviour */
                token
            }
            Ok((_, _)) => panic!("A non-canonical address must never be mapped!"),
            Err((error, _)) => panic!(
                "Unexpected error while mapping non-canonical address: {}",
                error
            ),
        };

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn unmap_non_canonical() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let mut paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        let token = match unsafe {
            paging.unmap(
                VirtualAddress::<usize>::new(0x01007ffffffff000usize as _),
                token,
            )
        } {
            Err((PagingError::InvalidAddress, token)) => {
                /* Expected behaviour */
                token
            }
            Ok(_) => panic!("A non-canonical address must never be mapped!"),
            Err((error, _)) => panic!(
                "Unexpected error while un-mapping non-canonical address: {}",
                error
            ),
        };

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }

    #[test]
    fn resolve_non_canonical() {
        let root_token = unsafe { RootToken::forge() };
        let (syscall_level, token) = SyscallLevel::enter(root_token);

        let paging: Paging<TestPageFrameAllocator> = Paging::new(VIRT_PHYS_SHIFT);

        match paging.resolve(VirtualAddress::<usize>::new(0x01007ffffffff000usize as _)) {
            Err(PagingError::InvalidAddress) => { /* Expected behaviour */ }
            Ok(_) => panic!("A non-canonical address must never be mapped!"),
            Err(error) => panic!(
                "Unexpected error while un-mapping non-canonical address: {}",
                error
            ),
        };

        let token = unsafe { paging.destroy(|_, _, t| t, token) };
        syscall_level.leave(token);
        assert!(!TestPageFrameAllocator::leaked());
    }
}
