//! Abstraction for an entry of the x86_64 *Global Descriptor Table* (GDT).
//!
//! Long mode keeps the GDT for the little it still decides: the privilege
//! level a selector loads, whether a code segment runs 64-bit code, and where
//! the task state segment lives. Base and limit survive as fields of a
//! code/data descriptor, but the CPU ignores them there — only a system
//! descriptor still uses them for anything.
//!
//! The table holds entries of two shapes: a code/data descriptor is 8 bytes,
//! a system descriptor 16, the second half carrying the top of a 64-bit base.
//! [`Descriptor`] is the eight bytes both shapes begin with, [`Extension`]
//! the half a system descriptor adds; the two go into the table back to
//! back. An array of `Descriptor` is therefore the wrong thing to hand the
//! CPU: the table is a packed byte sequence, and an entry contributes
//! [`Descriptor::size`] bytes to it, not always eight.
//!
//! [`Gdt`] is the table itself: one allocation, laid out the way `syscall`
//! and `sysret` need it and given a task state descriptor per core, with
//! every entry in it reachable as the descriptor or extension it holds.

use core::alloc::Layout;
use core::arch::asm;
use core::ffi::c_void;
use core::fmt::Display;
use core::mem::MaybeUninit;
use core::slice;

use bitfield_struct::bitfield;

use crate::{
    arch::generic::paging::{PrivilegeLevel, VirtualAddress},
    kernel::{
        bootinfo::BOOTINFO,
        locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken, ThreadLevelID},
        ticketlock::{ThreadTicketlock, Ticketlock},
    },
    mem::heap::Heap,
    utils::allocator::{Allocator, Error},
};

/// The largest value a segment limit can hold — it is a 20-bit field.
pub const LIMIT_MAX: u32 = 0xf_ffff;

/// The privilege level a descriptor is reachable from, as the two-bit `DPL`
/// field encodes it.
///
/// x86_64 offers four rings; the kernel uses two of them, which is what
/// [`PrivilegeLevel`] names. The middle two exist here because the field
/// encodes them and a descriptor read back from a table may hold one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Ring {
    /// Ring 0 — the kernel.
    #[default]
    Zero = 0b00,
    /// Ring 1 — unused.
    One = 0b01,
    /// Ring 2 — unused.
    Two = 0b10,
    /// Ring 3 — userspace.
    Three = 0b11,
}

impl Ring {
    /// The hardware encoding of this ring.
    pub const fn into_bits(self) -> u8 {
        self as u8
    }

    /// The ring `bits` encodes; every two-bit value is one.
    ///
    /// # Panics
    ///
    /// If `bits` does not fit in two bits.
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0b00 => Ring::Zero,
            0b01 => Ring::One,
            0b10 => Ring::Two,
            0b11 => Ring::Three,
            _ => panic!("ring out of range"),
        }
    }
}

impl Display for Ring {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ring {}", self.into_bits())
    }
}

impl From<PrivilegeLevel> for Ring {
    /// The ring the kernel's two privilege levels live in. The other
    /// direction is not offered: rings 1 and 2 are neither.
    fn from(privilege_level: PrivilegeLevel) -> Self {
        match privilege_level {
            PrivilegeLevel::Kernel => Ring::Zero,
            PrivilegeLevel::User => Ring::Three,
        }
    }
}

/// What a system descriptor describes, as encoded in the four-bit type field
/// of a descriptor whose [`code_or_data`](Descriptor::code_or_data) bit is
/// clear.
///
/// Only the kinds laid out like a [`Descriptor`] are named. A call, interrupt
/// or trap gate also takes an encoding of this field, but replaces base and
/// limit with an entrypoint offset and a selector, so it is a different
/// structure that happens to share the byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SystemKind {
    /// Local descriptor table (`LDT`).
    Ldt = 0b0010,

    /// A 64-bit task state segment that is not currently loaded.
    ///
    /// The kind to build a TSS descriptor with; `ltr` turns it into
    /// [`TaskStateBusy`](SystemKind::TaskStateBusy) as it loads it.
    TaskStateAvailable = 0b1001,

    /// A 64-bit task state segment that is loaded on some core.
    ///
    /// `ltr` refuses a descriptor already in this state, which is what stops
    /// two cores from sharing one TSS.
    TaskStateBusy = 0b1011,
}

impl SystemKind {
    /// The hardware encoding of this kind.
    pub const fn into_bits(self) -> u8 {
        self as u8
    }

    /// The kind `bits` encodes, or `None` for an encoding that is reserved,
    /// legacy-only, or a gate.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0b0010 => Some(SystemKind::Ldt),
            0b1001 => Some(SystemKind::TaskStateAvailable),
            0b1011 => Some(SystemKind::TaskStateBusy),
            _ => None,
        }
    }
}

impl Display for SystemKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SystemKind::Ldt => write!(f, "LDT"),
            SystemKind::TaskStateAvailable => write!(f, "TSS (available)"),
            SystemKind::TaskStateBusy => write!(f, "TSS (busy)"),
        }
    }
}

/// A single GDT entry.
///
/// This is the eight bytes every entry begins with; a system descriptor adds
/// an [`Extension`] holding the top of its base.
///
/// The fields sit where the hardware puts them, which is not where a reader
/// would: the base is cut into three pieces here and a fourth in the
/// extension, the limit into two. [`base`](Descriptor::base) and
/// [`limit`](Descriptor::limit) put back together what this half holds, and
/// the constructors below cover the descriptors a long-mode kernel actually
/// builds.
///
/// The four bits of [`kind`](Descriptor::kind) mean one thing in a code/data
/// descriptor and another in a system one, which is what
/// [`code_or_data`](Descriptor::code_or_data) selects between; see
/// [`system_kind`](Descriptor::system_kind).
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct Descriptor {
    /// Bits [15:0] of the segment limit (bits [15:0]).
    #[bits(16)]
    limit_low: u16,

    /// Bits [15:0] of the segment base (bits [31:16]).
    #[bits(16)]
    base_low: u16,

    /// Bits [23:16] of the segment base (bits [39:32]).
    #[bits(8)]
    base_middle: u8,

    /// Descriptor type (bits [43:40]).
    ///
    /// For a code/data descriptor this is the accessed, read/write,
    /// direction/conforming and executable bits, from bit 40 upwards. For a
    /// system descriptor it is a [`SystemKind`].
    #[bits(4)]
    pub kind: u8,

    /// Descriptor class (`S`, bit 44).
    ///
    /// Set for a code or data segment, clear for a system descriptor — which
    /// is also what makes the descriptor sixteen bytes long instead of eight.
    #[bits(1)]
    pub code_or_data: bool,

    /// Descriptor privilege level (`DPL`, bits [46:45]).
    ///
    /// The least privileged ring that may load this descriptor.
    #[bits(2)]
    pub dpl: Ring,

    /// Present (`P`, bit 47).
    ///
    /// Loading a selector for a descriptor with this clear raises `#NP`.
    #[bits(1)]
    pub present: bool,

    /// Bits [19:16] of the segment limit (bits [51:48]).
    #[bits(4)]
    limit_high: u8,

    /// Available for software (`AVL`, bit 52) — the hardware never reads it.
    #[bits(1)]
    pub available: bool,

    /// 64-bit code segment (`L`, bit 53).
    ///
    /// Set on a code segment running 64-bit code, which requires
    /// [`default_size`](Descriptor::default_size) to be clear. Meaningless on
    /// anything else.
    #[bits(1)]
    pub long_mode: bool,

    /// Default operand and address size (`D`/`B`, bit 54).
    ///
    /// 32-bit when set, 16-bit when clear. Must be clear on a 64-bit code
    /// segment, where [`long_mode`](Descriptor::long_mode) decides instead.
    #[bits(1)]
    pub default_size: bool,

    /// Granularity (`G`, bit 55).
    ///
    /// Scales the limit by 4 KiB when set, leaving it in bytes when clear.
    #[bits(1)]
    pub granularity: bool,

    /// Bits [31:24] of the segment base (bits [63:56]).
    #[bits(8)]
    base_high: u8,
}

impl Descriptor {
    /// The null descriptor, which every GDT begins with.
    ///
    /// A selector pointing at it loads no segment; using one to access memory
    /// raises `#GP`.
    pub const NULL: Self = Self::new();

    /// A flat 64-bit code segment reachable from `ring`.
    ///
    /// Long mode ignores base and limit here, but they are filled in flat
    /// anyway, so the descriptor still describes all of memory to anything
    /// that does read them.
    ///
    /// The accessed bit is set from the start: the CPU writes it on the first
    /// load otherwise, which faults if the GDT is mapped read-only.
    pub const fn code(ring: Ring) -> Self {
        Self::new()
            .with_kind(0b1011)
            .with_code_or_data(true)
            .with_dpl(ring)
            .with_present(true)
            .with_long_mode(true)
            .with_granularity(true)
            .with_limit_low(LIMIT_MAX as u16)
            .with_limit_high((LIMIT_MAX >> 16) as u8)
    }

    /// A flat 32-bit code segment reachable from `ring`.
    ///
    /// The compatibility-mode counterpart of [`code`](Descriptor::code): `L`
    /// clear and `D` set, so code loaded through it runs 32-bit. A 64-bit
    /// kernel carries one because `sysret` insists on it — see [`Gdt`] for
    /// where it has to sit.
    pub const fn compatibility_code(ring: Ring) -> Self {
        Self::new()
            .with_kind(0b1011)
            .with_code_or_data(true)
            .with_dpl(ring)
            .with_present(true)
            .with_default_size(true)
            .with_granularity(true)
            .with_limit_low(LIMIT_MAX as u16)
            .with_limit_high((LIMIT_MAX >> 16) as u8)
    }

    /// A flat data segment reachable from `ring`.
    ///
    /// As [`code`](Descriptor::code), except that a data segment has no
    /// 64-bit form: it keeps `D` set, and the CPU ignores the rest of it.
    pub const fn data(ring: Ring) -> Self {
        Self::new()
            .with_kind(0b0011)
            .with_code_or_data(true)
            .with_dpl(ring)
            .with_present(true)
            .with_default_size(true)
            .with_granularity(true)
            .with_limit_low(LIMIT_MAX as u16)
            .with_limit_high((LIMIT_MAX >> 16) as u8)
    }

    /// The two halves of a descriptor for the task state segment of `size`
    /// bytes at `base`.
    ///
    /// This is the sixteen-byte shape, so it takes an [`Extension`] to hold
    /// the top of the base; the halves belong in the table in the order they
    /// are returned. The limit is a byte count — the TSS is far smaller than
    /// the 4 KiB a granular limit would round it to — and the descriptor is
    /// built available, for `ltr` to mark busy.
    ///
    /// # Panics
    ///
    /// If `size` is zero, or larger than the limit field can express.
    pub const fn task_state(base: u64, size: u32) -> (Self, Extension) {
        assert!(size > 0, "a task state segment cannot be empty");

        let mut descriptor = Self::new()
            .with_kind(SystemKind::TaskStateAvailable.into_bits())
            .with_code_or_data(false)
            .with_dpl(Ring::Zero)
            .with_present(true);

        descriptor.set_base(base as u32);
        descriptor.set_limit(size - 1);

        (descriptor, Extension::for_base(base))
    }

    /// The number of bytes this entry occupies in the table: eight for a code
    /// or data descriptor, sixteen for a system one, whose [`Extension`]
    /// accounts for the other eight.
    ///
    /// The GDT is packed, so this is also how far the next entry sits from
    /// the start of this one — and, divided by eight, what a selector of the
    /// following entry has to count.
    pub const fn size(&self) -> usize {
        match self.code_or_data() {
            true => 8,
            false => 16,
        }
    }

    /// The eight bytes of the descriptor as they belong in the table — for a
    /// system descriptor, an [`Extension`] follows them.
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Bits [31:0] of the segment base, reassembled from the three pieces the
    /// layout cuts them into. A system descriptor keeps the rest of the base
    /// in its [`Extension`].
    ///
    /// Only meaningful on a system descriptor: long mode treats the base of a
    /// code or data segment as zero regardless of what stands here, `fs` and
    /// `gs` excepted, whose bases live in an MSR instead.
    pub const fn base(&self) -> u32 {
        self.base_low() as u32 | (self.base_middle() as u32) << 16 | (self.base_high() as u32) << 24
    }

    /// Sets bits [31:0] of the segment base, splitting them across the three
    /// fields that hold them.
    ///
    /// A base above 4 GiB needs the other half of a system descriptor as
    /// well; [`Extension::for_base`] is the piece that carries it.
    pub const fn set_base(&mut self, base: u32) {
        self.set_base_low(base as u16);
        self.set_base_middle((base >> 16) as u8);
        self.set_base_high((base >> 24) as u8);
    }

    /// The segment limit, reassembled from its two fields.
    ///
    /// This is the raw field: with [`granularity`](Descriptor::granularity)
    /// set it counts 4 KiB units rather than bytes.
    pub const fn limit(&self) -> u32 {
        self.limit_low() as u32 | (self.limit_high() as u32) << 16
    }

    /// Sets the segment limit, splitting it across its two fields.
    ///
    /// # Panics
    ///
    /// If `limit` exceeds [`LIMIT_MAX`]; the field is 20 bits, and a larger
    /// segment needs [`granularity`](Descriptor::granularity) instead of a
    /// larger number.
    pub const fn set_limit(&mut self, limit: u32) {
        assert!(limit <= LIMIT_MAX, "segment limit out of range");

        self.set_limit_low(limit as u16);
        self.set_limit_high((limit >> 16) as u8);
    }

    /// What this system descriptor describes, or `None` if it is a code or
    /// data descriptor, or holds a type this module does not name.
    pub const fn system_kind(&self) -> Option<SystemKind> {
        match self.code_or_data() {
            true => None,
            false => SystemKind::from_bits(self.kind()),
        }
    }
}

impl Display for Descriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x{:016x} (", self.0)?;

        match self.code_or_data() {
            true => write!(
                f,
                "code/data, kind: 0b{:04b}, long mode: {}, default size: {}",
                self.kind(),
                self.long_mode(),
                self.default_size()
            )?,
            false => match self.system_kind() {
                Some(kind) => write!(f, "system, kind: {}", kind)?,
                None => write!(f, "system, kind: 0b{:04b}", self.kind())?,
            },
        }

        write!(
            f,
            ", {}, present: {}, base: 0x...{:08x}, limit: 0x{:05x}, \
             granularity: {}, available: {})",
            self.dpl(),
            self.present(),
            self.base(),
            self.limit(),
            self.granularity(),
            self.available()
        )
    }
}

/// The second half of a system descriptor.
///
/// A 64-bit base does not fit in the eight bytes of the legacy layout, so a
/// system descriptor carries the top of it here, in the eight bytes that
/// follow the [`Descriptor`] in the table. A code or data descriptor has no
/// such half.
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct Extension {
    /// Bits [63:32] of the segment base (bits [31:0]).
    #[bits(32)]
    pub base_upper: u32,

    /// Reserved (bits [63:32]). Must be zero.
    #[bits(32)]
    __: u32,
}

impl Extension {
    /// The extension a segment based at `base` needs: the top half of that
    /// address, the bottom half being the [`Descriptor`]'s to hold.
    pub const fn for_base(base: u64) -> Self {
        Self::new().with_base_upper((base >> 32) as u32)
    }

    /// The eight bytes of the extension as they belong in the table, directly
    /// behind the descriptor they complete.
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl Display for Extension {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "0x{:016x} (base: 0x{:08x}...)",
            self.0,
            self.base_upper()
        )
    }
}

/// The sixteen bytes a system descriptor occupies in the table: a
/// [`Descriptor`] and the [`Extension`] that completes it, in the order the
/// CPU reads them.
///
/// The two are one type here because a slice holds one: the entries a table
/// repeats — a task state descriptor per core — are these pairs, where a
/// slice of descriptors and a slice of extensions would have to interleave.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SystemEntry {
    /// The descriptor, which the eight bytes of the entry begin with.
    pub descriptor: Descriptor,

    /// The extension completing it, in the eight bytes behind it.
    pub extension: Extension,
}

impl SystemEntry {
    /// An entry no selector may load: a null descriptor and an empty
    /// extension, which is what an unused slot in the table holds.
    pub const NULL: Self = Self::new(Descriptor::NULL, Extension::new());

    /// The entry a descriptor and its extension make up.
    pub const fn new(descriptor: Descriptor, extension: Extension) -> Self {
        Self {
            descriptor,
            extension,
        }
    }

    /// The entry for the task state segment of `size` bytes at `base`: the
    /// two halves of [`Descriptor::task_state`], put together.
    ///
    /// # Panics
    ///
    /// If `size` is zero, or larger than the limit field can express.
    pub const fn task_state(base: u64, size: u32) -> Self {
        let (descriptor, extension) = Descriptor::task_state(base, size);

        Self::new(descriptor, extension)
    }
}

impl Display for SystemEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} + {}", self.descriptor, self.extension)
    }
}

/// The kernel's global descriptor table.
///
/// # Layout
///
/// Long mode leaves the GDT one ordering constraint, and it comes from
/// `syscall` and `sysret`, which derive three selectors from one entry each
/// instead of taking them as operands. The table is laid out for it:
///
/// | Index | Bytes | Entry                                  |
/// |-------|-------|----------------------------------------|
/// | 0     | 0x00  | the null descriptor                    |
/// | 1     | 0x08  | kernel code — `CS` after `syscall`     |
/// | 2     | 0x10  | kernel data — `SS` after `syscall`     |
/// | 3     | 0x18  | 32-bit compatibility user code         |
/// | 4     | 0x20  | user data — `SS` after `sysret`        |
/// | 5     | 0x28  | user code — `CS` after `sysretq`       |
/// | 6..   | 0x30  | a task state descriptor per core       |
///
/// `syscall` loads `CS` from the entry `IA32_STAR` names and `SS` from the
/// one behind it, which is why the kernel's data descriptor follows its code
/// descriptor. `sysret` counts from the entry `IA32_STAR` names for the
/// return: `SS` from the one behind it and `CS` from the one behind that,
/// the 32-bit form taking `CS` from the named entry itself. That is what
/// puts a compatibility-mode code segment below the user data and 64-bit
/// code descriptors, whether or not anything ever loads it.
///
/// The task state descriptors are the sixteen-byte kind, so each core's
/// takes two entries; they start out empty, for a core to fill in with the
/// address of its own segment.
///
/// # Lifetime
///
/// The table is sized from [`BOOTINFO`] and then stays as it is: a GDT is
/// loaded for as long as the machine runs, and the CPU walks it behind every
/// selector, so the allocation is never freed and the entries are handed out
/// as `&'static mut`. Writing through one of them writes the table the CPU
/// reads.
#[allow(unused)]
#[derive(Debug)]
pub struct Gdt {
    // Start address in memory.
    start: VirtualAddress<c_void>,

    // Size in bytes.
    size: usize,

    /// The null descriptor at index 0.
    null: &'static mut Descriptor,

    /// The kernel's code segment.
    kernel_code: &'static mut Descriptor,

    /// The kernel's data segment.
    kernel_data: &'static mut Descriptor,

    /// Userspace's 32-bit code segment, which only `sysret` cares about.
    user_code_compatibility: &'static mut Descriptor,

    /// Userspace's data segment.
    user_data: &'static mut Descriptor,

    /// Userspace's 64-bit code segment.
    user_code: &'static mut Descriptor,

    /// One task state descriptor per core, indexed by core id.
    task_states: &'static mut [SystemEntry],
}

// SAFETY: the fields borrow one heap allocation that is never freed and that
// nothing else owns, so a `Gdt` carries no reference into the core that built
// it. The borrows are exclusive, which is what [`GDT`] hands out under its
// lock.
unsafe impl Send for Gdt {}

pub static GDT: ThreadTicketlock<MaybeUninit<Gdt>> =
    ThreadTicketlock::new(Ticketlock::new(), MaybeUninit::zeroed());

/// The ten bytes `lgdt` reads: where the table starts and how far it reaches.
#[repr(C, packed)]
struct Gdtr {
    /// The offset of the last byte of the table, i.e. its size less one.
    limit: u16,

    /// The address the table starts at.
    base: u64,
}

impl Gdt {
    /// The number of code/data descriptors the table begins with, the null
    /// descriptor included.
    pub const SEGMENTS: usize = 6;

    /// Builds the kernel's table and publishes it as [`GDT`], for
    /// [`load`](Gdt::load) to install on each core.
    ///
    /// # Safety
    ///
    /// This must run exactly once, on the core that boots. A second call
    /// overwrites the published table while other cores may already be
    /// running on it, and leaks the one it replaces — the allocation behind a
    /// `Gdt` is never freed.
    ///
    /// # Panics
    ///
    /// If the table cannot be allocated.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub unsafe fn init<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<ThreadLevelID as LockId>::Level> + PreviousToken,
    {
        match Self::try_new(token) {
            Ok((g, token)) => {
                let (mut gdt, t) = GDT.acquire(token);
                gdt.write(g);
                gdt.release(t)
            }
            Err((error, _token)) => {
                panic!("Unable to initialise GDT: {}", error);
            }
        }
    }

    /// Points this core's `GDTR` at the table [`init`](Gdt::init) built.
    ///
    /// Every core runs this for itself: `lgdt` is a per-core register, while
    /// the table behind it is shared.
    ///
    /// The selectors already loaded keep the descriptors they were loaded
    /// with — the CPU caches those — so this takes effect at the next load of
    /// one. Nothing here reloads `cs` or the data segments.
    ///
    /// # Safety
    ///
    /// [`init`](Gdt::init) must have run, and its table must still be the one
    /// [`GDT`] holds: this reads the static as an initialised `Gdt`, which a
    /// zeroed one is not.
    ///
    /// # Panics
    ///
    /// If the table is larger than a `GDTR` limit can express, which takes
    /// more cores than 64 KiB of entries hold.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub unsafe fn load<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<ThreadLevelID as LockId>::Level> + PreviousToken,
    {
        const KERNEL_CODE: u16 = 0x08;
        const KERNEL_DATA: u16 = 0x10;

        let (gdt, token) = GDT.acquire(token);

        // SAFETY: the caller guarantees that `init` has published a table.
        unsafe {
            let gdt = gdt.assume_init_ref();

            assert!(
                gdt.size <= u16::MAX as usize + 1,
                "the GDT is larger than a `GDTR` limit can express"
            );

            let gdtr = Gdtr {
                // The limit addresses the last byte of the table, so it is
                // one below its size: a table of `size` bytes ends there.
                limit: (gdt.size - 1) as u16,
                base: gdt.start.as_ptr() as u64,
            };

            asm!(
                // 1. Reload GDT.
                "lgdt [{gdtr}]",

                // 2. Reload CS.
                "push {code_sel}",
                "lea {tmp}, [rip + 2f]",
                "push {tmp}",
                "retfq",
                "2:",

                // 3. Reload ds/es/ss/fs/gs.
                "mov ds, {data_sel:x}",
                "mov es, {data_sel:x}",
                "mov ss, {data_sel:x}",

                gdtr     = in(reg) &gdtr,
                code_sel = in(reg) KERNEL_CODE as u64,
                data_sel = in(reg) KERNEL_DATA,
                tmp      = lateout(reg) _,
                options(nostack, preserves_flags),
            );
        }

        gdt.release(token)
    }

    /// The table for the core count in [`BOOTINFO`], allocated on the kernel
    /// [`Heap`].
    ///
    /// Shorthand for [`try_new_in`](Gdt::try_new_in) with the allocator and
    /// the core count the kernel runs with.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the heap cannot serve the table.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_new<Token>(token: Token) -> Result<(Self, Token), (Error, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        // SAFETY: the bootloader fills the boot information before the kernel
        // is entered, and nothing writes it afterwards.
        let bootinfo = unsafe { BOOTINFO.assume_init_ref() };

        Self::try_new_in(bootinfo.num_cpus, Heap, token)
    }

    /// The table for `num_cpus` cores, allocated from `alloc`.
    ///
    /// The block is never given back, so the allocator is not kept: see the
    /// [type documentation](Gdt).
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `alloc` cannot serve the table.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    ///
    /// # Panics
    ///
    /// If a table for `num_cpus` cores is too large to lay out at all.
    pub fn try_new_in<Alloc, Token>(
        num_cpus: usize,
        alloc: Alloc,
        token: Token,
    ) -> Result<(Self, Token), (Error, Token)>
    where
        Alloc: Allocator<MemoryManagementLevelID>,
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let (layout, task_states_offset) = Self::layout(num_cpus);

        let (table, token) = match alloc.allocate(layout, token) {
            Ok((table, token)) => (table.as_ptr(), token),
            Err(error) => return Err(error),
        };
        let start = VirtualAddress::new(table.cast());
        let size = layout.size();

        let segments = table.cast::<Descriptor>();

        let contents = [
            Descriptor::NULL,
            Descriptor::code(Ring::Zero),
            Descriptor::data(Ring::Zero),
            Descriptor::compatibility_code(Ring::Three),
            Descriptor::data(Ring::Three),
            Descriptor::code(Ring::Three),
        ];

        for (index, descriptor) in contents.iter().enumerate() {
            // SAFETY: the block begins with `SEGMENTS` descriptors, and the
            // array holds exactly that many, so every write lands in one of
            // them. Nothing has read the block before, it being fresh.
            unsafe { segments.add(index).write(*descriptor) };
        }

        // SAFETY: the layout puts the task state entries at this offset,
        // inside the same block, aligned for a `SystemEntry`.
        let task_states = unsafe { table.add(task_states_offset) }.cast::<SystemEntry>();

        for index in 0..num_cpus {
            // SAFETY: the block holds `num_cpus` entries from here on, so
            // every write stays in it — and initialises the entry a borrow is
            // taken of below.
            unsafe { task_states.add(index).write(SystemEntry::NULL) };
        }

        // SAFETY: the six descriptors and the `num_cpus` entries were all
        // written just above, so every borrow addresses an initialised entry;
        // no two of them address the same one, since each sits at its own
        // offset in the block; and the block is never freed, which is what
        // makes them `'static`.
        let gdt = unsafe {
            Self {
                start,
                size,
                null: &mut *segments.add(0),
                kernel_code: &mut *segments.add(1),
                kernel_data: &mut *segments.add(2),
                user_code_compatibility: &mut *segments.add(3),
                user_data: &mut *segments.add(4),
                user_code: &mut *segments.add(5),
                task_states: slice::from_raw_parts_mut(task_states, num_cpus),
            }
        };

        Ok((gdt, token))
    }

    /// The layout of the table for `num_cpus` cores, and the offset the task
    /// state descriptors start at.
    ///
    /// # Panics
    ///
    /// If the table cannot be laid out, which takes a core count whose
    /// entries alone overrun the address space.
    pub fn layout(num_cpus: usize) -> (Layout, usize) {
        let segments = Layout::array::<Descriptor>(Self::SEGMENTS)
            .expect("the GDT segments do not fit a layout");
        let task_states = Layout::array::<SystemEntry>(num_cpus)
            .expect("the task state descriptors do not fit a layout");

        segments
            .extend(task_states)
            .expect("the GDT does not fit a layout")
    }
}

/// A value loaded into a segment register (`cs`, `ss`, ...): which GDT
/// descriptor to use, and at what privilege.
///
/// Sixteen bits wide, matching the segment registers it is loaded into.
#[bitfield(u16)]
#[derive(PartialEq, Eq)]
pub struct SegmentSelector {
    /// Requested Privilege Level (bits [1:0]).
    ///
    /// The privilege level the selector is requested at. When loading `cs`,
    /// the RPL must match the code segment descriptor's DPL (via a call
    /// gate) or the current CPL. For data segments, the effective privilege
    /// used for access checks is `max(RPL, CPL)`.
    #[bits(2)]
    pub rpl: u8,

    /// Table Indicator (bit 2).
    ///
    /// `false` = GDT, `true` = LDT. BusyOS does not use an LDT, so this
    /// should always be `false`.
    #[bits(1)]
    pub table_indicator: bool,

    /// Index (bits [15:3]).
    ///
    /// The index of the descriptor within the GDT (or LDT), counted in
    /// units of 8-byte descriptor slots — NOT a byte offset. To get the
    /// byte offset into the table, multiply by 8 (equivalent to this
    /// field already being pre-shifted into the selector's bit position).
    #[bits(13)]
    pub index: u16,
}

impl SegmentSelector {
    /// A selector for the GDT descriptor at `index`, requested at `rpl`.
    pub const fn create(index: u16, rpl: Ring) -> Self {
        Self::new()
            .with_index(index)
            .with_table_indicator(false)
            .with_rpl(rpl as u8)
    }

    /// The kernel code segment: [`Gdt`]'s descriptor 1, ring 0.
    pub const KERNEL_CODE: Self = Self::create(1, Ring::Zero);

    /// The kernel data segment: [`Gdt`]'s descriptor 2, ring 0.
    pub const KERNEL_DATA: Self = Self::create(2, Ring::Zero);

    /// The user data segment: [`Gdt`]'s descriptor 4, ring 3.
    ///
    /// Descriptor 3, the 32-bit compatibility code segment, has no selector
    /// here — BusyOS runs no compatibility-mode code.
    pub const USER_DATA: Self = Self::create(4, Ring::Three);

    /// The user code segment: [`Gdt`]'s descriptor 5, ring 3.
    pub const USER_CODE: Self = Self::create(5, Ring::Three);
}

#[cfg(test)]
mod test {
    use crate::{
        kernel::locking::{EpilogueLevel, RootToken},
        utils::testing::HeapAllocator,
    };

    use super::*;

    extern crate std;

    /// Builds a table for `num_cpus` cores from the test allocator, panicking
    /// if it cannot be served. A macro rather than a function because the
    /// error arm cannot be unwrapped: a token is not [`Debug`].
    macro_rules! gdt {
        ($num_cpus:expr, $token:expr) => {
            match Gdt::try_new_in($num_cpus, HeapAllocator, $token) {
                Ok((gdt, token)) => (gdt, token),
                Err(_) => panic!("the test allocator could not serve the table"),
            }
        };
    }

    /// The encodings a long-mode GDT is expected to hold, byte for byte.
    #[test]
    fn the_flat_segments_encode_as_the_architecture_spells_them() {
        assert_eq!(Descriptor::code(Ring::Zero).raw(), 0x00af9b000000ffff);
        assert_eq!(Descriptor::data(Ring::Zero).raw(), 0x00cf93000000ffff);
        assert_eq!(Descriptor::code(Ring::Three).raw(), 0x00affb000000ffff);
        assert_eq!(Descriptor::data(Ring::Three).raw(), 0x00cff3000000ffff);
    }

    #[test]
    fn the_null_descriptor_is_zero() {
        assert_eq!(Descriptor::NULL.raw(), 0);
    }

    /// A code segment is 64-bit through `L`, and `D` has to stay clear for it.
    #[test]
    fn a_code_segment_is_a_64_bit_one() {
        let code = Descriptor::code(Ring::Zero);

        assert!(code.long_mode());
        assert!(!code.default_size());
    }

    /// The accessed bit is part of the descriptor, so that the first load of
    /// it does not have to write the table.
    #[test]
    fn the_flat_segments_are_accessed_from_the_start() {
        assert_eq!(Descriptor::code(Ring::Zero).kind() & 0b0001, 0b0001);
        assert_eq!(Descriptor::data(Ring::Zero).kind() & 0b0001, 0b0001);
    }

    /// The four pieces the base is cut into come back as the number that went
    /// in, the fourth from the half that carries it.
    #[test]
    fn a_base_survives_being_split_across_the_two_halves() {
        let mut descriptor = Descriptor::new();
        descriptor.set_base(0x1234_5678);

        assert_eq!(descriptor.base(), 0x1234_5678);
        assert_eq!(descriptor.base_low(), 0x5678);
        assert_eq!(descriptor.base_middle(), 0x34);
        assert_eq!(descriptor.base_high(), 0x12);

        let extension = Extension::for_base(0xdead_beef_1234_5678);

        assert_eq!(extension.base_upper(), 0xdead_beef);
    }

    /// A base that fits in the descriptor leaves the extension with nothing
    /// to carry, and its reserved half stays empty either way.
    #[test]
    fn an_extension_holds_only_the_top_of_a_base() {
        assert_eq!(Extension::for_base(0xffff_ffff).raw(), 0);
        assert_eq!(
            Extension::for_base(0xdead_beef_1234_5678).raw(),
            0xdead_beef
        );
    }

    #[test]
    fn a_limit_survives_being_split_across_the_layout() {
        let mut descriptor = Descriptor::new();
        descriptor.set_limit(LIMIT_MAX);

        assert_eq!(descriptor.limit(), LIMIT_MAX);
        assert_eq!(descriptor.limit_low(), 0xffff);
        assert_eq!(descriptor.limit_high(), 0xf);
    }

    #[test]
    #[should_panic(expected = "segment limit out of range")]
    fn a_limit_the_field_cannot_hold_is_refused() {
        Descriptor::new().set_limit(LIMIT_MAX + 1);
    }

    /// A TSS descriptor holds the address of the segment and the last byte
    /// offset in it, ungranular.
    #[test]
    fn a_task_state_descriptor_describes_the_segment_it_is_given() {
        let (descriptor, extension) = Descriptor::task_state(0xffff_8000_0001_0000, 104);

        assert_eq!(descriptor.base(), 0x0001_0000);
        assert_eq!(extension.base_upper(), 0xffff_8000);
        assert_eq!(descriptor.limit(), 103);
        assert!(!descriptor.granularity());
        assert!(descriptor.present());
        assert_eq!(descriptor.dpl(), Ring::Zero);
        assert_eq!(
            descriptor.system_kind(),
            Some(SystemKind::TaskStateAvailable)
        );
    }

    /// Which half of the descriptor is real depends on its class, and the
    /// table is packed accordingly.
    #[test]
    fn a_system_descriptor_is_twice_the_size_of_a_segment_one() {
        assert_eq!(Descriptor::code(Ring::Zero).size(), 8);
        assert_eq!(Descriptor::data(Ring::Zero).size(), 8);
        assert_eq!(Descriptor::task_state(0, 104).0.size(), 16);
    }

    /// The type field only names a system kind when it is one.
    #[test]
    fn a_segment_descriptor_has_no_system_kind() {
        assert_eq!(Descriptor::code(Ring::Zero).system_kind(), None);
        assert_eq!(
            Descriptor::new().with_kind(0b0000).system_kind(),
            None,
            "a reserved system type names nothing"
        );
    }

    /// A compatibility-mode code segment is the 32-bit one `sysret` wants:
    /// `D` set where a 64-bit code segment has `L`.
    #[test]
    fn a_compatibility_code_segment_is_a_32_bit_one() {
        let code = Descriptor::compatibility_code(Ring::Three);

        assert!(!code.long_mode());
        assert!(code.default_size());
        assert_eq!(code.dpl(), Ring::Three);
        assert_eq!(code.raw(), 0x00cffb000000ffff);
    }

    /// The two halves of a task state descriptor, as the table holds them.
    #[test]
    fn a_system_entry_is_the_sixteen_bytes_of_a_system_descriptor() {
        assert_eq!(size_of::<SystemEntry>(), 16);
        assert_eq!(align_of::<SystemEntry>(), align_of::<Descriptor>());

        let entry = SystemEntry::task_state(0xffff_8000_0001_0000, 104);

        assert_eq!(entry.descriptor.base(), 0x0001_0000);
        assert_eq!(entry.extension.base_upper(), 0xffff_8000);
        assert_eq!(entry.descriptor.size(), 16);
    }

    /// The table is the six segment descriptors and sixteen bytes per core,
    /// with the cores' entries behind the segments.
    #[test]
    fn the_table_is_sized_for_the_cores_it_serves() {
        let (layout, task_states) = Gdt::layout(4);

        assert_eq!(layout.size(), 6 * 8 + 4 * 16);
        assert_eq!(layout.align(), align_of::<Descriptor>());
        assert_eq!(task_states, 6 * 8);
    }

    /// Every entry the layout names is the descriptor the kernel expects to
    /// find behind its selector.
    #[test]
    fn a_table_holds_the_segments_syscall_and_sysret_need() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (gdt, token) = gdt!(2, token);

        assert_eq!(*gdt.null, Descriptor::NULL);
        assert_eq!(*gdt.kernel_code, Descriptor::code(Ring::Zero));
        assert_eq!(*gdt.kernel_data, Descriptor::data(Ring::Zero));
        assert_eq!(
            *gdt.user_code_compatibility,
            Descriptor::compatibility_code(Ring::Three)
        );
        assert_eq!(*gdt.user_data, Descriptor::data(Ring::Three));
        assert_eq!(*gdt.user_code, Descriptor::code(Ring::Three));

        level.leave(token);
    }

    /// A core gets an entry of its own, and it starts out loadable by no one.
    #[test]
    fn a_table_holds_an_empty_task_state_descriptor_per_core() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (gdt, token) = gdt!(3, token);

        assert_eq!(gdt.task_states.len(), 3);
        assert!(
            gdt.task_states
                .iter()
                .all(|entry| *entry == SystemEntry::NULL)
        );

        level.leave(token);
    }

    /// The entries are one packed table rather than seven objects: each sits
    /// where a selector counting entries from the null descriptor finds it,
    /// and writing through a borrow writes the table itself.
    #[test]
    fn the_entries_are_one_packed_table() {
        let root = unsafe { RootToken::forge() };
        let (level, token) = EpilogueLevel::enter(root);

        let (gdt, token) = gdt!(2, token);

        let base = (&*gdt.null as *const Descriptor).addr();
        let offset = |descriptor: &Descriptor| (descriptor as *const Descriptor).addr() - base;

        assert_eq!(offset(gdt.kernel_code), 0x08);
        assert_eq!(offset(gdt.kernel_data), 0x10);
        assert_eq!(offset(gdt.user_code_compatibility), 0x18);
        assert_eq!(offset(gdt.user_data), 0x20);
        assert_eq!(offset(gdt.user_code), 0x28);
        assert_eq!(offset(&gdt.task_states[0].descriptor), 0x30);
        assert_eq!(offset(&gdt.task_states[1].descriptor), 0x40);

        let entry = SystemEntry::task_state(0xffff_8000_0001_0000, 104);
        gdt.task_states[1] = entry;

        // SAFETY: the entry is sixteen bytes of the table this borrow of the
        // null descriptor points into, eight of them a descriptor.
        let raw = unsafe { (&*gdt.null as *const Descriptor).add(8).read() };

        assert_eq!(raw, entry.descriptor);

        level.leave(token);
    }

    #[test]
    fn the_kernel_s_privilege_levels_are_the_outer_two_rings() {
        assert_eq!(Ring::from(PrivilegeLevel::Kernel), Ring::Zero);
        assert_eq!(Ring::from(PrivilegeLevel::User), Ring::Three);
    }

    #[test]
    fn rings_round_trip_through_their_encoding() {
        for ring in [Ring::Zero, Ring::One, Ring::Two, Ring::Three] {
            assert_eq!(Ring::from_bits(ring.into_bits()), ring);
        }
    }
}
