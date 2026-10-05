//! x86_64 implementation of the generic [`CPU`](crate::arch::generic::cpu::CPU)
//! interface.

use crate::arch::generic::cpu::{InterruptFlag, Register};
use crate::arch::generic::paging::VirtualAddress;
use crate::arch::x86_64::gdt::SegmentSelector;
use crate::arch::x86_64::paging::{CR2, CR3};
use crate::arch::x86_64::rflags::RFLAGS;
use core::arch::asm;
use core::ffi::c_void;
use core::fmt::{Debug, Display, Formatter, Result as FmtResult};
use core::num::TryFromIntError;
use core::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};

/// The value `%rbp` holds in the outermost frame of every frame-pointer
/// chain, so that [`unwind`](crate::arch::generic::cpu::CPU::unwind) knows
/// where to stop.
///
/// `head.S` puts it into the first frame on each core's boot stack, and
/// `entry.S` into `%rbp` before calling
/// `__interrupt_handler`, so that a walk from inside an interrupt ends at
/// the interrupt entry instead of continuing into the code it interrupted.
///
/// A non-canonical address on purpose: anything that dereferences it by
/// mistake faults with `#GP` rather than reading memory. It also lies below
/// every kernel address, which is what `__unwind` checks.
#[unsafe(no_mangle)]
pub static CALL_STACK_END_MARKER: u64 = 0xdeadcafedeadc0de;

// Assembled by `build.rs` into the kernel binary only, so the bootloader,
// which links this crate as a library, and the host tests do without it.
#[cfg(all(not(test), not(feature = "library")))]
unsafe extern "C" {
    /// Walks the frame-pointer chain from the caller's frame outwards,
    /// storing one return address per frame into the `len` entries at
    /// `call_stack`, and returns how many it stored. See `unwind.S`.
    fn __unwind(call_stack: *mut usize, len: usize) -> usize;
}

#[derive(Debug)]
pub struct CPU;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct CPUID(u8);

#[derive(Debug)]
#[repr(C)]
pub struct State {
    rax: Register<u64>,
    rbx: Register<u64>,
    rcx: Register<u64>,
    rdx: Register<u64>,
    rdi: Register<u64>,
    rsi: Register<u64>,
    rsp: Register<u64>,
    rbp: Register<u64>,
    r8: Register<u64>,
    r9: Register<u64>,
    r10: Register<u64>,
    r11: Register<u64>,
    r12: Register<u64>,
    r13: Register<u64>,
    r14: Register<u64>,
    r15: Register<u64>,
    rflags: RFLAGS,
    cr2: CR2,
    cr3: CR3,
    cs: SegmentSelector,
    ss: SegmentSelector,
    fs: u64,
    gs: u64,
}

unsafe impl Send for State {}

unsafe impl Sync for State {}

const _: () = {
    use core::mem::offset_of;
    assert!(offset_of!(State, rax) == 0x00);
    assert!(offset_of!(State, rbx) == 0x08);
    assert!(offset_of!(State, rcx) == 0x10);
    assert!(offset_of!(State, rdx) == 0x18);
    assert!(offset_of!(State, rdi) == 0x20);
    assert!(offset_of!(State, rsi) == 0x28);
    assert!(offset_of!(State, rsp) == 0x30);
    assert!(offset_of!(State, rbp) == 0x38);
    assert!(offset_of!(State, r8) == 0x40);
    assert!(offset_of!(State, r9) == 0x48);
    assert!(offset_of!(State, r10) == 0x50);
    assert!(offset_of!(State, r11) == 0x58);
    assert!(offset_of!(State, r12) == 0x60);
    assert!(offset_of!(State, r13) == 0x68);
    assert!(offset_of!(State, r14) == 0x70);
    assert!(offset_of!(State, r15) == 0x78);
    assert!(offset_of!(State, rflags) == 0x80);
    assert!(offset_of!(State, cr2) == 0x88);
    assert!(offset_of!(State, cr3) == 0x90);
    assert!(offset_of!(State, cs) == 0x98);
    assert!(offset_of!(State, ss) == 0x9A);
    assert!(offset_of!(State, fs) == 0xA0);
    assert!(offset_of!(State, gs) == 0xA8);
};

impl Display for State {
    // The general-purpose registers go through one format site in a loop,
    // and the rest through a handful of small ones. A single `write!` with
    // every register builds an argument array that alone takes more than a
    // kilobyte of stack, which the panic path cannot spare.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        const GENERAL: [(&str, fn(&State) -> Register<u64>); 16] = [
            ("rax", |s| s.rax),
            ("rbx", |s| s.rbx),
            ("rcx", |s| s.rcx),
            ("rdx", |s| s.rdx),
            ("rdi", |s| s.rdi),
            ("rsi", |s| s.rsi),
            ("rbp", |s| s.rbp),
            ("rsp", |s| s.rsp),
            ("r8 ", |s| s.r8),
            ("r9 ", |s| s.r9),
            ("r10", |s| s.r10),
            ("r11", |s| s.r11),
            ("r12", |s| s.r12),
            ("r13", |s| s.r13),
            ("r14", |s| s.r14),
            ("r15", |s| s.r15),
        ];

        for pair in GENERAL.chunks(2) {
            let ((left, left_value), (right, right_value)) = (pair[0], pair[1]);
            writeln!(
                f,
                "{left}: 0x{:x} {right}: 0x{:x}",
                left_value(self),
                right_value(self)
            )?;
        }

        writeln!(f, "rflags: {}", self.rflags)?;
        writeln!(f, "cr2: {} cr3: {}", self.cr2, self.cr3)?;
        writeln!(f, "cs: {}", self.cs)?;
        writeln!(f, "ss: {}", self.ss)?;
        write!(f, "fs: 0x{:016x}", self.fs)?;
        write!(f, "gs: 0x{:016x}", self.gs)
    }
}

/// Id of the core the firmware starts the kernel on.
///
/// `EFI_MP_SERVICES_PROTOCOL` numbers the bootstrap processor zero, and the
/// bootloader passes that number through, so a core can tell whether it is the
/// one that has to do the work that happens once.
pub const BOOT_CPUID: CPUID = CPUID(0);

impl Display for CPUID {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<usize> for CPUID {
    type Error = TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        let raw = u8::try_from(value)?;
        Ok(Self(raw))
    }
}

impl Into<usize> for CPUID {
    fn into(self) -> usize {
        self.0 as _
    }
}

impl crate::arch::generic::cpu::CPU for CPU {
    /// Returns whether maskable interrupts are currently enabled.
    ///
    /// Determined by the `IF` bit of the [`RFLAGS`] register.
    #[inline]
    fn interrupt_flag() -> InterruptFlag {
        if RFLAGS::read().interrupt() {
            InterruptFlag::Enabled
        } else {
            InterruptFlag::Disabled
        }
    }

    /// Enables maskable interrupts by executing `STI`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the interrupt handling infrastructure (IDT,
    /// per-CPU state, ...) is fully initialized and that enabling interrupts
    /// does not break the invariants of an enclosing critical section.
    #[inline]
    unsafe fn raw_enable_interrupts() {
        unsafe {
            core::arch::asm!("sti", options(nostack, preserves_flags));
        }
    }

    /// Disables maskable interrupts by executing `CLI`.
    ///
    /// # Safety
    ///
    /// The caller is responsible for restoring the previous interrupt state,
    /// otherwise interrupts stay masked indefinitely.
    #[inline]
    unsafe fn raw_disable_interrupts() {
        unsafe {
            core::arch::asm!("cli", options(nostack, preserves_flags));
        }
    }

    /// Masks maskable interrupts with `CLI` and halts with `HLT`, forever.
    ///
    /// `HLT` alone is not the end: a non-maskable interrupt, a system
    /// management interrupt or an INIT/SIPI still wakes the core with
    /// interrupts masked, and execution would carry on after the
    /// instruction. So the `HLT` sits in a loop.
    #[inline]
    unsafe fn halt() -> ! {
        loop {
            unsafe {
                core::arch::asm!("cli", "hlt", options(nomem, nostack, preserves_flags));
            }
        }
    }

    /// Required minimum stack alignment.
    const STACK_ALIGNMENT: usize = 16;

    /// Kernel stack size.
    const KERNEL_STACK_SIZE: usize = 16 * 1024;

    type CPUID = CPUID;

    const CPUID_BITS: usize = 256;

    type State = State;

    #[inline(never)]
    fn state(state: &mut Self::State) {
        let ptr = state as *mut Self::State;
        unsafe {
            asm!(
                "mov [{p} + 0x00], rax",
                "mov [{p} + 0x08], rbx",
                "mov [{p} + 0x10], rcx",
                "mov [{p} + 0x18], rdx",
                "mov [{p} + 0x20], rdi",
                "mov [{p} + 0x28], rsi",
                "mov [{p} + 0x30], rsp",
                "mov [{p} + 0x38], rbp",
                "mov [{p} + 0x40], r8",
                "mov [{p} + 0x48], r9",
                "mov [{p} + 0x50], r10",
                "mov [{p} + 0x58], r11",
                "mov [{p} + 0x60], r12",
                "mov [{p} + 0x68], r13",
                "mov [{p} + 0x70], r14",
                "mov [{p} + 0x78], r15",

                // RFLAGS nur über den Stack erreichbar.
                "pushfq",
                "pop rax",
                "mov [{p} + 0x80], rax",

                "mov rax, cr2",
                "mov [{p} + 0x88], rax",
                "mov rax, cr3",
                "mov [{p} + 0x90], rax",

                // 16-Bit-Selektoren, direkt als Wort geschrieben.
                "mov ax, cs",
                "mov [{p} + 0x98], ax",
                "mov ax, ss",
                "mov [{p} + 0x9A], ax",

                // FS.BASE (MSR 0xC000_0100): rdmsr erwartet die
                // MSR-Nummer in ecx und liefert low in eax, high in edx.
                "mov ecx, 0xC0000100",
                "rdmsr",
                "shl rdx, 32",
                "or  rax, rdx",
                "mov [{p} + 0xA0], rax",

                // GS.BASE (MSR 0xC000_0101).
                //
                // Nach einem swapgs liegt die aktive Basis in
                // KERNEL_GS_BASE (0xC000_0102); hier wird immer
                // GS.BASE gelesen, unabhängig vom swapgs-Zustand.
                "mov ecx, 0xC0000101",
                "rdmsr",
                "shl rdx, 32",
                "or  rax, rdx",
                "mov [{p} + 0xA8], rax",
                p = in(reg) ptr,
                out("rax") _,
                out("rcx") _,
                out("rdx") _,
                // Neither `nostack` nor `preserves_flags`: `pushfq` uses the
                // stack, and `shl` and `or` change the status flags.
            );
        }
    }

    /// Walks the chain through `__unwind`, see `unwind.S`.
    ///
    /// Never inlined: `__unwind` starts at the frame of whoever called it,
    /// which is this function's own, so the first entry is the return
    /// address into this function's caller. Inlined, this frame would be
    /// gone and the first entry would silently skip a level.
    ///
    /// In the bootloader and the host tests, which have no `__unwind`, this
    /// records nothing.
    #[inline(never)]
    fn unwind(call_stack: &mut [VirtualAddress<c_void>]) -> &mut [VirtualAddress<c_void>] {
        #[cfg(all(not(test), not(feature = "library")))]
        // SAFETY: `VirtualAddress` is `#[repr(transparent)]` over a pointer,
        // so the entries can be written as `usize`, and `__unwind` writes at
        // most `call_stack.len()` of them and only reads the current stack.
        let len = unsafe { __unwind(call_stack.as_mut_ptr().cast(), call_stack.len()) };

        #[cfg(any(test, feature = "library"))]
        let len = 0;

        &mut call_stack[..len]
    }
}

impl CPU {
    /// Reads the current `GS` base address using `RDGSBASE`.
    ///
    /// # Safety
    ///
    /// `RDGSBASE` raises `#UD` unless `CR4.FSGSBASE` is set, which in turn
    /// requires the processor to support
    /// [`fsgsbase`](crate::arch::x86_64::cpuid::StructuredExtendedFeatureEBX::fsgsbase).
    #[inline]
    pub unsafe fn gs_base() -> usize {
        let base: usize;

        unsafe {
            core::arch::asm!(
                "rdgsbase {}",
                out(reg) base,
                options(nomem, nostack, preserves_flags)
            );
        }

        base
    }

    /// Sets the `GS` base address to `base` using `WRGSBASE`.
    ///
    /// # Safety
    ///
    /// As for [`gs_base`](CPU::gs_base), `WRGSBASE` requires `CR4.FSGSBASE`
    /// to be set.
    ///
    /// Additionally, `GS` is the anchor of this core's core-local storage: a
    /// base that does not point at this core's block makes every subsequent
    /// [`PerCPU`](crate::kernel::core_local::PerCPU) access read or write
    /// unrelated memory. `base` must be canonical, otherwise the write
    /// raises `#GP`.
    #[inline]
    pub unsafe fn set_gs_base(base: usize) {
        unsafe {
            core::arch::asm!(
                "wrgsbase {}",
                in(reg) base,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
}

/// The x86_64 architectural exceptions, numbered by the vector the CPU
/// raises them on.
///
/// Vectors without a variant here (`15`, `22`–`27`) are reserved by Intel/AMD
/// and never raised.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exceptions {
    DivisionError = 0,
    Debug = 1,
    NMI = 2,
    Breakpoint = 3,
    Overflow = 4,
    BoundRangeExceeded = 5,
    InvalidOpcode = 6,
    DeviceNotAvailable = 7,
    DoubleFault = 8,
    CoprocessorSegmentOverrun = 9,
    InvalidTSS = 10,
    SegmentNotPresent = 11,
    StackSegmentFault = 12,
    GeneralProtectionFault = 13,
    PageFault = 14,
    #[allow(non_camel_case_types)]
    x87FloatingPointException = 16,
    AlignmentCheck = 17,
    MachineCheck = 18,
    SimdFloatingPointException = 19,
    VirtualizationException = 20,
    ControlProtectionException = 21,
    HypervisorInjectionException = 28,
    VmmCommunicationException = 29,
    SecurityException = 30,
}

impl Display for Exceptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Exceptions::DivisionError => write!(f, "Division Error"),
            Exceptions::Debug => write!(f, "Debug Exception"),
            Exceptions::NMI => write!(f, "Non-maskable Interrupt"),
            Exceptions::Breakpoint => write!(f, "Breakpoint"),
            Exceptions::Overflow => write!(f, "Overflow Exception"),
            Exceptions::BoundRangeExceeded => write!(f, "Bound Range Exceeded"),
            Exceptions::InvalidOpcode => write!(f, "Invalid Opcode"),
            Exceptions::DeviceNotAvailable => write!(f, "Device not Available"),
            Exceptions::DoubleFault => write!(f, "Double Fault"),
            Exceptions::CoprocessorSegmentOverrun => write!(f, "Coprocessor Segment Overrun"),
            Exceptions::InvalidTSS => write!(f, "Invalid TSS"),
            Exceptions::SegmentNotPresent => write!(f, "Segment not Present"),
            Exceptions::StackSegmentFault => write!(f, "Stack Segment Fault"),
            Exceptions::GeneralProtectionFault => write!(f, "General Protection Fault"),
            Exceptions::PageFault => write!(f, "Page Fault"),
            Exceptions::x87FloatingPointException => write!(f, "x87 Floating Point Exception"),
            Exceptions::AlignmentCheck => write!(f, "Alignment Check"),
            Exceptions::MachineCheck => write!(f, "Machine Check"),
            Exceptions::SimdFloatingPointException => write!(f, "Simd Floating-Point Exception"),
            Exceptions::VirtualizationException => write!(f, "Virtualization Exception"),
            Exceptions::ControlProtectionException => write!(f, "Control-Protection Exception"),
            Exceptions::HypervisorInjectionException => write!(f, "Hypervisor-Injection Exception"),
            Exceptions::VmmCommunicationException => write!(f, "VMM Communication Exception"),
            Exceptions::SecurityException => write!(f, "Security Exception"),
        }
    }
}

/// x86_64's [`InterruptVector`](crate::arch::generic::cpu::InterruptVector):
/// a vector number as the CPU and `entry.S` see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterruptVector(u8);

impl Into<usize> for InterruptVector {
    fn into(self) -> usize {
        self.0 as _
    }
}

impl crate::arch::generic::cpu::InterruptVector for InterruptVector {
    /// The IDT holds 256 gates, and a vector is the `u8` that indexes it.
    const MAX_NUM: usize = 256;

    type Raw = u8;

    fn into_raw(self) -> Self::Raw {
        self.0
    }

    fn from_raw(raw: Self::Raw) -> Self {
        Self(raw)
    }

    fn is_division_by_zero(&self) -> bool {
        self.0 == Exceptions::DivisionError as _
    }

    fn is_breakpoint(&self) -> bool {
        self.0 == Exceptions::Breakpoint as _
    }

    fn is_invalid_instruction(&self) -> bool {
        self.0 == Exceptions::InvalidOpcode as _
    }

    fn is_page_fault(&self) -> bool {
        self.0 == Exceptions::PageFault as _
    }

    fn is_non_maskable(&self) -> bool {
        self.0 == Exceptions::NMI as _
    }

    fn is_invalid_alignmnet(&self) -> bool {
        self.0 == Exceptions::AlignmentCheck as _
    }

    /// Hands out the next unused vector, counting up from 32.
    ///
    /// Vectors 0 to 31 are the architecturally defined exceptions, so the
    /// first one software may claim is 32, and the counter starts there. It
    /// only ever moves forward, so every caller gets a vector of its own.
    ///
    /// The last vector handed out is 254. Reaching 255 is what marks the
    /// counter as exhausted, so 255 itself is never given to a driver, and
    /// it stays free to serve as the local APIC's spurious vector.
    fn allocate() -> Option<Self> {
        static VECTOR: AtomicU8 = AtomicU8::new(32);

        let mut cur = VECTOR.load(AtomicOrdering::Relaxed);
        loop {
            if cur == u8::MAX {
                return None;
            }

            match VECTOR.compare_exchange(
                cur,
                cur + 1,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return Some(Self(cur)),
                Err(next) => cur = next,
            }
        }
    }

    /// Vectors 0 to 31, which the architecture reserves for exceptions,
    /// whether or not one is currently defined for a given number.
    fn is_exception(&self) -> bool {
        self.0 <= 31
    }

    /// Vectors 32 to 255, the ones left to software.
    fn is_interrupt(&self) -> bool {
        self.0 >= 32
    }
}

/// The state `entry.S` saves before calling [`__interrupt_handler`], in the
/// order it lands on the stack.
///
/// `rip`/`cs`/`rflags`/`rsp`/`ss` are the frame the CPU itself pushes on any
/// interrupt or exception; `error_code` and the general-purpose registers
/// above it are pushed by `entry.S` (a `0` error code if the vector doesn't
/// supply one, to keep every vector's frame the same shape).
#[repr(C, packed)]
pub struct InterruptStackFrame {
    /// `%rax` registers.
    rax: u64,

    /// `%rcx` registers.
    rcx: u64,

    /// `%rdx` registers.
    rdx: u64,

    /// `%rsi` registers.
    rsi: u64,

    /// `%rdi` registers.
    rdi: u64,

    /// `%r8` registers.
    r8: u64,

    /// `%r9` registers.
    r9: u64,

    /// `%r10` registers.
    r10: u64,

    /// `%r11` registers.
    r11: u64,

    /// `%rbp` registers.
    rbp: u64,

    // Interrupt-related error (`0` for compatibility).
    error_code: u64,

    /// Instruction pointer.
    rip: u64,
    /// Code segment descriptor.
    cs: u64,
    /// Flags register.
    rflags: u64,
    /// Stack pointer.
    rsp: u64,
    /// Stack segment descriptor.
    ss: u64,
}

impl InterruptStackFrame {
    /// The code segment the interrupted context ran under.
    ///
    /// # Panics
    ///
    /// If `cs` is neither [`KERNEL_CODE`](SegmentSelector::KERNEL_CODE) nor
    /// [`USER_CODE`](SegmentSelector::USER_CODE) — the only two selectors
    /// `entry.S`'s `swapgs` handling accounts for.
    pub const fn cs(&self) -> SegmentSelector {
        let cs = SegmentSelector::from_bits(self.cs as _);
        assert!(
            cs.into_bits() == SegmentSelector::KERNEL_CODE.into_bits()
                || cs.into_bits() == SegmentSelector::USER_CODE.into_bits()
        );
        cs
    }

    /// The stack segment the interrupted context ran under.
    ///
    /// # Panics
    ///
    /// If `ss` is neither [`KERNEL_DATA`](SegmentSelector::KERNEL_DATA) nor
    /// [`USER_DATA`](SegmentSelector::USER_DATA).
    pub const fn ss(&self) -> SegmentSelector {
        let ss = SegmentSelector::from_bits(self.ss as _);
        assert!(
            ss.into_bits() == SegmentSelector::KERNEL_DATA.into_bits()
                || ss.into_bits() == SegmentSelector::USER_DATA.into_bits()
        );
        ss
    }

    /// The flags register at the point of interruption.
    pub const fn rflags(&self) -> RFLAGS {
        RFLAGS::from_bits(self.rflags)
    }
}

impl crate::arch::generic::cpu::InterruptStackFrame for InterruptStackFrame {
    type RawRegister = u64;

    type ErrorCode = u64;

    fn get_sp(&self) -> Register<Self::RawRegister> {
        Register::<Self::RawRegister>::from_raw(self.rsp)
    }

    fn set_sp(&mut self, sp: Register<Self::RawRegister>) {
        self.rsp = sp.into_raw();
    }

    fn get_ip(&self) -> Register<Self::RawRegister> {
        Register::<Self::RawRegister>::from_raw(self.rip)
    }

    fn set_ip(&mut self, ip: Register<Self::RawRegister>) {
        self.rip = ip.into_raw();
    }

    fn get_ret(&self) -> Register<Self::RawRegister> {
        Register::<Self::RawRegister>::from_raw(self.rax)
    }

    fn set_ret(&mut self, ret: Register<Self::RawRegister>) {
        self.rax = ret.into_raw();
    }

    fn get_error(&self) -> Self::ErrorCode {
        self.error_code
    }
}

/// Entry point every `entry.S` stub calls into after saving state.
///
/// `vector` is the vector number the stub was generated for; `state` is the
/// [`InterruptStackFrame`] it just built on the current stack.
///
/// # Safety (from the caller's side)
///
/// Must only be called by an `entry.S` stub, immediately after it has pushed
/// a complete [`InterruptStackFrame`] and with `state` pointing at it.
#[cfg(all(not(test), not(feature = "library")))]
#[unsafe(no_mangle)]
extern "C" fn __interrupt_handler(vector: u64, state: *mut InterruptStackFrame) {
    use crate::{arch::generic::cpu::InterruptVector as _, kernel::handler::handler};

    let vector = InterruptVector::from_raw(vector as _);

    handler(vector, state);
}
