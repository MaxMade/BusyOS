//! Abstraction for the x86_64 `RFLAGS` register.
//!
//! `RFLAGS` holds the processor status flags (carry, zero, sign, ...), the
//! system flags (interrupt enable, trap, I/O privilege level, ...) and a
//! number of reserved bits.
//!
//! The register cannot be accessed with a `mov`. It is instead transferred
//! through the stack using `PUSHFQ`/`POPFQ`, which is what [`RFLAGS::read`]
//! and [`RFLAGS::write`] do.

use core::fmt::Display;

use bitfield_struct::bitfield;

/// x86_64 `RFLAGS` register.
///
/// Bits `[63:22]` as well as the individually marked reserved bits are not
/// defined by the architecture and must be preserved. The recommended way to
/// modify the register is therefore to [`read`](RFLAGS::read) it, adjust the
/// fields of interest and [`write`](RFLAGS::write) it back.
#[bitfield(u64)]
pub struct RFLAGS {
    /// Carry Flag (`CF`).
    ///
    /// Set if an arithmetic operation generated a carry out of, or a borrow
    /// into, the most significant bit of the result.
    #[bits(1)]
    pub cf: bool,

    /// Reserved (bit 1). Always one.
    #[bits(1, default = true)]
    reserved_1: bool,

    /// Parity Flag (`PF`).
    ///
    /// Set if the least significant byte of the result contains an even
    /// number of one bits.
    #[bits(1)]
    pub pf: bool,

    /// Reserved (bit 3). Always zero.
    #[bits(1)]
    __: bool,

    /// Auxiliary Carry Flag (`AF`).
    ///
    /// Set if an arithmetic operation generated a carry out of, or a borrow
    /// into, bit 3 of the result. Used for binary-coded decimal arithmetic.
    #[bits(1)]
    pub af: bool,

    /// Reserved (bit 5). Always zero.
    #[bits(1)]
    __: bool,

    /// Zero Flag (`ZF`).
    ///
    /// Set if the result of an operation is zero.
    #[bits(1)]
    pub zf: bool,

    /// Sign Flag (`SF`).
    ///
    /// Set to the most significant bit of the result, which is the sign bit of
    /// a signed integer.
    #[bits(1)]
    pub sf: bool,

    /// Trap Flag (`TF`).
    ///
    /// If set, the processor raises a `#DB` exception after every instruction,
    /// enabling single-step debugging.
    #[bits(1)]
    pub tf: bool,

    /// Interrupt Enable Flag (`IF`).
    ///
    /// If set, the processor responds to maskable external interrupts. Cleared
    /// by `CLI`, set by `STI`.
    #[bits(1)]
    pub interrupt: bool,

    /// Direction Flag (`DF`).
    ///
    /// Controls whether string instructions increment (`0`) or decrement (`1`)
    /// their index registers.
    #[bits(1)]
    pub df: bool,

    /// Overflow Flag (`OF`).
    ///
    /// Set if the result of a signed arithmetic operation is too large or too
    /// small to fit into the destination operand.
    #[bits(1)]
    pub of: bool,

    /// I/O Privilege Level (`IOPL`, bits `[13:12]`).
    ///
    /// The maximum privilege level that may execute I/O instructions and
    /// modify [`interrupt`](RFLAGS::interrupt) directly.
    #[bits(2)]
    pub iopl: u8,

    /// Nested Task (`NT`).
    ///
    /// Legacy flag indicating that the current task is linked to a previously
    /// executed task. Unused in long mode.
    #[bits(1)]
    pub nt: bool,

    /// Reserved (bit 15). Always zero.
    #[bits(1)]
    __: bool,

    /// Resume Flag (`RF`).
    ///
    /// Temporarily suppresses instruction breakpoint `#DB` exceptions so that
    /// a faulting instruction can be restarted.
    #[bits(1)]
    pub rf: bool,

    /// Virtual-8086 Mode (`VM`).
    ///
    /// Legacy flag enabling virtual-8086 mode. Not available in long mode.
    #[bits(1)]
    pub vm: bool,

    /// Alignment Check / Access Control (`AC`).
    ///
    /// Together with `CR0.AM`, enables alignment checking of user-mode memory
    /// accesses. Also gates SMAP when `CR4.SMAP` is set.
    #[bits(1)]
    pub ac: bool,

    /// Virtual Interrupt Flag (`VIF`).
    ///
    /// Virtual image of [`interrupt`](RFLAGS::interrupt), used by the
    /// virtual-8086 mode extensions.
    #[bits(1)]
    pub vif: bool,

    /// Virtual Interrupt Pending (`VIP`).
    ///
    /// Indicates that an interrupt is pending while
    /// [`vif`](RFLAGS::vif) masks it.
    #[bits(1)]
    pub vip: bool,

    /// Identification Flag (`ID`).
    ///
    /// Writable if and only if the processor supports the `CPUID` instruction.
    #[bits(1)]
    pub id: bool,

    /// Reserved (bits `[63:22]`). Always zero.
    #[bits(42)]
    __: u64,
}

impl Display for RFLAGS {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const FLAGS: [(&str, fn(&RFLAGS) -> bool); 11] = [
            ("CF", RFLAGS::cf),
            ("PF", RFLAGS::pf),
            ("AF", RFLAGS::af),
            ("ZF", RFLAGS::zf),
            ("SF", RFLAGS::sf),
            ("TF", RFLAGS::tf),
            ("IF", RFLAGS::interrupt),
            ("DF", RFLAGS::df),
            ("OF", RFLAGS::of),
            ("NT", RFLAGS::nt),
            ("AC", RFLAGS::ac),
        ];

        write!(f, "0x{:016x} (iopl: {}", self.0, self.iopl())?;

        for (name, get) in FLAGS {
            if get(self) {
                write!(f, " {name}")?;
            }
        }

        write!(f, ")")
    }
}

impl RFLAGS {
    /// Reads the current value of the `RFLAGS` register.
    ///
    /// Pushes the register onto the stack with `PUSHFQ` and pops it into a
    /// general purpose register.
    #[inline]
    pub fn read() -> Self {
        let val: u64;

        unsafe {
            core::arch::asm!(
                "pushfq",
                "pop {}",
                out(reg) val,
                options(preserves_flags)
            );
        }

        Self(val)
    }

    /// Writes `self` to the `RFLAGS` register.
    ///
    /// Pushes the value onto the stack and loads it with `POPFQ`.
    ///
    /// Note that not every field is writable from every privilege level: at a
    /// current privilege level greater than [`iopl`](RFLAGS::iopl) writes to
    /// [`interrupt`](RFLAGS::interrupt) are ignored, and writes to
    /// [`iopl`](RFLAGS::iopl) itself are ignored outside of ring 0.
    ///
    /// # Safety
    ///
    /// This overwrites *all* flags, including the status flags the compiler
    /// relies on. The value should be derived from a previous
    /// [`read`](RFLAGS::read) rather than constructed from scratch, otherwise
    /// reserved bits may be cleared.
    ///
    /// Restoring [`interrupt`](RFLAGS::interrupt) may re-enable interrupts and
    /// therefore requires the same guarantees as `STI`.
    #[inline]
    pub unsafe fn write(&self) {
        unsafe {
            core::arch::asm!(
                "push {}",
                "popfq",
                in(reg) self.0,
                options(nostack)
            );
        }
    }
}
