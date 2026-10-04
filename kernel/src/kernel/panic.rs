use core::{
    cell::UnsafeCell,
    ffi::c_void,
    fmt::{self, Display, Formatter},
    mem::MaybeUninit,
    panic::PanicInfo,
};

use crate::{
    arch::{
        CPU,
        generic::{cpu::CPU as GenericCPU, paging::VirtualAddress},
    },
    driver::ksymbols::KSymbols,
    kernel::printk::LogLevel,
    printkln,
};

/// Room for the register state [`panic`] captures.
///
/// A `static` rather than a local, so that the panic path does not need the
/// stack space. There is a single core and a panic never returns, so only
/// one panic ever uses it.
struct StateCell(UnsafeCell<MaybeUninit<<CPU as GenericCPU>::State>>);

// SAFETY: only the panic handler touches it, and it runs at most once, see
// above.
unsafe impl Sync for StateCell {}

static CPU_STATE: StateCell = StateCell(UnsafeCell::new(MaybeUninit::zeroed()));

/// How many frames of the call stack a panic shows at most.
const BACKTRACE_DEPTH: usize = 16;

/// The return addresses [`panic`] records, shown one frame per line, each
/// with the demangled name of the symbol it lies in.
///
/// The first frame is the panic handler itself, followed by the panic
/// machinery of `core` and then the code that panicked.
struct Backtrace<'a>(&'a [VirtualAddress<c_void>]);

impl Display for Backtrace<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("  <none>");
        }

        for (depth, &addr) in self.0.iter().enumerate() {
            if depth > 0 {
                f.write_str("\n")?;
            }

            write!(f, "  #{depth:<2} 0x{:016x}", addr.addr())?;

            match KSymbols::emergency_lookup(addr) {
                // `{:#}` leaves out the hash suffix of a legacy-mangled name,
                // which only tells crate versions apart. A name that is not a
                // Rust symbol, such as one from assembly, is shown as it is.
                Some((name, offset)) => {
                    write!(f, " {:#}+0x{offset:x}", rustc_demangle::demangle(name))?
                }
                None => f.write_str(" <unknown>")?,
            }
        }

        Ok(())
    }
}

#[panic_handler]
fn panic(panic_info: &PanicInfo) -> ! {
    // SAFETY: nothing else refers to `CPU_STATE`, see `StateCell`, and every
    // field of the state is a plain integer or register value, for which all
    // zeros is valid.
    let cpu_state = unsafe { (*CPU_STATE.0.get()).assume_init_mut() };

    // First, before anything else overwrites the registers.
    CPU::state(cpu_state);

    let mut call_stack = [VirtualAddress::null(); BACKTRACE_DEPTH];
    let call_stack = CPU::unwind(&mut call_stack);

    // Halts after showing the message, see `__printk_emergency`.
    printkln!(
        LogLevel::Panic,
        "KERNEL PANIC: {}\n\nCPU State:\n{}\n\nBacktrace:\n{}",
        panic_info,
        cpu_state,
        Backtrace(call_stack)
    );

    // SAFETY: not reached, the message above halts already. Halting again
    // is the one thing left to do on the panic path either way.
    unsafe { CPU::halt() }
}
