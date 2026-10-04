use core::{cell::UnsafeCell, mem::MaybeUninit, panic::PanicInfo};

use crate::{
    arch::{CPU, generic::cpu::CPU as GenericCPU},
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

#[panic_handler]
fn panic(panic_info: &PanicInfo) -> ! {
    // SAFETY: nothing else refers to `CPU_STATE`, see `StateCell`, and every
    // field of the state is a plain integer or register value, for which all
    // zeros is valid.
    let cpu_state = unsafe { (*CPU_STATE.0.get()).assume_init_mut() };

    // First, before anything else overwrites the registers.
    CPU::state(cpu_state);

    // Halts after showing the message, see `__printk_emergency`.
    printkln!(
        LogLevel::Panic,
        "KERNEL PANIC: {}\n\nCPU State:\n{}",
        panic_info,
        cpu_state
    );

    // SAFETY: not reached, the message above halts already. Halting again
    // is the one thing left to do on the panic path either way.
    unsafe { CPU::halt() }
}
