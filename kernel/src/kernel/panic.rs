use core::panic::PanicInfo;

use crate::{
    arch::{CPU, generic::cpu::CPU as _},
    kernel::printk::LogLevel,
    printkln,
};

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Halts after showing the message, see `__printk_emergency`.
    printkln!(LogLevel::Panic, "KERNEL PANIC: {}", info);

    // SAFETY: not reached, the message above halts already. Halting again
    // is the one thing left to do on the panic path either way.
    unsafe { CPU::halt() }
}
