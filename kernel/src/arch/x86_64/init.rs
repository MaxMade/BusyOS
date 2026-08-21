use core::cell::RefCell;
use core::ffi::c_void;

use crate::arch::CPU;
use crate::arch::generic::cpu::CPU as _;
use crate::core_local;
use crate::kernel::core_local::init_block_base;
use crate::kernel::bootinfo::BOOTINFO;

unsafe extern "C" {
    /// First byte of the `.percpu` template, defined by the linker script.
    static _percpu_start: u8;
}

#[repr(C, align(16))] // TODO(@MaxMade): Somehow, use CPU::STACK_ALIGNMENT instead...
#[derive(Debug, Clone, Copy)]
pub struct Stack([u8; CPU::KERNEL_STACK_SIZE]);

core_local! {
    #[export_offset(__boot_stack_offset)]
    /// Stack the bootstrap stub runs on, one per core.
    pub static BOOT_STACK: Stack = Stack([0; CPU::KERNEL_STACK_SIZE]);
}

/// Points `gs` at the core-local block of `cpu_id`.
///
/// # Safety
///
/// `cpu_id` must be the id this core was handed, and below the number of
/// cores the bootloader replicated the core-local template for. Calling this
/// with another core's id points two cores at the same block.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __init_gs(cpu_id: usize) {
    // SAFETY: the caller guarantees that `cpu_id` is this core's own id and
    // that no core-local variable has been touched yet.
    let base = unsafe { init_block_base(cpu_id) };

    // SAFETY: `base` is the block belonging to this core, which is what `GS`
    // is expected to anchor; `CR4.FSGSBASE` is set per the contract above.
    unsafe { CPU::set_gs_base(base as _) };
}

#[unsafe(no_mangle)]
pub extern "C" fn __boot_stack(cpu_id: usize) -> *const c_void {
    let offset = BOOT_STACK.offset(cpu_id);
    let base = (&raw const _percpu_start) as usize;

    let stack = unsafe {
        &*((base + offset) as *mut RefCell<Option<Stack>>)
    };

    unsafe { stack.borrow().unwrap().0.as_ptr().byte_add(CPU::KERNEL_STACK_SIZE - CPU::STACK_ALIGNMENT) as _}
}

#[unsafe(no_mangle)]
pub extern "C" fn start() -> i32 {
    let bootinfo = unsafe { BOOTINFO.assume_init_ref() };

    loop { }
}


