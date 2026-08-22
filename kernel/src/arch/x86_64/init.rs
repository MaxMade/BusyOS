use crate::arch::CPU;
use crate::arch::generic::cpu::CPU as _;
use crate::core_local;
use crate::kernel::core_local::init_block_base;
use crate::kernel::bootinfo::BOOTINFO;

#[repr(C, align(16))] // TODO(@MaxMade): Somehow, use CPU::STACK_ALIGNMENT instead...
#[derive(Debug, Clone, Copy)]
pub struct Stack([u8; CPU::KERNEL_STACK_SIZE]);

core_local! {
    #[export_offset(__boot_stack_offset)]
    /// Stack the bootstrap stub runs on, one per core.
    ///
    /// Never read as a Rust value: the stub reaches it before any core has a
    /// `GS` base, as
    ///
    /// ```text
    /// _percpu_start + __boot_stack_offset(cpu_id) + __BOOT_STACK_SIZE
    /// ```
    ///
    /// which is the top of this core's stack, and grows down from there.
    pub static BOOT_STACK: Stack = Stack([0; CPU::KERNEL_STACK_SIZE]);
}

/// Size of one boot stack, for the bootstrap stub to turn
/// `__boot_stack_offset` into a stack pointer.
///
/// `__boot_stack_offset` names the *start* of a core's `BOOT_STACK` slot,
/// while a stack has to be entered at its top, so the stub needs the size as
/// well — and it cannot see `CPU::KERNEL_STACK_SIZE` itself, since `head.S`
/// is assembled on its own.
///
/// The slot holds a `PerCPU<Stack>`, i.e. the stack behind a header of
/// unspecified size, so the top computed this way can be up to that header's
/// size below the slot's end. It is never *past* it, which is what matters:
/// growing down from it stays inside this core's own slot. The stack stays
/// aligned, too — the slot is `align_of::<Stack>()`-aligned and the size is a
/// multiple of that.
#[unsafe(no_mangle)]
pub static __BOOT_STACK_SIZE: usize = size_of::<Stack>();

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
pub extern "C" fn start() -> i32 {
    let bootinfo = unsafe { BOOTINFO.assume_init_ref() };

    loop { }
}


