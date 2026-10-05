use crate::arch::BOOT_CPUID;
use crate::arch::CPU;
use crate::arch::CPUID;
use crate::arch::Paging;
use crate::arch::generic::cpu::CPU as _;
use crate::arch::generic::cpu::CPUID;
use crate::arch::generic::paging::Paging as _;
use crate::arch::x86_64::gdt::Gdt;
use crate::arch::x86_64::idt::Idt;
use crate::core_local;
use crate::driver::irq::IRQCapable;
use crate::driver::module::Modules;
use crate::driver::timer::{Timer as _, Timers};
use crate::kernel::bootinfo::BOOTINFO;
use crate::kernel::core_local::{PerCPU, init_block_base};
use crate::kernel::locking::InitLevel;
use crate::kernel::locking::RootToken;
use crate::kernel::locking::Token;
use crate::kernel::printk;
use crate::kernel::printk::LogLevel;
use crate::kernel::time::MilliSeconds;
use crate::kernel::time::TimeUnit;
use crate::mem::heap::Heap;
use crate::mem::page_frames::EarlyPageFrames;
use crate::mem::page_frames::PageFrames;
use crate::printkln;

use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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

/// Points `gs` at the core-local block of `cpu_id` and installs the id in it.
///
/// Publishing the id here is what makes it reachable as `CPUID.with(...)`
/// from anywhere on this core, instead of being threaded through call chains
/// from the entry point. It has to happen after `GS` is set, since the store
/// resolves this core's block through it like every other core-local access.
///
/// # Safety
///
/// `cpu_id` must be the id this core was handed, and below the number of
/// cores the bootloader replicated the core-local template for. Calling this
/// with another core's id points two cores at the same block.
///
/// This must further be the first Rust code to run on this core, and run
/// exactly once: the id is written into the block directly, without the
/// guards a regular core-local store takes.
///
/// # Panics
///
/// If `cpu_id` does not fit a [`CPUID`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __init_gs(cpu_id: usize) {
    // SAFETY: the caller guarantees that `cpu_id` is this core's own id and
    // that no core-local variable has been touched yet.
    let base = unsafe { init_block_base(cpu_id) };

    // SAFETY: `base` is the block belonging to this core, which is what `GS`
    // is expected to anchor; `CR4.FSGSBASE` is set per the contract above.
    unsafe { CPU::set_gs_base(base as _) };

    let id = match CPUID::try_from(cpu_id) {
        Ok(id) => id,
        Err(_) => panic!("core id {cpu_id} does not fit a `CPUID`"),
    };

    // Installed by hand instead of through `CPUID.set`, which wants a token
    // this early in boot and would mask interrupts to keep a prologue on this
    // core out of the slot — there is no lock hierarchy yet, and no prologue
    // to fend off. The slot is reached the way `offset` prescribes,
    // `_percpu_start + VAR.offset(cpu_id)`, which is how the stub finds this
    // core's boot stack as well.
    let slot = (&raw const _percpu_start) as usize + CPUID.offset(cpu_id);

    // SAFETY: `slot` is this core's copy of `CPUID`, so it is a `PerCPU`
    // living in the `.percpu` template `with_value` asks for. Nothing can
    // observe the write: this core has touched no core-local variable yet,
    // and no other core reaches this block.
    unsafe {
        core::ptr::with_exposed_provenance_mut::<PerCPU<CPUID>>(slot).write(PerCPU::with_value(id))
    };
}

static RAW_CPUID: AtomicUsize = AtomicUsize::new(0);

#[unsafe(no_mangle)]
pub extern "C" fn start() -> i32 {
    let root_token = unsafe { RootToken::forge() };
    let (init_level, token) = InitLevel::enter(root_token);

    // Get current cpu ID
    let (cpu_id, mut token) = CPUID.with_mut(token, |cpuid| {
        let id = CPUID::try_from(RAW_CPUID.fetch_add(1, AtomicOrdering::Relaxed)).unwrap();
        *cpuid = id;
        id
    });

    // TODO(@MaxMade): Remove me as soon as BOOTINFO is actually used.
    let bootinfo = unsafe { BOOTINFO.assume_init_ref() };

    // Initialise global heap allocator
    if cpu_id == BOOT_CPUID {
        token = unsafe { Heap::early_initialisation(token) };
    }

    // Setup kernel mapping
    let (paging, t): (Paging<EarlyPageFrames>, Token<_, _, _>) =
        match Paging::kernel_mapping(bootinfo, token) {
            Ok((paging, t)) => (paging, t),
            Err((error, _)) => panic!("Unable to setup kernel mapping: {}", error),
        };
    token = t;
    unsafe { paging.activate() };

    // Prepare page frames
    token = PageFrames::handover_from_early(token);
    token = unsafe { PageFrames::init_from_bootinfo(token) };

    // Prepare GDT
    token = unsafe { Gdt::init(token) };
    token = unsafe { Gdt::load(token) };

    // Prepare IDT
    token = unsafe { Idt::init(token) };
    token = unsafe { Idt::load(token) };

    // Prepare printk!
    token = printk::init(LogLevel::Info, token);
    crate::printkln!(LogLevel::Info, "BusyOS Kernel");

    // Initialise modules
    token = Modules::init(token);

    // Setup timer
    match Timers::get(token) {
        (Some(timer), t) => {
            let t = match timer.enable_irqs(t) {
                Ok(t) => t,
                Err((error, t)) => printkln!(
                    LogLevel::Error,
                    t,
                    "Unable to unmask system timer: {}",
                    error
                ),
            };

            let interval = MilliSeconds::from(10).to_nanoseconds_lossy();
            match timer.setup(interval, true, t) {
                Ok(t) => token = t,
                Err((error, t)) => {
                    token = printkln!(
                        LogLevel::Error,
                        t,
                        "Unable to setup system timer: {}",
                        error
                    );
                }
            }
        }
        (None, t) => {
            token = printkln!(
                LogLevel::Warn,
                t,
                "Unable to setup system timer: no timer found"
            );
        }
    }

    init_level.leave(token);

    unsafe { CPU::raw_enable_interrupts() };
    loop {}
}
