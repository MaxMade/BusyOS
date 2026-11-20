//! A thread of execution and the state kept for it while it does not run.

use core::{
    cell::UnsafeCell,
    ffi::c_void,
    fmt::Display,
    mem::offset_of,
    sync::atomic::{AtomicU8, AtomicUsize, Ordering as AtomicOrdering},
};

use crate::{
    arch::{
        CPU, Paging,
        generic::{
            cpu::CPU as GenericCPU,
            paging::{
                PageSize, Paging as GenericPaging, ReversePaging as GenericReversePaging,
                VirtualAddress,
            },
        },
    },
    kernel::{
        arc::Arc,
        locking::{CanAcquire, LockId, MemoryManagementLevelID, PreviousToken, RootToken},
    },
    mem::page_frames::{PAGE_FRAMES, PageFrames, PhysicalPageFrames},
    sched::scheduler::{Scheduler, SchedulerDriver},
    user::errno::{Errno, ToErrno},
};

/// What a new thread runs, see
/// [`CPU::context_prepare`](crate::arch::generic::cpu::CPU::context_prepare).
///
/// Gets the argument it was prepared with and the thread's [`RootToken`], with
/// the first switch into the thread already finished and interrupts enabled.
/// A thread never returns from it.
///
/// Runs on the epilogue level, which the first switch handed over, see
/// [`Epilogue`](crate::kernel::handler::Epilogue): kernel code always does,
/// so a timer tick never preempts it, and its epilogues wait until the level
/// is left. Only code off the level, such as user space or an idle core, is
/// preempted.
pub type ThreadEntry = fn(usize, RootToken) -> !;

/// Identifies a thread for as long as it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ThreadID(usize);

/// The identity the next thread gets, see [`Thread::new`].
static NEXT_THREAD_ID: AtomicUsize = AtomicUsize::new(0);

impl Display for ThreadID {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TID {}", self.0)
    }
}

/// What a thread runs and how the scheduler treats it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadType {
    /// Runs when nothing else is runnable. One per core, kept out of the run
    /// queue.
    Idle,
    /// Runs in the kernel only.
    Kernel,
    /// Runs in user space, entering the kernel through its kernel stack.
    User,
}

/// Whether a thread may run, as the scheduler tracks it.
///
/// Only the scheduler changes it, and only under its own lock, which it keeps
/// across a switch. A thread that blocks is therefore never seen blocked
/// before its switch away has saved where it stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ThreadState {
    /// Running, or waiting in the run queue for its turn.
    Runnable,
    /// Waiting for something, out of the run queue until it is runnable
    /// again.
    Blocked,
    /// Done for good. The scheduler gives up its handle once the switch away
    /// is finished, and the thread, with the kernel stack it owns, is freed
    /// with its last handle, see [`Thread::release`].
    Exited,
}

impl ThreadState {
    /// The state stored as `raw` by [`Thread::set_state`].
    const fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Runnable,
            1 => Self::Blocked,
            2 => Self::Exited,
            _ => panic!("Invalid thread state"),
        }
    }
}

/// A thread of execution.
///
/// While a thread does not run, everything needed to resume it lies on its
/// kernel stack, and the thread itself only keeps the pointer to it. See
/// [`CPU::context_switch`](crate::arch::generic::cpu::CPU::context_switch).
pub struct Thread {
    /// Identity of the thread.
    id: ThreadID,
    /// How the scheduler treats the thread.
    ty: ThreadType,
    /// Whether the thread may run, a [`ThreadState`].
    state: AtomicU8,
    /// Top of the thread's kernel stack, where it is empty.
    kernel_tos_stack_ptr: VirtualAddress<c_void>,
    /// Stack pointer the thread was suspended with. Written by the context
    /// switch only, while the thread does not run.
    kernel_stack_ptr: UnsafeCell<VirtualAddress<c_void>>,
    /// The page frames of the kernel stack the thread owns, from
    /// [`try_new`](Thread::try_new), until [`release`](Thread::release) frees
    /// them with the thread. `None` for a thread on a stack it does not own,
    /// such as a core's boot context on its boot stack.
    owned_kernel_stack: UnsafeCell<Option<PhysicalPageFrames>>,
}

// SAFETY: `kernel_tos_stack_ptr` is set when the thread is created and only
// read afterwards. `kernel_stack_ptr` is written by the context switch alone,
// for the thread being suspended, on the one core that ran it. No other core
// can switch to that thread before the scheduler has queued it again, which
// happens under the run queue's lock, after the write. Neither pointer is
// dereferenced outside the switch. `owned_kernel_stack` is only touched while
// the thread is created and once it is freed, both through `&mut`.
unsafe impl Send for Thread {}

// SAFETY: as above.
unsafe impl Sync for Thread {}

impl Thread {
    /// Offset of the saved stack pointer within a thread, for the context
    /// switch, which reaches it from assembly.
    pub const KERNEL_STACK_PTR_OFFSET: usize = offset_of!(Thread, kernel_stack_ptr);

    /// Creates a thread of type `ty` with a fresh identity, on a kernel stack
    /// of its own, taken from the page frame allocator.
    ///
    /// The thread cannot be switched to yet:
    /// [`CPU::context_prepare`](crate::arch::generic::cpu::CPU::context_prepare)
    /// sets it up to start. The stack is freed with the thread, by
    /// [`release`](Thread::release).
    ///
    /// # Errors
    ///
    /// [`Errno::ENOMEM`] if no frames are left for the stack.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn try_new<Token>(ty: ThreadType, token: Token) -> Result<(Self, Token), (Errno, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        const SIZE: usize = <CPU as GenericCPU>::KERNEL_STACK_SIZE;

        let page_size = match Paging::<PageFrames>::page_size(PageSize::Regular) {
            Some(page_size) => page_size,
            None => panic!("No regular page size for kernel stacks"),
        };

        let (mut page_frames, token) = PAGE_FRAMES.acquire(token);
        let frames = page_frames.allocate(PageSize::Regular, SIZE.div_ceil(page_size));
        let token = page_frames.release(token);

        let frames = match frames {
            Ok(frames) => frames,
            Err(error) => return Err((error.to_errno(), token)),
        };

        // SAFETY: the frames have just been allocated, out of the direct map,
        // which the kernel's page tables map.
        let stack = unsafe { Paging::<PageFrames>::phys_to_virt(frames.start()) };

        // SAFETY: the frames span at least the whole stack, so one past its
        // end is still part of them.
        let top = unsafe { stack.byte_add(SIZE) };

        // SAFETY: the frames are page aligned, span the whole stack, and
        // belong to this thread until `release` frees them with it.
        let mut thread = unsafe { Self::new(ty, top) };
        *thread.owned_kernel_stack.get_mut() = Some(frames);

        Ok((thread, token))
    }

    /// Creates a thread of type `ty` with a fresh identity, on the kernel
    /// stack that ends at `kernel_tos_stack_ptr`, which the thread does not
    /// own.
    ///
    /// For a context that already runs on its stack, such as a core's boot
    /// context: its first switch away saves where the thread continues.
    ///
    /// # Safety
    ///
    /// `kernel_tos_stack_ptr` must be the top of a stack of
    /// [`KERNEL_STACK_SIZE`](crate::arch::generic::cpu::CPU::KERNEL_STACK_SIZE)
    /// bytes, aligned for the architecture, that nothing but this thread uses
    /// for as long as the thread exists.
    pub unsafe fn new(ty: ThreadType, kernel_tos_stack_ptr: VirtualAddress<c_void>) -> Self {
        // Only has to be unique, so no ordering with anything else.
        let id = ThreadID(NEXT_THREAD_ID.fetch_add(1, AtomicOrdering::Relaxed));

        Self {
            id,
            ty,
            state: AtomicU8::new(ThreadState::Runnable as u8),
            kernel_tos_stack_ptr,
            kernel_stack_ptr: UnsafeCell::new(VirtualAddress::null()),
            owned_kernel_stack: UnsafeCell::new(None),
        }
    }

    /// Gives up a handle on a thread, freeing the thread, and the kernel
    /// stack it owns, with the last handle.
    ///
    /// Every handle on a thread that may be the last one has to go this way
    /// rather than through [`Arc::release`], which cannot free the stack, see
    /// [`Drop`](#impl-Drop-for-Thread).
    ///
    /// Nothing runs on the stack by the time the last handle goes: a running
    /// thread is held by [`CURRENT`](crate::sched::scheduler::CURRENT), a
    /// suspended one by the scheduler or whoever is to make it runnable
    /// again, and the scheduler gives up its handle on an exited thread only
    /// once the switch away from it is finished.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn release<Token>(this: Arc<Thread>, token: Token) -> Token
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut thread, token) = match Arc::into_inner(this, token) {
            (Some(thread), token) => (thread, token),
            (None, token) => return token,
        };

        let token = match thread.owned_kernel_stack.get_mut().take() {
            Some(frames) => {
                let (mut page_frames, token) = PAGE_FRAMES.acquire(token);

                // SAFETY: the frames came from the page frame allocator in
                // `try_new`, are taken out of the thread so they cannot be
                // freed twice, and nothing runs on them any more, see above.
                // The direct map is the only mapping of them.
                unsafe { page_frames.deallocate(frames) };

                page_frames.release(token)
            }
            None => token,
        };

        drop(thread);

        token
    }

    /// Returns the thread's identity.
    pub const fn id(&self) -> ThreadID {
        self.id
    }

    /// Returns how the scheduler treats the thread.
    pub const fn ty(&self) -> ThreadType {
        self.ty
    }

    /// Returns whether the thread may run.
    ///
    /// Up to date only under the scheduler's lock, see [`ThreadState`].
    pub fn state(&self) -> ThreadState {
        // The scheduler's lock orders the accesses that matter.
        ThreadState::from_raw(self.state.load(AtomicOrdering::Relaxed))
    }

    /// Sets whether the thread may run.
    ///
    /// For the scheduler alone, under its own lock, see [`ThreadState`].
    pub fn set_state(&self, state: ThreadState) {
        self.state.store(state as u8, AtomicOrdering::Relaxed);
    }

    /// Returns `true` for a core's idle thread.
    pub fn is_idle(&self) -> bool {
        self.ty == ThreadType::Idle
    }

    /// Sets the stack pointer the thread is resumed with.
    ///
    /// Takes `&mut self`, so it only reaches a thread nobody else knows of
    /// yet: one being set up, before its first switch.
    pub fn set_kernel_stack_ptr(&mut self, ptr: VirtualAddress<c_void>) {
        *self.kernel_stack_ptr.get_mut() = ptr;
    }

    /// Borrows the whole kernel stack, for setting a thread up before its
    /// first switch, see
    /// [`CPU::context_prepare`](crate::arch::generic::cpu::CPU::context_prepare).
    ///
    /// Takes `&mut self`, so it only reaches a thread nobody else knows of
    /// yet, which nothing runs on.
    pub fn kernel_stack(&mut self) -> &mut [u8] {
        const SIZE: usize = <CPU as GenericCPU>::KERNEL_STACK_SIZE;
        let stack = unsafe { self.kernel_tos_stack_ptr.byte_sub(SIZE) };

        unsafe { core::slice::from_raw_parts_mut(stack.as_ptr().cast(), SIZE) }
    }
}

impl Drop for Thread {
    /// Panics if the thread still owns its kernel stack.
    ///
    /// Freeing the stack needs a token, which `Drop` cannot be given, so the
    /// alternative would be to leak it silently. Use [`Thread::release`] for
    /// the last handle instead.
    fn drop(&mut self) {
        if self.owned_kernel_stack.get_mut().is_some() {
            panic!(
                "A thread must never be dropped with its kernel stack. Use Thread::release(...) instead!"
            );
        }
    }
}

/// Where every new thread starts.
///
/// Called by the trampoline the architecture's
/// [`CPU::context_prepare`](crate::arch::generic::cpu::CPU::context_prepare)
/// sets up, with what the thread's first switch returned — `prev` and `next`
/// as in [`CPU::context_switch`](crate::arch::generic::cpu::CPU::context_switch)
/// — and the `entry` and `arg` the thread was prepared with.
///
/// A resumed thread finishes the switch that resumed it at the end of its own
/// [`Scheduler::schedule`] call. A new thread has none, so this does the
/// same: it takes over both handles, lets the scheduler requeue `prev` and
/// unlock, and then enables interrupts — the switch does not carry the
/// interrupt flag, and a new thread has no saved one to restore.
///
/// The epilogue level the switch handed over is kept, not left: the thread is
/// kernel code, which runs on it, see [`ThreadEntry`].
///
/// # Panics
///
/// If no scheduler has registered, which no thread can have been switched to
/// without.
#[unsafe(export_name = "__thread_entry")]
#[allow(improper_ctypes_definitions)]
extern "C" fn thread_entry(
    prev: *const Thread,
    next: *const Thread,
    entry: ThreadEntry,
    arg: usize,
) -> ! {
    // SAFETY: this is the first code the thread runs, so this is its one root
    // token.
    let token = unsafe { RootToken::forge() };

    // SAFETY: both pointers were made by `into_raw` in the switch that started
    // this thread, which does not reclaim them itself.
    let (prev, next) = unsafe { (Arc::<Thread>::from_raw(prev), Arc::<Thread>::from_raw(next)) };

    // `next` is this thread, which `CURRENT` keeps alive, so this is never the
    // last handle and needs no token.
    drop(next);

    let (scheduler, token) = match SchedulerDriver::get(token) {
        (Some(scheduler), token) => (scheduler, token),
        (None, _) => panic!("Thread started without a scheduler"),
    };

    // SAFETY: this is the entry of a new thread, and nothing has touched the
    // scheduler since its first switch.
    let token = unsafe { scheduler.finish_first_switch(prev, token) };

    // The registered scheduler keeps a handle of its own.
    drop(scheduler);

    // SAFETY: the switch is finished, and the thread holds no lock yet.
    unsafe { CPU::raw_enable_interrupts() };

    entry(arg, token)
}
