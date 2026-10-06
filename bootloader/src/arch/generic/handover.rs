use busyos::kernel::{
    bootinfo::Bootinfo,
    locking::{CanAcquire, PreviousToken, level::Epilogue},
};

/// Prepares and performs the architecture-specific transition from the
/// bootloader environment into the kernel runtime.
///
/// A `HandOver` implementation encapsulates all state required to safely
/// transfer control to the kernel. This typically includes preparing a
/// temporary execution environment that remains valid while the kernel
/// establishes its own runtime state.
///
/// The handover consists of two phases:
///
/// 1. [`prepare`](Self::prepare) collects information from the bootloader and
///    allocates or constructs any temporary resources required for the
///    transition.
/// 2. [`handover`](Self::handover) performs the architecture-specific state
///    changes immediately before control is transferred to the kernel.
///
/// # Safety
///
/// The implementation must guarantee that the temporary execution environment
/// remains valid until the kernel has fully initialized its own state.
pub trait HandOver {
    /// Prepares the architecture-specific handover state.
    ///
    /// This method performs all work that can safely be completed before the
    /// actual transition to the kernel. Typical responsibilities include:
    ///
    /// - preparing the kernel stack,
    /// - constructing temporary page tables (for example, to transition into
    ///   the higher-half address space),
    /// - allocating architecture-specific transition data structures.
    ///
    /// The returned object contains all state required to complete the
    /// handover via [`handover`](Self::handover).
    ///
    /// The caller must hold the [`Epilogue`] lock level, ensuring that no
    /// further system initialization can race with the handover preparation.
    /// This implies access to all lower levels, such as the memory
    /// management level needed to build temporary page tables.
    fn prepare<Token>(bootinfo: &mut Bootinfo, token: Token) -> (Self, Token)
    where
        Self: Sized,
        Token: CanAcquire<Epilogue> + PreviousToken;

    /// Performs the architecture-specific handover, and indicating a regular
    /// shutdown (instead of reboot) is required.
    ///
    /// This method applies all processor state changes required immediately
    /// before transferring control to the kernel. Depending on the target
    /// architecture, this may include:
    ///
    /// - disabling interrupts,
    /// - loading temporary page tables (e.g. updating `CR3` on x86_64),
    /// - installing a temporary interrupt descriptor table,
    /// - loading a temporary global descriptor table,
    /// - saving or restoring architecture-specific processor state,
    /// - switching to the prepared stack,
    /// - establishing any other execution context required by the kernel.
    ///
    /// After this function returns (or transfers control), the previous
    /// execution environment should no longer be assumed to be usable.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - the instance was previously created by [`prepare`](Self::prepare),
    /// - all memory referenced by the temporary execution environment remains
    ///   valid,
    /// - no other execution context can concurrently modify the processor
    ///   state being installed,
    /// - the kernel entry point expects the processor state established by
    ///   this implementation.
    ///
    /// Violating these requirements may result in immediate undefined
    /// behavior, including processor exceptions or system reset.
    unsafe fn handover(&mut self, cpu_id: usize) -> bool;
}
