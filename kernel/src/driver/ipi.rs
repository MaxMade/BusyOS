//! Inter-processor interrupts: one core interrupting others.
//!
//! [`IPICapable`] is what every device that can send them offers, and [`IPIs`]
//! is the list of those that registered themselves. A driver joins it from its
//! own [`Module::init`], wrapped in its [`IPIDriver`] variant.
//!
//! The interface speaks in the kernel's own core numbers, a [`CPUSet`], not in
//! the hardware's addresses: turning one into the other is the driver's
//! business.

#[cfg(target_arch = "x86_64")]
use crate::driver::x86_64::x2apic::X2Apic;

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    arch::generic::cpu::{CPUID, CPUSet},
    driver::{irq::IRQCapable, module::Module},
    kernel::{
        arc::Arc,
        linked_list::LinkedList,
        locking::{
            CanAcquire, DriverLevelID, EpilogueLevelID, LockId, PreviousToken, PrologueLevelID,
            ReadGuard, Shared, Token,
        },
        ticketlock::{DriverRWTicketlock, RWTicketlock, RWTicketlockDriverID},
    },
    user::errno::{Errno, ToErrno},
};

/// What an inter-processor interrupt asks the cores it reaches to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Stop for good: the sending core is panicking, and nothing else may run
    /// alongside its last output.
    ///
    /// Delivered even to a core that has interrupts masked, for example
    /// because it spins on a lock the panicking core holds.
    Panic,
}

/// A device that can interrupt other cores.
///
/// Every method is generic over its token, so nothing here pins a device to
/// one token type. That is what keeps [`IPIs`] a list of [`IPIDriver`] rather
/// than of trait objects: a method generic over its token has no single
/// address to put in a vtable.
pub trait IPICapable: IRQCapable {
    /// Sends an interrupt asking every core in `target` to do what `mode`
    /// says.
    ///
    /// A core in `target` that the device cannot reach, for example because
    /// it never came up, is skipped. The calling core is interrupted too if it
    /// is in `target`, which for [`Mode::Panic`] means it stops as well.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn send<Token>(
        &self,
        target: CPUSet,
        mode: Mode,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;
}

/// A handle on a registered driver that can send inter-processor interrupts.
///
/// One variant per driver, for the reason [`IPICapable`] gives. Cloning a
/// handle is the [`Arc`] clone of the driver it names.
#[derive(Clone)]
pub enum IPIDriver {
    #[cfg(target_arch = "x86_64")]
    /// The x2APIC driver, see [`X2Apic`].
    X2Apic(Arc<X2Apic>),
}

// The enum is a `Module` so that it can be `IRQCapable` and `IPICapable`,
// which require one. Only `name` means anything on a handle, since the driver
// it names has long been initialised by the time a handle exists.
impl Module for IPIDriver {
    /// Never called.
    ///
    /// A handle names a driver that is already up, so there is nothing here
    /// to initialise. Each concrete driver is brought up through its own
    /// [`Module::init`], which is what creates the handle in the first
    /// place.
    ///
    /// # Panics
    ///
    /// Always.
    fn init<Token>(_: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        panic!("IPIDriver::init(...) must never be invoked directly.")
    }

    /// The name of the driver this handle names, see [`Module::name`].
    fn name(&self) -> &'static str {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.name(),
        }
    }
}

// Every method below matches on the variant and forwards to the concrete
// driver's method, handing `token` along.
impl IRQCapable for IPIDriver {
    /// Forwards to the concrete driver, see [`IRQCapable::enable_irqs`].
    fn enable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.enable_irqs(token),
        }
    }

    /// Forwards to the concrete driver, see [`IRQCapable::disable_irqs`].
    fn disable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.disable_irqs(token),
        }
    }

    /// Forwards to the concrete driver, see [`IRQCapable::irqs_enabled`].
    fn irqs_enabled<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.irqs_enabled(token),
        }
    }

    /// Forwards to the concrete driver, see [`IRQCapable::prologue`].
    fn prologue<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.prologue(token),
        }
    }

    /// Forwards to the concrete driver, see [`IRQCapable::epilogue`].
    fn epilogue<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<PrologueLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.epilogue(token),
        }
    }
}

impl IPICapable for IPIDriver {
    /// Forwards to the concrete driver, see [`IPICapable::send`].
    fn send<Token>(&self, target: CPUSet, mode: Mode, token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            IPIDriver::X2Apic(x2apic) => x2apic.send(target, mode, token),
        }
    }
}

/// Every driver that can send inter-processor interrupts and has registered
/// itself, in the order they did.
pub struct IPIs(LinkedList<IPIDriver>);

impl IPIs {
    /// Adds `driver` to the list of drivers the kernel may send
    /// inter-processor interrupts through.
    ///
    /// Takes an [`IPIDriver`], the enum naming every such driver, so a caller
    /// wraps the `Arc<Driver>` it already holds in that driver's variant and
    /// keeps using its own handle afterwards.
    ///
    /// # Errors
    ///
    /// [`Errno::ENOMEM`] if the list could not grow. The list is unchanged in
    /// that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn register<Token>(driver: IPIDriver, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut ipis, token) = IPIS.acquire(token);

        let token = match ipis.0.try_push_back(driver, token) {
            Ok(token) => token,
            Err((error, token)) => {
                let token = ipis.release(token);
                let errno = error.to_errno();
                return Err((errno, token));
            }
        };

        Ok(ipis.release(token))
    }

    /// Walks every registered driver, in the order they registered.
    ///
    /// The returned [`IPIsIter`] holds [`IPIS`] shared for as long as it
    /// lives, so no driver can register in the meantime, and yields a clone
    /// of each handle rather than a reference into the list.
    ///
    /// # Token
    ///
    /// The `token` is consumed and stored in the iterator. Hand the iterator
    /// to [`IPIsIter::release`] to unlock the list and get it back; merely
    /// dropping the iterator leaves the list locked for good.
    pub fn iter<Token>(token: Token) -> IPIsIter<Token>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (ipis, token) = IPIS.acquire_shared(token);

        IPIsIter {
            ipis,
            token,
            next: 0,
        }
    }

    /// Returns a handle on the driver the kernel is configured to send
    /// inter-processor interrupts through, or [`None`] if none registered.
    ///
    /// That is the first one to register for now.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn get<Token>(token: Token) -> (Option<IPIDriver>, Token)
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // TODO(@MaxMade): Pick the driver named by the build configuration
        // or the kernel command line, once either exists.
        let (ipis, token) = IPIS.acquire_shared(token);
        let driver = ipis.0.front().cloned();

        (driver, ipis.release(token))
    }
}

impl IPIs {
    /// Returns the driver [`get`](Self::get) would, without taking the lock.
    ///
    /// For the panic path, which has no token and may have panicked while
    /// holding [`IPIS`].
    ///
    /// # Safety
    ///
    /// Nothing may be registering a driver at the same time. That holds with
    /// interrupts masked on the core that registers them, the boot core, once
    /// it has brought the drivers up.
    pub unsafe fn emergency_get() -> Option<IPIDriver> {
        // SAFETY: see the function's contract. A registration the panic
        // interrupted halfway has either linked its node or not, and the
        // front of the list is valid either way.
        let ipis = unsafe { &*IPIS.data_ptr() };

        ipis.0.front().cloned()
    }
}

/// Cores that have come up, see [`core_online`].
static ONLINE_CORES: AtomicUsize = AtomicUsize::new(0);

/// Cores that have stopped for a panic, see [`acknowledge_stop`].
static STOPPED_CORES: AtomicUsize = AtomicUsize::new(0);

/// How long [`emergency_stop_others`] waits for the other cores, in spins.
///
/// A rough bound, not a measured time: the panic path has no timer it can
/// rely on. An NMI reaches a core within microseconds, so even at several
/// billion spins a second this leaves ample room, and a core that does not
/// answer in time is gone for good anyway.
const STOP_TIMEOUT_SPINS: usize = 50_000_000;

/// Counts the calling core as running, so that [`emergency_stop_others`]
/// knows how many cores to wait for.
///
/// Every core calls this once, while it comes up.
pub fn core_online() {
    ONLINE_CORES.fetch_add(1, Ordering::SeqCst);
}

/// Counts the calling core as stopped for a panic.
///
/// Called by a core that receives [`Mode::Panic`], right before it halts.
pub fn acknowledge_stop() {
    STOPPED_CORES.fetch_add(1, Ordering::SeqCst);
}

/// Stops every other running core with [`Mode::Panic`] and waits until each
/// has acknowledged it, or until [`STOP_TIMEOUT_SPINS`] have passed.
///
/// For the panic path: it takes no lock and no token, so that what the
/// panicking core prints next is the last thing that happens, without
/// another core drawing over it or changing what it shows. Without a driver
/// there is nobody to stop the other cores, and it returns at once.
///
/// Returns whether every other core acknowledged in time.
///
/// # Safety
///
/// For the panic path only: the other cores are stopped wherever they are,
/// holding whatever they hold. Interrupts must be masked on the calling
/// core, and it must not be stopped itself, which [`Mode::Panic`] would do.
pub unsafe fn emergency_stop_others() -> bool {
    let others = ONLINE_CORES.load(Ordering::SeqCst).saturating_sub(1);
    if others == 0 {
        return true;
    }

    // SAFETY: the drivers were registered by the boot core before the other
    // cores came up, so nothing registers one now.
    let Some(driver) = (unsafe { IPIs::emergency_get() }) else {
        return false;
    };

    let mut target = CPUSet::all();
    target.remove(CPUID.with(|cpuid| *cpuid));

    // The x2APIC's `send` takes no lock, so a forged token cannot get in the
    // way of another core: those are being stopped regardless.
    //
    // SAFETY: the panic path holds no token of its own, and nothing it does
    // from here on waits for another core.
    let token = unsafe { crate::kernel::locking::RootToken::forge() };
    let _ = driver.send(target, Mode::Panic, token);

    for _ in 0..STOP_TIMEOUT_SPINS {
        if STOPPED_CORES.load(Ordering::SeqCst) >= others {
            return true;
        }

        core::hint::spin_loop();
    }

    false
}

/// An iterator over every registered driver, see [`IPIs::iter`].
///
/// Yields a clone of each [`IPIDriver`], which the caller may keep or drop as
/// it likes, since [`IPIs`] still holds a handle of its own.
pub struct IPIsIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    ipis: ReadGuard<'static, IPIs, RWTicketlock<RWTicketlockDriverID>>,
    token: Token<RWTicketlockDriverID, From, Shared>,
    next: usize,
}

impl<From> IPIsIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    /// Unlocks [`IPIS`] and returns the token given to [`IPIs::iter`].
    pub fn release(self) -> From {
        self.ipis.release(self.token)
    }
}

impl<From> Iterator for IPIsIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    type Item = IPIDriver;

    fn next(&mut self) -> Option<Self::Item> {
        // The guard cannot lend the list for longer than a call, so the
        // position is kept as an index. A machine has a handful of such
        // drivers at most, which keeps walking up to it again cheap.
        let driver = self.ipis.0.iter().nth(self.next).cloned();
        self.next += 1;
        driver
    }
}

/// The one list of registered drivers that can send inter-processor
/// interrupts.
pub static IPIS: DriverRWTicketlock<IPIs> =
    DriverRWTicketlock::new(RWTicketlock::new(), IPIs(LinkedList::new()));
