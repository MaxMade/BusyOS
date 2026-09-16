//! Interrupt routing: which driver owns which vector, and the two halves an
//! interrupt is handled in.
//!
//! [`InterruptRouter`] answers the only question the interrupt path asks,
//! namely which [`IRQCapable`] driver is in charge of the vector that just
//! arrived. [`InterruptController`] is the other side of that, the chip which
//! delivers the vector, and it is what a driver goes through to have its own
//! vector masked or unmasked.
//!
//! Handling a vector is split in two. The prologue runs in interrupt context
//! under a [`PrologueToken`], does only what cannot wait, and reports whether
//! the rest is needed. The epilogue then runs under an [`EpilogueToken`], from
//! a level that may still take the driver's own locks.

use crate::{
    arch::{InterruptVector, generic::cpu::InterruptVector as GenericInterruptVector},
    driver::module::Module,
    kernel::{
        arc::Arc,
        locking::{
            CanAcquire, DriverLevelID, EpilogueLevelID, LockId, MemoryManagementLevelID,
            PreviousToken, PrologueLevelID, RootToken, Shared, Token,
        },
        ticketlock::{PrologueRWTicketlock, RWTicketlock},
    },
    user::errno::Errno,
};

/// The token the driver-level calls of a registered [`IRQCapable`] run under.
///
/// What `EpilogueLevel::enter` hands out, which is above the `Driver` level and
/// may therefore acquire a driver's own locks.
pub type DriverToken = Token<EpilogueLevelID, RootToken, Shared>;

/// The token a prologue runs under.
///
/// An interrupt entry holds no lock yet, so this is the root token: a prologue
/// has to reach the `Epilogue` level to request an epilogue, and only a token
/// above that level may.
pub type PrologueToken = RootToken;

/// The token an epilogue runs under: the `Epilogue` level it is named after,
/// from which it can still reach the `Prologue` level that the prologue shares
/// its state under.
pub type EpilogueToken = Token<EpilogueLevelID, RootToken, Shared>;

/// A driver that owns one or more interrupt vectors.
///
/// The token types are parameters of the trait rather than of each method: a
/// method generic over its token has no single address to put in a vtable, so a
/// trait whose methods are generic cannot be made into an object, and
/// [`InterruptRouter`] holds these as `dyn`. A driver implements the trait for
/// every token type that satisfies the bounds and so stays as general as it is
/// today. Only the table pins the three down.
pub trait IRQCapable<DriverToken, PrologueToken, EpilogueToken>: Module
where
    DriverToken: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    PrologueToken: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken,
    EpilogueToken: CanAcquire<<PrologueLevelID as LockId>::Level> + PreviousToken,
{
    /// Tells the device to start raising its interrupts.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn enable_irqs(&self, token: DriverToken) -> Result<DriverToken, (Errno, DriverToken)>;

    /// Tells the device to stop raising its interrupts.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn disable_irqs(&self, token: DriverToken) -> Result<DriverToken, (Errno, DriverToken)>;

    /// Whether the device is currently raising its interrupts.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn irqs_enabled(&self, token: DriverToken)
    -> Result<(bool, DriverToken), (Errno, DriverToken)>;

    /// The half that runs in interrupt context, straight off the vector.
    ///
    /// Does only what cannot be deferred, which is normally quieting the
    /// device, and reports whether [`epilogue`](Self::epilogue) has to run
    /// afterwards to finish the work.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms. It is the root
    /// token, so a prologue holds nothing on entry and has to reach the
    /// `Epilogue` level itself to request its epilogue.
    fn prologue(
        &self,
        token: PrologueToken,
    ) -> Result<(bool, PrologueToken), (Errno, PrologueToken)>;

    /// The deferred half, run once a prologue has asked for it.
    ///
    /// Runs at the `Epilogue` level instead of in interrupt context, so it may
    /// take the driver's own locks and do the work the prologue left.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn epilogue(&self, token: EpilogueToken) -> Result<EpilogueToken, (Errno, EpilogueToken)>;
}

/// A handle on a driver, as the router holds it.
pub type IRQDriver = Arc<dyn IRQCapable<DriverToken, PrologueToken, EpilogueToken>>;

/// The driver in charge of each interrupt vector.
///
/// # Filling it and reading it
///
/// A hold on a prologue-level lock owns the caller's token for as long as it
/// lasts (see [`PrologueLock`](crate::kernel::prologue_lock::PrologueLock)), so
/// nothing under such a hold can allocate. The table is therefore one flat slot
/// per vector, sized once at compile time from
/// [`InterruptVector::MAX_NUM`](GenericInterruptVector::MAX_NUM), so neither
/// [`register`](Self::register) nor [`driver_for`](Self::driver_for) needs the
/// heap while it holds the lock.
///
/// # Dropping
///
/// The slots hold [`Arc`] handles, and dropping the last handle on a driver
/// frees it, which needs a token that [`Drop`] has no way of being given. A
/// router therefore belongs in a `static`, which never goes out of scope.
pub struct InterruptRouter {
    vector_table: PrologueRWTicketlock<[Option<IRQDriver>; InterruptVector::MAX_NUM]>,
}

impl Default for InterruptRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl InterruptRouter {
    /// Creates a router that is in charge of nothing yet.
    ///
    /// The table is part of the router and starts out empty, so this allocates
    /// nothing and may be used to build a `static`.
    pub const fn new() -> Self {
        Self {
            vector_table: PrologueRWTicketlock::new(
                RWTicketlock::new(),
                [const { None }; InterruptVector::MAX_NUM],
            ),
        }
    }

    /// Puts `driver` in charge of `vector`.
    ///
    /// The slot has to be free. Two drivers sharing one vector is not
    /// supported, so an occupied slot is a bug in the caller rather than a
    /// condition to report back, and the returned [`Option`] is always [`None`]
    /// today. It names the driver that was replaced so that sharing can be
    /// handled here later without touching the callers.
    ///
    /// # Panics
    ///
    /// If another driver is already in charge of `vector`.
    ///
    /// # Token
    ///
    /// The `token` has to reach the `MemoryManagement` level, which is above
    /// the `Prologue` level the table is locked at. It is consumed for the
    /// write and returned in both arms.
    pub fn register<Token>(
        &self,
        vector: InterruptVector,
        driver: IRQDriver,
        token: Token,
    ) -> Result<(Option<IRQDriver>, Token), (Errno, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut vector_table = self.vector_table.acquire(token);

        let vector = vector.into_raw() as usize;
        let prev = vector_table[vector].replace(driver);
        if let Some(_prev) = prev {
            panic!("Detected un-wanted interrupt sharing");
        }

        let token = vector_table.release();
        Ok((None, token))
    }

    /// Returns a handle on the driver in charge of `vector`, or [`None`] if no
    /// driver claimed it.
    ///
    /// Reads the table under a shared hold, which masks interrupts for the
    /// lookup alone, and hands the token back with the answer. The handle is
    /// one of several, so the caller may drop it.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn driver_for<Token>(
        &self,
        vector: InterruptVector,
        token: Token,
    ) -> (Option<IRQDriver>, Token)
    where
        Token: CanAcquire<<PrologueLevelID as LockId>::Level> + PreviousToken,
    {
        let vector_table = self.vector_table.acquire_shared(token);
        let vector = vector.into_raw() as usize;
        let driver = match vector_table.get(vector) {
            Some(driver) => driver.clone(),
            None => {
                let token = vector_table.release();
                return (None, token);
            }
        };

        (driver, vector_table.release())
    }
}

/// The one routing table, which every interrupt entry and every
/// [`InterruptController`] works through.
pub static INTERRUPT_ROUTER: InterruptRouter = InterruptRouter::new();

/// The chip that delivers interrupt vectors, such as a PIC or an APIC.
///
/// The controller owns delivery and the [`INTERRUPT_ROUTER`] owns the mapping
/// from a vector to its driver, so the default methods below are the path a
/// caller takes to reach whoever is in charge of a vector. Only
/// [`acknowledge`](Self::acknowledge) is left to the concrete chip.
pub trait InterruptController: Module {
    /// Puts `driver` in charge of `vector` in the [`INTERRUPT_ROUTER`].
    ///
    /// Generic over the concrete driver instead of taking an [`IRQDriver`], so
    /// that a caller hands over the `Arc<Driver>` it already has and the
    /// unsizing to `dyn` happens here.
    ///
    /// # Panics
    ///
    /// If another driver is already in charge of `vector`.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn register<Token, Driver>(
        &self,
        token: Token,
        driver: Arc<Driver>,
        vector: InterruptVector,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
        Driver: 'static + IRQCapable<DriverToken, PrologueToken, EpilogueToken>,
    {
        let (prev, token) = INTERRUPT_ROUTER.register(vector, driver, token)?;
        if let Some(_prev) = prev {
            panic!("Detected interrupt-sharing");
        }
        Ok(token)
    }

    /// Tells the chip that `vector` has been handled and may be delivered
    /// again.
    ///
    /// Left to the implementation because only the concrete chip knows what
    /// acknowledging means, from a PIC's end-of-interrupt command to an APIC's
    /// write to its own register.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn acknowledge<Token>(
        &self,
        vector: InterruptVector,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;

    /// Tells the driver in charge of `vector` to stop raising it.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if no driver is in charge of `vector`, or whatever
    /// the driver itself reports.
    ///
    /// # Token
    ///
    /// Takes the [`DriverToken`] the table pins the driver's driver-level
    /// calls to, rather than any token that reaches the `Driver` level:
    /// [`InterruptRouter`] holds its drivers as `dyn`, so the token type of
    /// [`IRQCapable::disable_irqs`] is fixed and a generic one cannot reach it.
    /// The `token` is consumed and returned in both arms.
    fn disable_interrupt_vector(
        &self,
        vector: InterruptVector,
        token: DriverToken,
    ) -> Result<DriverToken, (Errno, DriverToken)> {
        let (driver, token) = match INTERRUPT_ROUTER.driver_for(vector, token) {
            (Some(driver), token) => (driver, token),
            (None, token) => {
                return Err((Errno::EINVAL, token));
            }
        };

        driver.disable_irqs(token)
    }

    /// Tells the driver in charge of `vector` to start raising it.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if no driver is in charge of `vector`, or whatever
    /// the driver itself reports.
    ///
    /// # Token
    ///
    /// Takes a [`DriverToken`], for the reason given on
    /// [`disable_interrupt_vector`](Self::disable_interrupt_vector). The
    /// `token` is consumed and returned in both arms.
    fn enable_interrupt_vector(
        &self,
        vector: InterruptVector,
        token: DriverToken,
    ) -> Result<DriverToken, (Errno, DriverToken)> {
        let (driver, token) = match INTERRUPT_ROUTER.driver_for(vector, token) {
            (Some(driver), token) => (driver, token),
            (None, token) => {
                return Err((Errno::EINVAL, token));
            }
        };

        driver.enable_irqs(token)
    }

    /// Asks the driver in charge of `vector` whether it is raising it.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if no driver is in charge of `vector`, or whatever
    /// the driver itself reports.
    ///
    /// # Token
    ///
    /// Takes a [`DriverToken`], for the reason given on
    /// [`disable_interrupt_vector`](Self::disable_interrupt_vector). The
    /// `token` is consumed and returned in both arms.
    fn enabled_interrupt_vector(
        &self,
        vector: InterruptVector,
        token: DriverToken,
    ) -> Result<(bool, DriverToken), (Errno, DriverToken)> {
        let (driver, token) = match INTERRUPT_ROUTER.driver_for(vector, token) {
            (Some(driver), token) => (driver, token),
            (None, token) => {
                return Err((Errno::EINVAL, token));
            }
        };

        driver.irqs_enabled(token)
    }
}
