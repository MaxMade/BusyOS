//! Interrupt routing: which driver owns which vector, and the two halves an
//! interrupt is handled in.
//!
//! [`InterruptVectorTable`] answers the only question the interrupt path
//! asks, namely which [`IRQCapable`] driver is in charge of the vector that
//! just arrived. [`InterruptController`] is the other side of that, the chip
//! which delivers the vector, and it is what a driver goes through to have
//! its own vector masked or unmasked.
//!
//! Handling a vector is split in two. The prologue runs in interrupt context,
//! does only what cannot wait, and reports whether the rest is needed. The
//! epilogue then runs from a level that may still take the driver's own
//! locks.

#[cfg(target_arch = "x86_64")]
use crate::{driver::x86_64::x2apic::X2Apic, kernel::arc::Arc};

use crate::{
    arch::{InterruptVector, generic::cpu::InterruptVector as GenericInterruptVector},
    driver::module::Module,
    kernel::{
        locking::{
            CanAcquire, DriverLevelID, EpilogueLevelID, LockId, MemoryManagementLevelID,
            PreviousToken, PrologueLevelID,
        },
        ticketlock::{PrologueRWTicketlock, RWTicketlock},
    },
    user::errno::Errno,
};

/// A driver that owns one or more interrupt vectors.
///
/// Every method is generic over its token, so nothing here pins a driver to
/// one token type. That is possible because [`InterruptVectorTable`] holds its
/// drivers as [`IRQCapableDriver`], the enum naming each of them, rather than
/// as a trait object: a method generic over its token has no single address to
/// put in a vtable, so a trait whose methods are generic cannot be made into
/// an object at all.
pub trait IRQCapable: Module {
    /// Tells the device to start raising its interrupts.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn enable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;

    /// Tells the device to stop raising its interrupts.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn disable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;

    /// Whether the device is currently raising its interrupts.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn irqs_enabled<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;

    /// The half that runs in interrupt context, straight off the vector.
    ///
    /// Does only what cannot be deferred, which is normally quieting the
    /// device, and reports whether [`epilogue`](Self::epilogue) has to run
    /// afterwards to finish the work.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms. A prologue holds
    /// nothing on entry and has to reach the `Epilogue` level itself to
    /// request its epilogue, which is what the bound asks for.
    fn prologue<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken;

    /// The deferred half, run once a prologue has asked for it.
    ///
    /// Runs at the `Epilogue` level instead of in interrupt context, so it may
    /// take the driver's own locks and do the work the prologue left.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn epilogue<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<PrologueLevelID as LockId>::Level> + PreviousToken;
}

/// A handle on a driver that owns interrupt vectors, as the
/// [`InterruptVectorTable`] holds it.
///
/// One variant per driver implementing [`IRQCapable`], rather than an
/// `Arc<dyn IRQCapable>`: the trait's methods are generic over their token and
/// therefore have no vtable entry, so a trait object cannot be formed. The
/// methods below match on the variant instead, which dispatches to the
/// concrete driver statically and lets each of them stay generic over its own
/// token — that is exactly what a `dyn` handle cost before, since it pinned
/// every call to one token type.
///
/// Cloning a handle is the [`Arc`] clone of the driver it names, which is
/// what makes a slot of the vector table cheap to read out.
///
/// Each driver that implements [`IRQCapable`] adds its variant here, gated on
/// the architecture it exists for, and one arm to every method.
#[derive(Clone)]
pub enum IRQCapableDriver {
    #[cfg(target_arch = "x86_64")]
    /// The x2APIC driver, see [`X2Apic`].
    X2Apic(Arc<X2Apic>),
}

// Every method below matches on the variant and forwards to the concrete
// driver's `IRQCapable` method, handing `token` along. On an architecture
// none of the variants exist for, each match is empty and diverges, and
// `token` is discarded to keep it from reading as unused there.
impl IRQCapableDriver {
    /// Tells the device to start raising its interrupts, see
    /// [`IRQCapable::enable_irqs`].
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn enable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let _ = token;
        match self {
            #[cfg(target_arch = "x86_64")]
            IRQCapableDriver::X2Apic(x2apic) => x2apic.enable_irqs(token),
        }
    }

    /// Tells the device to stop raising its interrupts, see
    /// [`IRQCapable::disable_irqs`].
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn disable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let _ = token;
        match self {
            #[cfg(target_arch = "x86_64")]
            IRQCapableDriver::X2Apic(x2apic) => x2apic.disable_irqs(token),
        }
    }

    /// Whether the device is currently raising its interrupts, see
    /// [`IRQCapable::irqs_enabled`].
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn irqs_enabled<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let _ = token;
        match self {
            #[cfg(target_arch = "x86_64")]
            IRQCapableDriver::X2Apic(x2apic) => x2apic.irqs_enabled(token),
        }
    }

    /// The half that runs in interrupt context, see [`IRQCapable::prologue`].
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn prologue<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken,
    {
        let _ = token;
        match self {
            #[cfg(target_arch = "x86_64")]
            IRQCapableDriver::X2Apic(x2apic) => x2apic.prologue(token),
        }
    }

    /// The deferred half, see [`IRQCapable::epilogue`].
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn epilogue<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<PrologueLevelID as LockId>::Level> + PreviousToken,
    {
        let _ = token;
        match self {
            #[cfg(target_arch = "x86_64")]
            IRQCapableDriver::X2Apic(x2apic) => x2apic.epilogue(token),
        }
    }
}

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
/// A slot holds an [`Arc`] handle on its driver, and dropping the last handle
/// on a driver frees it, which needs a token that [`Drop`] has no way of
/// being given. The table therefore lives in a `static` of this module,
/// which never goes out of scope.
pub struct InterruptVectorTable {
    vector_table: PrologueRWTicketlock<[Option<IRQCapableDriver>; InterruptVector::MAX_NUM]>,
}

impl InterruptVectorTable {
    /// Creates a table in which no vector has a driver yet.
    ///
    /// Every slot starts out empty and the table is one fixed-size array, so
    /// this allocates nothing and may be used to build a `static`.
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
        vector: InterruptVector,
        driver: IRQCapableDriver,
        token: Token,
    ) -> Result<(Option<IRQCapableDriver>, Token), (Errno, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut vector_table = INTERRUPT_VECTOR_TABLE.vector_table.acquire(token);

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
    ) -> (Option<IRQCapableDriver>, Token)
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
static INTERRUPT_VECTOR_TABLE: InterruptVectorTable = InterruptVectorTable::new();

/// The chip that delivers interrupt vectors, such as a PIC or an APIC.
///
/// The controller owns delivery and [`InterruptVectorTable`] owns the mapping
/// from a vector to its driver, so the default methods below are the path a
/// caller takes to reach whoever is in charge of a vector. Only
/// [`acknowledge`](Self::acknowledge) is left to the concrete chip.
pub trait InterruptController: Module {
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
    /// The `token` is consumed and returned in both arms. Any token reaching
    /// the `Driver` level does, since the router hands the driver back as an
    /// [`IRQCapableDriver`] whose calls are generic over their token.
    fn disable_interrupt_vector<Token>(
        &self,
        vector: InterruptVector,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (driver, token) = match INTERRUPT_VECTOR_TABLE.driver_for(vector, token) {
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
    /// The `token` is consumed and returned in both arms, as on
    /// [`disable_interrupt_vector`](Self::disable_interrupt_vector).
    fn enable_interrupt_vector<Token>(
        &self,
        vector: InterruptVector,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (driver, token) = match INTERRUPT_VECTOR_TABLE.driver_for(vector, token) {
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
    /// The `token` is consumed and returned in both arms, as on
    /// [`disable_interrupt_vector`](Self::disable_interrupt_vector).
    fn enabled_interrupt_vector<Token>(
        &self,
        vector: InterruptVector,
        token: Token,
    ) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (driver, token) = match INTERRUPT_VECTOR_TABLE.driver_for(vector, token) {
            (Some(driver), token) => (driver, token),
            (None, token) => {
                return Err((Errno::EINVAL, token));
            }
        };

        driver.irqs_enabled(token)
    }
}

/// A handle on the driver of an interrupt controller.
///
/// One variant per chip that can deliver vectors, for the reason
/// [`IRQCapableDriver`] gives: [`InterruptController`] is generic over its
/// token, so it has no vtable and cannot be held as a `dyn` value.
///
/// Cloning a handle is the [`Arc`] clone of the driver it names.
#[derive(Clone)]
pub enum InterruptControllerDriver {
    #[cfg(target_arch = "x86_64")]
    /// The x2APIC driver, see [`X2Apic`].
    X2Apic(Arc<X2Apic>),
}

// The enum is a `Module` so that it can be an `InterruptController`, which
// requires one. Only `name` means anything on a handle, since the driver it
// names has long been initialised by the time a handle exists.
impl Module for InterruptControllerDriver {
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
        panic!("InterruptControllerDriver::init(...) must never be invoked directly.")
    }

    /// The name of the driver this handle names, see [`Module::name`].
    fn name(&self) -> &'static str {
        match self {
            #[cfg(target_arch = "x86_64")]
            InterruptControllerDriver::X2Apic(x2apic) => x2apic.name(),
        }
    }
}

impl InterruptController for InterruptControllerDriver {
    /// Forwards to the concrete chip, see [`InterruptController::acknowledge`].
    fn acknowledge<Token>(
        &self,
        vector: InterruptVector,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            #[cfg(target_arch = "x86_64")]
            InterruptControllerDriver::X2Apic(x2_apic) => x2_apic.acknowledge(vector, token),
        }
    }
}

/// The chip that delivers interrupt vectors on this machine.
///
/// Room for one, not a list, and it is what the interrupt path acknowledges
/// a vector to. A machine with several controllers is not handled yet, see
/// [`register`](Self::register).
///
/// Locked at the `Prologue` level, like [`InterruptVectorTable`] and for the
/// same reason: the interrupt path reads it, so the read must not be
/// interrupted by the very vector it is about to acknowledge.
pub struct InterruptControllers(PrologueRWTicketlock<Option<InterruptControllerDriver>>);

impl InterruptControllers {
    /// Creates the slot with no controller in it yet.
    ///
    /// Allocates nothing and may be used to build a `static`.
    pub const fn new() -> Self {
        Self(PrologueRWTicketlock::new(RWTicketlock::new(), None))
    }

    /// Makes `driver` the controller every vector is acknowledged to.
    ///
    /// Takes an [`InterruptControllerDriver`], so a caller wraps the
    /// `Arc<Driver>` it already holds in that driver's variant and keeps its
    /// own handle.
    ///
    /// # Panics
    ///
    /// If a controller is already registered. Which of two controllers owns
    /// a given vector, and how a driver reaches the right one, is not
    /// decided yet, so a second one is refused rather than silently ignored
    /// or silently preferred.
    ///
    /// # Token
    ///
    /// The `token` has to reach the `MemoryManagement` level, which is above
    /// the `Prologue` level the slot is locked at. It is consumed for the
    /// write and returned in both arms.
    pub fn register<Token>(
        driver: InterruptControllerDriver,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<MemoryManagementLevelID as LockId>::Level> + PreviousToken,
    {
        let mut ic = INTERRUPT_CONTROLLERS.0.acquire(token);

        match *ic {
            Some(_) => {
                // TODO(@MaxMade): How to handle multiple interrupt controllers?
                todo!("Handle multiple interrupt controller");
            }
            None => {
                *ic = Some(driver);
            }
        }

        Ok(ic.release())
    }
}

/// The one interrupt controller, which every acknowledgement goes through.
static INTERRUPT_CONTROLLERS: InterruptControllers = InterruptControllers::new();
