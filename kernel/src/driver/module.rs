use crate::{
    driver::acpi::acpi::Acpi,
    kernel::{
        arc::Arc,
        linked_list::LinkedList,
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
        ticketlock::{DriverRWTicketlock, RWTicketlock},
    },
    user::errno::{Errno, ToErrno},
};

#[cfg(target_arch = "x86_64")]
use crate::driver::x86_64::x2apic::X2Apic;

/// Every driver that has registered itself, in the order they did.
///
/// A list rather than a table: a driver is looked up by walking it, which is
/// what the passes over the drivers do anyway, and nothing needs a driver by
/// index.
pub struct Modules(LinkedList<ModuleDriver>);

/// The one list of registered drivers.
pub static MODULES: DriverRWTicketlock<Modules> =
    DriverRWTicketlock::new(RWTicketlock::new(), Modules(LinkedList::new()));

unsafe extern "C" {
    /// First entry of the `init_modules_callbacks` array the linker script
    /// gathers the [`module!`](driver_macro::module) callbacks into.
    pub static __init_modules_callbacks_start: u64;

    /// One past the last entry of that array.
    pub static __init_modules_callbacks_end: u64;
}

impl Modules {
    /// Adds `module` to the list every later pass over the drivers walks.
    ///
    /// Takes a [`ModuleDriver`], the enum naming every registered driver, so a
    /// caller wraps the `Arc<Driver>` it already holds in that driver's variant
    /// and keeps using its own handle afterwards.
    ///
    /// # Errors
    ///
    /// [`Errno::ENOMEM`] if the list could not grow. The list is unchanged in
    /// that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn register<Token>(module: ModuleDriver, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut modules, token) = MODULES.acquire(token);

        let token = match modules.0.try_push_back(module, token) {
            Ok(token) => token,
            Err((error, token)) => {
                let token = modules.release(token);
                let errno = error.to_errno();
                return Err((errno, token));
            }
        };

        Ok(modules.release(token))
    }

    /// Runs the `init` callback of every driver built into the kernel.
    ///
    /// The callbacks are not held in a list that has to be built at run
    /// time. Each [`module!`](driver_macro::module) invocation emits one
    /// function pointer into an array the linker script gathers between
    /// [`__init_modules_callbacks_start`] and
    /// [`__init_modules_callbacks_end`], and this walks that array. A driver
    /// registers itself from its own callback, which is what fills
    /// [`MODULES`].
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned. It is not used here, but every
    /// callback needs one, and taking it is what proves no lock at or below
    /// the `Driver` level is held while they run.
    pub fn init<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let start: *const u64 = unsafe { &__init_modules_callbacks_start };
        let end: *const u64 = unsafe { &__init_modules_callbacks_end };

        let mut ptr = start;
        while ptr < end {
            let init: extern "C" fn() = unsafe { core::mem::transmute(*ptr) };

            init();

            ptr = unsafe { ptr.add(1) };
        }

        token
    }
}

/// A driver the kernel can bring up and keep track of.
///
/// Every driver implements this, whatever else it also is. An
/// [`IRQCapable`](crate::driver::irq::IRQCapable) device, an
/// [`InterruptController`](crate::driver::irq::InterruptController) and a
/// [`Timer`](crate::driver::timer::Timer) are all `Module` first, so a
/// driver that is several of those at once registers with each registry and
/// is still one object.
pub trait Module: Send + Sync {
    /// Brings the device up and registers the driver with every registry it
    /// belongs in.
    ///
    /// Called once per driver, from the callback
    /// [`module!`](driver_macro::module) emits, in the order the priorities
    /// given there impose. An implementation allocates its own state, wraps
    /// it in an [`Arc`], and hands a clone to each registry it joins, which
    /// is why this takes no `self`.
    ///
    /// # Errors
    ///
    /// Whatever the driver could not do. A device that is simply absent is
    /// not an error, so a driver that finds no hardware returns [`Ok`]
    /// without registering.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;

    /// The driver's name, as it appears in a log line.
    ///
    /// The same name the driver gives [`module!`](driver_macro::module), and
    /// meant for a human reader alone. Nothing looks a driver up by it.
    fn name(&self) -> &'static str;
}

/// A handle on a registered driver, as [`Modules`] holds it.
///
/// One variant per driver that registers itself, rather than an
/// `Arc<dyn Module>`: [`Module::init`] is generic over its token, so the trait
/// has no vtable to hold, and [`Arc`] cannot unsize to a `dyn` value in this
/// kernel anyway. Matching on the enum dispatches to the concrete driver
/// statically, so a driver's own methods stay generic over their token.
///
/// Cloning a handle is the [`Arc`] clone of the driver it names.
#[derive(Clone)]
pub enum ModuleDriver {
    /// The ACPI driver, see [`Acpi`].
    Acpi(Arc<Acpi>),

    #[cfg(target_arch = "x86_64")]
    /// The x2APIC driver, see [`X2Apic`].
    X2Apic(Arc<X2Apic>),
}
