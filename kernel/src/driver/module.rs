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

pub struct Modules(LinkedList<ModuleDriver>);

pub static MODULES: DriverRWTicketlock<Modules> =
    DriverRWTicketlock::new(RWTicketlock::new(), Modules(LinkedList::new()));

unsafe extern "C" {
    pub static __init_modules_callbacks_start: u64;
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

pub trait Module: Send + Sync {
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;
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
}
