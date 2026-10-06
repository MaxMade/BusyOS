//! Consoles: the devices the kernel can show text on.
//!
//! [`ConsoleOutput`] is what every one of them offers, and [`Consoles`] is
//! the list of those that registered themselves. A driver joins it from its
//! own [`Module::init`], wrapped in its [`ConsoleOutputDriver`] variant.

use crate::{
    driver::{framebuffer::Framebuffer, module::Module},
    kernel::{
        arc::Arc,
        linked_list::LinkedList,
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken, ReadGuard, Shared, Token},
        ticketlock::{DriverRWTicketlock, RWTicketlock, RWTicketlockDriverID},
    },
    user::errno::{Errno, ToErrno},
};

/// A device that can show text.
///
/// Every method is generic over its token, so nothing here pins a console to
/// one token type. That is what keeps [`Consoles`] a list of
/// [`ConsoleOutputDriver`] rather than of trait objects: a method generic
/// over its token has no single address to put in a vtable.
pub trait ConsoleOutput: Module {
    /// Shows `buffer` at the console's cursor, returning how many bytes of it
    /// were written.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn write<S, Token>(&self, buffer: &S, token: Token) -> Result<(usize, Token), (Errno, Token)>
    where
        S: AsRef<str>,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;

    /// Shows `buffer` at the console's cursor without taking any lock.
    ///
    /// For the panic path, see
    /// [`__printk_emergency`](crate::kernel::printk::__printk_emergency).
    /// Takes no lock and no token: whatever lock the console normally uses
    /// may be held, possibly by the very code that panicked, so `buffer` is
    /// written over whatever state the console is in.
    ///
    /// # Safety
    ///
    /// Nothing else may be writing to the console: interrupts have to be
    /// masked, and no other core may be running. Only for when nothing is
    /// going to run afterwards anyway.
    unsafe fn emergency_write<S>(&self, buffer: &S)
    where
        S: AsRef<str>;

    /// Blanks the console and moves its cursor back to the start.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn clear<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;
}

/// A handle on a registered console driver.
///
/// One variant per console, for the reason [`ConsoleOutput`] gives. Cloning a
/// handle is the [`Arc`] clone of the driver it names.
#[derive(Clone)]
pub enum ConsoleOutputDriver {
    /// The framebuffer driver, see [`Framebuffer`].
    Framebuffer(Arc<Framebuffer>),
}

// The enum is a `Module` so that it can be a `ConsoleOutput`, which requires
// one. Only `name` means anything on a handle, since the driver it names has
// long been initialised by the time a handle exists.
impl Module for ConsoleOutputDriver {
    /// Never called.
    ///
    /// A handle names a driver that is already up. Each concrete driver is
    /// brought up through its own [`Module::init`], which is what creates the
    /// handle in the first place.
    ///
    /// # Panics
    ///
    /// Always.
    fn init<Token>(_: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        panic!("ConsoleOutputDriver::init(...) must never be invoked directly.")
    }

    /// The name of the driver this handle names, see [`Module::name`].
    fn name(&self) -> &'static str {
        match self {
            ConsoleOutputDriver::Framebuffer(framebuffer) => framebuffer.name(),
        }
    }
}

impl ConsoleOutput for ConsoleOutputDriver {
    /// Forwards to the concrete console, see [`ConsoleOutput::write`].
    fn write<S, Token>(&self, buffer: &S, token: Token) -> Result<(usize, Token), (Errno, Token)>
    where
        S: AsRef<str>,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            ConsoleOutputDriver::Framebuffer(framebuffer) => framebuffer.write(buffer, token),
        }
    }

    /// Forwards to the concrete console, see
    /// [`ConsoleOutput::emergency_write`].
    unsafe fn emergency_write<S>(&self, buffer: &S)
    where
        S: AsRef<str>,
    {
        match self {
            // SAFETY: the caller's contract is the one of the concrete
            // console.
            ConsoleOutputDriver::Framebuffer(framebuffer) => unsafe {
                framebuffer.emergency_write(buffer)
            },
        }
    }

    /// Forwards to the concrete console, see [`ConsoleOutput::clear`].
    fn clear<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            ConsoleOutputDriver::Framebuffer(framebuffer) => framebuffer.clear(token),
        }
    }
}

/// Every console that has registered itself, in the order they did.
pub struct Consoles(LinkedList<ConsoleOutputDriver>);

impl Consoles {
    /// Adds `driver` to the list of consoles the kernel may use.
    ///
    /// Takes a [`ConsoleOutputDriver`], the enum naming every registered
    /// console, so a caller wraps the `Arc<Driver>` it already holds in that
    /// driver's variant and keeps using its own handle afterwards.
    ///
    /// # Errors
    ///
    /// [`Errno::ENOMEM`] if the list could not grow. The list is unchanged in
    /// that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn register<Token>(
        driver: ConsoleOutputDriver,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut consoles, token) = CONSOLES.acquire(token);

        let token = match consoles.0.try_push_back(driver, token) {
            Ok(token) => token,
            Err((error, token)) => {
                let token = consoles.release(token);
                let errno = error.to_errno();
                return Err((errno, token));
            }
        };

        Ok(consoles.release(token))
    }

    /// Walks every registered console, in the order they registered.
    ///
    /// The returned [`ConsolesIter`] holds [`CONSOLES`] shared for as long as
    /// it lives, so no console can register in the meantime, and yields a
    /// clone of each handle rather than a reference into the list.
    ///
    /// # Token
    ///
    /// The `token` is consumed and stored in the iterator. Hand the iterator
    /// to [`ConsolesIter::release`] to unlock the list and get it back;
    /// merely dropping the iterator leaves the list locked for good.
    pub fn iter<Token>(token: Token) -> ConsolesIter<Token>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (consoles, token) = CONSOLES.acquire_shared(token);

        ConsolesIter {
            consoles,
            token,
            next: 0,
        }
    }

    /// Returns a handle on the console the kernel is configured to use, or
    /// [`None`] if no console registered.
    ///
    /// That is the first one to register for now.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn get<Token>(token: Token) -> (Option<ConsoleOutputDriver>, Token)
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // TODO(@MaxMade): Pick the console named by the build configuration
        // or the kernel command line, once either exists.
        let (consoles, token) = CONSOLES.acquire_shared(token);
        let driver = consoles.0.front().cloned();

        (driver, consoles.release(token))
    }

    /// Returns the console [`get`](Self::get) would, without taking the
    /// lock.
    ///
    /// For the panic path, which has no token and may have panicked while
    /// holding [`CONSOLES`].
    ///
    /// # Safety
    ///
    /// Nothing may be registering a console at the same time. That holds
    /// with interrupts masked on the only core there is.
    pub unsafe fn emergency_get() -> Option<ConsoleOutputDriver> {
        // SAFETY: see the function's contract. A registration the panic
        // interrupted halfway has either linked its node or not, and the
        // front of the list is valid either way.
        let consoles = unsafe { &*CONSOLES.data_ptr() };

        consoles.0.front().cloned()
    }
}

/// An iterator over every registered console, see [`Consoles::iter`].
///
/// Yields a clone of each [`ConsoleOutputDriver`], which the caller may keep
/// or drop as it likes, since [`Consoles`] still holds a handle of its own.
pub struct ConsolesIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    consoles: ReadGuard<'static, Consoles, RWTicketlock<RWTicketlockDriverID>>,
    token: Token<RWTicketlockDriverID, From, Shared>,
    next: usize,
}

impl<From> ConsolesIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    /// Unlocks [`CONSOLES`] and returns the token given to
    /// [`Consoles::iter`].
    pub fn release(self) -> From {
        self.consoles.release(self.token)
    }
}

impl<From> Iterator for ConsolesIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    type Item = ConsoleOutputDriver;

    fn next(&mut self) -> Option<Self::Item> {
        // The guard cannot lend the list for longer than a call, so the
        // position is kept as an index. A machine has a handful of consoles
        // at most, which keeps walking up to it again cheap.
        let driver = self.consoles.0.iter().nth(self.next).cloned();
        self.next += 1;
        driver
    }
}

/// The one list of registered consoles.
pub static CONSOLES: DriverRWTicketlock<Consoles> =
    DriverRWTicketlock::new(RWTicketlock::new(), Consoles(LinkedList::new()));
