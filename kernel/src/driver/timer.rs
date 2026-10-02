//! Timers: the clock sources the kernel can ask to be interrupted by.
//!
//! A timer is a device that raises an interrupt after, or every, some
//! interval. [`Timer`] is what every one of them offers, and [`Timers`] is
//! the list of those that registered themselves. A timer may be per core,
//! like the local APIC's, or shared by all of them, like the HPET, which is
//! what [`TimerGroup`] is meant to express. It is not part of a request
//! yet: [`Timer::setup`] arms whatever the timer reaches from the calling
//! core, which for a per-core timer is that core's own.
//!
//! A timer is an [`IRQCapable`] device first: setting an interval only says
//! when the interrupt comes, and the two halves it is then handled in are
//! the ones every driver has.

#[cfg(target_arch = "x86_64")]
use crate::{driver::x86_64::x2apic::X2Apic, kernel::arc::Arc};

use crate::{
    arch::CPUID,
    driver::irq::IRQCapable,
    kernel::{
        linked_list::LinkedList,
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken, ReadGuard, Shared, Token},
        ticketlock::{DriverRWTicketlock, RWTicketlock, RWTicketlockDriverID},
        time::NanoSeconds,
    },
    user::errno::{Errno, ToErrno},
};

/// Which cores an interval is meant to fire on.
///
/// Meant to be asked for together with the interval, since the two are one
/// setting in the hardware: a per-core timer is programmed on the core it
/// fires on, so the choice decides which core's registers are written.
///
/// Not taken by [`Timer::setup`] yet, see the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimerGroup {
    /// Every core, each one interrupted on its own.
    All,

    /// Any one core, whichever is cheapest for the timer to pick.
    ///
    /// For work that has to happen somewhere but not anywhere in
    /// particular, such as a deadline the kernel keeps for itself.
    Any,

    /// One named core, and only that one.
    Specific(CPUID),
}

/// A device that can interrupt a core after a given interval.
///
/// Every method is generic over its token, so nothing here pins a timer to
/// one token type. That is what keeps [`Timers`] a list of [`TimerDriver`]
/// rather than of trait objects, for the reason [`IRQCapable`] gives: a
/// method generic over its token has no single address to put in a vtable.
pub trait Timer: IRQCapable {
    /// Arms the timer to interrupt after `interval`.
    ///
    /// The interval is what the caller wants, not what it gets. A timer
    /// counts in ticks of its own clock, so the interval is rounded to
    /// whatever that clock can express, and one too fine for the hardware to
    /// resolve is rejected rather than silently turned into a much shorter
    /// one.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if the timer cannot produce `interval`.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn setup<Token>(
        &self,
        interval: NanoSeconds,
        peridoc: bool,
        token: Token,
    ) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;
}

/// A handle on a registered timer driver.
///
/// One variant per timer, for the reason [`Timer`] gives. Cloning a handle
/// is the [`Arc`] clone of the driver it names.
#[derive(Clone)]
pub enum TimerDriver {
    #[cfg(target_arch = "x86_64")]
    /// The x2APIC driver, see [`X2Apic`].
    X2Apic(Arc<X2Apic>),
}

/// Every timer that has registered itself, in the order they did.
///
/// More than one is normal: a machine has both a per-core timer and a shared
/// one, and which of them suits a request depends on the [`TimerGroup`] it
/// asks for.
pub struct Timers(LinkedList<TimerDriver>);

impl Timers {
    /// Adds `driver` to the list of timers the kernel may use.
    ///
    /// Takes a [`TimerDriver`], the enum naming every registered timer, so a
    /// caller wraps the `Arc<Driver>` it already holds in that driver's
    /// variant and keeps using its own handle afterwards.
    ///
    /// # Errors
    ///
    /// [`Errno::ENOMEM`] if the list could not grow. The list is unchanged in
    /// that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn register<Token>(driver: TimerDriver, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut timers, token) = TIMERS.acquire(token);

        let token = match timers.0.try_push_back(driver, token) {
            Ok(token) => token,
            Err((error, token)) => {
                let token = timers.release(token);
                let errno = error.to_errno();
                return Err((errno, token));
            }
        };

        Ok(timers.release(token))
    }

    /// Walks every registered timer, in the order they registered.
    ///
    /// The returned [`TimersIter`] holds [`TIMERS`] shared for as long as it
    /// lives, so no timer can register in the meantime, and yields a clone
    /// of each handle rather than a reference into the list.
    ///
    /// # Token
    ///
    /// The `token` is consumed and stored in the iterator. Hand the iterator
    /// to [`TimersIter::release`] to unlock the list and get it back; merely
    /// dropping the iterator leaves the list locked for good.
    pub fn iter<Token>(token: Token) -> TimersIter<Token>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (timers, token) = TIMERS.acquire_shared(token);

        TimersIter {
            timers,
            token,
            next: 0,
        }
    }

    /// Returns a handle on the timer the kernel is configured to use, or
    /// [`None`] if no timer registered.
    ///
    /// That is the first one to register for now.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn get<Token>(token: Token) -> (Option<TimerDriver>, Token)
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // TODO(@MaxMade): Pick the timer named by the build configuration or
        // the kernel command line, once either exists.
        let (timers, token) = TIMERS.acquire_shared(token);
        let driver = timers.0.front().cloned();

        (driver, timers.release(token))
    }
}

/// An iterator over every registered timer, see [`Timers::iter`].
///
/// Yields a clone of each [`TimerDriver`], which the caller may keep or drop
/// as it likes, since [`Timers`] still holds a handle of its own.
pub struct TimersIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    timers: ReadGuard<'static, Timers, RWTicketlock<RWTicketlockDriverID>>,
    token: Token<RWTicketlockDriverID, From, Shared>,
    next: usize,
}

impl<From> TimersIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    /// Unlocks [`TIMERS`] and returns the token given to [`Timers::iter`].
    pub fn release(self) -> From {
        self.timers.release(self.token)
    }
}

impl<From> Iterator for TimersIter<From>
where
    From: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    type Item = TimerDriver;

    fn next(&mut self) -> Option<Self::Item> {
        // The guard cannot lend the list for longer than a call, so the
        // position is kept as an index. A machine has a handful of timers
        // at most, which keeps walking up to it again cheap.
        let driver = self.timers.0.iter().nth(self.next).cloned();
        self.next += 1;
        driver
    }
}

/// The one list of registered timers.
pub static TIMERS: DriverRWTicketlock<Timers> =
    DriverRWTicketlock::new(RWTicketlock::new(), Timers(LinkedList::new()));
