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
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
        ticketlock::{DriverRWTicketlock, RWTicketlock},
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
}

/// The one list of registered timers.
pub static TIMERS: DriverRWTicketlock<Timers> =
    DriverRWTicketlock::new(RWTicketlock::new(), Timers(LinkedList::new()));
