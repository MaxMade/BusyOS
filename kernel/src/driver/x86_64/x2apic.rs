//! The *Local Advanced Programmable Interrupt Controller* in x2APIC mode.
//!
//! Every logical processor owns one local APIC, which is the core's interface
//! to the interrupt subsystem. x2APIC is its extended mode: the register file
//! is reached through MSRs rather than a memory-mapped page, and the
//! identifier of a local APIC widens from 8 to 32 bits.
//!
//! What the module drives so far is the local APIC *timer*, the per-core
//! countdown whose expiry is the kernel's tick. The timer has no idea how
//! long one of its ticks is, so [`X2Apic::init`] measures that against the
//! [`PIT`], the one clock that is present and of known rate before anything
//! else has been brought up.
//!
//! [`LapicID`] is the other half of the module, and the older one: it names
//! a local APIC, so that the ACPI driver can record the identifiers the
//! firmware reports in its [`MADT`](crate::driver::acpi::madt::MADT) while
//! walking the tables.
//!
//! The driver registers itself four times over, once for each thing it is:
//! a [`Module`], an [`InterruptController`], an [`IRQCapable`] device that
//! owns the timer vector, and a [`Timer`].

use driver_macro::module;

use crate::{
    arch::{
        InterruptVector,
        generic::cpu::InterruptVector as GenericInterruptVector,
        x86_64::{
            msr::{
                MSR, X2ApicCurrentCount, X2ApicDivide, X2ApicDivideMode, X2ApicEOI,
                X2ApicInitialCount, X2ApicLVTTimer, X2ApicLVTTimerMode,
            },
            pit::PIT,
        },
    },
    driver::{
        irq::{
            IRQCapable, IRQCapableDriver, InterruptController, InterruptControllerDriver,
            InterruptControllers, InterruptVectorTable,
        },
        module::{Module, ModuleDriver, Modules},
        timer::{Timer, TimerDriver, Timers},
    },
    kernel::{
        arc::Arc,
        locking::{
            CanAcquire, DriverLevelID, EpilogueLevelID, LockId, PreviousToken, PrologueLevelID,
        },
        ticketlock::{DriverTicketlock, Ticketlock},
        time::{MilliSeconds, NanoSeconds, TimeUnit},
    },
    user::errno::Errno,
};

/// The 32-bit identifier of a local APIC in x2APIC mode.
///
/// The identifier is assigned by the hardware and is unique across the system.
/// It names the core a local APIC belongs to and is therefore what an
/// interrupt is addressed to.
///
/// The value is not an index: the firmware is free to leave gaps, so the
/// identifiers of `n` cores are not necessarily `0..n`.
pub struct LapicID(u32);

impl LapicID {
    /// Wraps a raw 32-bit identifier.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` is an identifier the platform
    /// actually reports, either through a firmware table such as the
    /// [`MADT`](crate::driver::acpi::madt::MADT) or through the local APIC's
    /// own ID register. Constructing an identifier no local APIC answers to
    /// is safe in the memory-safety sense, but interrupts addressed to it are
    /// never delivered.
    pub const unsafe fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

module! {
    name: "x2apic",
    priority: 10,
    driver: crate::driver::x86_64::x2apic::X2Apic,
}

/// Mutable driver state, behind the driver-level lock of [`X2Apic`].
///
/// Empty so far. None of the timer's registers are kept here: they are read
/// from and written to their MSRs where they are needed, so there is no copy
/// that could drift from what the hardware holds. What was measured of the
/// timer never changes after [`X2Apic::init`] and therefore lives outside the
/// lock, in [`X2Apic::ticks_per_ms`].
struct State {}

/// The local APIC of the core that brought the driver up.
///
/// The registers are not behind a lock: every core reaches only its own local
/// APIC through the x2APIC MSRs, so an access from one core cannot race with
/// one from another, and the methods below go to the MSRs directly. The rate
/// of the timer is measured once and read-only afterwards, so it needs no
/// lock either.
pub struct X2Apic {
    state: DriverTicketlock<State>,
    /// Timer ticks in one millisecond, as measured against the [`PIT`], at
    /// [`X2ApicDivideMode::Divide1`].
    ///
    /// This is what makes an interval in real time expressible at all: the
    /// rate is not architectural and differs from machine to machine, so it
    /// has to be measured on each one.
    ticks_per_ms: usize,
}

impl Module for X2Apic {
    /// Measures the timer's rate, claims a vector for it, and registers the
    /// driver.
    ///
    /// The rate is measured by running the timer down from its largest count
    /// while the [`PIT`] waits out a known interval, and seeing how far it
    /// got. One correction is taken off that, namely what the measurement
    /// costs when the interval is nothing at all: the call into the `PIT`,
    /// the programming of its channel 2, and the read of the timer back.
    /// That is not negligible at this resolution, since the divisor is set
    /// to one and a tick is a single bus cycle.
    ///
    /// The interval starts at 32 ms and is halved until the timer survives
    /// it without reaching zero, so that a fast timer is measured over a
    /// shorter span rather than pinned at zero.
    ///
    /// The timer is left masked, stopped and in
    /// [`X2ApicLVTTimerMode::OneShot`], with the allocated vector in its
    /// local vector table entry. [`IRQCapable::enable_irqs`] unmasks it and
    /// [`Timer::setup`] starts it.
    ///
    /// # Panics
    ///
    /// If the rate cannot be measured or comes out as zero, if no interrupt
    /// vector is left, if the driver cannot be allocated, or if any of the
    /// four registrations fails. All of them mean the core has no
    /// usable tick, which there is no way to carry on without.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        let state = State {};

        // Determine ticks per duration. The timer is masked throughout, so
        // the counts below expire without ever raising a vector: nothing is
        // ready to handle one yet.
        //
        // Every measurement counts from `u32::MAX`, and the timer counts
        // down, so what a measurement wants is the initial count less the
        // current one rather than the current one itself.
        let mut lvt_timer = unsafe { X2ApicLVTTimer::read() };
        lvt_timer.set_masked(true); // Mask interrupts
        lvt_timer.set_timer_mode(X2ApicLVTTimerMode::OneShot); // One-shot mode
        unsafe { lvt_timer.write() };

        let mut divider = unsafe { X2ApicDivide::read() };
        divider.set_divider(X2ApicDivideMode::Divide1); // Set smallest divider
        unsafe { divider.write() };

        let mut wait = MilliSeconds::from(32);
        let mut current_cnt = loop {
            // Halving has run out: even a wait of 1 ms outlasts the whole
            // 32-bit count, so the rate cannot be measured this way.
            if wait == MilliSeconds::from(0) {
                panic!("Unable to configure x2APIC timer frequency");
            }

            unsafe {
                let initial_cnt = X2ApicInitialCount::from_raw(u64::MAX);
                initial_cnt.write();
            }

            match PIT::try_wait(wait, token) {
                Ok(t) => token = t,
                Err((_, t)) => {
                    wait /= 2;
                    token = t;

                    continue;
                }
            }

            let current_cnt = unsafe { X2ApicCurrentCount::read() };
            if current_cnt.raw() > 0 {
                break current_cnt;
            }

            // The timer ran out before the wait did, so how far it would have
            // got is lost. Try again over half the span.
            wait /= 2;
        };
        let total = u32::MAX - current_cnt.raw() as u32;

        // Determine PIT overhead: what a wait of zero costs, which is the
        // call itself and the programming of channel 2, and is therefore
        // included in the measurement above.
        unsafe {
            let initial_cnt = X2ApicInitialCount::from_raw(u64::MAX);
            initial_cnt.write();
        }

        match PIT::try_wait(MilliSeconds::from(0), token) {
            Ok(t) => token = t,
            Err((_, t)) => {
                // Determine the PIT overhead is just best effort, ignore any error.
                token = t;
            }
        }
        unsafe { current_cnt = X2ApicCurrentCount::read() }
        let pit_overhead = u32::MAX - current_cnt.raw() as u32;

        // Disable Timer
        unsafe {
            let initial_cnt = X2ApicInitialCount::from_raw(0);
            initial_cnt.write();
        }

        // Determine timer frequency, as ticks per millisecond: the ticks the
        // wait accounted for, less what the measurement itself costs, over
        // the length of the wait.
        let mut ticks = total;
        ticks = ticks.saturating_sub(pit_overhead);
        ticks = ticks / usize::from(wait) as u32;
        if ticks == 0 {
            panic!("Unable to configure x2APIC timer frequency")
        }
        let ticks_per_ms = ticks as _;

        // Allocate timer interrupt
        let vector = match InterruptVector::allocate() {
            Some(vector) => vector,
            None => panic!("Unable to allocate interrupt vector for x2APIC timer"),
        };
        unsafe {
            lvt_timer.set_vector(vector.into_raw());
            lvt_timer.write();
        }

        let driver = Self {
            state: DriverTicketlock::new(Ticketlock::new(), state),
            ticks_per_ms,
        };
        let driver = match Arc::try_new(driver, token) {
            Ok((driver, t)) => {
                token = t;
                driver
            }
            Err((error, _)) => {
                panic!("Unable to allocate driver instance for x2APIC: {}", error);
            }
        };

        // Register as module
        match Modules::register(ModuleDriver::X2Apic(driver.clone()), token) {
            Ok(t) => token = t,
            Err((error, _)) => {
                panic!("Unable to register x2APIC driver as module: {}", error);
            }
        };

        // Register as interrupt controller
        match InterruptControllers::register(
            InterruptControllerDriver::X2Apic(driver.clone()),
            token,
        ) {
            Ok(t) => token = t,
            Err((error, _)) => {
                panic!(
                    "Unable to register x2APIC driver as interrupt controller: {}",
                    error
                );
            }
        };

        // Register as interrupt-capable device
        match InterruptVectorTable::register(
            vector,
            IRQCapableDriver::X2Apic(driver.clone()),
            token,
        ) {
            Ok((prev, t)) => {
                if prev.is_some() {
                    panic!("Detected invalid interrupt sharing");
                }
                token = t;
            }
            Err((error, _)) => {
                panic!(
                    "Unable to register x2APIC driver as interrupt-capable device: {}",
                    error
                );
            }
        };

        // Register as timer device
        match Timers::register(TimerDriver::X2Apic(driver.clone()), token) {
            Ok(t) => token = t,
            Err((error, _)) => {
                panic!(
                    "Unable to register x2APIC driver as timer device: {}",
                    error
                );
            }
        };

        Ok(token)
    }

    /// See [`Module::name`].
    fn name(&self) -> &'static str {
        "x2apic"
    }
}

impl IRQCapable for X2Apic {
    /// Unmasks the timer's entry in the local vector table, see
    /// [`IRQCapable::enable_irqs`].
    ///
    /// Only the mask bit changes. The vector and the timer mode are read
    /// back and written as they were.
    fn enable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // XXX: x2APIC is local; thus, no synchronisation
        let mut lvt_timer = unsafe { X2ApicLVTTimer::read() };
        lvt_timer.set_masked(false);
        unsafe { lvt_timer.write() };

        Ok(token)
    }

    /// Masks the timer's entry in the local vector table, see
    /// [`IRQCapable::disable_irqs`].
    ///
    /// The timer keeps counting. It only stops raising its vector.
    fn disable_irqs<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // XXX: x2APIC is local; thus, no synchronisation
        let mut lvt_timer = unsafe { X2ApicLVTTimer::read() };
        lvt_timer.set_masked(true);
        unsafe { lvt_timer.write() };

        Ok(token)
    }

    /// Whether the timer's entry in the local vector table is unmasked, see
    /// [`IRQCapable::irqs_enabled`].
    fn irqs_enabled<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // XXX: x2APIC is local; thus, no synchronisation
        let lvt_timer = unsafe { X2ApicLVTTimer::read() };
        let masked = lvt_timer.masked();

        Ok((!masked, token))
    }

    /// Nothing to do yet, see [`IRQCapable::prologue`].
    ///
    /// Always reports that no epilogue is needed.
    fn prologue<Token>(&self, token: Token) -> Result<(bool, Token), (Errno, Token)>
    where
        Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken,
    {
        // Nothing to do yet...
        Ok((false, token))
    }

    /// Nothing to do yet, see [`IRQCapable::epilogue`].
    fn epilogue<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<PrologueLevelID as LockId>::Level> + PreviousToken,
    {
        // Nothing to do yet...
        Ok(token)
    }
}

/// Every divisor the timer supports, finest first, with the factor it divides
/// the input clock by.
///
/// The order is what [`X2Apic::configuration`] relies on: the first divisor
/// whose count fits is the one with the best resolution.
const DIVIDERS: [(usize, X2ApicDivideMode); 8] = [
    (1, X2ApicDivideMode::Divide1),
    (2, X2ApicDivideMode::Divide2),
    (4, X2ApicDivideMode::Divide4),
    (8, X2ApicDivideMode::Divide8),
    (16, X2ApicDivideMode::Divide16),
    (32, X2ApicDivideMode::Divide32),
    (64, X2ApicDivideMode::Divide64),
    (128, X2ApicDivideMode::Divide128),
];

impl X2Apic {
    /// The divisor and initial count that come closest to `interval`.
    ///
    /// The interval is first expressed in ticks of the undivided clock, which
    /// is what [`ticks_per_ms`](X2Apic::ticks_per_ms) was measured in. The
    /// smallest divisor that brings that down into the 32 bits of
    /// [`X2ApicInitialCount`] is then taken, since every step up halves the
    /// resolution for no gain. The count is rounded to the nearest tick of
    /// the divided clock rather than truncated.
    ///
    /// `None` if the interval rounds to no tick at all, or if it does not fit
    /// even at the largest divisor.
    fn configuration(&self, interval: NanoSeconds) -> Option<(X2ApicDivideMode, u32)> {
        // Split off the whole milliseconds so that neither product overflows:
        // the remainder is below one million nanoseconds, and the whole
        // milliseconds are checked.
        let (ms, ns) = interval.to_milliseconds();
        let whole = usize::from(ms).checked_mul(self.ticks_per_ms)?;
        let partial = (usize::from(ns) * self.ticks_per_ms + 500_000) / 1_000_000;
        let ticks = whole.checked_add(partial)?;

        DIVIDERS
            .iter()
            .find_map(|&(factor, mode)| {
                let count = ticks.checked_add(factor / 2)? / factor;
                let count = u32::try_from(count).ok()?;

                Some((mode, count))
            })
            .filter(|&(_, count)| count > 0)
    }
}

impl Timer for X2Apic {
    /// Arms the timer to expire after `interval`, see [`Timer::setup`].
    ///
    /// The divisor and count are chosen by [`X2Apic::configuration`]. The
    /// divisor is written first, since writing the initial count is what
    /// starts the timer, and it should not start counting at the old rate.
    /// The timer mode and the mask in the local vector table are left as they
    /// are, so after [`X2Apic::init`] the timer fires once per call.
    ///
    /// The timer programmed is the one of the calling core, since every core
    /// only reaches its own local APIC.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if `interval` is shorter than half a tick of the
    /// undivided clock, or longer than `u32::MAX` ticks of the most divided
    /// one. The timer is left untouched in that case.
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
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mode, count) = match self.configuration(interval) {
            Some(configuration) => configuration,
            None => return Err((Errno::EINVAL, token)),
        };

        // XXX: x2APIC is local; thus, no synchronisation
        let mut divider = unsafe { X2ApicDivide::read() };
        divider.set_divider(mode);
        unsafe { divider.write() };

        // XXX: x2APIC is local; thus, no synchronisation
        unsafe {
            let initial_cnt = X2ApicInitialCount::from_raw(count as u64);
            initial_cnt.write();
        }

        // XXX: x2APIC is local; thus, no synchronisation
        unsafe {
            let mut lvtt_timer = X2ApicLVTTimer::read();
            match peridoc {
                true => lvtt_timer.set_timer_mode(X2ApicLVTTimerMode::Periodic),
                false => lvtt_timer.set_timer_mode(X2ApicLVTTimerMode::OneShot),
            }
            lvtt_timer.write();
        }

        Ok(token)
    }
}

impl InterruptController for X2Apic {
    /// Signals end of interrupt, see [`InterruptController::acknowledge`].
    ///
    /// The vector is ignored: a write to [`X2ApicEOI`] retires whichever
    /// interrupt the local APIC has in service at the highest priority, which
    /// is the one being handled.
    fn acknowledge<Token>(&self, _: InterruptVector, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        // XXX: x2APIC is local; thus, no synchronisation
        let eoi = unsafe { X2ApicEOI::from_raw(0) };
        unsafe { eoi.write() };

        Ok(token)
    }
}
