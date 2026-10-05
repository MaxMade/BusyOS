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

use bitfield_struct::bitfield;
use driver_macro::module;

use crate::{
    arch::{
        InterruptVector,
        generic::cpu::InterruptVector as GenericInterruptVector,
        x86_64::{
            cpuid::{CPUID as _, FeatureInformation},
            msr::MSR,
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

/// The *LVT Timer Register* of the local APIC in x2APIC mode.
///
/// The `IA32_APIC_BASE` MSR, which switches the local APIC on and selects
/// its mode.
///
/// [`enabled`](Self::enabled) and [`x2apic`](Self::x2apic) together give the
/// mode: both clear is disabled, only `enabled` is xAPIC, and both set is
/// x2APIC. Every x2APIC register (MSRs `0x800` to `0x8FF`) raises `#GP` until
/// both are set. Of the transitions between the modes only disabled to xAPIC
/// to x2APIC is allowed going up, so the two bits are set one write at a
/// time.
#[bitfield(u64)]
struct IA32ApicBase {
    /// Bits 7-0: Reserved.
    #[bits(8)]
    __: u8,

    /// Bit 8: Whether this core is the bootstrap processor (read-only).
    #[bits(1, access = RO)]
    pub bsp: bool,

    /// Bit 9: Reserved.
    #[bits(1)]
    __: bool,

    /// Bit 10: x2APIC mode enable (`EXTD`).
    #[bits(1)]
    pub x2apic: bool,

    /// Bit 11: APIC global enable (`EN`).
    #[bits(1)]
    pub enabled: bool,

    /// Bits 63-12: Physical page of the xAPIC register window, unused in
    /// x2APIC mode. Left as the firmware set it.
    #[bits(52)]
    pub base: u64,
}

impl MSR<0x1B> for IA32ApicBase {
    fn raw(&self) -> u64 {
        self.0
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// The *Spurious Interrupt Vector Register* of the local APIC in x2APIC
/// mode.
///
/// Besides the vector a spurious interrupt arrives on, it holds the
/// software enable of the local APIC. While that is clear, every local
/// vector table entry stays masked whatever is written to it, so the timer
/// could count but never raise its vector.
#[bitfield(u32)]
struct X2ApicSVR {
    /// Bits 7-0: Spurious vector.
    #[bits(8)]
    pub vector: u8,

    /// Bit 8: APIC software enable.
    #[bits(1)]
    pub enabled: bool,

    /// Bits 11-9: Reserved.
    #[bits(3)]
    __: u8,

    /// Bit 12: EOI-broadcast suppression.
    #[bits(1)]
    pub eoi_broadcast_suppression: bool,

    /// Bits 31-13: Reserved.
    #[bits(19)]
    __: u32,
}

impl MSR<0x80F> for X2ApicSVR {
    fn raw(&self) -> u64 {
        self.0 as _
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value as _)
    }
}

/// The vector a spurious interrupt of the local APIC arrives on.
///
/// The architecture recommends the top of the range, and a spurious
/// interrupt needs no end of interrupt.
const SPURIOUS_VECTOR: u8 = 0xFF;

/// One of the local vector table entries, and the one that says what the
/// core's own timer does when it expires: which vector it raises, whether
/// that vector is masked, and in which of the three modes
/// ([`X2ApicLVTTimerMode`]) the timer runs.
///
/// The register is 32 bits wide while an MSR is 64, so the upper half reads
/// as zero and must be written as zero.
#[bitfield(u32)]
struct X2ApicLVTTimer {
    /// Bits 7-0: Local vector number.
    #[bits(8)]
    pub vector: u8,

    /// Bits 11-8: Reserved.
    #[bits(4)]
    __: u8,

    /// Bit 12: Delivery status (read-only).
    #[bits(1, access = RO)]
    pub delivery_status: bool,

    /// Bit 15-13: Reserved.
    #[bits(3)]
    __: u8,

    /// Bit 16: Mask.
    #[bits(1)]
    pub masked: bool,

    /// Bits 18-17: Timer mode.
    #[bits(2)]
    pub timer_mode: X2ApicLVTTimerMode,

    #[bits(13)]
    __: u16,
}

/// How the local APIC timer counts, as encoded in
/// [`X2ApicLVTTimer::timer_mode`].
///
/// The numeric value of each variant is the encoding the hardware expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum X2ApicLVTTimerMode {
    /// Counts the initial count down once and then stays at zero, so the
    /// vector is raised exactly once per count that is written.
    OneShot = 0b00,

    /// Reloads the initial count every time it reaches zero, so the vector
    /// is raised at a fixed interval until the count is cleared.
    Periodic = 0b01,

    /// Fires when the TSC passes the deadline written to `IA32_TSC_DEADLINE`
    /// rather than when a count runs out. The initial count is unused in
    /// this mode.
    TscDeadline = 0b10,

    /// The encoding the architecture leaves undefined. Never written.
    Reserved = 0b11,
}

impl X2ApicLVTTimerMode {
    /// The encoding of this mode.
    const fn into_bits(self) -> u8 {
        self as u8
    }

    /// The mode `bits` encodes. Only the lower two bits are looked at.
    const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::OneShot,
            0b01 => Self::Periodic,
            0b10 => Self::TscDeadline,
            0b11 => Self::Reserved,
            _ => unreachable!(),
        }
    }
}

impl MSR<0x832> for X2ApicLVTTimer {
    fn raw(&self) -> u64 {
        self.0 as _
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value as _)
    }
}

/// The *Divide Configuration Register* of the local APIC in x2APIC mode.
///
/// Sets how far the core's bus or crystal clock is divided down before it
/// reaches the timer, which is what fixes the length of one timer tick.
/// [`X2ApicDivideMode::Divide1`] leaves the input undivided and gives the
/// finest resolution, which is what calibration wants.
#[bitfield(u32)]
struct X2ApicDivide {
    /// Bits 3-0: Divider Configuration.
    #[bits(4)]
    pub divider: X2ApicDivideMode,

    #[bits(28)]
    __: u32,
}

/// The divisor applied to the timer's input clock, as encoded in
/// [`X2ApicDivide::divider`].
///
/// The numeric value of each variant is the encoding the hardware expects.
/// Bit 2 of the field is reserved and takes no part in the encoding, which
/// is why the eight variants leave a gap between `0b0011` and `0b1000` and
/// why [`from_bits`](X2ApicDivideMode::from_bits) masks it off rather than
/// treating it as part of the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum X2ApicDivideMode {
    Divide2 = 0b0000,
    Divide4 = 0b0001,
    Divide8 = 0b0010,
    Divide16 = 0b0011,
    Divide32 = 0b1000,
    Divide64 = 0b1001,
    Divide128 = 0b1010,
    Divide1 = 0b1011,
}

impl X2ApicDivideMode {
    /// The encoding of this divisor.
    const fn into_bits(self) -> u8 {
        self as u8
    }

    /// The divisor `bits` encodes.
    ///
    /// Bit 2 is reserved, so it is masked off first and every one of the
    /// eight encodings that remain names a variant. A register value with
    /// that bit set therefore reads back as the divisor the other three bits
    /// name, rather than as an unknown one.
    const fn from_bits(bits: u8) -> Self {
        match bits & 0b1011 {
            0b0000 => Self::Divide2,
            0b0001 => Self::Divide4,
            0b0010 => Self::Divide8,
            0b0011 => Self::Divide16,
            0b1000 => Self::Divide32,
            0b1001 => Self::Divide64,
            0b1010 => Self::Divide128,
            0b1011 => Self::Divide1,
            _ => unreachable!(),
        }
    }
}

impl MSR<0x83E> for X2ApicDivide {
    fn raw(&self) -> u64 {
        self.0 as _
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value as _)
    }
}

/// The *Initial Count Register* of the local APIC in x2APIC mode.
///
/// Writing it starts the timer, which counts down from this value at one
/// step per tick of the divided input clock and raises its vector on
/// reaching zero. In [`X2ApicLVTTimerMode::Periodic`] the value is reloaded
/// and the timer runs again, and writing zero stops the timer outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct X2ApicInitialCount(u32);

impl X2ApicInitialCount {
    /// A count of zero, which stops the timer when written.
    const fn new() -> Self {
        Self(0)
    }
}

impl MSR<0x838> for X2ApicInitialCount {
    fn raw(&self) -> u64 {
        self.0 as _
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value as _)
    }
}

/// The *Current Count Register* of the local APIC in x2APIC mode.
///
/// How far the timer still has to go. It is loaded from
/// [`X2ApicInitialCount`] and counts *down*, so the number of ticks that
/// have passed since the timer was started is the initial count minus this
/// one, and a read of zero means the timer has already expired.
///
/// Read-only in the hardware. Writing it has no effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct X2ApicCurrentCount(u32);

impl MSR<0x839> for X2ApicCurrentCount {
    fn raw(&self) -> u64 {
        self.0 as _
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value as _)
    }
}

/// The *EOI Register* of the local APIC in x2APIC mode.
///
/// Writing it tells the local APIC that the interrupt currently in service
/// has been handled, so that the next one of equal or lower priority may be
/// delivered. Which vector is meant is not part of the write: the local APIC
/// retires whatever it has in service at the highest priority.
///
/// Write-only. In x2APIC mode the only value that may be written is zero,
/// anything else raises `#GP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct X2ApicEOI(u32);

impl X2ApicEOI {
    /// A value of zero, the only one the register accepts.
    const fn new() -> Self {
        Self(0)
    }
}

impl MSR<0x80B> for X2ApicEOI {
    fn raw(&self) -> u64 {
        self.0 as _
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value as _)
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

        // A core without x2APIC has no driver here. Not an error: the
        // driver simply does not register.
        //
        // SAFETY: leaf 1 exists on every x86_64 processor.
        let features = unsafe { FeatureInformation::read() };
        if !features.ecx.x2apic() {
            return Ok(token);
        }

        // Switch the local APIC into x2APIC mode, through xAPIC if the
        // firmware left it disabled, since going straight from disabled to
        // x2APIC raises `#GP`. Until this is done every x2APIC register
        // raises `#GP` as well.
        //
        // SAFETY: the processor reports x2APIC support above, and the steps
        // follow the allowed transitions.
        unsafe {
            let mut apic_base = IA32ApicBase::read();
            if !apic_base.enabled() {
                apic_base.set_enabled(true);
                apic_base.write();
            }
            if !apic_base.x2apic() {
                apic_base.set_x2apic(true);
                apic_base.write();
            }
        }

        // Software-enable the local APIC, without which no local vector table
        // entry can be unmasked.
        //
        // SAFETY: the local APIC is in x2APIC mode from here on.
        unsafe {
            let mut svr = X2ApicSVR::read();
            svr.set_vector(SPURIOUS_VECTOR);
            svr.set_enabled(true);
            svr.write();
        }

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
        unsafe { X2ApicInitialCount::new().write() };

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
        // XXX: x2APIC is local; thus, no synchronisation
        unsafe {
            let mut lvtt_timer = X2ApicLVTTimer::read();
            match peridoc {
                true => lvtt_timer.set_timer_mode(X2ApicLVTTimerMode::Periodic),
                false => lvtt_timer.set_timer_mode(X2ApicLVTTimerMode::OneShot),
            }
            lvtt_timer.write();
        }

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
        unsafe { X2ApicEOI::new().write() };

        Ok(token)
    }
}
