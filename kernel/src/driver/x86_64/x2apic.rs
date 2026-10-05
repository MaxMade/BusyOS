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

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};

use bitfield_struct::bitfield;
use driver_macro::module;

use crate::arch::BOOT_CPUID;
use crate::arch::generic::cpu::{CPU as GenericCPU, CPUID, CPUSet};
use crate::driver::ipi::{IPICapable, IPIDriver, IPIs, Mode};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// The *Interrupt Command Register* of the local APIC in x2APIC mode.
///
/// Writing it sends an inter-processor interrupt: to the core named by
/// [`destination`](Self::destination), with the kind of message
/// [`delivery_mode`](Self::delivery_mode) selects. In x2APIC mode the
/// register is one 64-bit MSR, and the write is the send. Unlike in xAPIC
/// mode there is no delivery-status bit to poll afterwards.
#[bitfield(u64)]
struct X2ApicICR {
    /// Bits 7-0: Vector. For a start-up IPI, the page the core starts at.
    #[bits(8)]
    pub vector: u8,

    /// Bits 10-8: Delivery mode.
    #[bits(3)]
    pub delivery_mode: X2ApicDeliveryMode,

    /// Bit 11: Destination mode, `false` for a physical APIC ID.
    #[bits(1)]
    pub logical: bool,

    /// Bits 13-12: Reserved. Bit 12 is the delivery status in xAPIC mode
    /// only.
    #[bits(2)]
    __: u8,

    /// Bit 14: Level. Set for every IPI except an INIT level de-assert,
    /// which x2APIC mode does not support.
    #[bits(1)]
    pub assert: bool,

    /// Bit 15: Trigger mode, `false` for edge.
    #[bits(1)]
    pub level_triggered: bool,

    /// Bits 17-16: Reserved.
    #[bits(2)]
    __: u8,

    /// Bits 19-18: Destination shorthand, `0` for none, so that
    /// [`destination`](Self::destination) is used.
    #[bits(2)]
    pub shorthand: u8,

    /// Bits 31-20: Reserved.
    #[bits(12)]
    __: u16,

    /// Bits 63-32: x2APIC ID of the destination.
    #[bits(32)]
    pub destination: u32,
}

impl MSR<0x830> for X2ApicICR {
    fn raw(&self) -> u64 {
        self.0
    }

    unsafe fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// What kind of message an inter-processor interrupt is, as encoded in
/// [`X2ApicICR::delivery_mode`].
///
/// The numeric value of each variant is the encoding the hardware expects.
/// Only the modes this driver sends are named. The others read back as
/// [`Fixed`](Self::Fixed), which is never a problem, since the register is
/// only ever written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum X2ApicDeliveryMode {
    /// An ordinary interrupt, on the vector given.
    Fixed = 0b000,

    /// Resets the destination core into its wait-for-SIPI state.
    Init = 0b101,

    /// A non-maskable interrupt, on the NMI vector whatever the vector says.
    Nmi = 0b100,

    /// Starts a core waiting for it at the page the vector names.
    StartUp = 0b110,
}

impl X2ApicDeliveryMode {
    /// The encoding of this mode.
    const fn into_bits(self) -> u8 {
        self as u8
    }

    /// The mode `bits` encodes, see the type's documentation for the ones
    /// it does not name.
    const fn from_bits(bits: u8) -> Self {
        match bits & 0b111 {
            0b100 => Self::Nmi,
            0b101 => Self::Init,
            0b110 => Self::StartUp,
            _ => Self::Fixed,
        }
    }
}

/// The *Local APIC ID Register* in x2APIC mode: the 32-bit identifier of the
/// calling core's local APIC. Read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct X2ApicID(u32);

impl MSR<0x802> for X2ApicID {
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

/// Marks a core in [`LAPIC_IDS`] whose local APIC is not known.
///
/// `0xFFFF_FFFF` is the x2APIC broadcast address, which no local APIC has as
/// its own identifier.
const UNKNOWN_LAPIC: u32 = u32::MAX;

/// The local APIC identifier of every core, indexed by the kernel's core
/// number, see [`X2Apic::record_local_id`].
///
/// What turns a [`CPUSet`] into the addresses an inter-processor interrupt
/// needs. Atomic, since every core fills in its own entry while it comes up,
/// while the boot core may already be sending.
static LAPIC_IDS: [AtomicU32; <crate::arch::CPU as GenericCPU>::CPUID_BITS] =
    [const { AtomicU32::new(UNKNOWN_LAPIC) }; <crate::arch::CPU as GenericCPU>::CPUID_BITS];

/// The local APIC of the core that brought the driver up.
///
/// The registers are not behind a lock: every core reaches only its own local
/// APIC through the x2APIC MSRs, so an access from one core cannot race with
/// one from another, and the methods below go to the MSRs directly. The rate
/// of the timer is measured once and read-only afterwards, so it needs no
/// lock either.
pub struct X2Apic {
    /// Timer ticks in one millisecond, as measured against the [`PIT`], at
    /// [`X2ApicDivideMode::Divide1`].
    ///
    /// This is what makes an interval in real time expressible at all: the
    /// rate is not architectural and differs from machine to machine, so it
    /// has to be measured on each one.
    ticks_per_ms: usize,

    /// Undivided timer ticks in all the periods that have expired so far,
    /// added by [`IRQCapable::prologue`] on every expiry of the boot core's
    /// timer. The other cores' timers are not counted, since this one counter
    /// is shared by all of them.
    ///
    /// Together with the progress of the running period, this is what
    /// [`Timer::nanoseconds_since`] converts. Counted at the undivided rate,
    /// like [`ticks_per_ms`](X2Apic::ticks_per_ms), so that a change of
    /// divisor between two periods does not change what a tick is worth.
    total_ticks: AtomicUsize,

    /// Undivided timer ticks in one period, as last set up by
    /// [`Timer::setup`]: the initial count times the divisor.
    ///
    /// Zero until the timer has been set up.
    period_ticks: AtomicUsize,
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
        // SAFETY: this runs once, on the boot core, before anything uses the
        // local APIC.
        if !unsafe { Self::enable_local() } {
            return Ok(token);
        }

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
            ticks_per_ms,
            total_ticks: AtomicUsize::new(0),
            period_ticks: AtomicUsize::new(0),
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

        // Register as inter-processor interrupt sender
        match IPIs::register(IPIDriver::X2Apic(driver.clone()), token) {
            Ok(t) => token = t,
            Err((error, _)) => {
                panic!(
                    "Unable to register x2APIC driver as inter-processor interrupt sender: {}",
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
        let cpuid = CPUID.with(|cpuid| *cpuid);
        if cpuid == BOOT_CPUID {
            // One period has expired. Added first thing, so that a reading of
            // the time on the way out of the interrupt already counts it.
            //
            // Release, pairing with the acquire loads in `nanoseconds_since`.
            let period = self.period_ticks.load(AtomicOrdering::Relaxed);
            self.total_ticks.fetch_add(period, AtomicOrdering::Release);
        }

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
    /// Switches the calling core's local APIC into x2APIC mode and
    /// software-enables it, returning whether the core supports x2APIC.
    ///
    /// Every core has to do this for its own local APIC before any of its
    /// x2APIC registers can be used: until then, each of them raises `#GP`.
    /// [`init`](Module::init) does it for the boot core, and every other
    /// core does it while it comes up. Doing it twice is harmless.
    ///
    /// # Safety
    ///
    /// Must not race with other code using the calling core's local APIC.
    pub unsafe fn enable_local() -> bool {
        // SAFETY: leaf 1 exists on every x86_64 processor.
        let features = unsafe { FeatureInformation::read() };
        if !features.ecx.x2apic() {
            return false;
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

        true
    }

    /// Records the calling core's local APIC identifier under its core
    /// number, so that [`IPICapable::send`] can address the core.
    ///
    /// Every core does this for itself while it comes up, right after
    /// [`enable_local`](Self::enable_local).
    ///
    /// # Safety
    ///
    /// As for [`local_id`](Self::local_id), and the core's `CPUID` must be
    /// set.
    pub unsafe fn record_local_id() {
        let cpu: usize = CPUID.with(|cpuid| *cpuid).into();

        // SAFETY: see the function's contract.
        let lapic = unsafe { Self::local_id() };

        LAPIC_IDS[cpu].store(lapic.0, AtomicOrdering::Release);
    }

    /// The identifier of the calling core's local APIC.
    ///
    /// # Safety
    ///
    /// The calling core's local APIC must be in x2APIC mode, see
    /// [`enable_local`](Self::enable_local).
    pub unsafe fn local_id() -> LapicID {
        // SAFETY: the register exists in x2APIC mode, see the contract.
        LapicID(unsafe { X2ApicID::read() }.0)
    }

    /// Sends an INIT IPI to the core whose local APIC is `destination`.
    ///
    /// The first step of starting another core: it resets the core into the
    /// state in which it waits for a start-up IPI, see
    /// [`send_startup_ipi`](Self::send_startup_ipi), whatever the firmware
    /// left it doing. The architecture asks for a wait of 10 ms before the
    /// start-up IPI follows.
    ///
    /// # Safety
    ///
    /// Resets `destination` on the spot, losing whatever it was doing, so it
    /// must be a core the kernel does not run on yet. Never the calling
    /// core.
    pub unsafe fn send_init_ipi(destination: &LapicID) {
        let icr = X2ApicICR::new()
            .with_delivery_mode(X2ApicDeliveryMode::Init)
            .with_assert(true)
            .with_destination(destination.0);

        // SAFETY: the local APIC is in x2APIC mode, so the register exists,
        // and the caller vouches for the destination.
        unsafe { icr.write() };
    }

    /// Sends a start-up IPI to the core whose local APIC is `destination`,
    /// making it start in real mode at physical address `page << 12`.
    ///
    /// Only a core that an INIT IPI has put into its wait-for-SIPI state
    /// acts on it. The architecture sends it twice, 200 µs apart, in case
    /// the first one is lost. A core that already runs ignores the second.
    ///
    /// # Safety
    ///
    /// The page must hold the code the core is to start with, below 1 MiB,
    /// and stay untouched until the core has left it.
    pub unsafe fn send_startup_ipi(destination: &LapicID, page: u8) {
        let icr = X2ApicICR::new()
            .with_vector(page)
            .with_delivery_mode(X2ApicDeliveryMode::StartUp)
            .with_assert(true)
            .with_destination(destination.0);

        // SAFETY: as for `send_init_ipi`, and the caller vouches for the page.
        unsafe { icr.write() };
    }

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

    /// Undivided ticks the running period has counted so far.
    ///
    /// Zero for a timer that is stopped, and for a one-shot period that has
    /// expired: the prologue counts that one in full.
    fn running_period_ticks() -> usize {
        // XXX: x2APIC is local; thus, no synchronisation
        let (initial, current, divider) = unsafe {
            (
                X2ApicInitialCount::read().raw() as usize,
                X2ApicCurrentCount::read().raw() as usize,
                X2ApicDivide::read().divider(),
            )
        };

        if current == 0 {
            return 0;
        }

        initial.saturating_sub(current) * divider_factor(divider)
    }

    /// Whether the timer's vector is pending, that is raised but not yet
    /// taken by the core.
    fn timer_pending() -> bool {
        // XXX: x2APIC is local; thus, no synchronisation
        let vector = unsafe { X2ApicLVTTimer::read() }.vector() as u32;

        // The interrupt request register is split over eight MSRs from
        // 0x820, 32 vectors each, and only the low half of each is used.
        let low: u32;
        // SAFETY: the local APIC is in x2APIC mode, so these MSRs exist, and
        // reading one has no side effects.
        unsafe {
            core::arch::asm!(
                "rdmsr",
                in("ecx") 0x820 + vector / 32,
                out("eax") low,
                out("edx") _,
                options(nomem, nostack, preserves_flags),
            );
        }

        low & (1 << (vector % 32)) != 0
    }

    /// Converts undivided timer ticks into nanoseconds.
    ///
    /// Split into whole milliseconds and a remainder, the same way
    /// [`configuration`](X2Apic::configuration) does it the other way round,
    /// so that the remainder is not lost and no product overflows: the
    /// remainder is below [`ticks_per_ms`](X2Apic::ticks_per_ms).
    fn ticks_to_nanoseconds(&self, ticks: usize) -> NanoSeconds {
        let ms = ticks / self.ticks_per_ms;
        let rest = ticks % self.ticks_per_ms;

        NanoSeconds::from(ms * 1_000_000 + rest * 1_000_000 / self.ticks_per_ms)
    }
}

/// The factor `mode` divides the timer's input clock by.
fn divider_factor(mode: X2ApicDivideMode) -> usize {
    DIVIDERS
        .iter()
        .find(|&&(_, candidate)| candidate == mode)
        .map(|&(factor, _)| factor)
        .unwrap_or(1)
}

impl Timer for X2Apic {
    /// Arms the timer to expire after `interval`, see [`Timer::setup`].
    ///
    /// The divisor and count are chosen by [`X2Apic::configuration`]. The
    /// mode in the local vector table, periodic or one-shot, is written
    /// first and the initial count last, since writing the count is what
    /// starts the timer, and it should not start in the old mode or at the
    /// old rate. The mask is left as it is: [`IRQCapable::enable_irqs`]
    /// unmasks the timer.
    ///
    /// The length of one period is recorded for
    /// [`nanoseconds_since`](Timer::nanoseconds_since).
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
        unsafe {
            let mut lvtt_timer = X2ApicLVTTimer::read();
            match peridoc {
                true => lvtt_timer.set_timer_mode(X2ApicLVTTimerMode::Periodic),
                false => lvtt_timer.set_timer_mode(X2ApicLVTTimerMode::OneShot),
            }
            lvtt_timer.write();
        }

        // XXX: x2APIC is local; thus, no synchronisation
        let mut divider = unsafe { X2ApicDivide::read() };
        divider.set_divider(mode);
        unsafe { divider.write() };

        // Recorded before the count is written, since writing it is what
        // starts the period.
        let period = count as usize * divider_factor(mode);
        self.period_ticks.store(period, AtomicOrdering::Relaxed);

        // XXX: x2APIC is local; thus, no synchronisation
        unsafe {
            let initial_cnt = X2ApicInitialCount::from_raw(count as u64);
            initial_cnt.write();
        }

        Ok(token)
    }

    /// The time the timer has counted since it was first set up, see
    /// [`Timer::nanoseconds_since`].
    ///
    /// Made up of the periods that have expired, which the prologue adds to
    /// [`total_ticks`](X2Apic::total_ticks), and the progress of the running
    /// period, which the count registers show. Precise to one tick of the
    /// undivided clock, rather than to a whole period or millisecond.
    ///
    /// Two races are dealt with:
    ///
    /// - An expiry handled while the registers are being read changes
    ///   `total_ticks`, and the reading is taken again.
    /// - An expiry that has happened but whose interrupt is still pending,
    ///   for example because the caller has interrupts masked, has already
    ///   reloaded the count without being in `total_ticks` yet. It is
    ///   recognised by the timer's vector being pending, and added here, so
    ///   that the time never appears to go backwards. The pending bit is read
    ///   before and after the count, and the reading is taken again if an
    ///   expiry fell in between.
    ///
    /// Only the boot core's expiries are counted, see
    /// [`IRQCapable::prologue`], while the count registers read are those of
    /// the calling core. So this is only correct on the boot core, which is
    /// the only one running so far.
    fn nanoseconds_since<Token>(&self, token: Token) -> (NanoSeconds, Token)
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        loop {
            // Acquire, pairing with the release in the prologue.
            let before = self.total_ticks.load(AtomicOrdering::Acquire);

            // Whether an expiry is pending, on both sides of the count: only
            // if the two agree do the count and the pending bit describe the
            // same side of an expiry. An expiry between the count and a
            // single check would either count the old period twice or miss
            // the reload, depending on the order.
            let pending_before = Self::timer_pending();
            let running = Self::running_period_ticks();
            let pending_after = Self::timer_pending();

            let after = self.total_ticks.load(AtomicOrdering::Acquire);
            if before == after && pending_before == pending_after {
                let pending = match pending_after {
                    true => self.period_ticks.load(AtomicOrdering::Relaxed),
                    false => 0,
                };
                let ticks = before + running + pending;
                return (self.ticks_to_nanoseconds(ticks), token);
            }
        }
    }
}

impl IPICapable for X2Apic {
    /// Sends one interrupt per core in `target`, see [`IPICapable::send`].
    ///
    /// Each core is addressed by the local APIC identifier it recorded while
    /// coming up, see [`X2Apic::record_local_id`]. A core that has none
    /// recorded did not come up, and is skipped.
    ///
    /// [`Mode::Panic`] is sent as a non-maskable interrupt: it has to reach a
    /// core that runs with interrupts masked, and the core halts on it, see
    /// `handler` in `kernel/handler.rs`.
    fn send<Token>(&self, target: CPUSet, mode: Mode, token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let icr = match mode {
            Mode::Panic => X2ApicICR::new()
                .with_delivery_mode(X2ApicDeliveryMode::Nmi)
                .with_assert(true),
        };

        for cpu in &target {
            let cpu: usize = cpu.into();
            let lapic = LAPIC_IDS[cpu].load(AtomicOrdering::Acquire);
            if lapic == UNKNOWN_LAPIC {
                continue;
            }

            // XXX: x2APIC is local; thus, no synchronisation
            //
            // SAFETY: the local APIC is in x2APIC mode, so the register
            // exists, and the destination is a core that came up.
            unsafe { icr.with_destination(lapic).write() };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The encodings of Intel SDM vol. 3, 8.4.4: an INIT is delivery mode
    /// `101` with the level asserted, `0x4500`, and the destination sits in
    /// the upper half.
    #[test]
    fn init_ipi_encoding() {
        let icr = X2ApicICR::new()
            .with_delivery_mode(X2ApicDeliveryMode::Init)
            .with_assert(true)
            .with_destination(5);

        assert_eq!(icr.raw(), 0x0000_0005_0000_4500);
    }

    /// The panic IPI is an NMI, delivery mode `100`, whose vector is ignored.
    #[test]
    fn nmi_ipi_encoding() {
        let icr = X2ApicICR::new()
            .with_delivery_mode(X2ApicDeliveryMode::Nmi)
            .with_assert(true)
            .with_destination(3);

        assert_eq!(icr.raw(), 0x0000_0003_0000_4400);
    }

    /// A start-up IPI is delivery mode `110` with the start page as its
    /// vector: `0x46` followed by the page.
    #[test]
    fn startup_ipi_encoding() {
        let icr = X2ApicICR::new()
            .with_vector(0x08)
            .with_delivery_mode(X2ApicDeliveryMode::StartUp)
            .with_assert(true)
            .with_destination(0x1_0000);

        assert_eq!(icr.raw(), 0x0001_0000_0000_4608);
    }
}
