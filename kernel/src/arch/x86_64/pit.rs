//! The *Programmable Interval Timer*, an 8254 or the chipset's equivalent.
//!
//! The chip holds three independent 16-bit counters fed by one 1.193182 MHz
//! input. Channel 0 drives IRQ 0 and channel 1 is a relic of DRAM refresh;
//! channel 2 is the only one whose output the kernel can read back directly,
//! through bit 5 of port `0x61`, and it is therefore the one a polled wait
//! uses.
//!
//! The PIT is not the kernel's clock. It is the coarse reference that is
//! always there before a better one exists, chiefly to calibrate the local
//! APIC timer, and [`PIT::try_wait`] spins for its whole duration: no
//! interrupt is involved and nothing is descheduled.

use bitfield_struct::bitfield;

use crate::{
    arch::x86_64::io::IOB,
    kernel::{
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
        ticketlock::{DriverTicketlock, Ticketlock},
        time::MilliSeconds,
    },
    user::errno::Errno,
};

/// Port `0x43`: the mode/command word, which selects a channel and says how
/// it counts and how its counter is written.
#[bitfield(u8)]
pub struct Command {
    /// Bit 0: BCD/Binary mode.
    pub bcd: bool,

    /// Bits 3-1: Operating mode.
    #[bits(3)]
    pub mode: OperatingMode,

    /// Bits 5-4: Access mode.
    #[bits(2)]
    pub access_mode: AccessMode,

    /// Bits 7-6: Channel selection.
    #[bits(2)]
    pub channel: Channel,
}

/// Which channel a [`Command`] programs, as encoded in
/// [`Command::channel`].
///
/// [`Channel::ReadBack`] is not a channel but the read-back command, which
/// latches the count or status of several channels at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Channel {
    Channel0 = 0b00,
    Channel1 = 0b01,
    Channel2 = 0b10,
    ReadBack = 0b11,
}

impl Channel {
    /// The encoding of this value.
    pub const fn into_bits(self) -> u8 {
        self as _
    }

    /// The value `bits` encodes. Bits outside the field are ignored.
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::Channel0,
            0b01 => Self::Channel1,
            0b10 => Self::Channel2,
            0b11 => Self::ReadBack,
            _ => unreachable!(),
        }
    }
}

/// How the 16-bit counter is reached through its 8-bit port, as encoded in
/// [`Command::access_mode`].
///
/// [`AccessMode::LatchCount`] is a command of its own: it freezes the
/// current count for a subsequent read and leaves the channel's mode alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessMode {
    LatchCount = 0b00,
    LowByte = 0b01,
    HighByte = 0b10,
    LowHighByte = 0b11,
}

impl AccessMode {
    /// The encoding of this value.
    pub const fn into_bits(self) -> u8 {
        self as _
    }

    /// The value `bits` encodes. Bits outside the field are ignored.
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::LatchCount,
            0b01 => Self::LowByte,
            0b10 => Self::HighByte,
            0b11 => Self::LowHighByte,
            _ => unreachable!(),
        }
    }
}

/// How a channel counts and drives its output, as encoded in
/// [`Command::mode`].
///
/// The `Alt` variants are the encodings `0b110` and `0b111`, which the 8254
/// treats as modes 2 and 3 again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OperatingMode {
    InterruptOnTerminalCount = 0b000,
    HardwareRetriggerableOneShot = 0b001,
    RateGenerator = 0b010,
    SquareWaveGenerator = 0b011,
    SoftwareTriggeredStrobe = 0b100,
    HardwareTriggeredStrobe = 0b101,
    RateGeneratorAlt = 0b110,
    SquareWaveGeneratorAlt = 0b111,
}

impl OperatingMode {
    /// The encoding of this value.
    pub const fn into_bits(self) -> u8 {
        self as _
    }

    /// The value `bits` encodes. Bits outside the field are ignored.
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b111 {
            0b000 => Self::InterruptOnTerminalCount,
            0b001 => Self::HardwareRetriggerableOneShot,
            0b010 => Self::RateGenerator,
            0b011 => Self::SquareWaveGenerator,
            0b100 => Self::SoftwareTriggeredStrobe,
            0b101 => Self::HardwareTriggeredStrobe,
            0b110 => Self::RateGeneratorAlt,
            0b111 => Self::SquareWaveGeneratorAlt,
            _ => unreachable!(),
        }
    }
}

/// Port `0x61`, as far as it concerns the PIT.
///
/// The register belongs to the keyboard controller and carries bits that
/// have nothing to do with the timer, several of them read-only. It is
/// therefore read, adjusted and written back rather than written outright,
/// which is what the unnamed fields below stand for: they keep whatever was
/// read.
#[bitfield(u8)]
pub struct Control {
    /// Bit 0: gate of channel 2. The counter only counts while it is set.
    pub gate: bool,

    /// Bit 1: speaker data enable.
    ///
    /// Channel 2 is wired to the PC speaker, so this is kept clear and the
    /// counter runs silently.
    pub speaker: bool,

    /// Bits 4-2: unrelated to the PIT.
    #[bits(3)]
    __: u8,

    /// Bit 5: output of channel 2.
    ///
    /// Read-only, and what a wait polls: under
    /// [`OperatingMode::InterruptOnTerminalCount`] it is low from the moment
    /// the command is written until the count reaches zero.
    pub output: bool,

    /// Bits 7-6: unrelated to the PIT.
    #[bits(2)]
    __: u8,
}

/// Port `0x43`: the mode/command register, write-only.
type CommandPort = IOB<0x43, Command>;

/// Port `0x42`: the counter of channel 2.
type CounterPort = IOB<0x42, u8>;

/// Port `0x61`: the gate and the output of channel 2.
type ControlPort = IOB<0x61, Control>;

/// The chip, with all of its ports behind one lock.
///
/// One lock for the whole chip rather than one per port, because a single
/// wait needs the command register, the counter of channel 2 and the gate
/// together: acquiring a lock yields a token that reaches only levels
/// *strictly below* the one just taken, so two locks at [`level::Driver`]
/// can never be held at once.
///
/// [`level::Driver`]: crate::kernel::locking::level::Driver
static CHIP: DriverTicketlock<PIT> = DriverTicketlock::new(Ticketlock::new(), PIT);

/// The PIT, as the value its one lock guards.
///
/// It holds no data, since the chip's state is all in its ports. Only a
/// reference obtained through the lock may reach them, which is what the
/// private accessors below rely on.
pub struct PIT;

impl PIT {
    /// Input frequency of every channel, in hertz.
    ///
    /// A third of the 3.579545 MHz colour burst crystal the PC inherited from
    /// NTSC, which is where the odd number comes from.
    const FREQUENCY: usize = 1_193_182;

    /// Largest count a single countdown can be given.
    ///
    /// The counter is 16 bits, so one pass spans at most
    /// `MAX_COUNT / FREQUENCY`, roughly 54.9 ms, and a longer wait is split
    /// into several.
    const MAX_COUNT: usize = u16::MAX as usize;

    /// Spins for `ms`, using channel 2 as the reference.
    ///
    /// The wait is never shorter than `ms` and overshoots by at most the
    /// rounding of one tick plus whatever the polling loop adds. A duration
    /// longer than one countdown is split, so `ms` is not bounded by the
    /// 54.9 ms a 16-bit counter covers.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if `ms` is so large that the tick count it stands
    /// for overflows a `usize`, which takes a duration of several centuries.
    pub fn try_wait<Token>(ms: MilliSeconds, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let Some(ticks) = Self::ticks(ms) else {
            return Err((Errno::EINVAL, token));
        };

        let (mut pit, token) = CHIP.acquire(token);

        let mut remaining = ticks;
        while remaining != 0 {
            let count = remaining.min(Self::MAX_COUNT);

            pit.count_down(count as u16);

            remaining -= count;
        }

        Ok(pit.release(token))
    }

    /// The number of input ticks `ms` amounts to, or [`None`] if that count
    /// does not fit a `usize`.
    ///
    /// Rounded up: truncating would cut up to one tick off every wait, and a
    /// caller calibrating another timer against this one needs the elapsed
    /// time to be at least what it asked for.
    fn ticks(ms: MilliSeconds) -> Option<usize> {
        let ticks = usize::from(ms).checked_mul(Self::FREQUENCY)?;

        Some(ticks.div_ceil(1_000))
    }

    /// Runs channel 2 down from `count` and spins until it reaches zero.
    fn count_down(&mut self, count: u16) {
        // Drop the gate before reprogramming, and keep the speaker off. A
        // counter with its gate low holds still, so the count written below
        // only starts running once the gate goes back up, and the setup in
        // between costs no time.
        let idle = self.control().with_gate(false).with_speaker(false);
        self.set_control(idle);

        // Mode 0 drives the output low as soon as the command is written and
        // raises it again at terminal count. That rising edge is the whole
        // signal, and it cannot be missed: the output stays high afterwards.
        self.set_command(
            Command::new()
                .with_channel(Channel::Channel2)
                .with_access_mode(AccessMode::LowHighByte)
                .with_mode(OperatingMode::InterruptOnTerminalCount)
                .with_bcd(false),
        );
        self.set_count(count);

        self.set_control(idle.with_gate(true));

        while !self.control().output() {
            core::hint::spin_loop();
        }
    }

    /// Writes the mode/command register.
    fn set_command(&mut self, command: Command) {
        // SAFETY: `&mut self` comes from the guard of `CHIP`, which is what
        // makes access to the PIT's ports exclusive.
        unsafe { CommandPort::write(command) };
    }

    /// Writes both bytes of the counter of channel 2, low byte first, as
    /// [`AccessMode::LowHighByte`] expects them.
    fn set_count(&mut self, count: u16) {
        // SAFETY: as in `set_command`.
        unsafe { CounterPort::write(count as u8) };

        // SAFETY: as in `set_command`.
        unsafe { CounterPort::write((count >> 8) as u8) };
    }

    /// Reads the gate and output register.
    fn control(&self) -> Control {
        // SAFETY: as in `set_command`.
        unsafe { ControlPort::read() }
    }

    /// Writes the gate and output register.
    fn set_control(&mut self, control: Control) {
        // SAFETY: as in `set_command`.
        unsafe { ControlPort::write(control) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The command word is laid out as the chip reads it.
    ///
    /// [`bitfield`] assigns the fields from the least significant bit
    /// upwards, in declaration order, while the 8254 documents its command
    /// register from the most significant bit down. The two therefore run
    /// opposite to one another, and a field declared in the documented order
    /// still type checks -- it just lands in the wrong bits.
    #[test]
    fn the_command_word_matches_the_hardware_layout() {
        let command = Command::new()
            .with_channel(Channel::Channel2)
            .with_access_mode(AccessMode::LowHighByte)
            .with_mode(OperatingMode::InterruptOnTerminalCount)
            .with_bcd(false);

        //                            channel 2 --. .-- both bytes
        //                                        | |
        assert_eq!(u8::from(command), 0b_10_11_000_0);
        //                                       | |
        //                                mode 0 -' '-- binary
    }

    /// The output of channel 2 is read from bit 5 of the control register.
    #[test]
    fn the_control_register_reports_the_output_in_bit_five() {
        assert!(Control::from_bits(0b0010_0000).output());
        assert!(!Control::from_bits(0b1101_1111).output());
    }

    /// Adjusting the gate leaves every other bit of the control register as
    /// it was read, which is what lets it be written back.
    #[test]
    fn the_control_register_keeps_the_bits_it_does_not_own() {
        let read = Control::from_bits(0b1111_1111);
        let idle = read.with_gate(false).with_speaker(false);

        assert_eq!(u8::from(idle), 0b1111_1100);
        assert_eq!(u8::from(idle.with_gate(true)), 0b1111_1101);
    }

    /// A duration is rounded up, so a wait is never shorter than asked for.
    #[test]
    fn a_duration_rounds_up_to_whole_ticks() {
        // A whole millisecond is 1193.182 ticks, so neither one nor two of
        // them come out even.
        assert_eq!(PIT::ticks(MilliSeconds::from(1)), Some(1_194));
        assert_eq!(PIT::ticks(MilliSeconds::from(2)), Some(2_387));

        assert_eq!(PIT::ticks(MilliSeconds::from(0)), Some(0));

        // 1000 ms is exactly `FREQUENCY` ticks, with nothing to round.
        assert_eq!(PIT::ticks(MilliSeconds::from(1_000)), Some(PIT::FREQUENCY));
    }

    /// A duration whose tick count does not fit a `usize` is rejected rather
    /// than wrapped into a short wait.
    #[test]
    fn an_unrepresentable_duration_is_rejected() {
        assert_eq!(PIT::ticks(MilliSeconds::from(usize::MAX)), None);
    }
}
