//! Time Handling
//!
//! # Overview
//!
//! Each unit — [`NanoSeconds`] through [`Days`] — is its own newtype over a
//! `usize`, so a count of milliseconds can never be passed where seconds are
//! expected. [`TimeUnit`] is what they have in common: it converts a value
//! into any other unit, and [`Timer`] is the interface a clock source
//! implements on top of them.
//!
//! # Conversions
//!
//! A conversion between two units is a single multiplication or division,
//! picked at compile time by [`conversion`] from how the two units relate:
//! going to a *finer* unit multiplies, going to a *coarser* one divides.
//! Every unit therefore only declares how many nanoseconds it is worth, and
//! the seven [`Conversion`] constants of its [`TimeUnit`] impl follow from
//! that — the factors are never written out by hand.
//!
//! Division truncates, so each conversion comes in two forms:
//!
//! - `to_*_lossy` yields just the converted value, dropping whatever did not
//!   fit into a whole unit.
//! - `to_*` yields that value *and* the remainder, in the original unit, so
//!   that no time is lost. The remainder of a multiplication is always zero.
//!
//! ```ignore
//! let (s, rest) = MilliSeconds::from(1500).to_seconds();
//! assert_eq!(s, Seconds::from(1));
//! assert_eq!(rest, MilliSeconds::from(500));
//! ```
//!
//! # Range
//!
//! A unit holds a plain `usize` and the arithmetic is unchecked, so it wraps
//! in release builds and panics in debug ones. This only becomes a practical
//! concern when converting a coarse value into a very fine one: a `usize` of
//! nanoseconds spans roughly 584 years, which leaves [`Days`] about 106_751
//! before [`TimeUnit::to_nanoseconds`] overflows.

use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

/// Nanoseconds in each of the units, the scale conversions are derived from.
const NANOSECOND: usize = 1;
const MICROSECOND: usize = 1_000 * NANOSECOND;
const MILLISECOND: usize = 1_000 * MICROSECOND;
const SECOND: usize = 1_000 * MILLISECOND;
const MINUTE: usize = 60 * SECOND;
const HOUR: usize = 60 * MINUTE;
const DAY: usize = 24 * HOUR;

/// How a value of one unit is turned into another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    /// The target unit is finer: multiply by the factor. Exact.
    Multiply(usize),

    /// The target unit is coarser: divide by the factor. Truncates, and the
    /// remainder is what the non-lossy conversions hand back.
    Divide(usize),
}

/// The [`Conversion`] from a unit worth `from` nanoseconds into one worth
/// `to`.
///
/// Both are exact multiples of one another — every unit here is a whole
/// number of the next finer one — so the factor is their plain ratio, and
/// the equal case falls out as `Multiply(1)`.
const fn conversion(from: usize, to: usize) -> Conversion {
    if from >= to {
        Conversion::Multiply(from / to)
    } else {
        Conversion::Divide(to / from)
    }
}

/// Emits the conversion methods of [`TimeUnit`], one pair per target unit.
///
/// Each entry names the two methods, the unit they produce, the [`TimeUnit`]
/// constant describing the step, and how to spell the unit in prose. Spelling
/// the pairs out by hand would invite a constant and a target type that do not
/// match — a mistake nothing else would catch, since every branch still type
/// checks.
macro_rules! conversion_methods {
    ($(($lossy:ident, $split:ident, $target:ident, $step:ident, $prose:literal)),* $(,)?) => {
        $(
            #[doc = concat!("Convert into ", $prose, ", dropping any remainder.")]
            fn $lossy(self) -> $target {
                match Self::$step {
                    Conversion::Multiply(factor) => $target(self.into() * factor),
                    Conversion::Divide(factor) => $target(self.into() / factor),
                }
            }

            #[doc = concat!("Split into whole ", $prose, " and the remainder.")]
            ///
            /// The remainder keeps the original unit and is what the
            /// conversion could not express, so the two together still
            /// describe the same duration. It is zero whenever the step is a
            /// [`Conversion::Multiply`], which cannot lose anything.
            fn $split(self) -> ($target, Self) {
                match Self::$step {
                    Conversion::Multiply(factor) => {
                        let remaining = Self::from(0);
                        let converted = $target(self.into() * factor);

                        (converted, remaining)
                    }
                    Conversion::Divide(factor) => {
                        let remaining = Self::from(self.into() % factor);
                        let converted = $target(self.into() / factor);

                        (converted, remaining)
                    }
                }
            }
        )*
    };
}

/// A duration in one particular unit.
///
/// The supertraits are what every unit provides: ordering and hashing so that
/// durations can be compared and keyed on, [`Display`] for the value with its
/// [`UNIT`](TimeUnit::UNIT) suffix, the conversions to and from the bare
/// `usize` the unit wraps, and arithmetic against both another duration of
/// the same unit and a plain `usize` scalar.
pub trait TimeUnit:
    Clone
    + Copy
    + Debug
    + Display
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Hash
    + Into<usize>
    + From<usize>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Add<usize, Output = Self>
    + AddAssign<usize>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Sub<usize, Output = Self>
    + SubAssign<usize>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Mul<usize, Output = Self>
    + MulAssign<usize>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Div<usize, Output = Self>
    + DivAssign<usize>
    + Rem<Self, Output = Self>
    + RemAssign<Self>
    + Rem<usize, Output = Self>
    + RemAssign<usize>
{
    /// The step from this unit to [`Days`].
    const TO_DAYS: Conversion;

    /// The step from this unit to [`Hours`].
    const TO_HOURS: Conversion;

    /// The step from this unit to [`Minutes`].
    const TO_MINUTES: Conversion;

    /// The step from this unit to [`Seconds`].
    const TO_SECONDS: Conversion;

    /// The step from this unit to [`MilliSeconds`].
    const TO_MILLISECONDS: Conversion;

    /// The step from this unit to [`MicroSeconds`].
    const TO_MICROSECONDS: Conversion;

    /// The step from this unit to [`NanoSeconds`].
    const TO_NANOSECONDS: Conversion;

    /// Suffix [`Display`] appends to the value, e.g. `"ms"`.
    const UNIT: &'static str;

    conversion_methods! {
        (to_days_lossy, to_days, Days, TO_DAYS, "days"),
        (to_hours_lossy, to_hours, Hours, TO_HOURS, "hours"),
        (to_minutes_lossy, to_minutes, Minutes, TO_MINUTES, "minutes"),
        (to_seconds_lossy, to_seconds, Seconds, TO_SECONDS, "seconds"),
        (
            to_milliseconds_lossy,
            to_milliseconds,
            MilliSeconds,
            TO_MILLISECONDS,
            "milliseconds"
        ),
        (
            to_microseconds_lossy,
            to_microseconds,
            MicroSeconds,
            TO_MICROSECONDS,
            "microseconds"
        ),
        (
            to_nanoseconds_lossy,
            to_nanoseconds,
            NanoSeconds,
            TO_NANOSECONDS,
            "nanoseconds"
        ),
    }
}

/// Emits one arithmetic operator for a unit, against both another value of
/// the same unit and a bare `usize` scalar.
///
/// The assigning form is written as `self.0 = self.0 <op> rhs`: `$op` is a
/// single token, and `+` and `=` cannot be pasted into `+=`.
macro_rules! binary_op {
    ($name:ident, $op_trait:ident, $op:ident, $assign_trait:ident, $assign:ident, $operator:tt) => {
        impl $op_trait for $name {
            type Output = Self;

            fn $op(self, rhs: Self) -> Self::Output {
                Self(self.0 $operator rhs.0)
            }
        }

        impl $assign_trait for $name {
            fn $assign(&mut self, rhs: Self) {
                self.0 = self.0 $operator rhs.0;
            }
        }

        impl $op_trait<usize> for $name {
            type Output = Self;

            fn $op(self, rhs: usize) -> Self::Output {
                Self(self.0 $operator rhs)
            }
        }

        impl $assign_trait<usize> for $name {
            fn $assign(&mut self, rhs: usize) {
                self.0 = self.0 $operator rhs;
            }
        }
    };
}

/// Declares a time unit: the newtype, its [`TimeUnit`] impl, and the
/// conversions, formatting and arithmetic that come with it.
///
/// `$scale` is the unit's worth in nanoseconds, which is all that
/// distinguishes one unit from another — every conversion constant is derived
/// from it — and `$suffix` is how [`Display`] spells it.
macro_rules! time_unit {
    ($(#[$attr:meta])* $name:ident, $scale:expr, $suffix:literal) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl TimeUnit for $name {
            const TO_DAYS: Conversion = conversion($scale, DAY);

            const TO_HOURS: Conversion = conversion($scale, HOUR);

            const TO_MINUTES: Conversion = conversion($scale, MINUTE);

            const TO_SECONDS: Conversion = conversion($scale, SECOND);

            const TO_MILLISECONDS: Conversion = conversion($scale, MILLISECOND);

            const TO_MICROSECONDS: Conversion = conversion($scale, MICROSECOND);

            const TO_NANOSECONDS: Conversion = conversion($scale, NANOSECOND);

            const UNIT: &'static str = $suffix;
        }

        impl From<$name> for usize {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl From<usize> for $name {
            fn from(value: usize) -> Self {
                Self(value)
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}{}", self.0, Self::UNIT)
            }
        }

        binary_op!($name, Add, add, AddAssign, add_assign, +);
        binary_op!($name, Sub, sub, SubAssign, sub_assign, -);
        binary_op!($name, Mul, mul, MulAssign, mul_assign, *);
        binary_op!($name, Div, div, DivAssign, div_assign, /);
        binary_op!($name, Rem, rem, RemAssign, rem_assign, %);
    };
}

time_unit! {
    /// A duration in nanoseconds.
    NanoSeconds, NANOSECOND, "ns"
}

time_unit! {
    /// A duration in microseconds.
    MicroSeconds, MICROSECOND, "us"
}

time_unit! {
    /// A duration in milliseconds.
    MilliSeconds, MILLISECOND, "ms"
}

time_unit! {
    /// A duration in seconds.
    Seconds, SECOND, "s"
}

time_unit! {
    /// A duration in minutes.
    Minutes, MINUTE, "min"
}

time_unit! {
    /// A duration in hours.
    Hours, HOUR, "h"
}

time_unit! {
    /// A duration in days.
    Days, DAY, "d"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived constants agree with the factors written out by hand.
    #[test]
    fn conversion_constants_match_the_hand_computed_factors() {
        assert_eq!(Days::TO_HOURS, Conversion::Multiply(24));
        assert_eq!(Days::TO_MINUTES, Conversion::Multiply(24 * 60));
        assert_eq!(Days::TO_SECONDS, Conversion::Multiply(24 * 60 * 60));
        assert_eq!(
            Days::TO_MILLISECONDS,
            Conversion::Multiply(24 * 60 * 60 * 1_000)
        );
        assert_eq!(
            Days::TO_MICROSECONDS,
            Conversion::Multiply(24 * 60 * 60 * 1_000_000)
        );
        assert_eq!(
            Days::TO_NANOSECONDS,
            Conversion::Multiply(24 * 60 * 60 * 1_000_000_000)
        );

        assert_eq!(NanoSeconds::TO_MICROSECONDS, Conversion::Divide(1_000));
        assert_eq!(NanoSeconds::TO_SECONDS, Conversion::Divide(1_000_000_000));
        assert_eq!(
            NanoSeconds::TO_DAYS,
            Conversion::Divide(24 * 60 * 60 * 1_000_000_000)
        );
    }

    /// A unit converted into itself is the identity, in every unit.
    #[test]
    fn a_unit_converts_into_itself_unchanged() {
        assert_eq!(NanoSeconds::TO_NANOSECONDS, Conversion::Multiply(1));
        assert_eq!(MicroSeconds::TO_MICROSECONDS, Conversion::Multiply(1));
        assert_eq!(MilliSeconds::TO_MILLISECONDS, Conversion::Multiply(1));
        assert_eq!(Seconds::TO_SECONDS, Conversion::Multiply(1));
        assert_eq!(Minutes::TO_MINUTES, Conversion::Multiply(1));
        assert_eq!(Hours::TO_HOURS, Conversion::Multiply(1));
        assert_eq!(Days::TO_DAYS, Conversion::Multiply(1));

        let (same, rest) = Seconds::from(42).to_seconds();
        assert_eq!(same, Seconds::from(42));
        assert_eq!(rest, Seconds::from(0));
    }

    /// Each unit converts into the next coarser one, so every step of the
    /// ladder is exercised rather than just the ends.
    #[test]
    fn every_step_of_the_ladder_converts() {
        assert_eq!(
            NanoSeconds::from(2_000).to_microseconds_lossy(),
            MicroSeconds::from(2)
        );
        assert_eq!(
            MicroSeconds::from(2_000).to_milliseconds_lossy(),
            MilliSeconds::from(2)
        );
        assert_eq!(
            MilliSeconds::from(2_000).to_seconds_lossy(),
            Seconds::from(2)
        );
        assert_eq!(Seconds::from(120).to_minutes_lossy(), Minutes::from(2));
        assert_eq!(Minutes::from(120).to_hours_lossy(), Hours::from(2));
        assert_eq!(Hours::from(48).to_days_lossy(), Days::from(2));
    }

    /// Going to a coarser unit truncates, and the non-lossy form hands the
    /// truncated part back in the original unit.
    #[test]
    fn a_coarser_unit_yields_the_remainder() {
        let (seconds, rest) = MilliSeconds::from(1_500).to_seconds();
        assert_eq!(seconds, Seconds::from(1));
        assert_eq!(rest, MilliSeconds::from(500));

        // The lossy form agrees on the value and simply drops the rest.
        assert_eq!(MilliSeconds::from(1_500).to_seconds_lossy(), seconds);

        let (hours, rest) = Minutes::from(150).to_hours();
        assert_eq!(hours, Hours::from(2));
        assert_eq!(rest, Minutes::from(30));

        // Less than one whole target unit: all of it is remainder.
        let (days, rest) = Hours::from(5).to_days();
        assert_eq!(days, Days::from(0));
        assert_eq!(rest, Hours::from(5));
    }

    /// Going to a finer unit is exact, so nothing is ever left over.
    #[test]
    fn a_finer_unit_leaves_no_remainder() {
        let (millis, rest) = Seconds::from(7).to_milliseconds();
        assert_eq!(millis, MilliSeconds::from(7_000));
        assert_eq!(rest, Seconds::from(0));

        let (nanos, rest) = Days::from(1).to_nanoseconds();
        assert_eq!(nanos, NanoSeconds::from(86_400_000_000_000));
        assert_eq!(rest, Days::from(0));
    }

    /// The remainder is what makes a conversion lossless: converted back and
    /// added on, it reproduces the original value.
    #[test]
    fn a_split_conversion_loses_no_time() {
        let original = MilliSeconds::from(3_999);
        let (seconds, rest) = original.to_seconds();

        assert_eq!(seconds.to_milliseconds_lossy() + rest, original);
    }

    /// `to_hours` reads `TO_HOURS`, not the constant of a neighbouring unit.
    #[test]
    fn each_conversion_reads_its_own_constant() {
        // Minutes are 1/60 of an hour but 1/1440 of a day: picking up
        // `TO_DAYS` here would yield zero hours and a 90 minute remainder.
        let (hours, rest) = Minutes::from(90).to_hours();
        assert_eq!(hours, Hours::from(1));
        assert_eq!(rest, Minutes::from(30));

        let (days, rest) = Minutes::from(90).to_days();
        assert_eq!(days, Days::from(0));
        assert_eq!(rest, Minutes::from(90));
    }

    /// `Display` appends the unit's own suffix.
    #[test]
    fn display_appends_the_unit_suffix() {
        assert_eq!(std::format!("{}", NanoSeconds::from(1)), "1ns");
        assert_eq!(std::format!("{}", MicroSeconds::from(2)), "2us");
        assert_eq!(std::format!("{}", MilliSeconds::from(3)), "3ms");
        assert_eq!(std::format!("{}", Seconds::from(4)), "4s");
        assert_eq!(std::format!("{}", Minutes::from(5)), "5min");
        assert_eq!(std::format!("{}", Hours::from(6)), "6h");
        assert_eq!(std::format!("{}", Days::from(7)), "7d");
    }

    /// The newtype is a transparent wrapper: `usize` goes in and comes back.
    #[test]
    fn a_unit_round_trips_through_usize() {
        let value: usize = Seconds::from(90).into();
        assert_eq!(value, 90);
        assert_eq!(Seconds::from(value), Seconds::from(90));
    }

    /// Arithmetic against another duration of the same unit.
    #[test]
    fn arithmetic_against_the_same_unit() {
        assert_eq!(Seconds::from(7) + Seconds::from(3), Seconds::from(10));
        assert_eq!(Seconds::from(7) - Seconds::from(3), Seconds::from(4));
        assert_eq!(Seconds::from(7) * Seconds::from(3), Seconds::from(21));
        assert_eq!(Seconds::from(7) / Seconds::from(3), Seconds::from(2));
        assert_eq!(Seconds::from(7) % Seconds::from(3), Seconds::from(1));

        let mut value = Seconds::from(7);
        value += Seconds::from(3);
        assert_eq!(value, Seconds::from(10));
        value -= Seconds::from(3);
        assert_eq!(value, Seconds::from(7));
        value *= Seconds::from(3);
        assert_eq!(value, Seconds::from(21));
        value /= Seconds::from(3);
        assert_eq!(value, Seconds::from(7));
        value %= Seconds::from(3);
        assert_eq!(value, Seconds::from(1));
    }

    /// Arithmetic against a bare `usize` scalar.
    #[test]
    fn arithmetic_against_a_scalar() {
        assert_eq!(Seconds::from(7) + 3, Seconds::from(10));
        assert_eq!(Seconds::from(7) - 3, Seconds::from(4));
        assert_eq!(Seconds::from(7) * 3, Seconds::from(21));
        assert_eq!(Seconds::from(7) / 3, Seconds::from(2));
        assert_eq!(Seconds::from(7) % 3, Seconds::from(1));

        let mut value = Seconds::from(7);
        value += 3;
        assert_eq!(value, Seconds::from(10));
        value -= 3;
        assert_eq!(value, Seconds::from(7));
        value *= 3;
        assert_eq!(value, Seconds::from(21));
        value /= 3;
        assert_eq!(value, Seconds::from(7));
        value %= 3;
        assert_eq!(value, Seconds::from(1));
    }

    /// Units order and compare by their value.
    #[test]
    fn units_order_by_value() {
        assert!(Seconds::from(1) < Seconds::from(2));
        assert_eq!(Seconds::from(2).max(Seconds::from(1)), Seconds::from(2));
    }

    /// The supertraits are strong enough to write code over any unit without
    /// naming a concrete one.
    #[test]
    fn a_generic_function_can_use_a_unit() {
        fn describe<T: TimeUnit>(value: T) -> (usize, &'static str) {
            let doubled = value * 2;
            ((doubled - value).into(), T::UNIT)
        }

        assert_eq!(describe(Seconds::from(21)), (21, "s"));
        assert_eq!(describe(Days::from(3)), (3, "d"));
    }
}
