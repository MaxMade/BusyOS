use crate::core_local;
use crate::kernel::bitset::{self, BitSet};
use crate::kernel::locking::PreviousToken;
use core::fmt::{Debug, Display};
use core::fmt::{LowerHex, UpperHex};
use core::hash::Hash;
use core::num::TryFromIntError;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Shl, Shr, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptFlag {
    Enabled,
    Disabled,
}

/// Interrupts masked, together with the token whose holder masked them.
///
/// Any token is accepted: masking interrupts touches nothing shared, so it
/// cannot take part in a deadlock cycle and needs no place in the lock
/// ordering. What matters is that the token is *consumed* for the duration,
/// so its holder can acquire nothing else until it is handed back.
#[derive(Debug)]
pub struct InterruptState<Token>
where
    Token: PreviousToken,
{
    flag: InterruptFlag,
    token: Token,
}

pub trait CPU {
    const STACK_ALIGNMENT: usize;

    const KERNEL_STACK_SIZE: usize;

    /// Stated as bounds on the associated type rather than as a
    /// `where Self::CPUID: ...` clause: the clause is self-referential —
    /// proving it requires normalising `Self::CPUID`, which brings in the
    /// very bounds being proven — and the solver gives up with `E0275`.
    type CPUID: PartialEq
        + Eq
        + Ord
        + PartialOrd
        + Debug
        + Display
        + Clone
        + Copy
        + TryFrom<usize, Error = TryFromIntError>
        + Into<usize>;

    const CPUID_BITS: usize;

    /// Stops the calling core for good, with interrupts masked.
    ///
    /// For the end of the panic path and anywhere else nothing sensible is
    /// left to do.
    ///
    /// # Safety
    ///
    /// Whatever the core holds stays held: locks are never released and work
    /// it took on is never finished, so other cores waiting on either wait
    /// forever.
    unsafe fn halt() -> !;

    fn disable_interrupts<Token>(token: Token) -> InterruptState<Token>
    where
        Token: PreviousToken,
    {
        let flag = Self::interrupt_flag();

        unsafe { Self::raw_disable_interrupts() };

        InterruptState { flag, token }
    }

    fn restore_interrupts<Token>(state: InterruptState<Token>) -> Token
    where
        Token: PreviousToken,
    {
        if state.flag == InterruptFlag::Enabled {
            unsafe { Self::raw_enable_interrupts() };
        }

        state.token
    }

    fn interrupt_flag() -> InterruptFlag;

    unsafe fn raw_enable_interrupts();

    unsafe fn raw_disable_interrupts();
}

core_local! {
    /// Id this core was handed by the bootloader.
    pub static CPUID: <crate::arch::CPU as CPU>::CPUID;
}

/// A set of cores, one bit per [`CPUID`](CPU::CPUID) the architecture can
/// name.
pub type CPUSet = BitSet<
    <crate::arch::CPU as CPU>::CPUID,
    { bitset::words(<crate::arch::CPU as CPU>::CPUID_BITS) },
>;

/// Architecture-neutral view of an interrupt/exception vector number.
///
/// Wraps whatever raw representation an architecture stores a vector in
/// (e.g. a `u8` on x86_64) behind named predicates for the exceptions
/// generic code needs to recognise, so a handler can act on "this was a page
/// fault" without knowing the concrete vector number an architecture
/// assigns it.
pub trait InterruptVector
where
    Self: Debug + Clone + Copy + PartialEq + Eq + Hash + Into<usize>,
{
    /// How many vectors the architecture has.
    ///
    /// Every vector number is below this, so a table with one slot per vector
    /// can be sized from it at compile time and indexed with
    /// [`into_raw`](Self::into_raw) without a bounds check of its own.
    const MAX_NUM: usize;

    /// The architecture's native representation of a vector number.
    type Raw: PartialEq + Eq;

    /// Converts to the architecture's native representation.
    fn into_raw(self) -> Self::Raw;

    /// Builds a vector from the architecture's native representation.
    fn from_raw(raw: Self::Raw) -> Self;

    /// Whether this vector is the division-by-zero exception.
    fn is_division_by_zero(&self) -> bool;

    /// Whether this vector is the breakpoint exception (`int3`).
    fn is_breakpoint(&self) -> bool;

    /// Whether this vector is the invalid-opcode exception.
    fn is_invalid_instruction(&self) -> bool;

    /// Whether this vector is the page-fault exception.
    fn is_page_fault(&self) -> bool;

    /// Whether this vector is the alignment-check exception.
    fn is_invalid_alignmnet(&self) -> bool;

    /// Hands out a vector no caller has been given before, or [`None`] once
    /// they are used up.
    ///
    /// Safe to call from several cores at once: two concurrent callers still
    /// get different vectors.
    ///
    /// A driver that needs an interrupt of its own asks here rather than
    /// picking a number, so two drivers cannot settle on the same vector.
    /// Only the vectors the architecture leaves to software are handed out,
    /// never one an exception already owns.
    ///
    /// A vector is never given back. There is no matching release, because
    /// nothing unregisters a driver yet, and a vector freed while the
    /// controller still routes it would be handed to a second driver that
    /// then sees the first one's interrupts.
    fn allocate() -> Option<Self>;
}

/// Architecture-neutral view of the state an interrupt/exception saved on
/// entry.
///
/// A handler reads and rewrites the interrupted context through this trait
/// instead of an architecture's raw frame layout, so generic fault-handling
/// code (page-fault resolution, signal delivery, ...) stays portable.
pub trait InterruptStackFrame {
    /// The architecture's native register width (e.g. `u64` on x86_64).
    type RawRegister: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord;

    /// The architecture's native error-code representation.
    type ErrorCode;

    /// The interrupted context's stack pointer.
    fn get_sp(&self) -> Register<Self::RawRegister>;

    /// Overwrites the interrupted context's stack pointer.
    fn set_sp(&mut self, sp: Register<Self::RawRegister>);

    /// The interrupted context's instruction pointer.
    fn get_ip(&self) -> Register<Self::RawRegister>;

    /// Overwrites the interrupted context's instruction pointer, e.g. to
    /// resume execution elsewhere.
    fn set_ip(&mut self, ip: Register<Self::RawRegister>);

    /// The register a call's return value is passed back in.
    fn get_ret(&self) -> Register<Self::RawRegister>;

    /// Overwrites the register a call's return value is passed back in.
    fn set_ret(&mut self, ret: Register<Self::RawRegister>);

    /// The error code the CPU pushed for this vector, or the architecture's
    /// placeholder if the vector does not push one.
    fn get_error(&self) -> Self::ErrorCode;
}

/// A CPU register's value, kept distinct from a bare integer so a stray
/// arithmetic operation or format specifier can't silently mix register
/// contents with ordinary data.
///
/// Implements the arithmetic, bitwise, and formatting traits `T` itself
/// implements, forwarding to `T`, so a `Register<T>` is used the same way as
/// the integer it wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Register<T>(T)
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord;

impl<T> Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord,
{
    /// Unwraps to the architecture's native representation.
    pub const fn into_raw(self) -> T {
        self.0
    }

    /// Wraps the architecture's native representation.
    pub fn from_raw(raw: T) -> Self {
        Self(raw)
    }
}

impl<T> Add for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<T> Add<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl<T> Sub for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<T> Sub<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl<T> Mul for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<T> Mul<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<T> Div for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl<T> Div<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl<T> BitAnd for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + BitAnd<Output = T>,
{
    type Output = Self;

    fn bitand(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<T> BitAnd<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + BitAnd<Output = T>,
{
    type Output = Self;

    fn bitand(self, rhs: T) -> Self::Output {
        Self(self.0 & rhs)
    }
}

impl<T> BitOr for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + BitOr<Output = T>,
{
    type Output = Self;

    fn bitor(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl<T> BitOr<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + BitOr<Output = T>,
{
    type Output = Self;

    fn bitor(self, rhs: T) -> Self::Output {
        Self(self.0 | rhs)
    }
}

impl<T> BitXor for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + BitXor<Output = T>,
{
    type Output = Self;

    fn bitxor(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<T> BitXor<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + BitXor<Output = T>,
{
    type Output = Self;

    fn bitxor(self, rhs: T) -> Self::Output {
        Self(self.0 ^ rhs)
    }
}

impl<T> Shl for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Shl<Output = T>,
{
    type Output = Self;

    fn shl(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 << rhs.0)
    }
}

impl<T> Shl<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Shl<Output = T>,
{
    type Output = Self;

    fn shl(self, rhs: T) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl<T> Shr for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Shr<Output = T>,
{
    type Output = Self;

    fn shr(self, rhs: Register<T>) -> Self::Output {
        Self(self.0 >> rhs.0)
    }
}

impl<T> Shr<T> for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Shr<Output = T>,
{
    type Output = Self;

    fn shr(self, rhs: T) -> Self::Output {
        Self(self.0 >> rhs)
    }
}

impl<T> Display for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> LowerHex for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + LowerHex,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Each byte is 2 hex digits.
        let width = size_of::<T>() * 2;
        write!(f, "{:0width$x}", self.0, width = width)
    }
}

impl<T> UpperHex for Register<T>
where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + UpperHex,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Each byte is 2 hex digits.
        let width = size_of::<T>() * 2;
        write!(f, "{:0width$X}", self.0, width = width)
    }
}
