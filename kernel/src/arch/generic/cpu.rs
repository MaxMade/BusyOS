use crate::kernel::locking::PreviousToken;
use core::fmt::{Debug,Display};

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
    type CPUID: PartialEq + Eq + Ord + PartialOrd + Debug + Display + Clone + Copy;

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
