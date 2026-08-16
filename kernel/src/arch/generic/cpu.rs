use crate::kernel::locking::{CanAcquire, PreviousToken, level::Prologue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptFlag {
    Enabled,
    Disabled,
}

#[derive(Debug)]
pub struct InterruptState<Token>
where
    Token: CanAcquire<Prologue> + PreviousToken,
{
    flag: InterruptFlag,
    token: Token,
}

pub trait CPU {
    fn disable_interrupts<Token>(token: Token) -> InterruptState<Token>
    where
        Token: CanAcquire<Prologue> + PreviousToken,
    {
        let flag = Self::interrupt_flag();

        unsafe { Self::raw_disable_interrupts() };

        InterruptState { flag, token }
    }

    fn restore_interrupts<Token>(state: InterruptState<Token>) -> Token
    where
        Token: CanAcquire<Prologue> + PreviousToken,
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
