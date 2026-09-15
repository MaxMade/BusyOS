use crate::{
    kernel::{
        arc::Arc,
        linked_list::LinkedList,
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
        ticketlock::{DriverRWTicketlock, RWTicketlock},
    },
    user::errno::{Errno, ToErrno},
};

pub struct Modules(LinkedList<Arc<dyn Module>>);

pub static MODULES: DriverRWTicketlock<Modules> =
    DriverRWTicketlock::new(RWTicketlock::new(), Modules(LinkedList::new()));

unsafe extern "C" {
    pub static __init_modules_callbacks_start: u64;
    pub static __init_modules_callbacks_end: u64;
}

impl Modules {
    pub fn register<Token>(module: Arc<dyn Module>, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut modules, token) = MODULES.acquire(token);

        let token = match modules.0.try_push_back(module, token) {
            Ok(token) => token,
            Err((error, token)) => {
                let token = modules.release(token);
                let errno = error.to_errno();
                return Err((errno, token));
            }
        };

        Ok(modules.release(token))
    }

    pub fn init<Token>(token: Token) -> Token
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let start: *const u64 = unsafe { &__init_modules_callbacks_start };
        let end: *const u64 = unsafe { &__init_modules_callbacks_end };

        let mut ptr = start;
        while ptr < end {
            let init: extern "C" fn() = unsafe { core::mem::transmute(*ptr) };

            init();

            ptr = unsafe { ptr.add(1) };
        }

        token
    }
}

pub trait Module: Send + Sync {
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken;
}
