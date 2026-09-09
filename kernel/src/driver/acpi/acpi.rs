use driver_macro::module;

use crate::{
    driver::module::Module,
    kernel::locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
    user::errno::Errno,
};

module! {
    name: "acpi",
    priority: 0,
    driver: crate::driver::acpi::acpi::Acpi,
}

pub struct Acpi {}

impl Module for Acpi {
    fn init<Token>(_: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        todo!()
    }
}
