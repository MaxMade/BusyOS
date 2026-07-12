#[cfg(all(not(test), not(feature = "library")))]
pub mod panic;

pub mod bootinfo;
pub mod locking;
pub mod spinlock;
pub mod ticketlock;
