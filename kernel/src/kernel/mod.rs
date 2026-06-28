#[cfg(all(not(test),not(feature="library")))]
pub mod panic;

pub mod locking;
pub mod spinlock;
pub mod ticketlock;
pub mod bootinfo;
