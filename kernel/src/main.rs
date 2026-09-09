#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

pub mod arch;
pub mod driver;
pub mod kernel;
pub mod mem;
pub mod user;
pub mod utils;

#[cfg(test)]
fn main() {
    // Must never be used - just to make the compiler happy
    loop {}
}
