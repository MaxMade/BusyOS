#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

pub mod kernel;

#[cfg(test)]
fn main() {
    // Must never be used - just to make the compiler happy
    loop {}
}
