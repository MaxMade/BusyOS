//! Port-mapped I/O.
//!
//! x86_64 keeps a second address space of 65536 byte-wide ports next to
//! memory, reached with `IN` and `OUT` rather than a load or a store. The
//! legacy devices the kernel cannot avoid live there, among them the
//! [`PIT`](crate::arch::x86_64::pit).
//!
//! [`outb`] and [`inb`] are the bare instructions, taking a port number
//! known only at run time. [`IOB`] is the typed form on top of them: it
//! fixes the port and the value type at compile time, so a register cannot
//! be read at the wrong width or written with a value meant for another
//! port.
//!
//! Neither form serialises anything. A port reached from two cores at once
//! is a data race in the hardware, and it is on the caller to hold whatever
//! lock guards the device, which is why every accessor here is `unsafe`.

use core::marker::PhantomData;

/// Writes `value` to the byte-wide port `port`, via `OUT`.
///
/// # Safety
///
/// A port write is a command to a device, so the caller must ensure that
///
/// - the device behind `port` expects this value, since a wrong one can
///   leave it in a state the kernel cannot get it out of,
/// - no other core is driving the same port at the same time.
pub unsafe fn outb(port: u16, value: u8) {
    unsafe {
        core::arch::asm!(
            "out dx, al",
            in("dx") port,
            in("al") value,
            options(nostack, nomem, preserves_flags),
        )
    };
}

/// Reads the byte-wide port `port`, via `IN`.
///
/// # Safety
///
/// A read is not always passive: some registers latch or clear on being
/// read, so the caller must ensure that
///
/// - reading `port` is harmless, or that the side effect is the one wanted,
/// - no other core is driving the same port at the same time.
pub unsafe fn inb(port: u16) -> u8 {
    let value: u8;

    unsafe {
        core::arch::asm!(
            "in al, dx",
            in("dx") port,
            out("al") value,
            options(nostack, nomem, preserves_flags),
        )
    };

    value
}

/// One byte-wide port, with its number and its value type fixed.
///
/// `PORT` is the port number and `Value` the type its contents are read and
/// written as, typically a [`bitfield`](bitfield_struct::bitfield) over `u8`
/// naming the register's fields. Both are compile-time parameters, so the
/// pairing of a port with its layout is checked once, where the alias is
/// declared, rather than at every access.
///
/// The type carries no data: [`read`](IOB::read) and [`write`](IOB::write)
/// are associated functions, and a value of `IOB` is only ever a handle to
/// put behind a lock. Holding it is what a driver uses to show it may touch
/// the port. It grants nothing on its own, since the port number is in the
/// type and anybody may name it.
pub struct IOB<const PORT: u16, Value: Into<u8> + From<u8>>(PhantomData<Value>);

impl<const PORT: u16, Value: Into<u8> + From<u8>> IOB<PORT, Value> {
    /// Creates a handle on the port.
    ///
    /// Allocates nothing and reads nothing, so it may be used to build a
    /// `static`.
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// Reads the port and interprets the byte as a `Value`.
    ///
    /// # Safety
    ///
    /// As [`inb`], for the port `PORT`.
    pub unsafe fn read() -> Value {
        unsafe { Value::from(inb(PORT)) }
    }

    /// Writes `value` to the port as a single byte.
    ///
    /// # Safety
    ///
    /// As [`outb`], for the port `PORT`.
    pub unsafe fn write(value: Value) {
        unsafe { outb(PORT, value.into()) }
    }
}
