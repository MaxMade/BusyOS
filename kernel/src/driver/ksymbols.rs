//! Kernel symbols: turning a code address back into `name+offset`.
//!
//! The bootloader builds the table from the kernel ELF's `.symtab`, filters
//! and sorts it, and hands it over in [`Bootinfo`](crate::kernel::bootinfo::Bootinfo)
//! as an array of [`KernelSymbol`]. Doing that there keeps ELF parsing and
//! sorting off the kernel's stack, and leaves the kernel a table that never
//! changes, so looking a symbol up takes no lock and works on the panic path.

use core::ffi::c_void;
use core::slice;
use core::str;

use driver_macro::module;

use crate::arch::Paging;
use crate::arch::generic::paging::{ReversePaging, VirtualAddress};
use crate::driver::module::{Module, ModuleDriver, Modules};
use crate::kernel::arc::Arc;
use crate::kernel::bootinfo::{BOOTINFO, KernelSymbol};
use crate::kernel::locking::{CanAcquire, DriverLevelID, LockId, PreviousToken};
use crate::mem::page_frames::PageFrames;
use crate::user::errno::Errno;

pub struct KSymbols;

module! {
    name: "ksymbols",
    priority: 10,
    driver: crate::driver::ksymbols::KSymbols,
}

impl KSymbols {
    /// Finds the symbol `addr` lies in, returning its name and how far into
    /// it `addr` is.
    ///
    /// Meant for turning a code address, such as a return address from a
    /// backtrace, into `name+offset`. See [`find`] for which symbol is picked
    /// when several could match.
    ///
    /// The table never changes after boot, so this takes no lock. The token
    /// is only threaded through, to keep the shape of the other driver
    /// calls.
    ///
    /// # Errors
    ///
    /// Gives the token back if no symbol covers `addr`.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    pub fn lookup<Token>(
        addr: VirtualAddress<c_void>,
        token: Token,
    ) -> Result<(&'static str, usize, Token), Token>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        match Self::emergency_lookup(addr) {
            Some((name, offset)) => Ok((name, offset, token)),
            None => Err(token),
        }
    }

    /// [`lookup`](Self::lookup) without a token, for the panic path.
    ///
    /// Safe, unlike other emergency paths: there is no lock to bypass, since
    /// the table is written once by the bootloader and only read here.
    pub fn emergency_lookup(addr: VirtualAddress<c_void>) -> Option<(&'static str, usize)> {
        let symbols = table();
        let (index, offset) = find(symbols, addr)?;

        Some((name(&symbols[index]), offset))
    }
}

/// The symbol table the bootloader handed over, or an empty one if there is
/// none.
fn table() -> &'static [KernelSymbol] {
    // SAFETY: the bootloader fills in the boot information before the kernel
    // runs, and nothing writes it afterwards.
    let bootinfo = unsafe { BOOTINFO.assume_init_ref() };

    if bootinfo.kernel_symbols.is_null() {
        return &[];
    }

    // SAFETY: the bootloader leaves `kernel_symbols_len` entries at
    // `kernel_symbols`, in memory the kernel never reuses, and the direct map
    // covers all physical memory.
    unsafe {
        let virt_addr = Paging::<PageFrames>::phys_to_virt(bootinfo.kernel_symbols);
        slice::from_raw_parts(virt_addr.as_ptr(), bootinfo.kernel_symbols_len)
    }
}

/// The name of `symbol`, which lies in the kernel ELF file.
fn name(symbol: &KernelSymbol) -> &'static str {
    // SAFETY: the bootloader points the name into the kernel ELF file, which
    // stays in place for as long as the kernel runs, and the direct map
    // covers all physical memory.
    let bytes = unsafe {
        let virt_addr = Paging::<PageFrames>::phys_to_virt(symbol.name);
        slice::from_raw_parts(virt_addr.as_ptr().cast_const(), symbol.name_len)
    };

    // The bootloader took the names from the ELF crate as `&str`, so they
    // are UTF-8, but a check costs nothing worth saving here.
    str::from_utf8(bytes).unwrap_or("<invalid symbol name>")
}

/// The index of the symbol in `symbols`, which is sorted by address, that
/// `addr` lies in, with the offset of `addr` into it.
///
/// A binary search for the last symbol starting at or before `addr`. Several
/// symbols can start at the same address, such as a function and a label the
/// linker script puts in front of it, so all of those are looked at:
///
/// - One whose size reaches `addr` wins.
/// - Failing that, one without a size, which is what assembly labels and
///   linker script symbols have, is taken to reach up to the next symbol.
/// - A sized symbol that ends before `addr` does not cover it, which keeps
///   padding between functions from being blamed on the function before it.
fn find(symbols: &[KernelSymbol], addr: VirtualAddress<c_void>) -> Option<(usize, usize)> {
    // Everything before `end` starts at or before `addr`.
    let end = symbols.partition_point(|symbol| symbol.addr <= addr);
    let start = symbols[..end].last()?.addr;

    let offset = addr.addr() - start.addr();
    let mut label = None;

    for (index, symbol) in symbols[..end]
        .iter()
        .enumerate()
        .rev()
        .take_while(|(_, symbol)| symbol.addr == start)
    {
        if offset < symbol.size {
            return Some((index, offset));
        }

        if symbol.size == 0 && label.is_none() {
            label = Some((index, offset));
        }
    }

    label
}

impl Module for KSymbols {
    /// Registers the driver. The table itself needs no setting up, see the
    /// [module documentation](self).
    ///
    /// # Panics
    ///
    /// If the driver cannot be allocated or registered.
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        let driver = match Arc::try_new(KSymbols, token) {
            Ok((driver, t)) => {
                token = t;
                driver
            }
            Err((error, _)) => panic!("Unable to create driver instance of ksymbols: {}", error),
        };

        match Modules::register(ModuleDriver::KSymbols(driver), token) {
            Ok(t) => token = t,
            Err((error, _)) => panic!("Unable to register driver ksymbols as module: {}", error),
        };

        Ok(token)
    }

    fn name(&self) -> &'static str {
        "ksymbols"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arch::generic::paging::PhysicalAddress;

    fn addr(value: usize) -> VirtualAddress<c_void> {
        VirtualAddress::new(value as _)
    }

    const fn symbol(addr: usize, size: usize) -> KernelSymbol {
        KernelSymbol {
            addr: VirtualAddress::new(addr as _),
            size,
            name: PhysicalAddress::null(),
            name_len: 0,
        }
    }

    const SYMBOLS: [KernelSymbol; 5] = [
        symbol(0, 0),
        // A label in front of a function at the same address.
        symbol(0x1000, 0),
        symbol(0x1000, 0x20),
        // Padding from 0x1020 up to the next function.
        symbol(0x1030, 0x10),
        symbol(0x2000, 0),
    ];

    #[test]
    fn an_address_inside_a_function_finds_it() {
        assert_eq!(find(&SYMBOLS, addr(0x1000)), Some((2, 0)));
        assert_eq!(find(&SYMBOLS, addr(0x101f)), Some((2, 0x1f)));
        assert_eq!(find(&SYMBOLS, addr(0x1034)), Some((3, 4)));
    }

    #[test]
    fn padding_after_a_function_falls_back_to_a_label_at_its_start() {
        // The function at 0x1000 ends at 0x1020, but the label in front of it
        // has no size, so it is taken to reach up to the next function.
        assert_eq!(find(&SYMBOLS, addr(0x1024)), Some((1, 0x24)));
    }

    #[test]
    fn padding_without_a_label_is_not_covered() {
        assert_eq!(find(&SYMBOLS, addr(0x1040)), None);
    }

    #[test]
    fn a_symbol_without_size_reaches_up_to_the_next_one() {
        assert_eq!(find(&SYMBOLS, addr(0x2345)), Some((4, 0x345)));
    }

    #[test]
    fn an_empty_table_finds_nothing() {
        assert_eq!(find(&[], addr(0x1000)), None);
    }
}
