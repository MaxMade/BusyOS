//! Handle x86_64 extensions and features.

use busyos::arch::x86_64::cpuid::CPUID;
use busyos::arch::x86_64::cpuid::ExtendedFunction;
use busyos::arch::x86_64::cpuid::StructuredExtendedFeature;
use busyos::arch::x86_64::cr4::CR4;
use busyos::arch::x86_64::msr::EFER;
use busyos::arch::x86_64::msr::MSR;

#[derive(Debug)]
pub struct Features;

impl crate::arch::generic::features::Features for Features {
    /// Checks whether all required features are available and actives them
    ///
    /// # Panics
    ///
    /// If one of the following features is unavailable, this function will `panic`:
    ///
    /// - `syscall`: support for `syscall`/`sysret` instructions.
    /// - `nx`: support for `non-executable` pages.
    /// - `pdpe1gb`: support for `1 GiB` pages at `pdp` (page directory pointer table)
    /// level.
    /// - `fsgsbase`: support for the `RDFSBASE`/`RDGSBASE`/`WRFSBASE`/`WRGSBASE`
    /// instructions.
    fn activate() {
        // Check if `syscall`/`sysret` instructions are available
        let extended_function = unsafe { ExtendedFunction::read() };
        if !extended_function.edx.syscall() {
            panic!("Required feature `syscall` is not available");
        }

        // Enable `syscall`/`sysret` instructions
        let mut efer = unsafe { EFER::read() };
        efer.set_sce(true);
        unsafe { efer.write() };

        // Check if `nx` bit (execute disable bit) is available
        let extended_function = unsafe { ExtendedFunction::read() };
        if !extended_function.edx.nx() {
            panic!("Required feature `nx` is not available");
        }

        // Enable `nx` bit (execute disable bit)
        let mut efer = unsafe { EFER::read() };
        efer.set_nxe(true);
        unsafe { efer.write() };

        // Check if `pdpe1gb` bit  is available
        if !extended_function.edx.pdpe1gb() {
            panic!("Required feature `pdpe1gb` is not available");
        }

        // Check if `RDGSBASE`/`WRGSBASE` instructions are available
        let structured_extended_feature = unsafe { StructuredExtendedFeature::read() };
        if !structured_extended_feature.ebx.fsgsbase() {
            panic!("Required feature `fsgsbase` is not available");
        }

        // Enable `RDGSBASE`/`WRGSBASE` instructions
        //
        // `cr4` is not reloaded during the hand-over, so the kernel's
        // bootstrap stub can rely on this bit being set.
        let mut cr4 = CR4::read();
        cr4.set_fsgsbase(true);
        unsafe { cr4.write() };
    }
}
