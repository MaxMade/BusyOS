//! Handle x86_64 extensions and features.

use busyos::arch::x86_64::cpuid::CPUID;
use busyos::arch::x86_64::cpuid::ExtendedFunction;
use busyos::arch::x86_64::cpuid::FeatureInformation;
use busyos::arch::x86_64::cpuid::StructuredExtendedFeature;
use busyos::arch::x86_64::cr4::CR4;
use busyos::arch::x86_64::msr::EFER;
use busyos::arch::x86_64::msr::MSR;
use busyos::arch::x86_64::paging::install_pat;

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
    /// - `pat`: support for the `IA32_PAT` MSR, which the caching mode of a
    /// mapping is encoded against.
    /// - `x2apic`: support for the x2APIC mode.
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

        // Check if the `IA32_PAT` MSR is available
        let feature_information = unsafe { FeatureInformation::read() };
        if !feature_information.edx.pat() {
            panic!("Required feature `pat` is not available");
        }

        // Check if the `x2APIC` mode is available
        if !feature_information.ecx.x2apic() {
            panic!("Required feature `x2apic` is not available");
        }

        // Install the memory types the kernel's caching modes name
        //
        // Like `EFER` above, this MSR is per core, so a core the bootloader
        // has not brought up has to be given the same layout before it uses a
        // mapping made with a caching mode of its own.
        //
        // Safety: `pat` was just checked, and the layout keeps the memory
        // types of the slots the current mappings select.
        unsafe { install_pat() };
    }
}
