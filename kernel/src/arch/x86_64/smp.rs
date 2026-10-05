//! Starting the other cores (application processors).
//!
//! A core is started by an INIT IPI followed by start-up IPIs, see
//! [`X2Apic::send_init_ipi`](crate::driver::x86_64::x2apic::X2Apic::send_init_ipi),
//! and begins in real mode at a page below 1 MiB. [`Trampoline`] prepares that
//! page: it copies the start-up code of `ap_trampoline.S` into the area the
//! bootloader reserved (see [`AP_TRAMPOLINE_PAGES`]), builds the temporary page
//! tables the code switches to long mode with, and hands each core the
//! parameters it needs to reach the kernel.
//!
//! The cores are started one at a time: they share the copy, its parameters and
//! its temporary stack, so the next one may only be started once the previous
//! one has [`started`](Trampoline::started).

use core::ptr;

use crate::{
    arch::{
        BOOT_CPUID, Paging,
        generic::paging::{PhysicalAddress, ReversePaging},
        x86_64::{
            cr4::CR4,
            msr::{EFER, MSR},
            paging::CR3,
            pit::PIT,
        },
    },
    driver::{acpi::acpi::Acpi, x86_64::x2apic::X2Apic},
    kernel::{
        bootinfo::{AP_TRAMPOLINE_PAGES, BOOTINFO},
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
        printk::LogLevel,
        time::MilliSeconds,
    },
    mem::page_frames::PageFrames,
    printkln,
};

unsafe extern "C" {
    // Only the addresses of these matter: the start-up code lies between the
    // first two, and the others name its parameters. See `ap_trampoline.S`.
    static ap_trampoline_start: u8;
    static ap_trampoline_end: u8;
    static ap_trampoline_cpu_id: u8;
    static ap_trampoline_cr3: u8;
    static ap_trampoline_cr4: u8;
    static ap_trampoline_efer: u8;
    static ap_trampoline_entry: u8;
    static ap_trampoline_started: u8;

    /// Where a core enters the kernel's half of the address space, see
    /// `ap_trampoline.S`.
    fn __ap_start();
}

/// How long a core may take to report in after its start-up IPIs before it
/// is given up on.
const START_TIMEOUT_MS: usize = 100;

/// Starts every enabled core the firmware reported, other than the calling
/// one, one at a time, and returns once each has either reported in or been
/// given up on.
///
/// Each core goes through the sequence of Intel SDM vol. 3, 8.4.4: an INIT
/// IPI, 10 ms, a start-up IPI, at least 200 µs, and a second start-up IPI if
/// the core has not got going yet. The core then has [`START_TIMEOUT_MS`] to
/// [`start`](Trampoline::started). One that misses it is sent another INIT,
/// which parks it again, so that it cannot wake up late in the area the next
/// core is already using.
///
/// The cores are numbered from [`BOOT_CPUID`] + 1 in the order the firmware
/// listed them, skipping the ones that did not come up, and never beyond the
/// number of core-local blocks the bootloader made.
///
/// Must be called on the boot core, once everything the other cores rely on
/// is in place: the kernel's page tables, the GDT and IDT, the heap and the
/// drivers, the x2APIC and ACPI ones in particular.
///
/// # Token
///
/// The `token` is consumed and returned.
pub fn start_other_cores<Token>(token: Token) -> Token
where
    Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    let Some(trampoline) = Trampoline::get() else {
        return printkln!(
            LogLevel::Warn,
            token,
            "Unable to start other cores: no start-up area below 1 MiB"
        );
    };

    let (acpi, mut token) = Acpi::get(token);
    let Some(acpi) = acpi else {
        return printkln!(
            LogLevel::Warn,
            token,
            "Unable to start other cores: no ACPI driver"
        );
    };

    // SAFETY: this is the boot core, with the kernel's page tables active,
    // and no other core runs yet.
    unsafe { trampoline.install() };

    // SAFETY: the x2APIC driver has switched the boot core's local APIC into
    // x2APIC mode, see the contract.
    let own = unsafe { X2Apic::local_id() };

    // SAFETY: the bootloader fills in the boot information before the kernel
    // runs, and nothing writes it afterwards.
    let num_cpus = unsafe { BOOTINFO.assume_init_ref() }.num_cpus;
    let boot: usize = BOOT_CPUID.into();
    let mut next = boot + 1;

    for lapic_id in acpi.x2apic_lapic_ids() {
        if lapic_id == own {
            continue;
        }

        if next >= num_cpus {
            token = printkln!(
                LogLevel::Warn,
                token,
                "Unable to start core with local APIC {:?}: no core-local block left",
                lapic_id
            );
            break;
        }

        // SAFETY: `install` has run, and the previous core has started or
        // been parked again, see below.
        unsafe { trampoline.prepare(next) };

        // SAFETY: the destination is another core, which the kernel does not
        // run on yet, and the trampoline holds the start-up code.
        unsafe { X2Apic::send_init_ipi(&lapic_id) };
        token = wait(10, token);

        // The PIT waits in whole milliseconds, which is more than the 200 µs
        // the sequence asks for between and after the start-up IPIs.
        unsafe { X2Apic::send_startup_ipi(&lapic_id, trampoline.page()) };
        token = wait(1, token);

        if !trampoline.started() {
            unsafe { X2Apic::send_startup_ipi(&lapic_id, trampoline.page()) };
            token = wait(1, token);
        }

        let mut waited = 0;
        while !trampoline.started() && waited < START_TIMEOUT_MS {
            token = wait(1, token);
            waited += 1;
        }

        if trampoline.started() {
            next += 1;
        } else {
            // SAFETY: as for the first INIT. Parks the core again, so that it
            // does not run into the area once the next core uses it.
            unsafe { X2Apic::send_init_ipi(&lapic_id) };

            token = printkln!(
                LogLevel::Error,
                token,
                "Core with local APIC {:?} did not start within {} ms",
                lapic_id,
                START_TIMEOUT_MS
            );
        }
    }

    token
}

/// Waits `ms` milliseconds on the PIT, best effort: a wait the PIT cannot
/// produce is skipped, since there is nothing better to wait on yet.
fn wait<Token>(ms: usize, token: Token) -> Token
where
    Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
{
    match PIT::try_wait(MilliSeconds::from(ms), token) {
        Ok(token) | Err((_, token)) => token,
    }
}

/// Size of one page of the trampoline area.
const PAGE_SIZE: usize = 0x1000;

/// Page of the trampoline area that holds the temporary PML4. The PDPT and
/// the page directory follow it, see [`AP_TRAMPOLINE_PAGES`].
const PML4_PAGE: usize = 1;

/// A present, writable page-table entry.
const PRESENT_WRITABLE: u64 = 0b11;

/// A present, writable entry that maps a 2 MiB page directly.
const PRESENT_WRITABLE_HUGE: u64 = PRESENT_WRITABLE | 1 << 7;

/// First PML4 entry of the kernel's half of the address space.
const KERNEL_HALF: usize = 256;

/// The area below 1 MiB the other cores start in.
pub struct Trampoline {
    /// Physical address of the area, which the start-up IPI names.
    phys: usize,

    /// The same area, reached through the direct map.
    virt: *mut u8,
}

impl Trampoline {
    /// The area the bootloader reserved, or [`None`] if it could not reserve
    /// one, in which case only the boot core can run.
    pub fn get() -> Option<Self> {
        // SAFETY: the bootloader fills in the boot information before the
        // kernel runs, and nothing writes it afterwards.
        let phys = unsafe { BOOTINFO.assume_init_ref() }.ap_trampoline;

        if phys.is_null() {
            return None;
        }

        // SAFETY: the direct map covers all physical memory.
        let virt = unsafe { Paging::<PageFrames>::phys_to_virt(phys) };

        Some(Self {
            phys: phys.addr(),
            virt: virt.as_ptr().cast(),
        })
    }

    /// The page a start-up IPI has to name for a core to start here.
    pub fn page(&self) -> u8 {
        // Below 1 MiB, so the page number fits, see the bootloader.
        (self.phys >> 12) as u8
    }

    /// Copies the start-up code into the area and builds its page tables.
    ///
    /// The parameters every core shares are filled in here too: the page
    /// tables, `CR4` and `EFER` of the calling core, which every other core
    /// takes over, and the address it enters the kernel at.
    ///
    /// # Safety
    ///
    /// Must be called on the boot core, with the kernel's page tables active,
    /// and before any core is started. No core may be running in the area.
    pub unsafe fn install(&self) {
        // SAFETY: the start-up code is a linked range of the kernel image.
        let code = unsafe {
            let start = &raw const ap_trampoline_start;
            let end = &raw const ap_trampoline_end;
            core::slice::from_raw_parts(start, end as usize - start as usize)
        };

        assert!(code.len() <= PAGE_SIZE, "AP start-up code exceeds a page");

        // SAFETY: the bootloader reserved `AP_TRAMPOLINE_PAGES` pages here
        // for this, and nothing runs in them yet.
        unsafe {
            ptr::write_bytes(self.virt, 0, AP_TRAMPOLINE_PAGES * PAGE_SIZE);
            ptr::copy_nonoverlapping(code.as_ptr(), self.virt, code.len());
        }

        // SAFETY: as above.
        unsafe { self.build_page_tables() };

        let cr3 = CR3::read().into_bits();
        let cr4 = CR4::read().into_bits();
        // SAFETY: `EFER` exists on every x86_64 processor.
        let efer = unsafe { EFER::read() }.into_bits();

        // SAFETY: the parameters lie in the copy just made.
        unsafe {
            self.write(&raw const ap_trampoline_cr3, cr3);
            self.write(&raw const ap_trampoline_cr4, cr4);
            self.write(&raw const ap_trampoline_efer, efer);
            self.write(
                &raw const ap_trampoline_entry,
                __ap_start as *const () as u64,
            );
        }
    }

    /// Prepares the area for starting the core `cpu_id`.
    ///
    /// # Safety
    ///
    /// [`install`](Self::install) must have run, and the previous core, if
    /// any, must have [`started`](Self::started).
    pub unsafe fn prepare(&self, cpu_id: usize) {
        // SAFETY: the parameters lie in the copy `install` made.
        unsafe {
            self.write(&raw const ap_trampoline_cpu_id, cpu_id as u64);
            self.flag().write_volatile(0);
        }
    }

    /// Whether the core last [`prepare`](Self::prepare)d for has got going:
    /// it has read its parameters and left the shared temporary stack, so
    /// that the next core may be started.
    pub fn started(&self) -> bool {
        // SAFETY: the flag lies in the copy `install` made, and the core
        // writes it with a single aligned store.
        unsafe { self.flag().read_volatile() != 0 }
    }

    /// Builds the temporary page tables in the pages behind the code.
    ///
    /// The PML4 maps the first 2 MiB onto themselves, through one PDPT and
    /// one page directory with a single 2 MiB page, which covers the area
    /// wherever below 1 MiB it is. Its upper half is a copy of the kernel's,
    /// so that kernel code, the boot stacks and the direct map are reachable
    /// as soon as long mode is active.
    ///
    /// # Safety
    ///
    /// As for [`install`](Self::install).
    unsafe fn build_page_tables(&self) {
        let table = |page: usize| -> *mut u64 {
            // SAFETY: every page index used below lies in the area.
            unsafe { self.virt.add(page * PAGE_SIZE).cast() }
        };
        let phys = |page: usize| (self.phys + page * PAGE_SIZE) as u64;

        let (pml4, pdpt, pd) = (PML4_PAGE, PML4_PAGE + 1, PML4_PAGE + 2);

        // SAFETY: the kernel's PML4 is the one `CR3` names, reached through
        // the direct map, and the tables written lie in the area.
        unsafe {
            let kernel_pml4 = Paging::<PageFrames>::phys_to_virt(PhysicalAddress::<u64>::new(
                (CR3::read().addr() << 12) as _,
            ))
            .as_ptr();

            for index in KERNEL_HALF..512 {
                *table(pml4).add(index) = *kernel_pml4.add(index);
            }

            *table(pml4) = phys(pdpt) | PRESENT_WRITABLE;
            *table(pdpt) = phys(pd) | PRESENT_WRITABLE;
            *table(pd) = PRESENT_WRITABLE_HUGE;
        }
    }

    /// Writes `value` into the copy, at the parameter `symbol` names.
    ///
    /// # Safety
    ///
    /// `symbol` must be one of the parameters of the start-up code, and
    /// [`install`](Self::install) must have copied the code.
    unsafe fn write(&self, symbol: *const u8, value: u64) {
        // SAFETY: see the function's contract.
        unsafe { self.parameter(symbol).cast::<u64>().write_volatile(value) };
    }

    /// The flag a core sets once it has started, in the copy.
    fn flag(&self) -> *mut u32 {
        self.parameter(&raw const ap_trampoline_started).cast()
    }

    /// Where the parameter `symbol` names lies in the copy.
    fn parameter(&self, symbol: *const u8) -> *mut u8 {
        let offset = symbol as usize - (&raw const ap_trampoline_start) as usize;

        // SAFETY: every parameter lies within the start-up code, which
        // `install` checks fits into the first page of the area.
        unsafe { self.virt.add(offset) }
    }
}

// SAFETY: the area is reserved for this type alone, and the boot core is the
// only one that uses it.
unsafe impl Send for Trampoline {}

// The area holds the code page, the three tables from `PML4_PAGE` on, and the
// temporary stack behind them, whose top `ap_trampoline.S` puts at 0x5000.
const _: () = assert!(AP_TRAMPOLINE_PAGES * PAGE_SIZE >= 0x5000);
const _: () = assert!(PML4_PAGE + 3 < AP_TRAMPOLINE_PAGES);
