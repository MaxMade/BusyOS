use uefi::prelude::*;
use uefi::println;

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();

    // Starting the kernel
    println!("Booting BUSYOS!");

    // Exit boot service.
    //
    // # Safety
    //
    // From then on, only UEFI configuration tables and runtime service can be used.
    let memory_map = unsafe { uefi::boot::exit_boot_services(None) };

    // Perform shutdown
    uefi::runtime::reset(runtime::ResetType::SHUTDOWN, Status::SUCCESS, None)
}
