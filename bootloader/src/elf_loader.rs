//! Helpers for loading the  Kernel *E*xecutable and *L*inkable *F*ormat file.

use core::ffi::c_void;

use alloc::{slice, vec::Vec};
use busyos::{
    arch::{
        REGULAR_PAGE_SIZE,
        generic::paging::{PhysicalAddress, VirtualAddress},
    },
    kernel::bootinfo::Bootinfo,
};
use elf::{ElfBytes, endian::AnyEndian};
use uefi::proto::media::file::File;

/// Loaded Kernel *E*xecutable and *L*inkable *F*ormat
pub struct ELF {
    data: Vec<u8>,
}

impl ELF {
    /// Try to read ELF from EFI root volume (identified by `image_handle`) at `path`.
    ///
    /// # Panics
    ///
    /// If any of the file system operations fails, this function will [`panic`].
    pub fn read(path: &str, image_handle: uefi::Handle) -> Self {
        let mut file_system = match uefi::boot::get_image_file_system(image_handle) {
            Ok(file_system) => file_system,
            Err(error) => {
                panic!("Unable to get image file system: {}", error);
            }
        };
        let mut volume = match file_system.open_volume() {
            Ok(volume) => volume,
            Err(error) => {
                panic!("Unable to get image file system: {}", error);
            }
        };
        let mut buf = [0u16; 128];
        let path = uefi::CStr16::from_str_with_buf(path, &mut buf).unwrap();
        let file = match volume.open(
            path,
            uefi::proto::media::file::FileMode::Read,
            uefi::proto::media::file::FileAttribute::empty(),
        ) {
            Ok(file) => file,
            Err(error) => {
                panic!("Unable to open file \"{}\": {}", path, error);
            }
        };
        let mut file = match file.into_type() {
            Ok(uefi::proto::media::file::FileType::Regular(file)) => file,
            Ok(uefi::proto::media::file::FileType::Dir(_)) => {
                panic!("Unable to open file \"{}\": Is a directory", path);
            }
            Err(error) => {
                panic!("Unable to open file \"{}\": {}", path, error);
            }
        };
        let mut buf = [0u8; 256];
        let file_info: &uefi::proto::media::file::FileInfo = match file.get_info(&mut buf) {
            Ok(file_info) => file_info,
            Err(error) => {
                panic!(
                    "Unable to query file information of \"{}\": {}",
                    path, error
                );
            }
        };
        let mut data = alloc::vec![0u8; file_info.file_size() as _];
        let data_len = match file.read(&mut data) {
            Ok(kernel_data_len) => kernel_data_len,
            Err(error) => {
                panic!("Unable to read \"{}\": {}", path, error);
            }
        };
        data.truncate(data_len);

        Self { data }
    }

    pub fn load(&self, bootinfo: &mut Bootinfo) {
        // Minimal ELF parsing
        let elf = match ElfBytes::<AnyEndian>::minimal_parse(&self.data) {
            Ok(elf) => elf,
            Err(error) => panic!("Unable to parse ELF: {}", error),
        };

        // Minimal validity check
        #[cfg(target_arch = "x86_64")]
        {
            if elf.ehdr.class != elf::file::Class::ELF64 {
                panic!("Unexpected ELF class: {:?}", elf.ehdr.class);
            }

            if elf.ehdr.e_type != elf::abi::ET_DYN {
                panic!("Unexpected ELF type: {:?}", elf.ehdr.e_type);
            }

            if elf.ehdr.e_machine != elf::abi::EM_X86_64 {
                panic!("Unexpected ELF machine: {:?}", elf.ehdr.e_machine);
            }
        }

        // Search for _kernel_start and _kernel_end symbol
        let (symbol_tbl, string_tbl) = match elf.symbol_table() {
            Ok(Some((symbol_tbl, string_tbl))) => (symbol_tbl, string_tbl),
            Ok(None) => panic!("Unable to parse symbol table: no such entry"),
            Err(error) => panic!("Unable to access symbol table of ELF: {}", error),
        };
        let get_symbol_value = |identifier| match symbol_tbl.iter().find(|symbol| match string_tbl
            .get(symbol.st_name as _)
        {
            Ok(symbol_identifier) => symbol_identifier == identifier,
            Err(_) => false,
        }) {
            Some(symbol) => VirtualAddress::new(symbol.st_value as *mut c_void),
            None => panic!("Unable find `{}` symbol", identifier),
        };
        let _kernel_start = get_symbol_value("_kernel_start");
        let _kernel_end = get_symbol_value("_kernel_end");

        if _kernel_start.addr() % REGULAR_PAGE_SIZE != 0 {
            panic!("Unexpected value for `_kernel_start`: {:p}", _kernel_start);
        }
        if _kernel_end.addr() % REGULAR_PAGE_SIZE != 0 {
            panic!("Unexpected value for `_kernel_end`: {:p}", _kernel_end);
        }

        // Calculate kernel size
        let kernel_size = match _kernel_end.addr().checked_sub(_kernel_start.addr()) {
            Some(kernel_size) => kernel_size,
            None => panic!(
                "Invalid kernel ELF: `_kernel_start` @ {:p} & `_kernel_end` @ {:p}",
                _kernel_start, _kernel_end
            ),
        };

        // Allocate the memory
        let mut num_pages = kernel_size / uefi::boot::PAGE_SIZE;
        if kernel_size % uefi::boot::PAGE_SIZE != 0 {
            num_pages += 1;
        }
        let mem = match uefi::boot::allocate_pages(
            uefi::boot::AllocateType::AnyPages,
            uefi::boot::MemoryType::LOADER_DATA,
            num_pages,
        ) {
            Ok(mem) => unsafe {
                slice::from_raw_parts_mut(mem.as_ptr(), num_pages * uefi::boot::PAGE_SIZE)
            },
            Err(error) => panic!("Unable to allocate memory for kernel segments: {}", error),
        };

        // Zero the memory
        mem.fill(0);

        // Copy all loadable segments
        if let Some(program_headers) = elf.segments() {
            for program_header in program_headers {
                // Skip all non-loadable segments
                if program_header.p_type != elf::abi::PT_LOAD {
                    continue;
                }

                let vaddr = VirtualAddress::new(program_header.p_vaddr as *mut c_void);
                let memory_offset = match vaddr.addr().checked_sub(_kernel_start.addr()) {
                    Some(memory_offset) => memory_offset,
                    None => {
                        panic!("Unexpected start address of loadable segment: {:p}", vaddr);
                    }
                };
                let file_offset = program_header.p_offset as usize;
                let file_size = program_header.p_filesz as usize;

                mem[memory_offset..memory_offset + file_size]
                    .copy_from_slice(&self.data[file_offset..file_offset + file_size]);
            }
        }

        // Update Bootinfo relying on UEFI running an identity mapping.
        bootinfo.kernel_elf_start = PhysicalAddress::new(self.data.as_ptr() as _);
        bootinfo.kernel_elf_size = self.data.len();
        bootinfo.kernel_virt_phys_offset = _kernel_start.addr() - mem.as_ptr().addr();
        #[cfg(target_arch = "x86_64")]
        {
            bootinfo.arch_bootinfo.uefi_cr3 = busyos::arch::amd64::paging::CR3::read();

            // TODO(@MaxMade): Save address of GDT

            // TODO(@MaxMade): Save address of IDT
        }
    }
}
