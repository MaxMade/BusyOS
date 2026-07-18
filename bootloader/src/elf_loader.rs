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

use crate::arch::generic::elf::ELF as _;

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
        crate::arch::ELF::check_header(&elf);

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

                // Get offset relative to beginning of `mem`
                let vaddr = VirtualAddress::new(program_header.p_vaddr as *mut c_void);
                let memory_offset = match vaddr.addr().checked_sub(_kernel_start.addr()) {
                    Some(memory_offset) => memory_offset,
                    None => {
                        panic!("Unexpected start address of loadable segment: {:p}", vaddr);
                    }
                };
                let file_offset = program_header.p_offset as usize;
                let file_size = program_header.p_filesz as usize;

                // Sanity check:
                let memory_size = program_header.p_memsz as usize;
                if memory_size < file_size {
                    panic!(
                        "Detected suspicious segment: memory size ({}) <  file_size ({})",
                        memory_size, file_size
                    );
                }

                mem[memory_offset..memory_offset + file_size]
                    .copy_from_slice(&self.data[file_offset..file_offset + file_size]);
            }
        }

        // Resolve relocations
        if let Some(section_headers) = elf.section_headers() {
            for section in section_headers.iter() {
                // Try to resolve REL
                if let Ok(rels) = elf.section_data_as_rels(&section) {
                    for rel in rels {
                        todo!("Handle relocation: {:?}", rel);
                    }
                }

                // Try to resolve RELA
                if let Ok(relas) = elf.section_data_as_relas(&section) {
                    for rela in relas {
                        todo!("Handle relocation (with addend): {:?}", rela);
                    }
                }
            }
        }

        // Update Bootinfo relying on UEFI running an identity mapping
        bootinfo.kernel_elf_start = PhysicalAddress::new(self.data.as_ptr() as _);
        bootinfo.kernel_elf_size = self.data.len();
        bootinfo.kernel_virt_phys_offset = _kernel_start.addr() - mem.as_ptr().addr();

        // Update bootinfo based on ELF file
        let _text_start = get_symbol_value("_text_start");
        let _text_end = get_symbol_value("_text_end");
        let text_size = _text_end.addr().checked_sub(_text_start.addr()).unwrap();
        let _text_start = PhysicalAddress::new(
            (mem.as_ptr().addr()
                + _text_start
                    .addr()
                    .checked_sub(_kernel_start.addr())
                    .unwrap()) as *mut c_void,
        );
        bootinfo.kernel_text_start = _text_start;
        bootinfo.kernel_text_size = text_size;

        let _rodata_start = get_symbol_value("_rodata_start");
        let _rodata_end = get_symbol_value("_rodata_end");
        let rodata_size = _rodata_end
            .addr()
            .checked_sub(_rodata_start.addr())
            .unwrap();
        let _rodata_start = PhysicalAddress::new(
            (mem.as_ptr().addr()
                + _rodata_start
                    .addr()
                    .checked_sub(_kernel_start.addr())
                    .unwrap()) as *mut c_void,
        );
        bootinfo.kernel_rodata_start = _rodata_start;
        bootinfo.kernel_rodata_size = rodata_size;

        let _data_start = get_symbol_value("_data_start");
        let _data_end = get_symbol_value("_data_end");
        let data_size = _data_end.addr().checked_sub(_data_start.addr()).unwrap();
        let _data_start = PhysicalAddress::new(
            (mem.as_ptr().addr()
                + _data_start
                    .addr()
                    .checked_sub(_kernel_start.addr())
                    .unwrap()) as *mut c_void,
        );
        bootinfo.kernel_data_start = _data_start;
        bootinfo.kernel_data_size = data_size;

        let _bss_start = get_symbol_value("_bss_start");
        let _bss_end = get_symbol_value("_bss_end");
        let bss_size = _bss_end.addr().checked_sub(_bss_start.addr()).unwrap();
        let _bss_start = PhysicalAddress::new(
            (mem.as_ptr().addr() + _bss_start.addr().checked_sub(_kernel_start.addr()).unwrap())
                as *mut c_void,
        );
        bootinfo.kernel_bss_start = _bss_start;
        bootinfo.kernel_bss_size = bss_size;
    }
}
