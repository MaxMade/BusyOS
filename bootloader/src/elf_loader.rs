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
use uefi::proto::{media::file::File, pi::mp::MpServices};

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

    pub fn load(&self) -> &'static mut Bootinfo {
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
        let mut kernel_size = match _kernel_end.addr().checked_sub(_kernel_start.addr()) {
            Some(kernel_size) => kernel_size,
            None => panic!(
                "Invalid kernel ELF: `_kernel_start` @ {:p} & `_kernel_end` @ {:p}",
                _kernel_start, _kernel_end
            ),
        };

        // Determine number of cores
        let handle = match uefi::boot::get_handle_for_protocol::<MpServices>() {
            Ok(handle) => handle,
            Err(error) => panic!("Unable to determine number of core: {}", error),
        };
        let mp = match uefi::boot::open_protocol_exclusive::<MpServices>(handle) {
            Ok(mp) => mp,
            Err(error) => panic!("Unable to determine number of core: {}", error),
        };

        let num_cpus = match mp.get_number_of_processors()  {
            Ok(count) => count,
            Err(error) => panic!("Unable to determine number of core: {}", error),
        };

        // Get size of a single entry for core local storage
        let _percpu_start = get_symbol_value("_percpu_start");
        let _percpu_size = get_symbol_value("_percpu_size");
        let _percpu_stride = get_symbol_value("_percpu_stride");
        if _percpu_start.addr() % REGULAR_PAGE_SIZE != 0 {
            panic!("Unexpected value for `_percpu_start`: {:p}", _percpu_start);
        }
        if _percpu_stride.addr() % REGULAR_PAGE_SIZE != 0 {
            panic!("Unexpected value for `_percpu_stride`: {:p}", _percpu_stride);
        }
        if _percpu_size.addr() > _percpu_stride.addr() {
            panic!(
                "Core-local block ({:p}) larger than its stride ({:p})",
                _percpu_size, _percpu_stride
            );
        }
        kernel_size += _percpu_stride.addr() * num_cpus.total.saturating_sub(1);

        // Allocate the memory
        let mut num_pages = kernel_size / uefi::boot::PAGE_SIZE;
        if kernel_size % uefi::boot::PAGE_SIZE != 0 {
            num_pages += 1;
        }
        let mem = match uefi::boot::allocate_pages(
            uefi::boot::AllocateType::Address(1024 * 1024 * 1024),
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

        // Copy bitwise the remaining core-local storage entries
        //
        // `.percpu` is the last section of the image, so the extra blocks are
        // simply appended: block `i` starts one stride further along than
        // block `i - 1`, and block 0 is the template that the segment loop
        // above has just written. The space for them was added to
        // `kernel_size` before the allocation.
        let percpu_offset = match _percpu_start.addr().checked_sub(_kernel_start.addr()) {
            Some(percpu_offset) => percpu_offset,
            None => panic!(
                "`_percpu_start` @ {:p} lies before `_kernel_start` @ {:p}",
                _percpu_start, _kernel_start
            ),
        };
        let percpu_size = _percpu_size.addr();
        let percpu_stride = _percpu_stride.addr();

        for core in 1..num_cpus.total {
            let dst = percpu_offset + core * percpu_stride;

            if dst + percpu_size > mem.len() {
                panic!(
                    "Core-local block {} ends at {:#x}, past the {:#x} byte allocation",
                    core,
                    dst + percpu_size,
                    mem.len()
                );
            }

            mem.copy_within(percpu_offset..percpu_offset + percpu_size, dst);
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
        let kernel_virt_phys_offset = _kernel_start.addr() - mem.as_ptr().addr();
        let bootinfo = unsafe {
            let ptr: VirtualAddress<Bootinfo> = get_symbol_value("BOOTINFO")
                .byte_sub(kernel_virt_phys_offset)
                .cast();
            ptr.as_ptr().as_mut_unchecked()
        };

        // Save the address of `_start` (kernel entry)
        bootinfo.kernel_start_symbol = VirtualAddress::new(elf.ehdr.e_entry as _);

        // Update Bootinfo relying on UEFI running an identity mapping
        bootinfo.kernel_elf_start = PhysicalAddress::new(self.data.as_ptr() as _);
        bootinfo.kernel_elf_size = self.data.len();
        bootinfo.kernel_virt_phys_offset = kernel_virt_phys_offset;

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

        bootinfo
    }
}
