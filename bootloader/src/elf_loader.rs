//! Helpers for loading the  Kernel *E*xecutable and *L*inkable *F*ormat file.

use alloc::vec::Vec;
use uefi::proto::media::file::File;

/// Loaded Kernel *E*xecutable and *L*inkable *F*ormat
pub struct ELF {
    data: Vec<u8>,
}

impl ELF {
    /// Try to load ELF from EFI root volume (identified by `image_handle`) at `path`.
    ///
    /// # Panics
    ///
    /// If any of the file system operations fails, this function will [`panic`].
    pub fn load(path: &str, image_handle: uefi::Handle) -> Self {
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
}
