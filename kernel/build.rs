use std::env;
use std::process::Command;

fn main() {
    #[cfg(not(feature = "unittest"))]
    {
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

        if target_arch == "x86_64" {
            let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
            let arch_dir = format!("{manifest_dir}/src/arch/x86_64");

            println!("cargo:rustc-link-arg=-T{arch_dir}/kernel.ld");
            println!("cargo:rerun-if-changed={arch_dir}/kernel.ld");

            assemble(&arch_dir, "head.S");
            assemble(&arch_dir, "entry.S");
        }
    }
}

/// Assemble `<dir>/<file>` into `OUT_DIR` and hand the object straight to the linker.
///
/// The object is passed as a plain link argument instead of being archived into a static
/// library: nothing in the Rust code references `_start`, so an archive member would be
/// dropped during linking.
#[allow(dead_code)]
fn assemble(dir: &str, file: &str) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let src = format!("{dir}/{file}");
    let obj = format!("{out_dir}/{}.o", file);

    // Honour the usual cross-compilation escape hatch.
    let cc = env::var("CC").unwrap_or_else(|_| String::from("cc"));

    let status = Command::new(&cc)
        .args([
            "-c",
            "-m64",
            "-ffreestanding",
            "-mno-red-zone",
            "-g",
            "-Wall",
            "-Werror",
            &src,
            "-o",
            &obj,
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run assembler \"{cc}\": {e}"));

    assert!(status.success(), "failed to assemble {src}");

    println!("cargo:rustc-link-arg={obj}");
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=CC");
}
