fn main() {
    #[cfg(not(feature = "unittest"))] 
    {
        #[cfg(target_arch = "x86_64")]
        {
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            println!("cargo:rustc-link-arg=-T{manifest_dir}/src/arch/x86_64/kernel.ld");
            println!("cargo:rerun-if-changed={manifest_dir}/src/arch/x86_64/kernel.ld");
        }
    }
}
