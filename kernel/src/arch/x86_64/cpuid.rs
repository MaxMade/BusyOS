use bitfield_struct::bitfield;

/// Typed interface for a single CPUID leaf.
///
/// `EAX` is the primary leaf index and `ECX` is the sub-leaf index passed
/// to the `CPUID` instruction. Most leaves use `ECX = 0`.
///
/// Implementors define how the four raw output registers map to typed fields
/// via [`from_raw`](CPUID::from_raw).
pub trait CPUID<const EAX: u32, const ECX: u32>
where
    Self: Sized,
{
    /// Returns the raw EAX output register value.
    fn eax(&self) -> u32;

    /// Returns the raw EBX output register value.
    fn ebx(&self) -> u32;

    /// Returns the raw ECX output register value.
    fn ecx(&self) -> u32;

    /// Returns the raw EDX output register value.
    fn edx(&self) -> u32;

    /// Constructs a typed leaf value from the four raw CPUID output registers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the values were produced by executing
    /// `CPUID` with leaf `EAX` and sub-leaf `ECX`. Passing arbitrary values
    /// is safe in the memory-safety sense but will produce a meaningless result.
    unsafe fn from_raw(eax: u32, ebx: u32, ecx: u32, edx: u32) -> Self;

    /// Executes `CPUID` with the leaf and sub-leaf encoded in the type
    /// parameters and returns the typed result.
    ///
    /// # Safety
    ///
    /// The caller must verify that the CPU supports this leaf before calling:
    /// - For basic leaves (`EAX < 0x80000000`): leaf `0x00000000` EAX must
    ///   be >= `EAX`.
    /// - For extended leaves (`EAX >= 0x80000000`): leaf `0x80000000` EAX
    ///   must be >= `EAX`.
    ///
    /// Executing `CPUID` with an unsupported leaf returns undefined values.
    unsafe fn read() -> Self {
        let (eax, ebx, ecx, edx): (u32, u32, u32, u32);
        unsafe {
            core::arch::asm!(
                "push rbx",
                "cpuid",
                "mov {ebx_out:e}, ebx",
                "pop rbx",
                inout("eax") EAX     => eax,
                inout("ecx") ECX     => ecx,
                out("edx")   edx,
                ebx_out = out(reg)   ebx,
                options(nostack, nomem, preserves_flags),
            );
        }
        unsafe { Self::from_raw(eax, ebx, ecx, edx) }
    }
}

/// CPUID leaf `0x80000001` ECX — extended feature identifiers.
///
/// Note: EAX (extended processor signature) and EBX (reserved on Intel,
/// brand/package on AMD) are exposed as raw `u32` values since their
/// sub-fields are either reserved or not required for feature detection.
#[bitfield(u32)]
pub struct ExtendedFunctionECX {
    /// `LAHF`/`SAHF` in 64-bit mode (bit 0).
    ///
    /// When set, the `LAHF` and `SAHF` instructions are available in 64-bit
    /// mode.
    #[bits(1, access = RO)]
    pub lahf_lm: bool,

    /// Reserved (bits [4:1]).
    #[bits(4)]
    __: u8,

    /// `LZCNT` instruction (bit 5).
    ///
    /// When set, the `LZCNT` (leading zero count) instruction is available.
    #[bits(1, access = RO)]
    pub lzcnt: bool,

    /// Reserved (bits [7:6]).
    #[bits(2)]
    __: u8,

    /// `PREFETCHW` instruction (bit 8).
    ///
    /// When set, the `PREFETCHW` instruction is available for write-intent
    /// prefetching.
    #[bits(1, access = RO)]
    pub prefetchw: bool,

    /// Reserved (bits [31:9]).
    #[bits(23)]
    __: u32,
}

/// CPUID leaf `0x80000001` EDX — extended feature identifiers.
#[bitfield(u32)]
pub struct ExtendedFunctionEDX {
    /// Reserved (bits [10:0]).
    #[bits(11)]
    __: u16,

    /// `SYSCALL`/`SYSRET` in 64-bit mode (bit 11).
    ///
    /// When set, `SYSCALL` and `SYSRET` are available in 64-bit mode.
    /// Both this bit and [`EFER::sce`] must be set for `SYSCALL` to work.
    #[bits(1, access = RO)]
    pub syscall: bool,

    /// Reserved (bits [19:12]).
    #[bits(8)]
    __: u8,

    /// No-Execute page protection (`NX`/`XD`, bit 20).
    ///
    /// When set, the Execute Disable bit (bit 63) in page table entries is
    /// supported. Must be set before enabling [`EFER::nxe`].
    #[bits(1, access = RO)]
    pub nx: bool,

    /// Reserved (bits [25:21]).
    #[bits(5)]
    __: u8,

    /// 1 GiB page support (`PDPE1GB`, bit 26).
    ///
    /// When set, the PS bit in PDPE entries can be used to map 1 GiB pages.
    /// Must be verified before using [`Granularity::Gigantic`].
    #[bits(1, access = RO)]
    pub pdpe1gb: bool,

    /// `RDTSCP` instruction and `IA32_TSC_AUX` MSR (bit 27).
    #[bits(1, access = RO)]
    pub rdtscp: bool,

    /// Reserved (bit 28).
    #[bits(1)]
    __: u8,

    /// Long mode (`LM`, bit 29).
    ///
    /// When set, the CPU supports 64-bit long mode. Must be set before
    /// enabling [`EFER::lme`].
    #[bits(1, access = RO)]
    pub lm: bool,

    /// Reserved (bits [31:30]).
    #[bits(2)]
    __: u8,
}

/// Typed result of CPUID leaf `0x80000001`.
///
/// Provides extended processor and feature information
pub struct ExtendedFunction {
    eax: u32,
    ebx: u32,
    /// Extended Feature `ecx` register.
    pub ecx: ExtendedFunctionECX,
    /// Extended Feature `edx` register.
    pub edx: ExtendedFunctionEDX,
}

impl CPUID<0x80000001, 0x0> for ExtendedFunction {
    fn eax(&self) -> u32 {
        self.eax
    }

    fn ebx(&self) -> u32 {
        self.ebx
    }

    fn ecx(&self) -> u32 {
        self.ecx.into_bits()
    }

    fn edx(&self) -> u32 {
        self.edx.into_bits()
    }

    unsafe fn from_raw(eax: u32, ebx: u32, ecx: u32, edx: u32) -> Self {
        Self {
            eax,
            ebx,
            ecx: ExtendedFunctionECX::from_bits(ecx),
            edx: ExtendedFunctionEDX::from_bits(edx),
        }
    }
}

/// CPUID leaf `0x00000007` sub-leaf `0x0` EBX — structured extended feature
/// identifiers.
#[bitfield(u32)]
pub struct StructuredExtendedFeatureEBX {
    /// `FSGSBASE` instructions (bit 0).
    ///
    /// When set, the `RDFSBASE`, `RDGSBASE`, `WRFSBASE` and `WRGSBASE`
    /// instructions are supported. Executing them additionally requires
    /// `CR4.FSGSBASE` to be set, otherwise a `#UD` exception is raised.
    #[bits(1, access = RO)]
    pub fsgsbase: bool,

    /// Reserved (bits [31:1]).
    #[bits(31)]
    __: u32,
}

/// Typed result of CPUID leaf `0x00000007` sub-leaf `0x0`.
///
/// Provides structured extended feature information.
///
/// Note: EAX (maximum supported sub-leaf), ECX and EDX are exposed as raw
/// `u32` values since their sub-fields are not required for feature detection.
pub struct StructuredExtendedFeature {
    eax: u32,
    /// Structured Extended Feature `ebx` register.
    pub ebx: StructuredExtendedFeatureEBX,
    ecx: u32,
    edx: u32,
}

impl CPUID<0x00000007, 0x0> for StructuredExtendedFeature {
    fn eax(&self) -> u32 {
        self.eax
    }

    fn ebx(&self) -> u32 {
        self.ebx.into_bits()
    }

    fn ecx(&self) -> u32 {
        self.ecx
    }

    fn edx(&self) -> u32 {
        self.edx
    }

    unsafe fn from_raw(eax: u32, ebx: u32, ecx: u32, edx: u32) -> Self {
        Self {
            eax,
            ebx: StructuredExtendedFeatureEBX::from_bits(ebx),
            ecx,
            edx,
        }
    }
}
