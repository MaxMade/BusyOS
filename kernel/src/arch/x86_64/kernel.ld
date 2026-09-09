/*
 * kernel.ld — linker script for a higher-half x86_64 kernel
 *
 * Virtual memory layout
 * ---------------------
 *  0xFFFF800000000000  kernel load address (−2 GiB, fits in a 2 GiB PML4 slot)
 *  .text              executable code, 4 KiB aligned
 *  .rodata            read-only data
 *  .data              mutable initialised data
 *  .bss               zero-initialised data (no space in the ELF file)
 *  .percpu            core-local storage template (see below)
 *  _kernel_end        first byte after the kernel image
 */

/* -- Constants ------------------------------------------------------------ */

/*
 * Higher-half base: first 2 GiB of the 64-bit upper-half address space.
 * Placing the kernel here allows a single PML4 entry (index 256) and a
 * single PDPE (1 GiB slot) to cover the entire kernel image, which makes
 * early boot page-table setup much simpler.
 */
KERNEL_VMA_START  = 0xFFFF800000000000;

/*
 * Physical load address.  The bootloader pins the kernel image to 1 GiB
 * (`AllocateType::Address`), and the temporary hand-over mapping maps
 * physical 1 GiB onto KERNEL_VMA_START, so that a section's link address is
 * also the address it runs at once the higher half is active.
 *
 * This constant only feeds the AT() clauses, i.e. the `p_paddr` field of the
 * program headers; the loader places segments by `p_vaddr`, so it is
 * informational.  Keep it equal to the bootloader's allocation address so
 * that `p_paddr` and debuggers agree with reality.
 */
KERNEL_PHYS_BASE = 0x0000000040000000;

/* -- Entry point ---------------------------------------------------------- */

ENTRY(_start)

/* -- Output format -------------------------------------------------------- */

OUTPUT_FORMAT("elf64-x86-64")
OUTPUT_ARCH(i386:x86-64)

/* -- Section layout ------------------------------------------------------- */

SECTIONS
{
    /*
     * Start the virtual address at the higher-half base.  The AT() directive
     * pins the *physical* (load) address to KERNEL_PHYS_BASE so the ELF
     * program headers tell the bootloader where to write the image in RAM.
     */
    . = KERNEL_VMA_START;

    _kernel_start = .;

    /* -- .text — executable code ------------------------------------------ */

    .text ALIGN(4K) : AT(ADDR(.text) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _text_start = .;

        /*
         * The bootstrap stub must come first so the entry point is at the
         * very start of the image and the bootloader can jump straight to it.
         */
        KEEP(*(.text.boot))

        /*
         * Registered driver module for automatic initialization.
         */
        KEEP(*(SORT_BY_NAME(.init.modules.text.*)))

        *(.text .text.*)
        *(.gnu.linkonce.t.*)

        _text_end = ALIGN(4K);
    }

    /* -- .rodata — read-only data ----------------------------------------- */

    .rodata ALIGN(4K) : AT(ADDR(.rodata) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _rodata_start = .;

        /*
         * Registered driver module callbacks for automatic initialization.
         */
        __init_modules_callbacks_start = .;
        KEEP(*(SORT_BY_NAME(.init.modules.rodata.*)))
        __init_modules_callbacks_end = .;

        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)

        /*
         * Keep DWARF and GCC exception-handling tables.  These are read-only
         * and must not be stripped even in release builds if you want
         * backtraces or C++ exceptions to work.
         */
        *(.eh_frame_hdr)
        *(.eh_frame)
        *(.gcc_except_table .gcc_except_table.*)

        _rodata_end = ALIGN(4K);
    }

    /* -- .data — mutable initialised data --------------------------------- */

    .data ALIGN(4K) : AT(ADDR(.data) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _data_start = .;

        *(.data .data.*)
        *(.gnu.linkonce.d.*)

        /*
         * Keep GOT/PLT entries.  A kernel linked with -static should not
         * produce a PLT, but the GOT is still used for certain relocations.
         */
        *(.got)
        *(.got.plt)

        _data_end = ALIGN(4K);;
    }

    /* -- .bss — zero-initialised data ------------------------------------- */

    /*
     * .bss takes no space in the ELF file (NOBITS).  The bootstrap stub
     * must zero the range [_bss_start, _bss_end) before calling kernel_main,
     * because the bootloader may not have zeroed it.  Use the exported
     * symbols for this loop.
     */
    .bss ALIGN(4K) : AT(ADDR(.bss) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _bss_start = .;

        *(.bss .bss.*)
        *(.gnu.linkonce.b.*)

        /*
         * COMMON symbols (unresolved external variables in C translation
         * units compiled without -fno-common) end up here.
         */
        *(COMMON)

        _bss_end = ALIGN(4K);
    }

    /* -- .percpu — core-local storage template ---------------------------- */

    /*
     * Template for a single core's block of core-local variables.
     *
     * Linked at a normal virtual address like every other section, so the
     * bootloader's PT_LOAD handling needs no special case.  Variables inside
     * are never addressed directly; they are reached as
     *
     *     GS base of the current core + (&VAR - _percpu_start)
     *
     * so only the offset within the block matters, not the absolute address.
     * The first eight bytes of a block are reserved for its own base address
     * (see below) and hold no variable.
     *
     * No storage is reserved here: the blocks themselves are allocated at
     * boot, once the real core count is known, and the template is copied
     * into each of them.  This section only describes what to copy
     * (_percpu_start, _percpu_size) and how far apart the copies must sit
     * (_percpu_stride).
     *
     * Placed last, after .bss, so that the template forms a page-aligned tail
     * of the image: the copies are simply appended behind _kernel_end, which
     * needs no hole in the middle of the kernel and keeps all blocks in one
     * contiguous range.  The template is not spare storage — core 0 runs on
     * it (see `block_base`), so the pages stay live for as long as any core
     * does.
     */

    .percpu ALIGN(4K) : AT(ADDR(.percpu) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _percpu_start = .;

        /*
         * Offset 0 of a block holds that block's own base address, so that a
         * core-local access can resolve the current core's block with a plain
         * `gs:0` load instead of an `RDGSBASE`.
         *
         * The slot is reserved here, rather than declared as a core-local
         * variable, because nothing else guarantees that a given variable ends
         * up at offset 0.  The template carries a zero; the bootstrap stub
         * fills in the address of the block belonging to the core it runs on.
         */
        QUAD(0);

        *(.percpu .percpu.*)

        . = ALIGN(4K);
        _percpu_end = .;
    }

    _percpu_size = _percpu_end - _percpu_start;

    /*
     * Distance between two blocks.  Rounded up to a page — or to the
     * section's own alignment when that is stricter — so that variables
     * belonging to different cores never share a cache line.
     */
    _percpu_stride = ALIGN(_percpu_size, MAX(4K, ALIGNOF(.percpu)));

    /* -- Exported layout symbols ---------------------------------------- */

    /*
     * _kernel_end is the first virtual address after the kernel image.
     * The physical frame allocator should treat frames below
     *     _kernel_end - KERNEL_VMA_START + KERNEL_PHYS_BASE
     * as reserved.
     */
    _kernel_end = ALIGN(4K);

    /* -- Discard -------------------------------------------------------- */

    /DISCARD/ :
    {
        *(.comment)           /* Compiler version strings */
        *(.note.GNU-stack)    /* Stack-executable marking (not needed) */
        *(.note.gnu.build-id) /* Build ID note (optional — remove if you want it) */
    }
}

