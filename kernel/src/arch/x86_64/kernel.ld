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
 * Physical load address.  1 MiB avoids the BIOS data area and the first
 * 640 KiB of conventional memory.  Must match whatever your bootloader
 * actually loads the kernel at.
 */
KERNEL_PHYS_BASE = 0x0000000000100000;

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

        *(.text .text.*)
        *(.gnu.linkonce.t.*)

        _text_end = ALIGN(4K);
    }

    /* -- .rodata — read-only data ----------------------------------------- */

    .rodata ALIGN(4K) : AT(ADDR(.rodata) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _rodata_start = .;

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
     *
     * No storage is reserved here: the blocks themselves are allocated at
     * boot, once the real core count is known, and the template is copied
     * into each of them.  This section only describes what to copy
     * (_percpu_start, _percpu_size) and how far apart the copies must sit
     * (_percpu_stride).
     *
     * Placed last, after .bss, so that the template forms a page-aligned tail
     * of the image: once every core has its copy the pages in
     * [_percpu_start, _kernel_end) are dead and can be handed back to the
     * frame allocator without leaving a hole in the middle of the kernel.
     */

    .percpu ALIGN(4K) : AT(ADDR(.percpu) - KERNEL_VMA_START + KERNEL_PHYS_BASE)
    {
        _percpu_start = .;

        *(.percpu .percpu.*)

        . = ALIGN(64);
        _percpu_end = .;
    }

    _percpu_size = _percpu_end - _percpu_start;

    /*
     * Distance between two blocks.  Rounded up to a cache line — or to the
     * section's own alignment when that is stricter — so that variables
     * belonging to different cores never share a cache line.
     */
    _percpu_stride = ALIGN(_percpu_size, MAX(64, ALIGNOF(.percpu)));

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

