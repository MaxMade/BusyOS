# Start-up code for the other cores, see `smp.rs`.
#
# A core that receives a start-up IPI begins in 16-bit real mode, at the page
# below 1 MiB the IPI names. The code between `ap_trampoline_start` and
# `ap_trampoline_end` is copied there by the kernel, together with its data,
# and takes the core from real mode through protected mode into long mode.
#
# It never relies on where it was linked, only on where it was copied to:
# every address it uses is an offset from `ap_trampoline_start`, and the few
# absolute addresses protected and long mode need are worked out at run time
# from %cs and patched into the copy.
#
# Long mode is entered with temporary page tables that the kernel builds in
# the pages behind the copy (see `AP_TRAMPOLINE_PAGES`). They identity-map the
# first 2 MiB, where the copy runs, and carry the kernel's own upper-half
# entries, so that the core can jump straight into kernel code. Only there, in
# `__ap_start`, does it switch to the kernel's page tables, which have no
# identity mapping for the copy.

.section .text

.code16
.global ap_trampoline_start
ap_trampoline_start:
        cli
        cld

        # %cs is the page the IPI named, as a real-mode segment, and %ip is 0
        mov %cs, %ax
        mov %ax, %ds

        # Linear address of the copy, for protected and long mode
        xor %ebx, %ebx
        mov %cs, %bx
        shl $4, %ebx

        # Patch the absolute addresses the copy needs
        mov %ebx, %eax
        add $(ap_gdt - ap_trampoline_start), %eax
        mov %eax, (ap_gdtr - ap_trampoline_start + 2)

        mov %ebx, %eax
        add $(ap_protected - ap_trampoline_start), %eax
        mov %eax, (ap_protected_ptr - ap_trampoline_start)

        mov %ebx, %eax
        add $(ap_long - ap_trampoline_start), %eax
        mov %eax, (ap_long_ptr - ap_trampoline_start)

        # Enter protected mode
        lgdtl (ap_gdtr - ap_trampoline_start)
        mov %cr0, %eax
        or $1, %eax
        mov %eax, %cr0
        ljmpl *(ap_protected_ptr - ap_trampoline_start)

.code32
ap_protected:
        mov $0x10, %ax
        mov %ax, %ds
        mov %ax, %es
        mov %ax, %fs
        mov %ax, %gs
        mov %ax, %ss

        # Physical address extension, which long mode requires
        mov %cr4, %eax
        or $(1 << 5), %eax
        mov %eax, %cr4

        # Temporary page tables, in the page right behind the copy. Below
        # 1 MiB, so that the 32-bit load reaches them.
        lea 0x1000(%ebx), %eax
        mov %eax, %cr3

        # Long mode, and no-execute, which the kernel's entries use
        mov $0xC0000080, %ecx
        rdmsr
        or $((1 << 8) | (1 << 11)), %eax
        wrmsr

        # Paging and write protection, which activates long mode
        mov %cr0, %eax
        or $((1 << 31) | (1 << 16)), %eax
        mov %eax, %cr0

        ljmpl *(ap_long_ptr - ap_trampoline_start)(%ebx)

.code64
ap_long:
        # The upper halves of the registers are undefined after the switch
        mov %ebx, %ebx

        # Temporary stack at the end of the area, shared by all cores. Only
        # one core starts at a time, and it reports in only once it has left
        # this stack, see `__ap_start`.
        lea 0x5000(%rbx), %rsp

        # Hand the parameters over in registers: the kernel's page tables do
        # not map this copy, so `__ap_start` cannot read them from here once
        # it has switched to those tables.
        mov (ap_trampoline_cpu_id - ap_trampoline_start)(%rbx), %rdi
        mov (ap_trampoline_cr3 - ap_trampoline_start)(%rbx), %rsi
        mov (ap_trampoline_cr4 - ap_trampoline_start)(%rbx), %rdx
        mov (ap_trampoline_efer - ap_trampoline_start)(%rbx), %rcx
        lea (ap_trampoline_started - ap_trampoline_start)(%rbx), %r8
        mov (ap_trampoline_entry - ap_trampoline_start)(%rbx), %rax

        jmp *%rax

# Temporary GDT: just enough for the two mode switches. The core loads the
# kernel's own GDT later, like the boot core does.
.align 8
ap_gdt:
        .quad 0
        .quad 0x00CF9A000000FFFF        # 0x08: 32-bit code
        .quad 0x00CF92000000FFFF        # 0x10: data
        .quad 0x00AF9A000000FFFF        # 0x18: 64-bit code
ap_gdt_end:

ap_gdtr:
        .word ap_gdt_end - ap_gdt - 1
        .long 0                         # base, patched at run time

# Far pointers for the two jumps: offset, then selector. The offsets are
# patched at run time.
ap_protected_ptr:
        .long 0
        .word 0x08
ap_long_ptr:
        .long 0
        .word 0x18

# Parameters, written by the kernel into the copy before each start-up IPI,
# see `Trampoline` in `smp.rs`.
.align 8
.global ap_trampoline_cpu_id
ap_trampoline_cpu_id:
        .quad 0                         # id of the core being started
.global ap_trampoline_cr3
ap_trampoline_cr3:
        .quad 0                         # the kernel's page tables
.global ap_trampoline_cr4
ap_trampoline_cr4:
        .quad 0                         # the boot core's CR4
.global ap_trampoline_efer
ap_trampoline_efer:
        .quad 0                         # the boot core's EFER
.global ap_trampoline_entry
ap_trampoline_entry:
        .quad 0                         # virtual address of `__ap_start`
.global ap_trampoline_started
ap_trampoline_started:
        .long 0                         # set once the core has left this area

.global ap_trampoline_end
ap_trampoline_end:

# Entry of the other cores into the kernel's half of the address space,
# reached from `ap_long` above with the temporary page tables still active.
#
#   %rdi: id of this core
#   %rsi: the kernel's CR3
#   %rdx: the boot core's CR4
#   %rcx: the boot core's EFER
#   %r8:  address of `ap_trampoline_started` in the copy
#   %rsp: the temporary stack in the trampoline area
.code64
.global __ap_start
.type __ap_start, @function
__ap_start:
        # Keep the parameters in callee-saved registers across the calls
        mov %rdi, %r12
        mov %rsi, %r13
        mov %rdx, %r14
        mov %rcx, %r15
        mov %r8, %rbx

        # Get top of this core's boot stack, as in head.S:
        #   _percpu_start + __boot_stack_offset(cpu_id) + __BOOT_STACK_SIZE
        mov %r12, %rdi
        call __boot_stack_offset
        lea _percpu_start(%rip), %rcx
        add %rcx, %rax
        add __BOOT_STACK_SIZE(%rip), %rax
        mov %rax, %rsp

        # Off the shared temporary stack: the boot core may start the next
        # core now. Still reachable, since the temporary page tables map it.
        movl $1, (%rbx)

        # The kernel's page tables, and the boot core's features
        mov %r13, %cr3
        mov %r14, %cr4
        mov $0xC0000080, %ecx
        mov %r15d, %eax
        mov %r15, %rdx
        shr $32, %rdx
        wrmsr

        # Outermost frame of this core's frame-pointer chain, as in head.S,
        # keeping the stack 16-byte aligned for the calls below
        mov CALL_STACK_END_MARKER(%rip), %rbp
        push %rbp
        mov %rsp, %rbp
        sub $8, %rsp

        # Initialize core-local storage
        mov %r12, %rdi
        call __init_gs

        # Jump to Rust code, which never returns
        mov %r12, %rdi
        call start_ap
        ud2
.size __ap_start, . - __ap_start
