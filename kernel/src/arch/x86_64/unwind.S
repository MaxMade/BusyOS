# Frame-pointer stack walk, see `CPU::unwind` in `cpu.rs`.
#
# The kernel is built with `-C force-frame-pointers=y`, so every function
# starts with `push %rbp; mov %rsp, %rbp`. Each frame therefore looks like
#
#   +----------------+
#   | return address | <- %rbp + 8, into the function that made the call
#   +----------------+
#   | caller's %rbp  | <- %rbp, links to the caller's frame
#   +----------------+
#
# and following the saved %rbp values walks from the innermost frame
# outwards.

.text
.global __unwind
.type __unwind, @function
# size_t __unwind(uint64_t *array, size_t len)
#   %rdi: array, %rsi: len
#   %rax: number of entries written
#
# Walks the frame-pointer chain starting at the caller's frame and stores one
# return address per frame. Stops when the array is full, when a return address
# lies outside .text, or when the next frame does not lie above the current one.
# The stack grows down, so a caller's frame always lies above its callee's.
# That last check also catches null, a loop in a corrupted chain and
# CALL_STACK_END_MARKER, which lies below every kernel address.
#
# Uses only caller-saved registers besides %rbp, which it restores.
__unwind:
        pushq %rbp
        movq %rsp, %rbp

        # Return addresses must point into .text
        leaq _text_start(%rip), %r10
        leaq _text_end(%rip), %r11

        # Start at the caller's frame, skipping __unwind's own
        movq (%rbp), %rax
        xorq %rcx, %rcx

1:
        # Stop when the array is full
        cmpq %rsi, %rcx
        jae 2f

        # Return address of this frame, which must lie in .text
        movq 8(%rax), %r8
        cmpq %r10, %r8
        jb 2f
        cmpq %r11, %r8
        jae 2f

        movq %r8, (%rdi, %rcx, 8)
        incq %rcx

        # The caller's frame must lie above this one
        movq (%rax), %r8
        cmpq %rax, %r8
        jbe 2f

        movq %r8, %rax
        jmp 1b

2:
        movq %rcx, %rax
        popq %rbp
        ret
.size __unwind, . - __unwind
