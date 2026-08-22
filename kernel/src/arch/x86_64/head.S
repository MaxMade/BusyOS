.section ".text.boot","ax"

# int _start(unsigned long cpu_id)
# Caution: Call from UEFI binary: Microsoft x64, UEFI X64!
# cpu_id: %rcx
# return: %rax
#
# Everything below `start` is compiled for SysV, so this stub has to bridge
# the two calling conventions. Of the registers UEFI expects back unchanged
# — %rbx, %rbp, %rsp, %rdi, %rsi, %r12-%r15 and %xmm6-%xmm15 — SysV keeps
# %rbx, %rbp, %rsp and %r12-%r15 for us, so only %rdi and %rsi are saved
# here and restored before returning. %xmm6-%xmm15 are deliberately left
# alone: the kernel never touches an SSE register, so there is nothing to
# preserve — should that ever change, they have to be saved here too, since
# SysV treats them as scratch. The control bits of %mxcsr and the x87
# control word are non-volatile under both conventions and need no handling.
# UEFI's own volatile registers (%rax, %rcx, %rdx, %r8-%r11) are ours to
# clobber.
#
# Frame on the UEFI stack:
#
#   -8(%rbp)               cpu_id
#   -16(%rbp)              %rdi
#   -24(%rbp)              %rsi
#   -32(%rbp)              padding, keeps the frame 16 byte aligned
.global _start
_start:
        # Create function prologue on UEFI stack
        #
        # 32 bytes, a multiple of the 16 byte stack alignment, so that the
        # calls below see the alignment both conventions ask for.
        push %rbp
        mov %rsp, %rbp
        sub $32, %rsp

        # Stash cpu_id, which arrives in a register SysV code may clobber
        mov %rcx, -8(%rbp)

        # Save the registers SysV code will not preserve for UEFI
        mov %rdi, -16(%rbp)
        mov %rsi, -24(%rbp)

        # Initialize core-local storage
        mov -8(%rbp), %rdi
        call __init_gs

        # Get top of this core's boot stack:
        #   _percpu_start + __boot_stack_offset(cpu_id) + __BOOT_STACK_SIZE
        mov -8(%rbp), %rdi
        call __boot_stack_offset
        lea _percpu_start(%rip), %rcx
        add %rcx, %rax
        add __BOOT_STACK_SIZE(%rip), %rax

        # Switch kernel pointer
        mov %rsp, %rdi
        mov %rbp, %rsi
        mov %rax, %rsp

        # Create function prologue on kernel stack
        push %rbp
        mov %rsp, %rbp
        sub $24, %rsp

        # Save UEFI stack frame
        mov %rdi, -8(%rbp) # %rsp
        mov %rsi, -16(%rbp) # %rbp

        # Jump to Rust code
        call start

        # Switch back UEFI stack
        mov -8(%rbp), %rsp
        mov -16(%rbp), %rbp

        # Restore the registers UEFI expects back unchanged
        mov -16(%rbp), %rdi
        mov -24(%rbp), %rsi

        mov %rbp, %rsp
        pop %rbp
        ret
