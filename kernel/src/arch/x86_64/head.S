.section ".text.boot","ax"

# int _start(unsigned long core_id)
.global _start
_start:
        # Create function prologue on UEFI stack
        push %rbp
        mov %rsp, %rbp
        sub $16, %rsp

        # Initialize core-local storage
        mov %rsi, -8(%rbp)
        call __init_gs

        # Get stack pointer
        mov -8(%rbp), %rsi
        call __boot_stack

        # Switch kernel pointer
        mov %rsp, %rdi
        mov %rax, %rsp

        # Create function prologue on kernel stack
        push %rbp
        mov %rsp, %rbp
        sub $16, %rsp

        # Save UEFI stack pointer
        mov %rdi, -8(%rbp)

        # Jump to Rust code
        call start
        
        # Restore UEFI stack pointer (no need for function epilogue of kernel stack)
        mov -8(%rbp), %rsp

        mov %rbp, %rsp
        pop %rbp
        ret
