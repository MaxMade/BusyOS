.section ".text.boot","ax"

# int _start(void)
.global _start
_start:
        # TODO(@MaxMade): Remove as soon as BOOTINFO is acutally used.
        # Otherwise, the symbol will be remove from binary.
        lea BOOTINFO(%rip), %rax
        jmp _start

        xor %rax, %rax
        ret
        
