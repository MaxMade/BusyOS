.section ".text.boot","ax"

# int _start(void)
.global _start
_start:
        nop
        jmp _start
        
