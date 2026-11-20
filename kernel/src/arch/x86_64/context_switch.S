# Kernel-stack switch between two threads, see `CPU::context_switch` in
# `cpu.rs`.
#
# A thread that is not running is suspended inside this function: its kernel
# stack ends in the callee-saved registers pushed below, and the thread's
# saved stack pointer points at the last of them. Switching to it loads that
# stack pointer, pops the registers and returns into whatever called
# `__context_switch` on that thread — so `ret` lands in the *next* thread's
# call, not in this one's.
#
#   +----------------+
#   | return address | into the thread's call of `__context_switch`
#   +----------------+
#   | %r15           |
#   | %r14           |
#   | %r13           |
#   | %r12           |
#   | %rbp           |
#   | %rbx           | <- saved stack pointer
#   +----------------+
#
# A new thread's stack is laid out the same way by `CPU::context_prepare`,
# with the return address pointing at `__thread_trampoline` below.

.text
.global __context_switch
.type __context_switch, @function
# struct switched __context_switch(struct thread *prev, struct thread *next,
#                                  size_t stack_offset)
#   %rdi: thread to suspend
#   %rsi: thread to resume
#   %rdx: offset of the saved stack pointer within a thread
#   %rax: `prev` of the call that resumes, i.e. the thread switched away from
#   %rdx: `next` of the call that resumes, i.e. the thread now running
#
# Only the registers the System V ABI makes callee-saved are kept. The Rust
# caller does not expect any of the caller-saved ones to survive.
__context_switch:
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbp
	pushq %rbx

	# Suspend `prev` and continue on the stack of `next`.
	movq %rsp, (%rdi, %rdx)
	movq (%rsi, %rdx), %rsp

	# Both pointers come back as a 16-byte struct in %rax:%rdx. %rdi and
	# %rsi still hold the arguments of *this* call, which is the one the
	# resumed thread sees return.
	movq %rdi, %rax
	movq %rsi, %rdx

	popq %rbx
	popq %rbp
	popq %r12
	popq %r13
	popq %r14
	popq %r15

	retq
.size __context_switch, . - __context_switch

.global __thread_trampoline
.type __thread_trampoline, @function
# void __thread_trampoline(void)
#   %rax: `prev` of the switch that started the thread
#   %rdx: `next` of that switch, i.e. this thread
#   %r12: the thread's entry function
#   %r13: the argument for it
#   %rbp: CALL_STACK_END_MARKER, saved by `__thread_entry` as its caller's
#         frame, so that a backtrace stops there
#
# Where a new thread's first switch returns to, see `CPU::context_prepare` in
# `cpu.rs`. It is reached through `ret`, not `call`, so the switch's return
# values are still in %rax:%rdx, and what `context_prepare` stored for the
# thread is in the callee-saved registers the switch has just popped. All four
# go to `__thread_entry` (`thread_entry` in `sched/thread.rs`), which finishes
# the switch and never returns.
#
# `context_prepare` leaves %rsp 16-byte aligned here, as a call requires.
__thread_trampoline:
	movq %rax, %rdi
	movq %rdx, %rsi
	movq %r12, %rdx
	movq %r13, %rcx
	call __thread_entry
	ud2
.size __thread_trampoline, . - __thread_trampoline
