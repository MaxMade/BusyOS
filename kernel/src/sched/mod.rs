//! Threads and the scheduler that decides which of them runs.
//!
//! # Overview
//!
//! - [`thread`] — a [`Thread`](thread::Thread): its identity, its state and
//!   the kernel stack it is suspended on while it does not run, which it owns
//!   and frees with its last handle.
//! - [`scheduler`] — the [`Scheduler`](scheduler::Scheduler) interface, the
//!   per-core state every scheduler works with — the running thread and the
//!   core's idle thread — and the hand-over of a core from its boot context
//!   to the threads, [`schedule`](scheduler::schedule).
//! - [`round_robin`] — [`RoundRobin`](round_robin::RoundRobin), a first-in,
//!   first-out run queue.
//!
//! Threads are shared as [`Arc<Thread>`](crate::kernel::arc::Arc): the run
//! queue, the per-core state and a thread suspended in a switch each hold a
//! handle, and the stack switch itself is
//! [`CPU::context_switch`](crate::arch::generic::cpu::CPU::context_switch).
//!
//! # Locking
//!
//! Scheduling runs at the `Scheduler` lock level, the boundary between
//! blocking and spinning locks: the run queue is a spinning lock, and a thread
//! that holds anything below `Scheduler` cannot be switched away from.
//!
//! Every switch happens on the epilogue level, see
//! [`Epilogue`](crate::kernel::handler::Epilogue), and hands it over to the
//! thread switched to. Kernel code always runs on it, so the timer, whose
//! epilogue schedules, only preempts code off the level: user space and an
//! idle core. A tick that arrives on the level has its epilogue deferred
//! until the level is left, which is what keeps a thread holding a spinning
//! lock from being switched away from.

pub mod round_robin;
pub mod scheduler;
pub mod thread;
