//! A round-robin scheduler: runnable threads take turns in the order they
//! became runnable.

use driver_macro::module;

use crate::{
    arch::{CPU, generic::cpu::CPU as _},
    driver::module::Module,
    kernel::{
        arc::Arc,
        compiler::debug::debug_build,
        linked_list::{LinkedList, Node},
        locking::{
            CanAcquire, DriverLevelID, Exclusive, LockId, PreviousToken, SchedulerLevelID, Token,
            WriteGuard,
        },
        ticketlock::{SchedulerTicketlock, Ticketlock, TicketlockSchedulerID},
    },
    mem::{cache::Cache, heap::Heap},
    sched::{
        scheduler::{CURRENT, IDLE_THREAD, Scheduler, SchedulerDriver},
        thread::{Thread, ThreadState},
    },
    user::errno::{Errno, ToErrno},
};

/// A [`Scheduler`] over a single first-in, first-out run queue.
///
/// [`schedule`](Scheduler::schedule) runs the thread at the front and puts the
/// one it replaces at the back, so every runnable thread gets its turn before
/// any gets a second one. The idle thread never enters the queue: a core runs
/// it only when its running thread stops being runnable — so far only the
/// boot context, which ends with [`start`](Scheduler::start) — and the queue
/// is empty, and the timer moves the core on as soon as the queue is not.
///
/// The queue's nodes come from a [`Cache`], so that requeueing a thread on
/// every switch reuses the node just freed instead of going to the heap.
///
/// Brought up through [`module!`], which registers it as the kernel's
/// scheduler, see [`SchedulerDriver::register`].
pub struct RoundRobin(SchedulerTicketlock<RunQueue>);

/// The runnable threads, in the order they get their turn.
type RunQueue = LinkedList<Arc<Thread>, Cache<Node<Arc<Thread>>, Heap>>;

impl RoundRobin {
    /// Creates a scheduler with an empty run queue.
    ///
    /// Allocates nothing.
    pub const fn new() -> Self {
        Self(SchedulerTicketlock::new(
            Ticketlock::new(),
            LinkedList::new_in(Cache::new()),
        ))
    }

    /// Switches this core away from the running thread, which stops running
    /// as `state` says, with the run queue locked as `queue`.
    ///
    /// The next thread is the one at the front of the queue. With the queue
    /// empty, a thread that stays runnable keeps the core, and one that does
    /// not hands it to the core's idle thread.
    ///
    /// The state is set under the run queue's lock, which the switch keeps
    /// until the next thread has finished it, see [`finish_switch`]. Nobody
    /// sees the thread blocked before its stack pointer is saved. Returns
    /// once the thread runs again, with the lock released.
    ///
    /// # Panics
    ///
    /// If the idle thread is to stop being runnable, or the core has no idle
    /// thread to fall back to.
    ///
    /// [`finish_switch`]: RoundRobin::finish_switch
    fn switch_away<From>(
        mut queue: WriteGuard<'_, RunQueue, Ticketlock<TicketlockSchedulerID>>,
        state: ThreadState,
        token: Token<TicketlockSchedulerID, From, Exclusive>,
    ) -> From
    where
        From: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let current_is_idle = CURRENT.with(|current| {
            current.set_state(state);
            current.is_idle()
        });

        if current_is_idle && state != ThreadState::Runnable {
            panic!("Idle thread must always be runnable");
        }

        let (next, token) = match queue.pop_front(token) {
            (Some(next), token) => (next, token),
            (None, token) if state == ThreadState::Runnable => {
                // Nothing else to run: the running thread keeps the core.
                return queue.release(token);
            }
            (None, token) => match IDLE_THREAD.with_mut(token, |idle| idle.take()) {
                (Some(idle), token) => (idle, token),
                (None, _) => panic!("Core has no idle thread to fall back to"),
            },
        };

        let (prev, token) =
            CURRENT.with_mut(token, |current| core::mem::replace(current, next.clone()));

        // Perform actual context switch. The queue stays locked across it:
        // whichever thread runs next unlocks it in `finish_switch`, and this
        // one in turn unlocks the hold of the thread that switches back to
        // it. `prev` is then that thread, not the one switched away from
        // above.
        let prev = CPU::context_switch(prev, next);

        Self::finish_switch(queue, prev, token)
    }

    /// Finishes a switch in the thread it switched to: puts `prev` where its
    /// state says and unlocks the run queue, which the switching thread locked
    /// and could not unlock any more.
    ///
    /// An idle thread is parked in [`IDLE_THREAD`], a runnable one goes to the
    /// back of the queue, and the handle on a blocked or exited one is given
    /// up: a blocked thread is held by whoever is to wake it, and an exited
    /// one is freed with its last handle.
    ///
    /// Every thread runs this first thing once it runs again: a resumed one
    /// at the end of its own [`schedule`](Scheduler::schedule), a new one from
    /// [`finish_first_switch`](Scheduler::finish_first_switch). It has to
    /// happen here, after the switch: only now is the stack pointer of `prev`
    /// saved, so that another core that picks it up resumes it where it
    /// stopped.
    ///
    /// # Panics
    ///
    /// If `prev` cannot be requeued for lack of memory, or is an idle thread
    /// while the core already has one.
    fn finish_switch<From>(
        mut queue: WriteGuard<'_, RunQueue, Ticketlock<TicketlockSchedulerID>>,
        prev: Arc<Thread>,
        token: Token<TicketlockSchedulerID, From, Exclusive>,
    ) -> From
    where
        From: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let token = match (prev.is_idle(), prev.state()) {
            (true, _) => match IDLE_THREAD.set(token, Some(prev)) {
                (None, token) => token,
                (Some(_), _) => panic!("Core has two idle threads"),
            },
            (false, ThreadState::Runnable) => match queue.try_push_back(prev, token) {
                Ok(token) => token,
                Err((error, _)) => panic!("Unable to requeue thread: {}", error),
            },
            (false, ThreadState::Blocked | ThreadState::Exited) => Thread::release(prev, token),
        };

        queue.release(token)
    }
}

impl Module for RoundRobin {
    /// Creates the scheduler and registers it as the kernel's.
    ///
    /// # Panics
    ///
    /// If the scheduler cannot be allocated, or another one has registered
    /// already. A kernel without a scheduler, or with two, cannot go on.
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        let driver = match Arc::try_new(RoundRobin::new(), token) {
            Ok((driver, t)) => {
                token = t;
                driver
            }
            Err((error, _)) => {
                panic!(
                    "Unable to create scheduler instance of round_robin: {}",
                    error
                )
            }
        };

        match SchedulerDriver::RoundRobin(driver).register(token) {
            Ok(t) => token = t,
            Err((_, error, _)) => {
                panic!("Unable to register round_robin as scheduler: {}", error)
            }
        };

        Ok(token)
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }
}

impl Scheduler for RoundRobin {
    type LockedToken<From: PreviousToken> = Token<TicketlockSchedulerID, From, Exclusive>;

    fn enqueue<Token>(&self, thread: Arc<Thread>, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut queue, token) = self.0.acquire(token);

        if debug_build() {
            if queue.iter().find(|t| t.id() == thread.id()).is_some() {
                panic!("Found duplicate entry in scheduler queue: {}", thread.id());
            }
        }

        // A thread that blocked is out of its switch by now: the switch kept
        // the lock, and only released it once it was finished.
        thread.set_state(ThreadState::Runnable);

        let token = match queue.try_push_back(thread, token) {
            Ok(token) => token,
            Err((error, token)) => {
                let token = queue.release(token);
                return Err((error.to_errno(), token));
            }
        };

        Ok(queue.release(token))
    }

    fn try_dequeue<Token>(
        &self,
        thread: Arc<Thread>,
        token: Token,
    ) -> Result<(Arc<Thread>, Token), (Arc<Thread>, Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut queue, token) = self.0.acquire(token);

        // `retain` keeps what the predicate accepts, so only `thread` itself
        // may be rejected. The queue's handle on it is never the last one —
        // the caller holds another — so dropping it needs no token.
        let mut found = false;
        let token = queue.retain(
            |t| {
                if t.id() == thread.id() {
                    found = true;
                    return false;
                }
                return true;
            },
            token,
        );

        let token = queue.release(token);

        match found {
            true => Ok((thread, token)),
            false => Err((thread, Errno::EINVAL, token)),
        }
    }

    fn schedule<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let (queue, token) = self.0.acquire(token);

        Ok(Self::switch_away(queue, ThreadState::Runnable, token))
    }

    fn try_block<Token, Block>(&self, token: Token, block: Block) -> Token
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
        Block: FnOnce(&Arc<Thread>, Self::LockedToken<Token>) -> (bool, Self::LockedToken<Token>),
    {
        let (queue, token) = self.0.acquire(token);

        // `CURRENT` keeps the running thread alive, so this handle is never
        // the last one and needs no token.
        let current = CURRENT.with(|current| current.clone());
        let (blocks, token) = block(&current, token);
        drop(current);

        match blocks {
            true => Self::switch_away(queue, ThreadState::Blocked, token),
            false => queue.release(token),
        }
    }

    fn start<Token>(&self, token: Token) -> !
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let (queue, token) = self.0.acquire(token);

        let _ = Self::switch_away(queue, ThreadState::Exited, token);

        panic!("Boot context was switched to again");
    }

    unsafe fn finish_first_switch<Token>(&self, prev: Arc<Thread>, token: Token) -> Token
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        // SAFETY: per the caller's promise, the run queue is still locked by
        // the switch that started this thread, which will never unlock it.
        let (queue, token) = unsafe { self.0.adopt(token) };

        Self::finish_switch(queue, prev, token)
    }
}

module! {
    name: "round_robin",
    priority: 50,
    driver: crate::sched::round_robin::RoundRobin,
}
