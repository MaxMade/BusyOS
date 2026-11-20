//! The scheduler interface and the per-core state it works with.

use crate::{
    arch::{CPU, generic::cpu::CPU as _},
    core_local,
    kernel::{
        arc::Arc,
        handler::Epilogue,
        locking::{
            CanAcquire, EpilogueLevelID, LockId, PreviousToken, RootToken, SchedulerLevelID,
            ThreadLevelID,
        },
        ticketlock::{RWTicketlock, SchedulerRWTicketlock},
    },
    sched::{
        round_robin::RoundRobin,
        thread::{Thread, ThreadType},
    },
    user::errno::Errno,
};

core_local! {
    /// The thread running on this core.
    pub static CURRENT: Arc<Thread>;
}

core_local! {
    /// This core's idle thread while it does not run, `None` while it does.
    pub static IDLE_THREAD: Option<Arc<Thread>>;
}

/// Hands the calling core over from its boot context to the threads, for
/// good.
///
/// The end of every core's start-up, which runs on the epilogue level and
/// hands it over as `epilogue`. The core's idle thread gets a stack of its own
/// and starts in `idle_thread`. The boot context becomes a [`Thread`] on the
/// boot stack it already runs on, [`CURRENT`] until it switches away, as any
/// thread does, through [`Scheduler::start`], which ends it. The next thread —
/// the first runnable one, or the idle thread — takes over the epilogue level.
///
/// # Panics
///
/// If the threads cannot be allocated, if no scheduler has registered, or if
/// the core is already scheduling.
pub fn schedule<Token>(mut epilogue: Epilogue<Token>) -> !
where
    Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken,
{
    epilogue.execute(|token| -> ((), Token) {
        let (mut idle, token) = match Thread::try_new(ThreadType::Idle, token) {
            Ok(result) => result,
            Err((error, _)) => panic!("Unable to create idle thread: {}", error),
        };
        CPU::context_prepare(&mut idle, idle_thread, 0);

        let (idle, token) = match Arc::try_new(idle, token) {
            Ok(result) => result,
            Err((error, _)) => panic!("Unable to create idle thread: {}", error),
        };

        // SAFETY: the boot stack is this core's alone, and the boot context
        // never runs again once it has switched away.
        let boot = unsafe { Thread::new(ThreadType::Kernel, CPU::boot_stack_top()) };

        let (boot, token) = match Arc::try_new(boot, token) {
            Ok(result) => result,
            Err((error, _)) => panic!("Unable to create boot thread: {}", error),
        };

        let token = match CURRENT.set(token, boot) {
            (None, token) => token,
            (Some(_), _) => panic!("Core is already scheduling"),
        };

        let token = match IDLE_THREAD.set(token, Some(idle)) {
            (None, token) => token,
            (Some(_), _) => panic!("Core is already scheduling"),
        };

        let (scheduler, token) = match SchedulerDriver::get(token) {
            (Some(scheduler), token) => (scheduler, token),
            (None, _) => panic!("Unable to schedule: no scheduler registered"),
        };

        scheduler.start(token)
    });

    unreachable!("Boot context was switched to again");
}

/// What a core's idle thread runs.
///
/// Leaves the epilogue level the thread's first switch handed over, which
/// runs the epilogues deferred until then, and waits for interrupts for good.
/// From then on, the timer moves the core on: while the idle thread waits,
/// nothing is on the epilogue level, so the timer's epilogue runs right away
/// and schedules.
fn idle_thread(_: usize, token: RootToken) -> ! {
    // SAFETY: this thread is new, and its first switch handed it the
    // epilogue level, see `ThreadEntry`.
    let epilogue = unsafe { Epilogue::adopt(token) };
    let _token = epilogue.leave();

    loop {
        // SAFETY: the idle thread holds no lock and is off the epilogue
        // level.
        unsafe { CPU::idle() };
    }
}

/// Decides which runnable thread a core runs next.
///
/// A scheduler owns the threads that are runnable but not running; the
/// running thread is in [`CURRENT`] and the idle thread in [`IDLE_THREAD`]
/// instead.
pub trait Scheduler {
    /// The token of the scheduler's own lock, on top of the caller's `From`,
    /// as [`try_block`](Scheduler::try_block) hands it to its closure.
    ///
    /// Reaches every level below `Scheduler`, so that the closure can lock the
    /// queue it puts the thread into and allocate for it.
    type LockedToken<From: PreviousToken>: CanAcquire<<ThreadLevelID as LockId>::Level>
        + PreviousToken;

    /// Makes `thread` runnable, e.g. to wake it up once it has blocked in
    /// [`try_block`](Scheduler::try_block).
    ///
    /// A thread that is still on its way into a block is only queued once its
    /// switch away is finished, see [`try_block`](Scheduler::try_block).
    ///
    /// # Errors
    ///
    /// An [`Errno`] if the thread cannot be queued, e.g. for lack of memory.
    /// `thread` is dropped in that case.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn enqueue<Token>(&self, thread: Arc<Thread>, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken;

    /// Takes `thread` out of the run queue, so that it does not get another
    /// turn until it is queued again. Its [`ThreadState`] stays as it is.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if `thread` is not in the run queue, e.g. because it
    /// is running. `thread` is handed back in both arms.
    ///
    /// [`ThreadState`]: crate::sched::thread::ThreadState
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn try_dequeue<Token>(
        &self,
        thread: Arc<Thread>,
        token: Token,
    ) -> Result<(Arc<Thread>, Token), (Arc<Thread>, Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken;

    /// Switches this core from the running thread to the next runnable one.
    /// The running thread stays runnable, and keeps running if there is no
    /// other.
    ///
    /// Returns once the calling thread runs again.
    ///
    /// Must be called on the epilogue level, see [`Epilogue`]. The switch
    /// hands the level over to the next thread, which leaves it once it is
    /// done with it: a resumed thread on the way out of its own call, an idle
    /// thread when it starts, a new kernel thread never.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    fn schedule<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken;

    /// Blocks the running thread if `block` decides so, deciding and blocking
    /// as one step.
    ///
    /// `block` runs under the scheduler's lock, with the running thread and
    /// the lock's token, and returns whether the thread is to block, along
    /// with the token. If not, this returns right away. If so, the thread
    /// stops being runnable and the core switches to the next runnable one,
    /// or to its idle thread, as in [`schedule`](Scheduler::schedule); this
    /// returns once the thread has been made runnable again, see
    /// [`enqueue`](Scheduler::enqueue), and runs again.
    ///
    /// The lock is kept from `block` until the switch away is finished, so
    /// whoever is to wake the thread — `block` typically puts it into a
    /// waiting queue — can neither miss the decision nor queue the thread
    /// before it has stopped running.
    ///
    /// Must be called on the epilogue level, as
    /// [`schedule`](Scheduler::schedule), and never by an idle thread.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    fn try_block<Token, Block>(&self, token: Token, block: Block) -> Token
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
        Block: FnOnce(&Arc<Thread>, Self::LockedToken<Token>) -> (bool, Self::LockedToken<Token>);

    /// Hands the calling core over from its boot context to the threads, for
    /// good. Never returns.
    ///
    /// The boot context, which has to be [`CURRENT`], stops as an exited
    /// thread, and the core switches to the next runnable thread, or to its
    /// idle thread, which has to be in [`IDLE_THREAD`], if there is none.
    /// Must be called on the epilogue level, as
    /// [`schedule`](Scheduler::schedule), see [`schedule`](schedule()).
    fn start<Token>(&self, token: Token) -> !
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken;

    /// Finishes the switch into a thread that has never run before.
    ///
    /// [`schedule`](Scheduler::schedule) keeps its own state locked across
    /// the switch, and the thread that comes out of it finishes the switch:
    /// it puts `prev`, the thread it replaced, back where it belongs and
    /// unlocks. A resumed thread does that at the end of its own `schedule`
    /// call. A new thread has none to return to, so its entry code calls this
    /// instead, see [`ThreadEntry`](crate::sched::thread::ThreadEntry).
    ///
    /// # Safety
    ///
    /// Only from the entry of a new thread, once, before anything else
    /// touches the scheduler, with the `prev` its first switch handed over.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    unsafe fn finish_first_switch<Token>(&self, prev: Arc<Thread>, token: Token) -> Token
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken;
}

/// A handle on a scheduler, as `SCHEDULER` holds it.
///
/// One variant per scheduler rather than an `Arc<dyn Scheduler>`: every
/// method of [`Scheduler`] is generic over its token, so the trait has no
/// vtable to hold. The handle is a [`Scheduler`] itself and forwards to the
/// one it names.
///
/// Cloning a handle is the [`Arc`] clone of the scheduler it names.
#[derive(Clone)]
pub enum SchedulerDriver {
    /// The round-robin scheduler, see [`RoundRobin`].
    RoundRobin(Arc<RoundRobin>),
}

/// The kernel's scheduler, once one has registered itself, see
/// [`SchedulerDriver::register`].
///
/// A lock at the `Scheduler` level, so that it can be read on the way into
/// [`Scheduler::schedule`], which runs at that level.
static SCHEDULER: SchedulerRWTicketlock<Option<SchedulerDriver>> =
    SchedulerRWTicketlock::new(RWTicketlock::new(), None);

impl SchedulerDriver {
    /// Makes `self` the kernel's scheduler.
    ///
    /// Meant to be called once, from the [`Module::init`] of the scheduler
    /// the kernel is built with.
    ///
    /// # Errors
    ///
    /// [`Errno::EEXISTS`] if a scheduler has registered already. That one
    /// stays, and `self` is handed back.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned in both arms.
    ///
    /// [`Module::init`]: crate::driver::module::Module::init
    pub fn register<Token>(self, token: Token) -> Result<Token, (Self, Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut scheduler, token) = SCHEDULER.acquire(token);

        if scheduler.is_some() {
            let token = scheduler.release(token);
            return Err((self, Errno::EEXISTS, token));
        }

        *scheduler = Some(self);

        Ok(scheduler.release(token))
    }

    /// Returns a handle on the kernel's scheduler, or [`None`] if none has
    /// registered yet.
    ///
    /// # Token
    ///
    /// The `token` is consumed and returned.
    pub fn get<Token>(token: Token) -> (Option<SchedulerDriver>, Token)
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        let (scheduler, token) = SCHEDULER.acquire_shared(token);
        let driver = scheduler.clone();

        (driver, scheduler.release(token))
    }
}

// Every method below matches on the variant and forwards to the concrete
// scheduler's method, handing `token` along.
impl Scheduler for SchedulerDriver {
    /// The round-robin scheduler's, the only one so far. A scheduler added
    /// as a variant has to hand out the same token type.
    type LockedToken<From: PreviousToken> = <RoundRobin as Scheduler>::LockedToken<From>;

    /// Forwards to the concrete scheduler, see [`Scheduler::enqueue`].
    fn enqueue<Token>(&self, thread: Arc<Thread>, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            SchedulerDriver::RoundRobin(round_robin) => round_robin.enqueue(thread, token),
        }
    }

    /// Forwards to the concrete scheduler, see [`Scheduler::try_dequeue`].
    fn try_dequeue<Token>(
        &self,
        thread: Arc<Thread>,
        token: Token,
    ) -> Result<(Arc<Thread>, Token), (Arc<Thread>, Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            SchedulerDriver::RoundRobin(round_robin) => round_robin.try_dequeue(thread, token),
        }
    }

    /// Forwards to the concrete scheduler, see [`Scheduler::schedule`].
    fn schedule<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            SchedulerDriver::RoundRobin(round_robin) => round_robin.schedule(token),
        }
    }

    /// Forwards to the concrete scheduler, see [`Scheduler::try_block`].
    fn try_block<Token, Block>(&self, token: Token, block: Block) -> Token
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
        Block: FnOnce(&Arc<Thread>, Self::LockedToken<Token>) -> (bool, Self::LockedToken<Token>),
    {
        match self {
            SchedulerDriver::RoundRobin(round_robin) => round_robin.try_block(token, block),
        }
    }

    /// Forwards to the concrete scheduler, see [`Scheduler::start`].
    fn start<Token>(&self, token: Token) -> !
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        match self {
            SchedulerDriver::RoundRobin(round_robin) => round_robin.start(token),
        }
    }

    /// Forwards to the concrete scheduler, see
    /// [`Scheduler::finish_first_switch`].
    unsafe fn finish_first_switch<Token>(&self, prev: Arc<Thread>, token: Token) -> Token
    where
        Token: CanAcquire<<SchedulerLevelID as LockId>::Level> + PreviousToken,
    {
        // SAFETY: forwarded as is, under the caller's promise.
        match self {
            SchedulerDriver::RoundRobin(round_robin) => unsafe {
                round_robin.finish_first_switch(prev, token)
            },
        }
    }
}
