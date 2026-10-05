use crate::{
    arch::{
        CPU, InterruptStackFrame, InterruptVector,
        generic::cpu::{CPU as _, InterruptFlag, InterruptVector as GenericInterruptVector},
    },
    core_local,
    driver::{
        irq::{
            IRQCapable as _, IRQCapableDriver, InterruptController, InterruptControllers,
            InterruptVectorTable,
        },
        module::Module as _,
    },
    kernel::{
        bitset::{self, BitSet},
        locking::{CanAcquire, EpilogueLevelID, LockId, PreviousToken, RootToken, SyscallLevel},
        printk::LogLevel,
        spsc::SPSC,
    },
    printkln,
};

pub struct Epilogue<Token>
where
    Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken,
{
    token: Option<Token>,
    vector: Option<InterruptVector>,
    prev_on_epilogue_level: bool,
    interrupt_flag: InterruptFlag,
}

core_local! {
    static ON_EPILOGUE_LEVEL: bool = false;
    static PENDING_INTERRUPTS: BitSet<InterruptVector, { bitset::words(<InterruptVector as GenericInterruptVector>::MAX_NUM) }> = BitSet::empty();
    static EPILOGUE_QUEUE: SPSC<(IRQCapableDriver, InterruptVector), { <InterruptVector as GenericInterruptVector>::MAX_NUM }> = SPSC::new();
}

impl<Token: CanAcquire<<EpilogueLevelID as LockId>::Level> + PreviousToken> Epilogue<Token> {
    pub fn enter_from_vector(token: Token, vector: InterruptVector) -> Self {
        let interrupt_flag = CPU::interrupt_flag();

        // Reading and setting the flag has to be one step with respect to this
        // core: an interrupt in between would also see `false`, and both would
        // then act as the outermost epilogue level. `with_mut` masks interrupts
        // for the closure, and the flag is core-local, so no other core
        // touches this copy.
        let (prev_on_epilogue_level, token) = ON_EPILOGUE_LEVEL.with_mut(token, |epilogue_level| {
            let prev = *epilogue_level;
            *epilogue_level = true;
            prev
        });

        Self {
            token: Some(token),
            interrupt_flag,
            prev_on_epilogue_level,
            vector: Some(vector),
        }
    }

    pub fn execute_or_defer(&mut self, driver: &IRQCapableDriver) {
        if self.prev_on_epilogue_level {
            let vector = match self.vector.take() {
                Some(vector) => vector,
                None => panic!("Detected to re-use epilogue level for already processed interrupt"),
            };
            let token = self.token.take().unwrap();
            let (already_pending, token) = PENDING_INTERRUPTS
                .with_mut(token, |pending_interrupts| {
                    !pending_interrupts.insert(vector)
                });
            let token = match already_pending {
                true => {
                    // Nothing to do here!
                    token
                }
                false => {
                    let (result, token) = EPILOGUE_QUEUE.with_mut(token, |epilogue_queue| unsafe {
                        epilogue_queue.try_push((driver.clone(), vector))
                    });
                    assert!(result.is_ok());

                    token
                }
            };
            self.token = Some(token);
        } else {
            let token = self.token.take().unwrap();

            // Enable interrupts
            unsafe { CPU::raw_enable_interrupts() };

            let token = match driver.epilogue(token) {
                Ok(token) => token,
                Err((error, token)) => {
                    printkln!(
                        LogLevel::Error,
                        "Failed {}::epilogue(): {}",
                        driver.name(),
                        error
                    );
                    token
                }
            };

            // Disable interrupts
            unsafe { CPU::raw_disable_interrupts() };

            self.token = Some(token);
        }
    }

    pub fn leave(mut self) -> Token {
        let mut token = self.token.take().unwrap();

        if !self.prev_on_epilogue_level {
            loop {
                // Disable interrupts
                unsafe { CPU::raw_disable_interrupts() };

                let (epilogue, t) = EPILOGUE_QUEUE
                    .with_mut(token, |epilogue_queue| unsafe { epilogue_queue.pop() });
                token = t;

                let (driver, vector) = match epilogue {
                    Some(epilogue) => epilogue,
                    None => break,
                };

                let (interrupt_pending, t) = PENDING_INTERRUPTS
                    .with_mut(token, |pending_interrupts| {
                        pending_interrupts.remove(vector)
                    });
                assert!(interrupt_pending);
                token = t;

                // Enable interrupts
                unsafe { CPU::raw_enable_interrupts() };

                token = match driver.epilogue(token) {
                    Ok(token) => token,
                    Err((error, token)) => {
                        printkln!(
                            LogLevel::Error,
                            "Failed {}::epilogue(): {}",
                            driver.name(),
                            error
                        );
                        token
                    }
                };
            }

            let (prev_on_epilogue_level, t) = ON_EPILOGUE_LEVEL.with_mut(token, |epilogue_level| {
                let prev = *epilogue_level;
                *epilogue_level = false;
                prev
            });
            token = t;
            assert!(prev_on_epilogue_level);
        }

        if self.interrupt_flag == InterruptFlag::Enabled {
            // Re-enable interrupts
            unsafe { CPU::raw_enable_interrupts() };
        }

        token
    }
}

pub fn handler(vector: InterruptVector, _stack_frame: *mut InterruptStackFrame) {
    assert!(CPU::interrupt_flag() == InterruptFlag::Disabled);

    // A non-maskable interrupt is only ever sent to stop this core, by a
    // core that is panicking, see `ipi::Mode::Panic`. Nothing is
    // acknowledged and nothing else runs: the panicking core's output is the
    // last thing that is to happen.
    if vector.is_non_maskable() {
        // Tell the panicking core that this one is quiet, see
        // `ipi::emergency_stop_others`.
        crate::driver::ipi::acknowledge_stop();

        // SAFETY: the panicking core does not wait for anything this core
        // holds.
        unsafe { CPU::halt() }
    }

    // Enter fake system level
    let root = unsafe { RootToken::forge() };
    let (level, mut token) = SyscallLevel::enter(root);

    // Get driver for interrupt/exception
    let driver = match InterruptVectorTable::driver(vector, token) {
        (Some(driver), t) => {
            token = t;
            driver
        }
        _ => panic!(
            "Unable to handler interrupt {:#x}: missing driver!",
            vector.into_raw()
        ),
    };

    // Execute prologue
    let epilogue_required = match driver.prologue(token) {
        Ok((epilogue_required, t)) => {
            token = t;
            epilogue_required
        }
        Err((errno, t)) => {
            printkln!(
                LogLevel::Error,
                "Failed {}::prologue() (Interrupt vector: {}): {}",
                driver.name(),
                vector.into_raw(),
                errno
            );
            token = t;
            false
        }
    };

    if vector.is_interrupt() {
        token = match InterruptControllers::get(token) {
            (Some(interrupt_controller), token) => {
                match interrupt_controller.acknowledge(vector, token) {
                    Ok(token) => token,
                    Err((errno, _)) => {
                        panic!(
                            "Failed {}::acknowledge() (Interrupt vector: {}): {}",
                            interrupt_controller.name(),
                            vector.into_raw(),
                            errno
                        );
                    }
                }
            }
            (None, token) => token,
        };
    }

    let mut epilogue = Epilogue::enter_from_vector(token, vector);
    if epilogue_required {
        epilogue.execute_or_defer(&driver);
    }
    let token = epilogue.leave();

    level.leave(token);
}
