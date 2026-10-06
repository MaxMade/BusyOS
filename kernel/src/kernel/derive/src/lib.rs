//! Derives a compile-time enforced locking hierarchy from an enum.
//!
//! Enforced statically:
//! - Locks are acquired strictly top-down (variant order of the enum).
//! - Holding a level grants access to *every* level below it: a bound of
//!   `CanAcquire<Epilogue>` implies `CanAcquire<MemoryManagement>`, so
//!   generic code only ever names the highest level it needs.
//! - At most one lock per level per thread (tokens are consumed on acquire).
//! - Release happens in exact reverse order (previous token embedded in type).
//! - Tokens can only be released through the lock identity that minted them.
//! - Shared holds are counted at the type level: returning to a higher
//!   level requires the nested-share count to be zero, while descending
//!   to lower levels is permitted with outstanding shares.
//! - Tokens are `!Send`/`!Sync` and cannot be forged in safe code.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput};

/// Derives the complete locking hierarchy infrastructure from an enum.
///
/// Variants define lock levels in **descending order**: the first variant
/// is the highest level; each level is acquirable from any token of a
/// strictly higher level, and from `RootToken`, which holds nothing and
/// may therefore enter the hierarchy at any level.
///
/// The capability is downward-closed and, crucially, *stays* that way
/// through a generic bound: a `T: CanAcquire<Epilogue>` parameter also
/// satisfies `CanAcquire<MemoryManagement>` and `CanAcquire<Prologue>`,
/// so a caller only names the highest level it needs.
///
/// Derive this exactly **once** per module — it emits module-level items
/// (`Token`, `RootToken`, `Lock`, ...) that would collide if generated
/// twice in the same scope.
///
/// ```rust
/// #[derive(Locking)]
/// pub enum Level {
///     Syscall,
///     Epilogue,
///     MemoryManagement,
///     Prologue,
/// }
/// ```
#[proc_macro_derive(Locking)]
pub fn derive_locking(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input.ident, "#[derive(Locking)] only supports enums")
            .to_compile_error()
            .into();
    };

    let variants: Vec<_> = data.variants.iter().map(|v| v.ident.clone()).collect();
    if variants.is_empty() {
        return syn::Error::new_spanned(&input.ident, "enum needs at least one lock level")
            .to_compile_error()
            .into();
    }

    // ------------------------------------------------------------------
    // Downward chain: `Lower1`, `Lower2`, ... name the levels below a
    // level, nearest first. The chain saturates at the lowest level, so
    // entries past the bottom simply repeat it.
    //
    // This exists purely to make the implication in `CanAcquire` work.
    // Rust elaborates *supertraits* only: from `T: CanAcquire<Epilogue>`
    // in a where-clause it will never reach a different instantiation of
    // the same trait, no matter which impls exist, because impls cannot
    // be run backwards against an opaque type parameter. Naming the lower
    // levels as associated types lets `CanAcquire<L>` list all of them in
    // its own supertrait bounds, which the compiler *does* elaborate.
    // ------------------------------------------------------------------
    let depth = variants.len() - 1;
    let lower_names: Vec<_> = (1..=depth).map(|k| format_ident!("Lower{}", k)).collect();

    // ------------------------------------------------------------------
    // Level marker structs + LockLevel/LowerLevels impls
    // ------------------------------------------------------------------
    let level_markers = variants.iter().enumerate().map(|(i, v)| {
        let doc = format!("Marker type for the `{v}` lock level.");
        let chain = lower_names.iter().enumerate().map(|(k, name)| {
            let below = &variants[usize::min(i + k + 1, variants.len() - 1)];
            quote! { type #name = #below; }
        });
        quote! {
            #[doc = #doc]
            pub struct #v;
            impl super::LockLevel for #v {}
            impl super::LowerLevels for #v {
                #(#chain)*
            }
        }
    });

    // ------------------------------------------------------------------
    // Ordering:
    //   - RootToken: Reaches<level::Vj>     for every level
    //   - level::Vi: Above<level::Vj>       for all i < j
    //   - One blanket impl: Token<I,P,K>: Reaches<L>
    //                       where I::Level: Above<L>
    //   Using Above on the concrete level marker types avoids the
    //   coherence conflict that arises from multiple blanket impls on
    //   Token<I,P,K> with different where-clauses.
    // ------------------------------------------------------------------
    let orderings = variants.iter().enumerate().flat_map(|(j, lower)| {
        let root = quote! {
            impl Reaches<level::#lower> for RootToken {}
        };
        let variants = variants.clone();
        let lower = lower.clone();
        let highers = (0..j).map(move |i| {
            let higher = &variants[i];
            let lower = &lower;
            quote! {
                impl Above<level::#lower> for level::#higher {}
            }
        });
        core::iter::once(root).chain(highers)
    });

    // ------------------------------------------------------------------
    // `CanAcquire<L>` bundles the raw `Reaches` capability for `L` with
    // the one for every level below it, as supertraits. Both the trait
    // header and the blanket impl need the identical list.
    // ------------------------------------------------------------------
    let lower_bounds: Vec<_> = lower_names
        .iter()
        .map(|name| quote! { Reaches<<Level as LowerLevels>::#name> })
        .collect();
    let lower_decls = lower_names.iter().enumerate().map(|(k, name)| {
        let doc = format!(
            "The level {} step(s) below `Self`, saturating at the lowest level.",
            k + 1
        );
        quote! {
            #[doc = #doc]
            type #name;
        }
    });

    quote! {
        // ==============================================================
        // Type-level naturals (nested shared-hold counter)
        // ==============================================================

        /// Type-level zero.
        pub struct Z;

        /// Type-level successor: `S<Z>` = 1, `S<S<Z>>` = 2, ...
        pub struct S<N>(::core::marker::PhantomData<N>);

        // ==============================================================
        // Ownership-kind markers
        // ==============================================================

        /// Marker for exclusive lock ownership.
        pub struct Exclusive;

        /// Marker for shared lock ownership, carrying the number of
        /// outstanding nested shared holds at the type level.
        ///
        /// Returning to a higher level requires `Shared<Z>` — all nested
        /// holds must be released first.
        pub struct Shared<Count = Z>(::core::marker::PhantomData<Count>);

        // ==============================================================
        // Core traits
        // ==============================================================

        /// Marker trait for lock level types.
        pub trait LockLevel: LowerLevels {}

        /// Names the levels below a level, nearest first, as associated
        /// types (`Lower1`, `Lower2`, ...). The chain saturates at the
        /// lowest level: entries past the bottom repeat it.
        ///
        /// Implementation detail of the [`CanAcquire`] implication chain,
        /// generated for every level; not meant to be named directly.
        pub trait LowerLevels {
            #(#lower_decls)*
        }

        /// `Higher: Above<Lower>` — a lock at `Higher` level may be held
        /// while acquiring a lock at `Lower` level.
        ///
        /// Implemented on the concrete level marker types (not on `Token`),
        /// which avoids coherence conflicts when multiple higher levels can
        /// all reach the same lower level.
        #[diagnostic::on_unimplemented(
            message = "lock level `{Self}` is not above `{Lower}`",
            label = "a lock at `{Lower}` may only be acquired while holding \
                     a strictly higher level"
        )]
        pub trait Above<Lower: LockLevel>: LockLevel {}

        /// A unique lock identity, tied to a hierarchy level.
        ///
        /// Declare with `#[lock_id(LevelName)] pub struct MyLockId;`.
        pub trait LockId {
            /// The hierarchy level this lock identity belongs to.
            type Level: LockLevel;
        }

        /// `T: Reaches<L>` — the raw, single-level fact that `T`'s own
        /// level is strictly [`Above`] `L`.
        ///
        /// For [`Token`]: a single blanket rule keyed on [`Above`], so
        /// there is exactly one impl regardless of how many levels are
        /// above `L`. For [`RootToken`]: one impl per level — the root
        /// holds nothing and may therefore enter anywhere.
        ///
        /// Use [`CanAcquire`] in bounds; this trait carries no implication
        /// to lower levels on its own.
        #[diagnostic::on_unimplemented(
            message = "`{Self}` may not acquire a lock at level `{Level}`",
            label = "this token's own level is not strictly above `{Level}`",
            note = "a token may only descend the hierarchy: it can acquire \
                    levels below the one it currently holds"
        )]
        pub trait Reaches<Level> {}

        /// Blanket rule: a token reaches any level that its own level
        /// is [`Above`].
        impl<Level, I, P, K> Reaches<Level> for Token<I, P, K>
        where
            Level: LockLevel,
            I: LockId,
            <I as LockId>::Level: Above<Level>,
        {}

        /// `T: CanAcquire<L>` — a token of type `T` may acquire a lock at
        /// level `L`, **and at every level below `L`**.
        ///
        /// The lower levels are supertraits, so the implication survives
        /// into generic code: a function bounded by `CanAcquire<Epilogue>`
        /// can hand its token to one requiring `CanAcquire<MemoryManagement>`
        /// without restating the bound. Elaboration is what makes this
        /// work, hence the [`LowerLevels`] chain — a plain blanket impl
        /// would only ever apply to concrete token types.
        #[diagnostic::on_unimplemented(
            message = "`{Self}` may not acquire a lock at level `{Level}`",
            label = "requires a token holding a level strictly above `{Level}`",
            note = "locks are acquired strictly top-down; bound the token by \
                    the highest level it must acquire, which implies all \
                    lower ones"
        )]
        pub trait CanAcquire<Level: LockLevel>: Reaches<Level> #(+ #lower_bounds)* {}

        /// Blanket rule: the bundle holds exactly when every part does.
        impl<T, Level> CanAcquire<Level> for T
        where
            Level: LockLevel,
            T: Reaches<Level> #(+ #lower_bounds)*,
        {}

        /// Marker types for each lock level, highest first.
        pub mod level {
            #(#level_markers)*
        }

        #(#orderings)*

        // ==============================================================
        // Tokens
        // ==============================================================

        /// A token branded with the identity of the lock that produced it.
        ///
        /// Not `Copy`/`Clone` and `!Send`/`!Sync`: it cannot be duplicated
        /// or moved to another thread, and only the release path of the
        /// owning lock can consume it to recover the previous-level token.
        pub struct Token<Id: LockId, Previous, Kind = Exclusive> {
            _id:       ::core::marker::PhantomData<Id>,
            _previous: ::core::marker::PhantomData<Previous>,
            _kind:     ::core::marker::PhantomData<Kind>,
            _not_send: ::core::marker::PhantomData<*mut ()>,
        }

        impl<Id: LockId, Previous, Kind> Token<Id, Previous, Kind> {
            /// # Safety
            ///
            /// Must only be called immediately after the lock identified
            /// by `Id` has actually been acquired.
            unsafe fn forge() -> Self {
                Self {
                    _id:       ::core::marker::PhantomData,
                    _previous: ::core::marker::PhantomData,
                    _kind:     ::core::marker::PhantomData,
                    _not_send: ::core::marker::PhantomData,
                }
            }
        }

        /// One nested shared hold, branded with the lock identity.
        ///
        /// Exactly as many `SharedToken`s exist as the counter in the
        /// primary token's `Shared<N>` kind says: they are forged only on
        /// increment and consumed only on decrement, and cannot be cloned.
        pub struct SharedToken<Id: LockId> {
            _id:       ::core::marker::PhantomData<Id>,
            _not_send: ::core::marker::PhantomData<*mut ()>,
        }

        impl<Id: LockId> SharedToken<Id> {
            /// # Safety
            ///
            /// Must only be called while the lock identified by `Id` is
            /// held in shared mode, immediately after incrementing the
            /// reader count.
            unsafe fn forge() -> Self {
                Self {
                    _id:       ::core::marker::PhantomData,
                    _not_send: ::core::marker::PhantomData,
                }
            }
        }

        /// The root token — entry point into the locking hierarchy.
        ///
        /// Exactly one per thread, created at thread entry.
        pub struct RootToken {
            _private:  (),
            _not_send: ::core::marker::PhantomData<*mut ()>,
        }

        impl RootToken {
            /// # Safety
            ///
            /// Must be called exactly once per thread, at thread entry,
            /// before any hierarchy lock is touched. A second `RootToken`
            /// on the same thread re-enables same-level hold-and-wait
            /// (and thereby ABBA deadlocks).
            pub unsafe fn forge() -> Self {
                Self {
                    _private:  (),
                    _not_send: ::core::marker::PhantomData,
                }
            }
        }

        mod __locking_sealed {
            pub trait Sealed {}
            impl Sealed for super::RootToken {}
            impl<I: super::LockId, P, K> Sealed for super::Token<I, P, K> {}
        }

        /// Reconstructs the previous token on release. Sealed.
        pub trait PreviousToken: __locking_sealed::Sealed {
            /// # Safety
            ///
            /// Only callable from release paths, where the previous lock
            /// is provably still held (it was embedded in the type of the
            /// token just consumed).
            unsafe fn forge_previous() -> Self;
        }

        impl PreviousToken for RootToken {
            unsafe fn forge_previous() -> Self {
                Self {
                    _private:  (),
                    _not_send: ::core::marker::PhantomData,
                }
            }
        }

        impl<I: LockId, P, K> PreviousToken for Token<I, P, K> {
            unsafe fn forge_previous() -> Self {
                unsafe { Token::forge() }
            }
        }

        // ==============================================================
        // Raw lock trait + blanket token-checked interface
        // ==============================================================

        /// Implemented by lock types to plug into the hierarchy.
        ///
        /// Provide only the raw lock mechanics; all token logic comes from
        /// the blanket [`HierarchicalLockExt`] implementation.
        pub trait HierarchicalLock {
            /// Unique identity of this lock; determines its level.
            type Id: LockId;

            /// # Safety
            /// Only via [`HierarchicalLockExt::acquire`].
            unsafe fn raw_lock(&self);
            /// Attempt to lock exclusively without blocking. Returns whether
            /// the lock was taken; on `false` nothing was acquired.
            ///
            /// # Safety
            /// Only via [`HierarchicalLockExt::try_acquire`].
            unsafe fn raw_try_lock(&self) -> bool;
            /// # Safety
            /// Only via [`HierarchicalLockExt::release`].
            unsafe fn raw_unlock(&self);
            /// # Safety
            /// Only via [`HierarchicalLockExt::acquire_shared`].
            unsafe fn raw_lock_shared(&self);
            /// Attempt to lock in shared mode without blocking. Returns
            /// whether the lock was taken; on `false` nothing was acquired.
            ///
            /// # Safety
            /// Only via [`HierarchicalLockExt::try_acquire_shared`].
            unsafe fn raw_try_lock_shared(&self) -> bool;
            /// # Safety
            /// Only via [`HierarchicalLockExt::release_shared`].
            unsafe fn raw_unlock_shared(&self);
            /// # Safety
            /// Only via [`HierarchicalLockExt::acquire_shared_nested`].
            unsafe fn raw_lock_shared_nested(&self);
            /// # Safety
            /// Only via [`HierarchicalLockExt::release_shared_nested`].
            unsafe fn raw_unlock_shared_nested(&self);
        }

        /// Token-checked locking interface, blanket-implemented for every
        /// [`HierarchicalLock`]. Do not implement manually.
        pub trait HierarchicalLockExt: HierarchicalLock + Sized {
            /// Acquire exclusively, consuming the incoming token.
            fn acquire<From>(&self, token: From) -> Token<Self::Id, From, Exclusive>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>;

            /// Acquire exclusively if the lock is free, consuming the
            /// incoming token.
            ///
            /// On success the token is replaced by the target-level one, as
            /// in [`acquire`](HierarchicalLockExt::acquire). On failure the
            /// incoming token is handed back untouched, so the caller keeps
            /// exactly the capability it came with.
            fn try_acquire<From>(
                &self,
                token: From,
            ) -> Result<Token<Self::Id, From, Exclusive>, From>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>;

            /// Release, recovering the previous token.
            ///
            /// Only accepts tokens produced by a lock with the same `Id`.
            fn release<P: PreviousToken>(
                &self,
                token: Token<Self::Id, P, Exclusive>,
            ) -> P;

            /// Acquire in shared mode, consuming the incoming token.
            ///
            /// The nested-hold count starts at zero.
            fn acquire_shared<From>(
                &self,
                token: From,
            ) -> Token<Self::Id, From, Shared<Z>>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>;

            /// Acquire in shared mode if no writer holds the lock,
            /// consuming the incoming token.
            ///
            /// On success the nested-hold count starts at zero, as in
            /// [`acquire_shared`](HierarchicalLockExt::acquire_shared). On
            /// failure the incoming token is handed back untouched.
            fn try_acquire_shared<From>(
                &self,
                token: From,
            ) -> Result<Token<Self::Id, From, Shared<Z>>, From>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>;

            /// Release the shared lock, recovering the previous token.
            ///
            /// Requires the nested-hold count to be **zero** (`Shared<Z>`)
            /// — releasing with outstanding [`SharedToken`]s is a type
            /// error.
            fn release_shared<P: PreviousToken>(
                &self,
                token: Token<Self::Id, P, Shared<Z>>,
            ) -> P;

            /// Take an additional shared hold.
            ///
            /// Consumes the primary token and returns it with the count
            /// incremented, alongside the new [`SharedToken`].
            fn acquire_shared_nested<P, N>(
                &self,
                token: Token<Self::Id, P, Shared<N>>,
            ) -> (Token<Self::Id, P, Shared<S<N>>>, SharedToken<Self::Id>);

            /// Return a nested shared hold.
            ///
            /// Consumes one [`SharedToken`] together with the primary
            /// token and returns the primary with the count decremented.
            fn release_shared_nested<P, N>(
                &self,
                token:  Token<Self::Id, P, Shared<S<N>>>,
                shared: SharedToken<Self::Id>,
            ) -> Token<Self::Id, P, Shared<N>>;
        }

        impl<L: HierarchicalLock> HierarchicalLockExt for L {
            fn acquire<From>(&self, _token: From) -> Token<Self::Id, From, Exclusive>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>,
            {
                // SAFETY: hierarchy proven by the consumed token; the forged
                // token corresponds to the lock acquired above.
                unsafe {
                    self.raw_lock();
                    Token::forge()
                }
            }

            fn try_acquire<From>(
                &self,
                token: From,
            ) -> Result<Token<Self::Id, From, Exclusive>, From>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>,
            {
                // SAFETY: hierarchy proven by the token passed in; a token is
                // forged only when the lock was actually taken.
                if unsafe { self.raw_try_lock() } {
                    Ok(unsafe { Token::forge() })
                } else {
                    // Nothing was acquired, so the caller stays where it was:
                    // hand its own token straight back.
                    Err(token)
                }
            }

            fn release<P: PreviousToken>(
                &self,
                _token: Token<Self::Id, P, Exclusive>,
            ) -> P {
                // SAFETY: consumed token proves this lock is held; `P` was
                // embedded at acquire time and is still held by this thread.
                unsafe {
                    self.raw_unlock();
                    P::forge_previous()
                }
            }

            fn acquire_shared<From>(
                &self,
                _token: From,
            ) -> Token<Self::Id, From, Shared<Z>>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>,
            {
                // SAFETY: as in `acquire`, shared mode, count zero.
                unsafe {
                    self.raw_lock_shared();
                    Token::forge()
                }
            }

            fn try_acquire_shared<From>(
                &self,
                token: From,
            ) -> Result<Token<Self::Id, From, Shared<Z>>, From>
            where
                From: CanAcquire<<Self::Id as LockId>::Level>,
            {
                // SAFETY: as in `try_acquire`, shared mode, count zero.
                if unsafe { self.raw_try_lock_shared() } {
                    Ok(unsafe { Token::forge() })
                } else {
                    Err(token)
                }
            }

            fn release_shared<P: PreviousToken>(
                &self,
                _token: Token<Self::Id, P, Shared<Z>>,
            ) -> P {
                // SAFETY: count is Z so no nested holds are outstanding;
                // the primary shared hold ends here.
                unsafe {
                    self.raw_unlock_shared();
                    P::forge_previous()
                }
            }

            fn acquire_shared_nested<P, N>(
                &self,
                _token: Token<Self::Id, P, Shared<N>>,
            ) -> (Token<Self::Id, P, Shared<S<N>>>, SharedToken<Self::Id>) {
                // SAFETY: the consumed token proves shared mode is active;
                // the reader count and the type-level count are incremented
                // together, keeping them in sync.
                unsafe {
                    self.raw_lock_shared_nested();
                    (Token::forge(), SharedToken::forge())
                }
            }

            fn release_shared_nested<P, N>(
                &self,
                _token:  Token<Self::Id, P, Shared<S<N>>>,
                _shared: SharedToken<Self::Id>,
            ) -> Token<Self::Id, P, Shared<N>> {
                // SAFETY: one SharedToken consumed per decrement keeps the
                // runtime reader count and type-level count in sync.
                unsafe {
                    self.raw_unlock_shared_nested();
                    Token::forge()
                }
            }
        }

        // ==============================================================
        // RAII data-carrying lock wrapper
        // ==============================================================

        /// A data-carrying hierarchical lock.
        ///
        /// [`acquire`](Lock::acquire) returns a `(guard, token)` pair: the
        /// guard grants data access, the token proves the level and can be
        /// used to acquire lower-level locks while the guard is held.
        /// Unlocking is explicit: the guard's `release` consumes both the
        /// guard and the token, recovering the previous-level token.
        pub struct Lock<T, L: HierarchicalLock> {
            raw:  L,
            data: ::core::cell::UnsafeCell<T>,
        }

        // SAFETY: sending the Lock sends the T inside it.
        unsafe impl<T: Send, L: HierarchicalLock + Send> Send for Lock<T, L> {}

        // SAFETY: read guards hand out `&T` on multiple threads (T: Sync);
        // write guards allow moving values out via `&mut T` (T: Send).
        // Identical bounds to std's RwLock.
        unsafe impl<T: Send + Sync, L: HierarchicalLock + Sync> Sync for Lock<T, L> {}

        impl<T, L: HierarchicalLock> Lock<T, L> {
            /// Creates a new lock around `value`.
            pub const fn new(raw: L, value: T) -> Self {
                Self {
                    raw,
                    data: ::core::cell::UnsafeCell::new(value),
                }
            }

            /// Consumes the lock, returning the inner value.
            ///
            /// Safe without a token: `self` by value proves no guards exist.
            pub fn into_inner(self) -> T {
                self.data.into_inner()
            }

            /// Returns a raw pointer to the underlying data, without locking.
            ///
            /// Like [`UnsafeCell::get`](::core::cell::UnsafeCell::get), the
            /// call itself is safe and the pointer is valid for as long as the
            /// lock is. Dereferencing it while somebody may hold the lock is
            /// a data race, which only a caller that has nothing left to lose,
            /// such as the panic path, may accept.
            pub const fn data_ptr(&self) -> *mut T {
                self.data.get()
            }

            /// Returns a mutable reference to the underlying data.
            ///
            /// Since this call borrows the Mutex mutably, no actual locking
            /// needs to take place – the mutable borrow statically guarantees
            /// no new locks can be acquired while this reference exists.
            pub const fn get_mut(&mut self) -> &mut T {
                unsafe { self.data.get_mut() }
            }

            /// Exclusive access.
            ///
            /// Returns the data guard and the target-level token as a pair.
            /// Release them together via [`WriteGuard::release`].
            pub fn acquire<From>(
                &self,
                token: From,
            ) -> (WriteGuard<'_, T, L>, Token<L::Id, From, Exclusive>)
            where
                From: CanAcquire<<L::Id as LockId>::Level>,
            {
                let token = HierarchicalLockExt::acquire(&self.raw, token);
                (WriteGuard { lock: self }, token)
            }

            /// Exclusive access if the lock is free.
            ///
            /// On success, the `(guard, token)` pair of
            /// [`acquire`](Lock::acquire); on failure, the caller's own
            /// token back, unchanged.
            pub fn try_acquire<From>(
                &self,
                token: From,
            ) -> Result<(WriteGuard<'_, T, L>, Token<L::Id, From, Exclusive>), From>
            where
                From: CanAcquire<<L::Id as LockId>::Level>,
            {
                let token = HierarchicalLockExt::try_acquire(&self.raw, token)?;
                Ok((WriteGuard { lock: self }, token))
            }

            /// Shared access. The nested-hold count starts at zero.
            ///
            /// Returns the data guard and the target-level token as a pair.
            /// Release them together via [`ReadGuard::release`].
            pub fn acquire_shared<From>(
                &self,
                token: From,
            ) -> (ReadGuard<'_, T, L>, Token<L::Id, From, Shared<Z>>)
            where
                From: CanAcquire<<L::Id as LockId>::Level>,
            {
                let token = HierarchicalLockExt::acquire_shared(&self.raw, token);
                (ReadGuard { lock: self }, token)
            }

            /// Shared access if no writer holds the lock.
            ///
            /// On success, the `(guard, token)` pair of
            /// [`acquire_shared`](Lock::acquire_shared); on failure, the
            /// caller's own token back, unchanged.
            pub fn try_acquire_shared<From>(
                &self,
                token: From,
            ) -> Result<(ReadGuard<'_, T, L>, Token<L::Id, From, Shared<Z>>), From>
            where
                From: CanAcquire<<L::Id as LockId>::Level>,
            {
                let token = HierarchicalLockExt::try_acquire_shared(&self.raw, token)?;
                Ok((ReadGuard { lock: self }, token))
            }

            /// Additional shared access alongside an existing shared hold.
            ///
            /// Consumes the primary token and returns it with the count
            /// incremented, alongside a nested data guard. Release via
            /// [`NestedReadGuard::release`].
            pub fn acquire_shared_nested<P, N>(
                &self,
                token: Token<L::Id, P, Shared<N>>,
            ) -> (NestedReadGuard<'_, T, L>, Token<L::Id, P, Shared<S<N>>>) {
                let (token, shared) =
                    HierarchicalLockExt::acquire_shared_nested(&self.raw, token);
                (NestedReadGuard { lock: self, token: shared }, token)
            }
        }

        /// Exclusive data guard. Dereferences to `T` (mutably).
        ///
        /// Release explicitly via [`WriteGuard::release`], together with
        /// the level token.
        pub struct WriteGuard<'a, T, L: HierarchicalLock> {
            lock: &'a Lock<T, L>,
        }

        impl<'a, T, L: HierarchicalLock> WriteGuard<'a, T, L> {
            /// Unlocks, consuming the guard and the level token, recovering
            /// the previous-level token.
            ///
            /// The token must stem from the same lock identity (`Id` is
            /// checked at the type level).
            pub fn release<P: PreviousToken>(
                self,
                token: Token<L::Id, P, Exclusive>,
            ) -> P {
                HierarchicalLockExt::release(&self.lock.raw, token)
            }
        }

        impl<'a, T, L: HierarchicalLock> ::core::ops::Deref for WriteGuard<'a, T, L> {
            type Target = T;
            fn deref(&self) -> &T {
                // SAFETY: exclusive lock held for the guard's lifetime.
                unsafe { &*self.lock.data.get() }
            }
        }

        impl<'a, T, L: HierarchicalLock> ::core::ops::DerefMut for WriteGuard<'a, T, L> {
            fn deref_mut(&mut self) -> &mut T {
                // SAFETY: exclusive lock held; `&mut self` ensures uniqueness.
                unsafe { &mut *self.lock.data.get() }
            }
        }

        /// Shared data guard. Dereferences to `T` (immutably).
        ///
        /// Release explicitly via [`ReadGuard::release`], together with the
        /// level token — which must have a nested-hold count of **zero**.
        pub struct ReadGuard<'a, T, L: HierarchicalLock> {
            lock: &'a Lock<T, L>,
        }

        impl<'a, T, L: HierarchicalLock> ReadGuard<'a, T, L> {
            /// Unlocks, consuming the guard and the level token, recovering
            /// the previous-level token.
            ///
            /// Accepts only `Shared<Z>` — all nested holds must have been
            /// released first.
            pub fn release<P: PreviousToken>(
                self,
                token: Token<L::Id, P, Shared<Z>>,
            ) -> P {
                HierarchicalLockExt::release_shared(&self.lock.raw, token)
            }
        }

        impl<'a, T, L: HierarchicalLock> ::core::ops::Deref for ReadGuard<'a, T, L> {
            type Target = T;
            fn deref(&self) -> &T {
                // SAFETY: shared lock held for the guard's lifetime.
                unsafe { &*self.lock.data.get() }
            }
        }

        /// Additional shared data guard.
        ///
        /// Release via [`NestedReadGuard::release`], together with the
        /// counted level token, which is returned decremented.
        pub struct NestedReadGuard<'a, T, L: HierarchicalLock> {
            lock:  &'a Lock<T, L>,
            token: SharedToken<L::Id>,
        }

        impl<'a, T, L: HierarchicalLock> NestedReadGuard<'a, T, L> {
            /// Returns this shared hold, decrementing the type-level count
            /// and the runtime reader count together.
            pub fn release<P, N>(
                self,
                token: Token<L::Id, P, Shared<S<N>>>,
            ) -> Token<L::Id, P, Shared<N>> {
                HierarchicalLockExt::release_shared_nested(
                    &self.lock.raw,
                    token,
                    self.token,
                )
            }
        }

        impl<'a, T, L: HierarchicalLock> ::core::ops::Deref
            for NestedReadGuard<'a, T, L>
        {
            type Target = T;
            fn deref(&self) -> &T {
                // SAFETY: shared lock held for the guard's lifetime.
                unsafe { &*self.lock.data.get() }
            }
        }
    }
    .into()
}

/// Declares a lock identity at a given level.
///
/// ```rust
/// #[lock_id(Paging)]
/// pub struct PageTableLockId;
/// ```
#[proc_macro_attribute]
pub fn lock_id(attr: TokenStream, item: TokenStream) -> TokenStream {
    let level = parse_macro_input!(attr as syn::Ident);
    let input = parse_macro_input!(item as syn::ItemStruct);
    let name = &input.ident;

    quote! {
        #input

        impl LockId for #name {
            type Level = level::#level;
        }
    }
    .into()
}
