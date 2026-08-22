use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Derives a complete pointer-wrapper API for a newtype struct wrapping `*mut T`.
///
/// The target struct must be of the form `struct Foo<T>(*mut T)`.
///
/// Besides the pointer methods this emits the comparison, hashing and
/// formatting impls, and the byte arithmetic a `Range` needs from its `Base`:
/// `Self + usize -> Self` and `Self - Self -> usize`.
#[proc_macro_derive(Address)]
pub fn derive_address(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #name #ty_generics #where_clause {

            /// Wraps a raw `*mut T` pointer.
            ///
            /// Unlike `NonNull`, this does **not** require the pointer to be non-null.
            /// The caller is responsible for ensuring validity before dereferencing.
            pub const fn new(ptr: *mut T) -> Self {
                Self(ptr)
            }

            /// Create a `null` pointer.
            pub const fn null() -> Self {
                Self(core::ptr::null_mut())
            }

            /// Creates an address from a shared reference.
            ///
            /// The resulting pointer is valid for the lifetime of `r`, but the
            /// type system does not enforce this — the caller must ensure the
            /// address is not used after `r` is dropped.
            pub const fn from_ref(r: &T) -> Self {
                unsafe { Self::new(r as *const T as *mut T) }
            }

            /// Creates an address from a mutable reference.
            ///
            /// Consumes the exclusive borrow for the duration of the raw pointer's use.
            /// No other references to the same data may exist while the pointer is live.
            pub const fn from_mut(r: &mut T) -> Self {
                unsafe { Self::new(r as *mut T) }
            }

            /// Creates a well-aligned but non-dereferenceable dangling pointer.
            ///
            /// Useful as a placeholder or sentinel value. The address is guaranteed
            /// to be non-null and correctly aligned for `T`, but must never be
            /// dereferenced.
            pub const fn dangling() -> Self {
                unsafe {
                    Self::new(core::ptr::dangling_mut::<T>())
                }
            }

            /// Returns the underlying `*mut T`.
            pub const fn as_ptr(self) -> *mut T {
                self.0
            }

            /// Check if the pointer is null.
            pub const fn is_null(self) -> bool {
                self.0.is_null()
            }

            /// Dereferences the pointer as a shared reference.
            ///
            /// # Safety
            ///
            /// - The pointer must be non-null and correctly aligned for `T`.
            /// - The memory must contain a valid, initialized `T`.
            /// - The returned reference must not outlive the data it points to.
            /// - No mutable references to the same memory may exist for the
            ///   lifetime `'a`.
            pub const unsafe fn as_ref<'a>(&self) -> &'a T {
                & *self.as_ptr()
            }

            /// Dereferences the pointer as a mutable reference.
            ///
            /// # Safety
            ///
            /// - The pointer must be non-null and correctly aligned for `T`.
            /// - The memory must contain a valid, initialized `T`.
            /// - The returned reference must not outlive the data it points to.
            /// - No other references (shared or mutable) to the same memory may
            ///   exist for the lifetime `'a`.
            pub const unsafe fn as_mut<'a>(&mut self) -> &'a mut T {
                &mut *self.as_ptr()
            }

            /// Casts the pointer to a different pointee type `U`.
            ///
            /// Returns a new address of type `Self<U>` pointing to the same
            /// memory location. The caller must ensure the new type `U` is
            /// compatible with the memory at that address before dereferencing.
            pub fn cast<U>(self) -> #name<U> {
                unsafe { #name::new(self.0 as *mut U) }
            }

            /// Offsets the pointer forward by `count` elements of type `T`.
            ///
            /// # Safety
            ///
            /// The resulting pointer must remain within the bounds of the same
            /// allocated object (or one past the end). See [`*mut T::add`].
            pub const unsafe fn add(self, count: usize) -> Self {
                unsafe { Self::new(self.0.add(count)) }
            }

            /// Offsets the pointer forward by `count` bytes.
            ///
            /// # Safety
            ///
            /// The resulting pointer must remain within the bounds of the same
            /// allocated object (or one past the end). See [`*mut T::byte_add`].
            pub const unsafe fn byte_add(self, count: usize) -> Self {
                unsafe { Self::new(self.0.byte_add(count)) }
            }

            /// Offsets the pointer backward by `count` elements of type `T`.
            ///
            /// # Safety
            ///
            /// The resulting pointer must remain within the bounds of the same
            /// allocated object. See [`*mut T::sub`].
            pub const unsafe fn sub(self, count: usize) -> Self {
                unsafe { Self::new(self.0.sub(count)) }
            }

            /// Offsets the pointer backward by `count` bytes.
            ///
            /// # Safety
            ///
            /// The resulting pointer must remain within the bounds of the same
            /// allocated object. See [`*mut T::byte_sub`].
            pub const unsafe fn byte_sub(self, count: usize) -> Self {
                unsafe { Self::new(self.0.byte_sub(count)) }
            }

            /// Offsets the pointer by a signed element count.
            ///
            /// # Safety
            ///
            /// The resulting pointer must remain within the bounds of the same
            /// allocated object (or one past the end). See [`*mut T::offset`].
            pub const unsafe fn offset(self, count: isize) -> Self {
                unsafe { Self::new(self.0.offset(count)) }
            }

            /// Returns the signed element distance between `self` and `from`.
            ///
            /// Equivalent to `(self - from)` in units of `T`.
            ///
            /// # Safety
            ///
            /// Both pointers must point into or one past the end of the same
            /// allocated object. See [`*mut T::offset_from`].
            pub const unsafe fn offset_from(self, from: Self) -> isize {
                unsafe { self.0.offset_from(from.0) }
            }

            /// Returns the pointer's address as a `usize`.
            pub fn addr(self) -> usize {
                self.0.addr()
            }

            /// Reads the value at the pointer.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null, aligned, and point to a valid
            /// initialized `T`. See [`*mut T::read`].
            pub const unsafe fn read(self) -> T
            where T: Sized {
                self.0.read()
            }

            /// Performs a volatile read of the value at the pointer.
            ///
            /// Volatile reads are never elided or reordered by the compiler,
            /// making this suitable for memory-mapped I/O registers.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and aligned. See [`*mut T::read_volatile`].
            pub unsafe fn read_volatile(self) -> T
            where T: Sized {
                self.0.read_volatile()
            }

            /// Reads the value at the pointer without requiring alignment.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and point to a valid initialized `T`,
            /// but need not be aligned. See [`*mut T::read_unaligned`].
            pub const unsafe fn read_unaligned(self) -> T
            where T: Sized {
                self.0.read_unaligned()
            }

            /// Writes `val` to the pointer's location.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null, aligned, and valid for writes of `T`.
            /// See [`*mut T::write`].
            pub const unsafe fn write(self, val: T)
            where T: Sized {
                self.0.write(val)
            }

            /// Performs a volatile write of `val` to the pointer's location.
            ///
            /// Volatile writes are never elided or reordered by the compiler,
            /// making this suitable for memory-mapped I/O registers.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and aligned. See [`*mut T::write_volatile`].
            pub unsafe fn write_volatile(self, val: T)
            where T: Sized {
                self.0.write_volatile(val)
            }

            /// Writes `val` to the pointer's location without requiring alignment.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and valid for writes of `T`, but need
            /// not be aligned. See [`*mut T::write_unaligned`].
            pub const unsafe fn write_unaligned(self, val: T)
            where T: Sized {
                self.0.write_unaligned(val)
            }

            /// Writes `src` to the pointer's location, returning the previous value.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null, aligned, and point to a valid
            /// initialized `T`. See [`*mut T::replace`].
            pub const unsafe fn replace(self, src: T) -> T {
                self.0.replace(src)
            }

            /// Swaps the values at `self` and `src`.
            ///
            /// # Safety
            ///
            /// Both pointers must be non-null, aligned, valid for reads and writes
            /// of `T`, and must not overlap. See [`*mut T::swap`].
            pub const unsafe fn swap(self, src: Self) {
                self.0.swap(src.0);
            }

            // --- Alignment ---

            /// Returns `true` if the pointer is aligned to `T`'s natural alignment.
            pub fn is_aligned(self) -> bool {
                self.0.is_aligned()
            }

            /// Returns the number of bytes needed to advance the pointer to the
            /// next address aligned to `align`.
            ///
            /// `align` must be a power of two.
            pub fn align_offset(self, align: usize) -> usize {
                self.0.align_offset(align)
            }
        }

        impl #impl_generics core::fmt::Pointer for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Pointer::fmt(&self.0, f)
            }
        }

        impl #impl_generics core::fmt::Debug for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Pointer::fmt(&self.0, f)
            }
        }

        impl #impl_generics Clone for #name #ty_generics #where_clause {
            fn clone(&self) -> Self { *self }
        }

        impl #impl_generics Copy for #name #ty_generics #where_clause {}

        impl #impl_generics PartialEq for #name #ty_generics #where_clause {
            fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }

        impl #impl_generics Eq for #name #ty_generics #where_clause {}

        impl #impl_generics PartialOrd for #name #ty_generics #where_clause {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl #impl_generics Ord for #name #ty_generics #where_clause {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }

        impl #impl_generics core::hash::Hash for #name #ty_generics #where_clause {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state)
            }
        }

        // --- Range arithmetic ---
        //
        // `Range<Self, usize>` measures its length in *bytes*, so these count
        // bytes as well, unlike `add`/`sub`, which step in elements of `T`.
        //
        // Wrapping, and therefore safe: `add` and friends are `unsafe`
        // because they promise to stay within one allocated object, which an
        // address that merely delimits a region need not do.

        impl #impl_generics core::ops::Add<usize> for #name #ty_generics #where_clause {
            type Output = Self;

            fn add(self, bytes: usize) -> Self {
                Self(self.0.wrapping_byte_add(bytes))
            }
        }

        /// The distance from `origin` to `self` in bytes.
        ///
        /// # Panics
        ///
        /// If `self` lies below `origin`, in builds with overflow checks. This
        /// is the half-open `end - base` of a range, which is never negative.
        impl #impl_generics core::ops::Sub for #name #ty_generics #where_clause {
            type Output = usize;

            fn sub(self, origin: Self) -> usize {
                self.addr() - origin.addr()
            }
        }
    }
    .into()
}
