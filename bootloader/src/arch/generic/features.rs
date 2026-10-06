//! Handle architecture-specific extensions and features.

pub trait Features {
    /// Checks whether all required features are available and actives them.
    ///
    /// # Panics
    ///
    /// If any of the required features is missing, this function will `panic`.
    fn activate();
}
