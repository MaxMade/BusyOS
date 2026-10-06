//! Common error return values.

use core::error::Error;
use core::fmt::Display;

/// Common error return values.
#[derive(Debug, PartialEq, Eq)]
pub enum Errno {
    /// Cannot allocate memory.
    ENOMEM = 12,
    /// Bad address.
    EFAULT = 14,
    /// Resource exists.
    EEXISTS = 17,
    /// Invalid argument.
    EINVAL,
}

impl Display for Errno {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Errno::ENOMEM => write!(f, "cannot allocate memory"),
            Errno::EFAULT => write!(f, "bad address"),
            Errno::EEXISTS => write!(f, "resource exists"),
            Errno::EINVAL => write!(f, "invalid argument"),
        }
    }
}

impl Into<usize> for Errno {
    fn into(self) -> usize {
        self as usize
    }
}

impl Into<isize> for Errno {
    fn into(self) -> isize {
        self as isize
    }
}

/// Trait to convert [`Error`]s to [`Errno`] value.
pub trait ToErrno: Error {
    /// Convert [`Error`] to [`Errno`].
    fn to_errno(&self) -> Errno;
}
