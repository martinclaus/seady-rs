//! Trait for types that represent a domain mask, i.e. data that specifies
//! wether a particular location is part of the computational domain, and the default implementation
//! using the enum [`DomainMask`].

use std::{fmt::Display, ops::Mul};

/// Type to flag a location as being inside or outside the computational domain.
///
/// For convenience, `core::convert::From<DomainMask>` is implemented for `f64` and
/// `std::ops::Mul<f64>` for `DomainMask` to allow for easy use in arithmetic expressions.
/// Here, `DomainMask::Inside` is converted to `1f64` and `DomainMask::Outside` to `0f64`.
///
/// # Examples
/// To create a domain mask based on the Arr2D type, where the rim of the
/// domain is flagged as closed, you can do the following:
/// ```
/// use seady::field::{ArrND, Field, shape};
/// use seady::mask::{DomainMask, Mask};
///
/// let mut mask = ArrND::full(DomainMask::outside(), shape([5, 10]));
/// for j in 1..4 {
///     for i in 1..9 {
///         mask[[j, i]] = DomainMask::inside()
///     }
/// }
///
/// assert!(mask[[1, 1]].is_inside());
/// assert!(mask[[0, 1]].is_outside());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DomainMask {
    /// Marks data to be outside of the domain.
    Outside,
    /// Marks Data to be inside the domain.
    Inside,
}

/// Trait for types representing a domain mask.
///
/// The default implementation of the constructor `new` is to return
/// "inside".
pub trait Mask: Copy + PartialEq + Into<f64> {
    /// Return "inside" by default.
    fn new() -> Self {
        Self::inside()
    }
    /// Return an "inside" value.
    fn inside() -> Self;
    /// Return an "outside" value.
    fn outside() -> Self;
    /// Check if the mask value corresponds to "inside".
    fn is_inside(&self) -> bool {
        *self == Self::inside()
    }
    /// Check if the mask value corresponds to "outside".
    fn is_outside(&self) -> bool {
        !self.is_inside()
    }
}

impl Mask for DomainMask {
    /// Returns [`DomainMask::Inside`]
    fn inside() -> Self {
        DomainMask::Inside
    }

    /// Returns [`DomainMask::Outside`]
    fn outside() -> Self {
        DomainMask::Outside
    }
}

/// Mask values are interpreted as 0f64 if outside and 1f64 if inside the domain.
///
/// # Examples
/// ```
/// use seady::mask::DomainMask;
///
/// assert_eq!(f64::from(DomainMask::Outside), 0.0);
/// assert_eq!(f64::from(DomainMask::Inside), 1.0);
/// ```
impl From<DomainMask> for f64 {
    fn from(mask: DomainMask) -> Self {
        match mask {
            DomainMask::Inside => 1f64,
            DomainMask::Outside => 0f64,
        }
    }
}

/// Implement multiplication with float as rhs.
///
/// Mask values are interpreted as 0f64 if outside and 1f64 if inside the domain.
///
/// # Examples
/// ```
/// use seady::mask::{DomainMask, Mask};
///
/// assert_eq!(DomainMask::outside() * 5.0, 0.0);
/// assert_eq!(DomainMask::inside() * 5.0, 5.0);
/// ```
impl Mul<f64> for DomainMask {
    type Output = f64;

    fn mul(self, rhs: f64) -> Self::Output {
        f64::from(self) * rhs
    }
}

impl Display for DomainMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DomainMask::Outside => write!(f, "{:7}", "Outside"),
            DomainMask::Inside => write!(f, "{:7}", "Inside"),
        }
    }
}
