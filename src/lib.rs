//! A framework for quickly developing high performance ocean and atmosphere models

pub mod field;
pub mod grid;
pub mod mask;
pub mod state;
pub mod timestepping;
pub mod variable;

use std::ops::{Add, Div, Mul, Sub};

/// Super Trait for numeric data types. It only contains a single method
/// but the trait bounds include all the necessary traits to ensure
/// that we can do number crunching with this type.
pub trait Numeric:
    Copy
    + std::fmt::Debug
    + Add<Output = Self>
    + Mul<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Add<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Div<f64, Output = Self>
    + From<f64>
    + Into<f64>
{
    /// Returns zero of this type.
    fn zero() -> Self;
}

impl Numeric for f64 {
    fn zero() -> Self {
        0f64
    }
}
