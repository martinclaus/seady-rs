//! A framework for quickly developing high performance ocean and atmosphere models

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
    + Copy
{
    /// Returns zero of this type.
    fn zero() -> Self;
}

impl Numeric for f64 {
    fn zero() -> Self {
        0f64
    }
}

pub mod field {
    //! Provides the [`Arr2D`] type, which is the basic data type to deal with 2D arrays,
    //! and the [`Field`] trait which must be implemented by any type that shall be used to store
    //! data of model variables.

    use std::{
        fmt::Display,
        ops::{Add, AddAssign, Index, IndexMut},
    };

    /// Type alias for Array sizes
    pub type Size2D = (usize, usize);

    /// Type alias for index tuples
    pub type Ix2 = [usize; 2];

    pub fn cyclic_shift(idx: usize, shift: isize, len: usize) -> usize {
        let len = len as isize;
        match (idx as isize) + shift {
            i if i < 0 => (i + len) as usize,
            i if i >= len => (i - len) as usize,
            i => i as usize,
        }
    }
    /// Trait for array backends.
    pub trait Field<I>
    where
        Self: Sized + Index<Ix2, Output = I> + IndexMut<Ix2, Output = I>,
    {
        /// Create a new field with all elements set to a constant value.
        fn full(item: I, size: Size2D) -> Self;

        /// Return a tuple of the size of an array.
        fn size(&self) -> Size2D;
    }

    /// Simple 2D array with linear contiguous memory layout. The data is stored in
    /// a boxed slice and available via indexing with an array of indices.
    ///
    /// # Examples
    /// Create an array, filled with a value.
    /// ```
    /// use seady::field::Arr2D;
    /// use seady::field::Field;
    ///
    /// let arr = Arr2D::full(1f64, (2, 2));
    ///
    /// assert_eq!(arr[[0, 0]], 1.0);
    /// assert_eq!(arr[[1, 0]], 1.0);
    /// assert_eq!(arr[[0, 1]], 1.0);
    /// assert_eq!(arr[[1, 1]], 1.0);
    /// ```
    #[derive(Clone, Debug)]
    pub struct Arr2D<I> {
        size: (usize, usize),
        data: Box<[I]>,
    }

    impl<I> Index<Ix2> for Arr2D<I> {
        type Output = I;
        #[inline]
        fn index(&self, index: Ix2) -> &I {
            &self.data[self.size.1 * index[0] + index[1]]
        }
    }

    impl<I> IndexMut<Ix2> for Arr2D<I> {
        #[inline]
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            &mut self.data[self.size.1 * index[0] + index[1]]
        }
    }

    impl<I: Add<Output = I> + Copy> Add for Arr2D<I> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            assert_eq!(self.size, rhs.size);
            let data: Box<[I]> = self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| *a + *b)
                .collect();
            Self {
                size: self.size,
                data,
            }
        }
    }

    impl<I: AddAssign + Copy> AddAssign<&Arr2D<I>> for Arr2D<I> {
        fn add_assign(&mut self, rhs: &Self) {
            for j in 0..self.size.0 {
                for i in 0..self.size.1 {
                    self[[j, i]] += rhs[[j, i]];
                }
            }
        }
    }

    impl<I: AddAssign + Copy> AddAssign for Arr2D<I> {
        fn add_assign(&mut self, rhs: Self) {
            self.data
                .iter_mut()
                .zip(rhs.data.iter())
                .for_each(|(a, b)| *a += *b);
        }
    }

    impl<I: Copy> Field<I> for Arr2D<I> {
        fn full(item: I, size: Size2D) -> Self {
            Arr2D {
                size,
                data: vec![item; size.0 * size.1].into_boxed_slice(),
            }
        }

        fn size(&self) -> Size2D {
            self.size
        }
    }

    impl<I: Display> Display for Arr2D<I> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let ilen = self.size.1;
            write!(f, "[")?;
            for j in 0..self.size.0 {
                let s = &self.data[j * ilen..(j + 1) * ilen]
                    .iter()
                    .map(|i| format!("{}", i))
                    .collect::<Vec<_>>()
                    .join(",");
                write!(f, "[{}]", s)?;
                if j != self.size.0 - 1 {
                    write!(f, ",\n ")?;
                } else {
                    write!(f, "]")?;
                }
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod test {
        use super::{cyclic_shift, Arr2D, Field};

        #[test]
        fn display_output() {
            let mut arr = Arr2D::full(1f64, (3, 5));
            arr[[1, 2]] = 0f64;
            assert_eq!(
                format!("{}", arr),
                "[[1,1,1,1,1],\n [1,1,0,1,1],\n [1,1,1,1,1]]"
            );
        }

        #[test]
        fn test_cyclic_shift() {
            let size: usize = 10;
            assert_eq!(cyclic_shift(0, 1, size), 1);
            assert_eq!(cyclic_shift(1, 1, size), 2);
            assert_eq!(cyclic_shift(1, -1, size), 0);
            assert_eq!(cyclic_shift(0, -1, size), size - 1);
            assert_eq!(cyclic_shift(size - 1, 2, size), 1);
            assert_eq!(cyclic_shift(1, 3, size), 4);
        }
    }
}
