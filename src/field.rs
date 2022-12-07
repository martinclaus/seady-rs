//! Provides the [`ArrND`] type, which is the basic data type to deal with 2D arrays,
//! and the [`Field`] trait which must be implemented by any type that shall be used to store
//! data of model variables.

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut, RangeFrom},
};

/// Index tuples
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ix<const ND: usize>([usize; ND]);

impl<const ND: usize> Ix<ND> {
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<usize> {
        self.0.iter()
    }

    #[inline]
    pub fn cshift(&self, dim: usize, shift: isize, shape: Shape<ND>) -> Self {
        assert!(dim < ND);
        let mut shifted = self.clone();
        shifted[dim] = cyclic_shift(shifted[dim], shift, shape[dim]);
        shifted
    }
}

impl<const ND: usize> Index<usize> for Ix<ND> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const ND: usize> IndexMut<usize> for Ix<ND> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const ND: usize> From<[usize; ND]> for Ix<ND> {
    fn from(from: [usize; ND]) -> Self {
        Ix(from)
    }
}

impl<const ND: usize> Into<[usize; ND]> for Ix<ND> {
    fn into(self) -> [usize; ND] {
        self.0
    }
}

/// Array shape
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape<const ND: usize>([usize; ND]);

impl<const ND: usize> Shape<ND> {
    /// Return an iterator over all possible indices of an array with shape `self`.
    pub fn iter(&self) -> NDIndexer<ND> {
        self.into_iter()
    }

    /// Return the number of the elements of the array
    pub fn size(&self) -> usize {
        let mut size = 1;
        self.0.iter().for_each(|i| size *= i);
        size
    }

    /// Return number of dimensions
    pub const fn ndim(&self) -> usize {
        ND
    }
}

impl<const ND: usize> AsRef<[usize; ND]> for Shape<ND> {
    fn as_ref(&self) -> &[usize; ND] {
        &self.0
    }
}

impl<const ND: usize> IntoIterator for Shape<ND> {
    type Item = Ix<ND>;

    type IntoIter = NDIndexer<ND>;

    fn into_iter(self) -> Self::IntoIter {
        NDIndexer {
            inner: None,
            shape: self,
        }
    }
}

impl<const ND: usize> Index<usize> for Shape<ND> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        self.as_ref().index(index)
    }
}

impl<const ND: usize> Index<RangeFrom<usize>> for Shape<ND> {
    type Output = [usize];

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        self.as_ref().index(index)
    }
}

pub struct NDIndexer<const ND: usize> {
    inner: Option<Ix<ND>>,
    shape: Shape<ND>,
}

impl<const ND: usize> Iterator for NDIndexer<ND> {
    type Item = Ix<ND>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            Some(ix) => {
                let shape = self.shape;
                let mut ix = ix;
                ix[ND - 1] += 1;
                (1..ND).rev().for_each(|d| {
                    if ix[d] >= shape[d] {
                        ix[d] = 0;
                        ix[d - 1] += 1
                    }
                });
                if ix[0] >= shape[0] {
                    None
                } else {
                    self.inner = Some(ix);
                    self.inner
                }
            }
            None => {
                self.inner = Some(Ix([0 as usize; ND]));
                self.inner
            }
        }
    }
}

impl<const ND: usize, T> From<T> for Shape<ND>
where
    T: Into<[usize; ND]>,
{
    fn from(from: T) -> Self {
        Shape(from.into())
    }
}

#[inline]
fn cyclic_shift(idx: usize, shift: isize, len: usize) -> usize {
    let len = len as isize;
    match (idx as isize) + shift {
        i if i < 0 => (i + len) as usize,
        i if i >= len => (i - len) as usize,
        i => i as usize,
    }
}

/// Trait for array backends.
pub trait Field<const ND: usize>
where
    Self: Sized + Index<Ix<ND>, Output = Self::Item> + IndexMut<Ix<ND>, Output = Self::Item>,
{
    type Item;
    /// Create a new field with all elements set to a constant value.
    fn full(item: Self::Item, shape: impl Into<Shape<ND>>) -> Self;

    /// Return the shape of the array.
    fn shape(&self) -> Shape<ND>;
}

/// N-dimensional Array with linear contiguous memory layout.
///
/// The data is stored in a boxed slice and available via indexing with an
/// array of indices. The indexing is column-major.
///
/// # Examples
/// Create an array, filled with a value.
/// ```
/// use seady::field::{ArrND, Field};
///
/// let arr = ArrND::full(1f64, [2, 2]);
///
/// assert_eq!(arr[[0, 0]], 1.0);
/// assert_eq!(arr[[1, 0]], 1.0);
/// assert_eq!(arr[[0, 1]], 1.0);
/// assert_eq!(arr[[1, 1]], 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct ArrND<const ND: usize, I> {
    shape: Shape<ND>,
    data: Box<[I]>,
}

impl<const ND: usize, I> ArrND<ND, I> {
    #[inline]
    fn flatten_index(&self, index: Ix<ND>) -> usize {
        let shape = self.shape;
        let mut sum = 0;
        let mut prod: usize;
        for d in 0..ND {
            prod = shape[d + 1..].iter().product();
            sum += index[d] * prod;
        }
        sum
    }
}

impl<const ND: usize, T, I> Index<T> for ArrND<ND, I>
where
    T: Into<Ix<ND>>,
{
    type Output = I;
    #[inline]
    fn index(&self, index: T) -> &I {
        &self.data[self.flatten_index(index.into())]
    }
}

impl<const ND: usize, T, I> IndexMut<T> for ArrND<ND, I>
where
    T: Into<Ix<ND>>,
{
    #[inline]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.data[self.flatten_index(index.into())]
    }
}

impl<const ND: usize, I: Add<Output = I> + Copy> Add for ArrND<ND, I> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.shape, rhs.shape);
        let data: Box<[I]> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Self {
            shape: self.shape,
            data,
        }
    }
}

impl<const ND: usize, I: AddAssign + Copy> AddAssign<&ArrND<ND, I>> for ArrND<ND, I> {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.shape.size() {
            self.data[i] += rhs.data[i]
        }
    }
}

impl<const ND: usize, I: AddAssign + Copy> AddAssign for ArrND<ND, I> {
    fn add_assign(&mut self, rhs: Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(a, b)| *a += *b);
    }
}

impl<const ND: usize, I: Copy> Field<ND> for ArrND<ND, I> {
    type Item = I;

    fn full(item: I, shape: impl Into<Shape<ND>>) -> Self {
        let shape = shape.into();
        ArrND {
            shape,
            data: vec![item; shape.size()].into_boxed_slice(),
        }
    }

    fn shape(&self) -> Shape<ND> {
        self.shape
    }
}

impl<const ND: usize, I: Display + Copy> Display for ArrND<ND, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice = ArrSlice {
            data: &self.data[..],
            shape: &self.shape.0[..],
            fixed_dims: 0,
        };
        write!(f, "{}", slice)
    }
}

struct ArrSlice<'a, I> {
    data: &'a [I],
    shape: &'a [usize],
    fixed_dims: usize,
}

impl<'a, I> Display for ArrSlice<'a, I>
where
    I: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shape.len() == 1 {
            write!(
                f,
                "[{}]",
                self.data
                    .iter()
                    .map(|i| format!("{}", i))
                    .collect::<Vec<_>>()
                    .join(","),
            )?
        } else {
            write!(f, "[")?;
            for i in 0..self.shape[0] {
                let size: usize = self.shape[1..].iter().product();
                let slice = ArrSlice {
                    data: &self.data[i * size..(i + 1) * size],
                    shape: &self.shape[1..],
                    fixed_dims: self.fixed_dims + 1,
                };
                write!(f, "{}", slice)?;
                if i != self.shape[0] - 1 {
                    write!(f, ",\n")?
                }
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::field::{Ix, Shape};

    use super::{ArrND, Field};

    #[test]
    fn new_shape_from_array() {
        let shape: Shape<2> = [2, 3].into();
        assert_eq!(shape, Shape([2, 3]));
    }

    #[test]
    fn nd_indexer_produces_correct_values_1d() {
        use super::NDIndexer;
        let shape = [2].into();

        let mut indexer = NDIndexer { inner: None, shape };

        assert_eq!(indexer.next(), Some(Ix([0])));
        assert_eq!(indexer.next(), Some(Ix([1])));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }

    #[test]
    fn nd_indexer_produces_correct_values_2d() {
        use super::NDIndexer;
        let shape = [2, 3].into();

        let mut indexer = NDIndexer { inner: None, shape };

        assert_eq!(indexer.next(), Some(Ix([0, 0])));
        assert_eq!(indexer.next(), Some(Ix([0, 1])));
        assert_eq!(indexer.next(), Some(Ix([0, 2])));
        assert_eq!(indexer.next(), Some(Ix([1, 0])));
        assert_eq!(indexer.next(), Some(Ix([1, 1])));
        assert_eq!(indexer.next(), Some(Ix([1, 2])));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }

    #[test]
    fn nd_indexer_produces_correct_values_3d() {
        use super::NDIndexer;
        let shape = [2, 2, 2].into();

        let mut indexer = NDIndexer { inner: None, shape };

        assert_eq!(indexer.next(), Some(Ix([0, 0, 0])));
        assert_eq!(indexer.next(), Some(Ix([0, 0, 1])));
        assert_eq!(indexer.next(), Some(Ix([0, 1, 0])));
        assert_eq!(indexer.next(), Some(Ix([0, 1, 1])));
        assert_eq!(indexer.next(), Some(Ix([1, 0, 0])));
        assert_eq!(indexer.next(), Some(Ix([1, 0, 1])));
        assert_eq!(indexer.next(), Some(Ix([1, 1, 0])));
        assert_eq!(indexer.next(), Some(Ix([1, 1, 1])));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }

    #[test]
    fn shape_size_is_correct() {
        assert_eq!(Shape([2, 3]).size(), 6)
    }

    #[test]
    fn create_arrnd() {
        let shape = [2, 2];

        let arr = ArrND::full(0f64, shape);
        assert_eq!(arr.shape, shape.into());
        assert_eq!(arr[Ix([0, 0])], 0f64);
        assert_eq!(arr[Ix([0, 1])], 0f64);
        assert_eq!(arr[Ix([1, 0])], 0f64);
        assert_eq!(arr[Ix([1, 1])], 0f64);
    }

    #[test]
    fn display_output() {
        let mut arr = ArrND::full(1f64, [2, 3, 2]);
        arr[Ix([1, 2, 1])] = 0f64;
        assert_eq!(
            format!("{}", arr),
            "[[[1,1],\n[1,1],\n[1,1]],\n[[1,1],\n[1,1],\n[1,0]]]"
        );
    }

    #[test]
    fn index_via_array() {
        let mut arr = ArrND::full(1f64, [2, 3, 2]);
        arr[[1, 2, 1]] = 0f64;
        assert_eq!(arr[[1, 2, 1]], 0f64);
        assert_eq!(arr[[1, 1, 1]], 1f64);
    }

    #[test]
    fn test_cyclic_shift() {
        use super::cyclic_shift;
        let size: usize = 10;
        assert_eq!(cyclic_shift(0, 1, size), 1);
        assert_eq!(cyclic_shift(1, 1, size), 2);
        assert_eq!(cyclic_shift(1, -1, size), 0);
        assert_eq!(cyclic_shift(0, -1, size), size - 1);
        assert_eq!(cyclic_shift(size - 1, 2, size), 1);
        assert_eq!(cyclic_shift(1, 3, size), 4);
    }
}
