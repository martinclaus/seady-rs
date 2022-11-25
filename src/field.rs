//! Provides the [`Arr2D`] type, which is the basic data type to deal with 2D arrays,
//! and the [`Field`] trait which must be implemented by any type that shall be used to store
//! data of model variables.

use std::ops::{Add, AddAssign, Index, IndexMut, RangeFrom};

/// Type alias for index tuples
pub type Ix<const ND: usize> = [usize; ND];

pub fn shape<const ND: usize>(shape: [usize; ND]) -> Shape<ND> {
    Shape(shape)
}

/// Array shape
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape<const ND: usize>(Ix<ND>);

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
}

impl<const ND: usize> AsRef<Ix<ND>> for Shape<ND> {
    fn as_ref(&self) -> &Ix<ND> {
        &self.0
    }
}

impl<const ND: usize> IntoIterator for Shape<ND> {
    type Item = Ix<ND>;

    type IntoIter = NDIndexer<ND>;

    fn into_iter(self) -> Self::IntoIter {
        NDIndexer {
            inner: None,
            shape: self.0,
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
    shape: Ix<ND>,
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
                self.inner = Some([0 as usize; ND]);
                self.inner
            }
        }
    }
}

/// Trait to allow for conversion into a Shape type
pub trait IntoShape<const ND: usize> {
    fn into_shape(self) -> Shape<ND>;
}

impl<const ND: usize> IntoShape<ND> for Shape<ND> {
    fn into_shape(self) -> Shape<ND> {
        self
    }
}

impl<const ND: usize> IntoShape<ND> for [usize; ND] {
    fn into_shape(self) -> Shape<ND> {
        Shape(self)
    }
}

impl<const ND: usize> IntoShape<ND> for &[usize] {
    fn into_shape(self) -> Shape<ND> {
        let shape = self.try_into().expect("Dimension should match");
        Shape(shape)
    }
}

pub fn cyclic_shift(idx: usize, shift: isize, len: usize) -> usize {
    let len = len as isize;
    match (idx as isize) + shift {
        i if i < 0 => (i + len) as usize,
        i if i >= len => (i - len) as usize,
        i => i as usize,
    }
}

/// Trait for array backends.
pub trait Field<const ND: usize, I>
where
    Self: Sized + Index<Ix<ND>, Output = I> + IndexMut<Ix<ND>, Output = I>,
{
    /// Create a new field with all elements set to a constant value.
    fn full(item: I, shape: impl IntoShape<ND>) -> Self;

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
/// use seady::field::{ArrND, shape};
/// use seady::field::Field;
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
        let mut sum = index[ND - 1];
        let mut prod: usize;
        for d in 0..ND - 1 {
            prod = shape[d + 1..].iter().product();
            sum += index[d] * prod;
        }
        sum
    }
}

impl<const ND: usize, I> Index<Ix<ND>> for ArrND<ND, I> {
    type Output = I;
    #[inline]
    fn index(&self, index: Ix<ND>) -> &I {
        &self.data[self.flatten_index(index)]
    }
}

impl<const ND: usize, I> IndexMut<Ix<ND>> for ArrND<ND, I> {
    #[inline]
    fn index_mut(&mut self, index: Ix<ND>) -> &mut Self::Output {
        &mut self.data[self.flatten_index(index)]
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

impl<const ND: usize, I: Copy> Field<ND, I> for ArrND<ND, I> {
    fn full(item: I, shape: impl IntoShape<ND>) -> Self {
        let shape = shape.into_shape();
        ArrND {
            shape,
            data: vec![item; shape.size()].into_boxed_slice(),
        }
    }

    fn shape(&self) -> Shape<ND> {
        self.shape
    }
}

// impl<const ND: usize, I: Display + Copy> Display for ArrND<ND, I> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         if ND > 1 {
//             let shape = self.shape;
//             let s = (0..shape[0])
//                 .map(|i| {
//                     let data = self.data[i * IntoShape::<ND>::into_shape(&shape[1..]).size()
//                         ..(i + 1) * IntoShape::<ND>::into_shape(&shape[1..]).size()]
//                         .iter()
//                         .map(|i| *i)
//                         .collect::<Vec<I>>()
//                         .into_boxed_slice();
//                     let slice = ArrND {
//                         data,
//                         shape: shape[1..].into_shape(),
//                     };
//                     format!("[{}]", slice)
//                 })
//                 .collect::<Vec<_>>()
//                 .join(",\n");
//             write!(f, "[{}]", s)
//         } else {
//             let s = &self
//                 .data
//                 .iter()
//                 .map(|i| format!("{}", i))
//                 .collect::<Vec<_>>()
//                 .join(",");
//             write!(f, "[{}]", s)
//         }
//     }
// }

#[cfg(test)]
mod test {
    use crate::field::IntoShape;

    use super::{shape, ArrND, Field};

    #[test]
    fn new_shape_from_array() {
        let shape = shape([2, 3]);
        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 3);
    }

    #[test]
    fn nd_indexer_produces_correct_values_1d() {
        use super::NDIndexer;
        let shape = [2];

        let mut indexer = NDIndexer { inner: None, shape };

        assert_eq!(indexer.next(), Some([0]));
        assert_eq!(indexer.next(), Some([1]));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }

    #[test]
    fn nd_indexer_produces_correct_values_2d() {
        use super::NDIndexer;
        let shape = [2, 3];

        let mut indexer = NDIndexer { inner: None, shape };

        assert_eq!(indexer.next(), Some([0, 0]));
        assert_eq!(indexer.next(), Some([0, 1]));
        assert_eq!(indexer.next(), Some([0, 2]));
        assert_eq!(indexer.next(), Some([1, 0]));
        assert_eq!(indexer.next(), Some([1, 1]));
        assert_eq!(indexer.next(), Some([1, 2]));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }

    #[test]
    fn nd_indexer_produces_correct_values_3d() {
        use super::NDIndexer;
        let shape = [2, 2, 2];

        let mut indexer = NDIndexer { inner: None, shape };

        assert_eq!(indexer.next(), Some([0, 0, 0]));
        assert_eq!(indexer.next(), Some([0, 0, 1]));
        assert_eq!(indexer.next(), Some([0, 1, 0]));
        assert_eq!(indexer.next(), Some([0, 1, 1]));
        assert_eq!(indexer.next(), Some([1, 0, 0]));
        assert_eq!(indexer.next(), Some([1, 0, 1]));
        assert_eq!(indexer.next(), Some([1, 1, 0]));
        assert_eq!(indexer.next(), Some([1, 1, 1]));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }

    #[test]
    fn shape_size_is_correct() {
        assert_eq!(shape([2, 3]).size(), 6)
    }

    #[test]
    fn create_arrnd() {
        let shape = [2, 2];

        let arr = ArrND::full(0f64, shape);
        assert_eq!(arr.shape, shape.into_shape());
        assert_eq!(arr[[0, 0]], 0f64);
        assert_eq!(arr[[0, 1]], 0f64);
        assert_eq!(arr[[1, 0]], 0f64);
        assert_eq!(arr[[1, 1]], 0f64);
    }

    // #[test]
    // fn display_output() {
    //     let mut arr = ArrND::full(1f64, (3, 5));
    //     arr[[1, 2]] = 0f64;
    //     assert_eq!(
    //         format!("{}", arr),
    //         "[[1,1,1,1,1],\n [1,1,0,1,1],\n [1,1,1,1,1]]"
    //     );
    // }

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
