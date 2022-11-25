//! Basic traits and data structures to describe [Grids](`Grid`).

use crate::field::{ArrND, Field, IntoShape, Shape};
use crate::mask::Mask;
use crate::Numeric;

/// Trait defining the interface of grid-like structs.
///
/// A grid defines the coordinates of a [`Field`], e.g. the latitude or longitude, and
/// a mask to flag grid points, e.g. to be inside or outside the domain.
pub trait Grid<const ND: usize, I, M>
where
    Self: Sized,
    Self::Coord: Field<ND, I>,
    Self::MaskContainer: Field<ND, M>,
    M: Mask,
{
    type Coord;
    type MaskContainer;

    /// Return a reference to the coordinate of the corresponding dimension
    fn get_coord(&self, dim: usize) -> &Self::Coord;
    /// Return a reference to the grid spacing in of the corrsponding dimension
    fn get_delta(&self, dim: usize) -> &Self::Coord;
    /// Return a reference to the mask field
    fn get_mask(&self) -> &Self::MaskContainer;
    /// Return a mutable reference to the mask field
    fn get_mask_mut(&mut self) -> &mut Self::MaskContainer;
    /// Set the mask field and return a mutable reference to the grid
    fn with_mask(&mut self, mask: Self::MaskContainer) -> &mut Self {
        *self.get_mask_mut() = mask;
        self
    }
    /// Return the shape of the grid, i.e. the number of grid points along each dimension
    fn size(&self) -> Shape<ND> {
        self.get_mask().shape()
    }
}

//// Defines the relative relation of multiple [Grids](`Grid`) for a curvilinear grid.
////
//// For example, the Arakawa C grid consists of four grids, one defining the center of a grid box,
//// two define the mid-point of the "vertical" and "horizontal" faces and the fourth are the corners
//// of the grid box.
// pub trait GridTopology<const ND: usize, I, M>
// where
//     Self::Grid: Grid<ND, I, M>,
//     M: Mask,
// {
//     type Grid;
//     /// Return an [Rc] smart pointer to the grid box centers
//     fn get_center(&self) -> Rc<Self::Grid>;

//     /// Return an [Rc] smart pointer to the grid box's face orthogonal to a vector pointing along dimension `dim`.
//     fn get_face(&self, dim: usize) -> Rc<Self::Grid>;

//     /// Return an [Rc] smart pointer to the grid box's edges parallel to a vector pointing along dimension 'dim'
//     fn get_edge(&self, dim: usize) -> Rc<Self::Grid>;

//     /// Return an [Rc] smart pointer to the grid box's corners
//     fn get_corner(&self) -> Rc<Self::Grid>;

//     /// Check if the face of a grid box at position `pos` is inside the domain
//     fn is_face_inside(
//         pos: Ix<ND>,
//         dim: usize,
//         center_mask: &<Self::Grid as Grid<ND, I, M>>::MaskContainer,
//     ) -> bool {
//         let ip1 = cyclic_shift(pos[dim], 1, center_mask.shape()[0]);
//         let new_pos = pos.clone();
//         new_pos[dim] = ip1;
//         center_mask[pos].is_inside() && center_mask[new_pos].is_inside()
//     }
//     /// Check if a corner of grid box at position `pos` is inside the domain
//     fn is_corner_inside(
//         pos: Ix<ND>,
//         center_mask: &<Self::Grid as Grid<ND, I, M>>::MaskContainer,
//     ) -> bool {
//         let shape = center_mask.shape();

//         // All neighboring grid boxes of a corned are obtained by iterating through all possible
//         // values of a bit mask of length ND. If the bit is set, the index is inreased by one, adhering
//         // to cyclic boundary conditions.
//         (0..((1 as usize) << ND))
//             .map(|i| {
//                 (0..ND)
//                     .map(|n| {
//                         if i & (1 << n) {
//                             cyclic_shift(pos[n], 1, shape[n])
//                         } else {
//                             pos[n]
//                         }
//                     })
//                     .collect::<Ix<ND>>()
//             })
//             .all(|idx| center_mask[pos].is_inside())
//     }

//     /// Create a mask field from a mask by evaluating the predicate function.
//     fn make_mask(
//         base_mask: &<Self::Grid as Grid<ND, I, M>>::MaskContainer,
//         predicate: fn(Ix<ND>, &<Self::Grid as Grid<ND, I, M>>::MaskContainer) -> bool,
//     ) -> <Self::Grid as Grid<ND, I, M>>::MaskContainer {
//         let mut mask =
//             <Self::Grid as Grid<ND, I, M>>::MaskContainer::full(M::inside(), base_mask.shape());
//         for j in 0..base_mask.shape().0 {
//             for i in 0..base_mask.shape().1 {
//                 mask[[j, i]] = match predicate([j, i], base_mask) {
//                     true => M::inside(),
//                     false => M::outside(),
//                 };
//             }
//         }
//         mask
//     }
// }

/// N-dimensional Grid
#[derive(Clone, Debug)]
pub struct GridND<const ND: usize, I, M> {
    coords: [ArrND<ND, I>; ND],
    delta: [ArrND<ND, I>; ND],
    mask: ArrND<ND, M>,
}

impl<const ND: usize, I, M> Grid<ND, I, M> for GridND<ND, I, M>
where
    I: Copy,
    M: Mask,
{
    type Coord = ArrND<ND, I>;
    type MaskContainer = ArrND<ND, M>;

    fn get_mask(&self) -> &Self::MaskContainer {
        &self.mask
    }

    fn get_mask_mut(&mut self) -> &mut Self::MaskContainer {
        &mut self.mask
    }

    fn get_coord(&self, dim: usize) -> &Self::Coord {
        debug_assert!(dim < ND);
        &self.coords[dim]
    }

    fn get_delta(&self, dim: usize) -> &Self::Coord {
        debug_assert!(dim < ND);
        &self.delta[dim]
    }
}

impl<const ND: usize, I, M> GridND<ND, I, M>
where
    I: Numeric,
    M: Mask,
{
    /// Return a Cartesian grid, i.e. a rectangular grid where the grid points are evenly spaced.
    pub fn cartesian(
        shape: impl IntoShape<ND>,
        start: [I; ND],
        delta: [I; ND],
        inside_domain: M,
    ) -> Self
    where
        I: std::fmt::Debug,
    {
        let shape = shape.into_shape();
        GridND {
            coords: (0..ND)
                .map(|dim| {
                    let mut res = ArrND::full(I::zero(), shape);
                    let start = start[dim];
                    let delta = delta[dim];
                    for ind in shape {
                        res[ind] = start + delta * (ind[dim] as f64)
                    }
                    res
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("Number of dimensions must match"), // {
            delta: delta.map(|delta| ArrND::full(delta, shape)),
            mask: {
                let mut res = ArrND::full(inside_domain, shape);
                shape.iter().for_each(|idx| {
                    res[idx] = if idx
                        .iter()
                        .zip(shape.as_ref().iter())
                        .any(|(&i, &ni)| (i == ni - 1) | (i == 0))
                    {
                        M::outside()
                    } else {
                        M::inside()
                    }
                });
                res
            },
        }
    }
}

//// Grid topology for curvilinear grids.
// #[derive(Debug)]
// pub struct StaggeredGrid<G> {
//     center: Rc<G>,
//     h_side: Rc<G>,
//     v_side: Rc<G>,
//     corner: Rc<G>,
// }

// impl<const ND: usize, I, M, G> GridTopology<ND, I, M> for StaggeredGrid<G>
// where
//     G: Grid<ND, I, M>,
//     M: Mask,
//     I: Numeric,
// {
//     type Grid = G;
//     fn get_center(&self) -> Rc<G> {
//         Rc::clone(&self.center)
//     }

//     fn get_h_face(&self) -> Rc<G> {
//         Rc::clone(&self.h_side)
//     }

//     fn get_v_face(&self) -> Rc<G> {
//         Rc::clone(&self.v_side)
//     }

//     fn get_corner(&self) -> Rc<G> {
//         Rc::clone(&self.corner)
//     }
// }

// impl<const ND: usize, I, M> StaggeredGrid<GridND<ND, I, M>>
// where
//     I: Numeric,
//     M: Mask,
// {
//     /// Return the topology of a Cartesian grid
//     pub fn cartesian(size: Shape<ND>, x_start: I, y_start: I, dx: I, dy: I) -> Self {
//         let x_shift = x_start + dx * 0.5;
//         let y_shift = y_start + dy * 0.5;
//         let center = GridND::cartesian(size, x_start, y_start, dx, dy);

//         let mut h_side = GridND::cartesian(size, x_shift, y_start, dx, dy);
//         *h_side.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_h_inside);
//         let h_side = Rc::new(h_side);

//         let mut v_side = GridND::cartesian(size, x_start, y_shift, dx, dy);
//         *v_side.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_v_inside);
//         let v_side = Rc::new(v_side);

//         let mut corner = GridND::cartesian(size, x_shift, y_shift, dx, dy);
//         *corner.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_corner_inside);
//         let corner = Rc::new(corner);

//         StaggeredGrid {
//             center: Rc::new(center),
//             h_side,
//             v_side,
//             corner,
//         }
//     }
// }

#[cfg(test)]
mod test {
    use crate::mask::{DomainMask, Mask};

    use super::{Grid, GridND};

    #[test]
    fn cartesian_grid_creation_set_correct_coords() {
        let g = GridND::cartesian([3, 3], [1f64, 0f64], [2.0, 3.0], DomainMask::inside());

        assert_eq!(g.get_coord(1)[[0, 0]], 0.0);
        assert_eq!(g.get_coord(1)[[1, 0]], 0.0);
        assert_eq!(g.get_coord(1)[[0, 1]], 3.0);
        assert_eq!(g.get_coord(1)[[0, 2]], 6.0);
        assert_eq!(g.get_coord(0)[[0, 0]], 1.0);
        assert_eq!(g.get_coord(0)[[1, 0]], 3.0);
        assert_eq!(g.get_coord(0)[[2, 2]], 5.0);
    }

    #[test]
    fn cartesian_grid_creation_set_correct_delta() {
        let g = GridND::cartesian([3, 3], [1f64, 0f64], [2.0, 3.0], DomainMask::inside());

        assert_eq!(g.get_delta(0)[[1, 1]], 2.0);
        assert_eq!(g.get_delta(1)[[1, 1]], 3.0);
    }
}
