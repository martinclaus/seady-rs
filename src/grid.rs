//! Basic traits and data structures to describe [Grids](`Grid`).

use std::rc::Rc;

use crate::field::{ArrND, Field, IntoShape, Ix, Shape};
use crate::mask::Mask;
use crate::Numeric;

/// Trait defining the interface of grid-like structs.
///
/// A grid defines the coordinates of a [`Field`], e.g. the latitude or longitude, and
/// a mask to flag grid points, e.g. to be inside or outside the domain.
pub trait Grid<const ND: usize>
where
    Self: Sized,
{
    type MaskValue: Mask;
    type Coord: Field<ND>;
    type MaskContainer: Field<ND, Item = Self::MaskValue>;

    /// Return a reference to the coordinate of the corresponding dimension
    fn get_coord(&self, dim: usize) -> &Self::Coord;
    /// Return a reference to the grid spacing in of the corrsponding dimension
    fn get_delta(&self, dim: usize) -> &Self::Coord;
    /// Return a reference to the mask field
    fn get_mask(&self) -> &Self::MaskContainer;
    /// Return a mutable reference to the mask field
    fn get_mask_mut(&mut self) -> &mut Self::MaskContainer;
    /// Set the mask field and return a mutable reference to the grid
    fn set_mask(&mut self, mask: Self::MaskContainer) {
        *self.get_mask_mut() = mask;
    }
    /// Return the shape of the grid, i.e. the number of grid points along each dimension
    fn shape(&self) -> Shape<ND> {
        self.get_mask().shape()
    }
}

/// Defines the relative relation of multiple [Grids](`Grid`) for a curvilinear grid.
///
/// For example, the Arakawa C grid consists of four grids, one defining the center of a grid box,
/// two define the mid-point of the "vertical" and "horizontal" faces and the fourth are the corners
/// of the grid box.
pub trait GridTopology<const ND: usize> {
    type Grid: Grid<ND>;
    /// Return an [Rc] smart pointer to the grid box centers
    fn get_center(&self) -> Rc<Self::Grid>;

    /// Return an [Rc] smart pointer to the grid box's face orthogonal to a vector pointing along dimension `dim`.
    fn get_face(&self, dim: usize) -> Rc<Self::Grid>;

    /// Return an [Rc] smart pointer to the grid box's corners
    fn get_corner(&self) -> Rc<Self::Grid>;

    /// Check if the face of a grid box at position `pos` is inside the domain
    fn is_face_inside(
        pos: Ix<ND>,
        dim: usize,
        center_mask: &<Self::Grid as Grid<ND>>::MaskContainer,
        cyclic_dims: usize,
    ) -> bool {
        let new_pos = if dim >= (ND - cyclic_dims) {
            pos.cshift(dim, 1, center_mask.shape())
        } else {
            pos.clone()
        };
        center_mask[pos].is_inside() && center_mask[new_pos].is_inside()
    }
    /// Check if a corner of grid box at position `pos` is inside the domain
    fn is_corner_inside(
        pos: Ix<ND>,
        center_mask: &<Self::Grid as Grid<ND>>::MaskContainer,
        cyclic_dims: usize,
    ) -> bool {
        let shape = center_mask.shape();

        // All neighboring grid boxes of a corner are obtained by iterating through all possible
        // values of a bit mask of length ND. If the bit is set, the index is inreased by one, adhering
        // to cyclic boundary conditions for the second right-most dimensions.
        (0..((1 as usize) << ND))
            .map(|i| {
                let mut idx = pos.clone();
                for n in 0..ND {
                    if ((i & (1 << n)) != 0) & (n >= ND - cyclic_dims) {
                        idx = idx.cshift(n, 1, shape);
                    }
                }
                idx
            })
            .all(|idx| center_mask[idx].is_inside())
    }

    /// Create a mask field from a mask by evaluating the predicate function.
    fn make_mask<F>(
        base_mask: &<Self::Grid as Grid<ND>>::MaskContainer,
        predicate: F,
    ) -> <Self::Grid as Grid<ND>>::MaskContainer
    where
        F: Fn(Ix<ND>, &<Self::Grid as Grid<ND>>::MaskContainer) -> bool,
    {
        let mut mask = <Self::Grid as Grid<ND>>::MaskContainer::full(
            <Self::Grid as Grid<ND>>::MaskValue::inside(),
            base_mask.shape(),
        );
        for ind in base_mask.shape() {
            mask[ind] = if predicate(ind, base_mask) {
                <Self::Grid as Grid<ND>>::MaskValue::inside()
            } else {
                <Self::Grid as Grid<ND>>::MaskValue::outside()
            };
        }
        mask
    }
}

/// N-dimensional Grid
#[derive(Clone, Debug)]
pub struct GridND<const ND: usize, I, M> {
    coords: [ArrND<ND, I>; ND],
    delta: [ArrND<ND, I>; ND],
    mask: ArrND<ND, M>,
}

impl<const ND: usize, I, M> Grid<ND> for GridND<ND, I, M>
where
    I: Copy,
    M: Mask,
{
    type MaskValue = M;
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

/// Grid topology for curvilinear finite volume grids.
#[derive(Debug)]
pub struct FiniteVolumeGrid<const ND: usize, G> {
    center: Rc<G>,
    face: [Rc<G>; ND],
    corner: Rc<G>,
}

impl<const ND: usize, G> GridTopology<ND> for FiniteVolumeGrid<ND, G>
where
    G: Grid<ND>,
{
    type Grid = G;
    fn get_center(&self) -> Rc<G> {
        self.center.clone()
    }

    fn get_corner(&self) -> Rc<G> {
        self.corner.clone()
    }

    fn get_face(&self, dim: usize) -> Rc<Self::Grid> {
        self.face[dim].clone()
    }
}

/// Builder for [`FiniteVolumeGrid`] types
pub struct FiniteVolumeGridBuilder<const ND: usize, G>
where
    G: Grid<ND>,
{
    shape: Shape<ND>,
    center_coords: Vec<G::Coord>,
    center_delta: Vec<G::Coord>,
    center_mask: Option<G::MaskContainer>,
}

impl<const ND: usize, I, M> FiniteVolumeGridBuilder<ND, GridND<ND, I, M>>
where
    I: Copy,
    M: Mask,
{
    /// Build [`FiniteVolumeGrid`] with given `shape`.
    ///
    /// This is the first method in the build chain.
    pub fn shape(shape: impl IntoShape<ND>) -> Self {
        Self {
            shape: shape.into_shape(),
            center_coords: Vec::with_capacity(ND),
            center_delta: Vec::with_capacity(ND),
            center_mask: None,
        }
    }

    /// Define Cartesian coordinates, i.e. evenly sized grid boxes.
    pub fn cartesian_coordinates(self, start: [I; ND], delta: [I; ND]) -> Self
    where
        I: Numeric,
    {
        let shape = self.shape;
        Self {
            shape: shape,
            center_coords: (0..ND)
                .map(|dim| {
                    let mut res = <GridND<ND, I, M> as Grid<ND>>::Coord::full(I::zero(), shape);
                    let start = start[dim];
                    let delta = delta[dim];
                    for ind in shape.into_shape() {
                        res[ind] = start + delta * (ind[dim] as f64)
                    }
                    res
                })
                .collect(),
            center_delta: delta
                .iter()
                .map(|&d| <GridND<ND, I, M> as Grid<ND>>::Coord::full(d, shape))
                .collect(),
            center_mask: None,
        }
    }

    /// Define mask by providing a closure that takes an index array as argument and returns the mask value
    pub fn mask<F>(self, predicate: F) -> Self
    where
        F: Fn(Ix<ND>) -> M,
    {
        let mut mask = <GridND<ND, I, M> as Grid<ND>>::MaskContainer::full(M::inside(), self.shape);
        for ind in self.shape {
            mask[ind] = predicate(ind)
        }
        Self {
            center_mask: Some(mask),
            ..self
        }
    }

    /// Build the [`FiniteVolumeGrid`] object. If no grid is defined, a default grid with closed boundaries will be specified.
    ///
    /// `closed_boundaries` corresponds to the number of right-most dimensions which have closed boundaries on both ends.
    /// E.g., for an ocean model, `ND=3` and `closed_boundary=2` means that the horizonal dimensions (the right-most two dimensions)
    /// and ocean bottom are closed.
    pub fn build(self, closed_boundaries: usize) -> FiniteVolumeGrid<ND, GridND<ND, I, M>>
    where
        I: Numeric,
        M: std::fmt::Debug,
    {
        assert_eq!(self.center_coords.len(), ND);

        match self.center_mask {
            None => {
                let shape = self.shape.clone();
                return self
                    .mask(|idx| default_mask::<ND, M>(idx, shape, closed_boundaries))
                    .build(closed_boundaries);
            }
            Some(center_mask) => {
                let Self {
                    shape,
                    center_coords,
                    center_delta,
                    center_mask: _,
                } = self;

                let center = Rc::new(GridND {
                    coords: center_coords.try_into().expect("Dimensions should match"),
                    delta: center_delta.try_into().expect("Dimensions should match"),
                    mask: center_mask,
                });

                let delta = &center.delta;

                let face = (0..ND)
                    .map(|ndim| {
                        let mut face_grid = center.as_ref().clone();
                        let coord = &mut face_grid.coords[ndim];
                        for idx in shape {
                            coord[idx] = coord[idx] + delta[ndim][idx] * 0.5;
                        }
                        *face_grid.get_mask_mut() =
                            FiniteVolumeGrid::<ND, GridND<ND, I, M>>::make_mask(
                                center.get_mask(),
                                |pos, mask| {
                                    FiniteVolumeGrid::<ND, GridND<ND, I, M>>::is_face_inside(
                                        pos,
                                        ndim,
                                        mask,
                                        closed_boundaries,
                                    )
                                },
                            );
                        Rc::new(face_grid)
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("Should coerce to array");

                let corner = Rc::new({
                    let mut corner_grid = center.as_ref().clone();
                    for ndim in 0..ND {
                        let coord = &mut corner_grid.coords[ndim];
                        for idx in shape {
                            coord[idx] = coord[idx] + delta[ndim][idx] * 0.5;
                        }
                    }
                    *corner_grid.get_mask_mut() =
                        FiniteVolumeGrid::<ND, GridND<ND, I, M>>::make_mask(
                            center.get_mask(),
                            |pos, mask| {
                                FiniteVolumeGrid::<ND, GridND<ND, I, M>>::is_corner_inside(
                                    pos,
                                    mask,
                                    closed_boundaries,
                                )
                            },
                        );
                    corner_grid
                });

                FiniteVolumeGrid {
                    center,
                    face,
                    corner,
                }
            }
        }
    }
}

/// Default mask.
fn default_mask<const ND: usize, M>(idx: Ix<ND>, shape: Shape<ND>, closed_boundaries: usize) -> M
where
    M: Mask,
{
    if idx.iter().zip(shape.as_ref().iter()).enumerate().any(
        |(ndim, (&i, &ni))| {
            (i==0)                               // close boundary at lowest index
        | ((i == ni - 1) & (ndim >= ND - closed_boundaries))
        }, // closed bounday for largest index of two right-most dimensions (typically horizontal dimensions)
    ) {
        M::outside()
    } else {
        M::inside()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        field::IntoShape,
        mask::{DomainMask, Mask},
    };

    use super::{FiniteVolumeGridBuilder, Grid, GridND, GridTopology};

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

    #[test]
    fn cartesian_grid_set_correct_mask() {
        let shape = [4, 4];
        let g = GridND::cartesian(shape, [1f64, 0f64], [2.0, 3.0], DomainMask::inside());

        for j in 1..shape[0] - 1 {
            for i in 1..shape[1] - 1 {
                if (i == 0) | (j == 0) | (i == shape[1] - 1) | (j == shape[0] - 1) {
                    // check boundaries
                    assert!(g.get_mask()[[j, i]].is_outside())
                } else {
                    // check interior
                    assert!(g.get_mask()[[j, i]].is_inside())
                }
            }
        }
    }

    #[test]
    fn cartesian_staggered_grid_set_correct_default_center_mask() {
        let shape = [4, 4, 4];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates([0.0, 1.0, 2.0], [1., 2., 3.])
            .build(2);

        let g: std::rc::Rc<GridND<3, f64, DomainMask>> = sg.get_center();

        println!("{}", g.get_mask());

        for idx in shape.into_shape() {
            if idx.iter().any(|&i| i == 0)
                | idx
                    .iter()
                    .enumerate()
                    .any(|(ndim, &i)| (i == shape[ndim] - 1) & (ndim >= 1))
            {
                assert!(g.get_mask()[idx].is_outside());
            } else {
                assert!(g.get_mask()[idx].is_inside());
            }
        }
    }

    #[test]
    fn cartesian_staggered_grid_set_correct_corner_mask() {
        let shape = [4, 4, 4];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates([0.0, 1.0, 2.0], [1., 2., 3.])
            .build(2);

        let g: std::rc::Rc<GridND<3, f64, DomainMask>> = sg.get_corner();

        println!("{}", g.get_mask());

        for idx in shape.into_shape() {
            if idx.iter().any(|&i| i == 0)
                | idx
                    .iter()
                    .enumerate()
                    .any(|(ndim, &i)| (i >= shape[ndim] - 2) & (ndim >= 1))
            {
                assert!(g.get_mask()[idx].is_outside());
            } else {
                assert!(g.get_mask()[idx].is_inside());
            }
        }
    }

    #[test]
    fn cartesian_staggered_grid_set_correct_face_mask() {
        let shape = [4, 4, 4];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates([0.0, 1.0, 2.0], [1., 2., 3.])
            .build(2);

        for face_dim in 0..shape.len() {
            let g: std::rc::Rc<GridND<3, f64, DomainMask>> = sg.get_face(face_dim);
            println!("{}, {}\n", face_dim, g.get_mask());
            for idx in shape.into_shape() {
                if idx.iter().any(|&i| i == 0)
                    | idx.iter().enumerate().any(|(dim, &i)| {
                        if dim >= 3 - 2 {
                            if dim == face_dim {
                                i >= shape[dim] - 2
                            } else {
                                i >= shape[dim] - 1
                            }
                        } else {
                            (dim == face_dim) & (i >= shape[dim] - 2) & (face_dim >= 1)
                        }
                    })
                {
                    println!("True");
                    assert!(g.get_mask()[idx].is_outside());
                } else {
                    println!("False");
                    assert!(g.get_mask()[idx].is_inside());
                }
            }
        }
    }

    #[test]
    fn cartesian_staggered_grid_set_correct_face_coords() {
        let shape = [4, 4, 4];
        let delta = [1., 2., 3.];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates([0.0, 1.0, 2.0], [1., 2., 3.])
            .build(2);
        let center: std::rc::Rc<GridND<3, f64, DomainMask>> = sg.get_center();
        for face_dim in 0..shape.len() {
            let g = sg.get_face(face_dim);
            for dim in 0..shape.len() {
                if dim != face_dim {
                    for idx in shape.into_shape() {
                        assert_eq!(center.get_coord(dim)[idx], g.get_coord(dim)[idx]);
                    }
                } else {
                    for idx in shape.into_shape() {
                        assert_eq!(
                            g.get_coord(dim)[idx],
                            center.get_coord(dim)[idx] + delta[dim] * 0.5
                        )
                    }
                }
            }
        }
    }
    #[test]
    fn cartesian_staggered_grid_set_correct_center_coords() {
        let shape = [4, 4, 4];
        let delta = [1., 2., 3.];
        let start = [0.0, 1.0, 1.0];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates(start, delta)
            .build(2);
        let g: std::rc::Rc<GridND<3, f64, DomainMask>> = sg.get_center();
        for dim in 0..shape.len() {
            for idx in shape.into_shape() {
                assert_eq!(
                    g.get_coord(dim)[idx],
                    start[dim] + delta[dim] * (idx[dim] as f64)
                )
            }
        }
    }
    #[test]
    fn cartesian_staggered_grid_set_correct_corner_coords() {
        let shape = [4, 4, 4];
        let delta = [1., 2., 3.];
        let start = [0.0, 1.0, 1.0];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates(start, delta)
            .build(2);
        let g: std::rc::Rc<GridND<3, f64, DomainMask>> = sg.get_corner();
        for dim in 0..shape.len() {
            for idx in shape.into_shape() {
                assert_eq!(
                    g.get_coord(dim)[idx],
                    start[dim] + delta[dim] * (idx[dim] as f64 + 0.5)
                )
            }
        }
    }

    #[test]
    fn cartesian_staggered_grid_set_correct_mask() {
        let shape = [4, 4, 4];
        let delta = [1., 2., 3.];
        let start = [0.0, 1.0, 1.0];
        let sg = FiniteVolumeGridBuilder::shape(shape)
            .cartesian_coordinates(start, delta)
            .mask(|idx| {
                if idx[0] < 2 {
                    DomainMask::inside()
                } else {
                    DomainMask::outside()
                }
            })
            .build(2);

        let g = sg.get_center();

        for idx in shape.into_shape() {
            assert_eq!(g.get_mask()[idx].is_inside(), idx[0] < 2)
        }
    }
}
