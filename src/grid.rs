use std::rc::Rc;

use crate::field::{Arr2D, Field, Ix2, Size2D};
use crate::mask::Mask;
use crate::Numeric;

pub trait Grid<I, M>
where
    Self: Sized,
    Self::Coord: Field<I>,
    Self::MaskContainer: Field<M>,
    M: Mask,
{
    type Coord;
    type MaskContainer;

    fn get_x(&self) -> &Self::Coord;
    fn get_dx(&self) -> &Self::Coord;
    fn get_y(&self) -> &Self::Coord;
    fn get_dy(&self) -> &Self::Coord;
    fn get_mask(&self) -> &Self::MaskContainer;
    fn get_mask_mut(&mut self) -> &mut Self::MaskContainer;
    fn with_mask(&mut self, mask: Self::MaskContainer) -> &mut Self {
        *self.get_mask_mut() = mask;
        self
    }
    fn size(&self) -> Size2D {
        self.get_mask().size()
    }
}

pub trait GridTopology<I, M>
where
    Self::Grid: Grid<I, M>,
    M: Mask,
{
    type Grid;
    fn get_center(&self) -> Rc<Self::Grid>;

    fn get_h_face(&self) -> Rc<Self::Grid>;

    fn get_v_face(&self) -> Rc<Self::Grid>;

    fn get_corner(&self) -> Rc<Self::Grid>;

    // fn cartesian(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self;

    fn is_v_inside(pos: Ix2, center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool {
        let jp1 = (pos[0] + 1) % center_mask.size().0;
        center_mask[pos].is_inside() && center_mask[[jp1, pos[1]]].is_inside()
    }
    fn is_h_inside(pos: Ix2, center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool {
        let ip1 = (pos[1] + 1) % center_mask.size().1;
        center_mask[pos].is_inside() && center_mask[[pos[0], ip1]].is_inside()
    }
    fn is_corner_inside(pos: Ix2, center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool {
        let ip1 = (pos[1] + 1) % center_mask.size().1;
        let jp1 = (pos[0] + 1) % center_mask.size().0;
        center_mask[pos].is_inside()
            && center_mask[[pos[0], ip1]].is_inside()
            && center_mask[[jp1, ip1]].is_inside()
            && center_mask[[jp1, pos[1]]].is_inside()
    }

    fn make_mask(
        base_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
        is_inside_strategy: fn(Ix2, &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool,
    ) -> <Self::Grid as Grid<I, M>>::MaskContainer {
        let mut mask =
            <Self::Grid as Grid<I, M>>::MaskContainer::full(M::inside(), base_mask.size());
        for j in 0..base_mask.size().0 {
            for i in 0..base_mask.size().1 {
                mask[[j, i]] = match is_inside_strategy([j, i], base_mask) {
                    true => M::inside(),
                    false => M::outside(),
                };
            }
        }
        mask
    }
}

#[derive(Clone)]
pub struct Grid2D<I, M> {
    x: Arr2D<I>,
    y: Arr2D<I>,
    dx: Arr2D<I>,
    dy: Arr2D<I>,
    mask: Arr2D<M>,
}

impl<I, M> Grid<I, M> for Grid2D<I, M>
where
    I: Copy,
    M: Mask,
{
    type Coord = Arr2D<I>;
    type MaskContainer = Arr2D<M>;

    fn get_x(&self) -> &Self::Coord {
        &self.x
    }

    fn get_dx(&self) -> &Self::Coord {
        &self.dx
    }

    fn get_y(&self) -> &Self::Coord {
        &self.y
    }

    fn get_dy(&self) -> &Self::Coord {
        &self.dy
    }

    fn get_mask(&self) -> &Self::MaskContainer {
        &self.mask
    }

    fn get_mask_mut(&mut self) -> &mut Self::MaskContainer {
        &mut self.mask
    }
}

impl<I, M> Grid2D<I, M>
where
    I: Numeric,
    M: Mask,
{
    pub fn cartesian(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self {
        Grid2D {
            x: {
                let mut res = Arr2D::full(I::zero(), size);
                (0..size.0).for_each(|j| {
                    (0..size.1).for_each(|i| {
                        res[[j, i]] = x_start + dx * (i as f64);
                    })
                });
                res
            },
            y: {
                let mut res = Arr2D::full(I::zero(), size);
                (0..size.0).for_each(|j| {
                    (0..size.1).for_each(|i| {
                        res[[j, i]] = y_start + dy * (j as f64);
                    })
                });
                res
            },
            dx: Arr2D::full(dx, size),
            dy: Arr2D::full(dy, size),
            mask: {
                let mut res = Arr2D::full(M::inside(), size);
                (0..size.0).for_each(|j| {
                    (0..size.1).for_each(|i| {
                        res[[j, i]] =
                            match (j == 1) | (j == size.0 - 1) | (i == 0) | (i == size.1 - 1) {
                                true => M::outside(),
                                false => M::inside(),
                            };
                    })
                });
                res
            },
        }
    }
}

pub struct StaggeredGrid<G> {
    center: Rc<G>,
    h_side: Rc<G>,
    v_side: Rc<G>,
    corner: Rc<G>,
}

impl<I, M, G> GridTopology<I, M> for StaggeredGrid<G>
where
    G: Grid<I, M>,
    M: Mask,
    I: Numeric,
{
    type Grid = G;
    fn get_center(&self) -> Rc<G> {
        Rc::clone(&self.center)
    }

    fn get_h_face(&self) -> Rc<G> {
        Rc::clone(&self.h_side)
    }

    fn get_v_face(&self) -> Rc<G> {
        Rc::clone(&self.v_side)
    }

    fn get_corner(&self) -> Rc<G> {
        Rc::clone(&self.corner)
    }
}

impl<I, M> StaggeredGrid<Grid2D<I, M>>
where
    I: Numeric,
    M: Mask,
{
    pub fn cartesian(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self {
        let x_shift = x_start + dx * 0.5;
        let y_shift = y_start + dy * 0.5;
        let center = Grid2D::cartesian(size, x_start, y_start, dx, dy);

        let mut h_side = Grid2D::cartesian(size, x_shift, y_start, dx, dy);
        *h_side.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_h_inside);
        let h_side = Rc::new(h_side);

        let mut v_side = Grid2D::cartesian(size, x_start, y_shift, dx, dy);
        *v_side.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_v_inside);
        let v_side = Rc::new(v_side);

        let mut corner = Grid2D::cartesian(size, x_shift, y_shift, dx, dy);
        *corner.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_corner_inside);
        let corner = Rc::new(corner);

        StaggeredGrid {
            center: Rc::new(center),
            h_side,
            v_side,
            corner,
        }
    }
}
