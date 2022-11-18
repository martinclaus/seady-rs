use std::ops::{Add, AddAssign};
use std::rc::Rc;

use crate::field::Field;
use crate::grid::Grid;
use crate::mask::Mask;
use crate::Numeric;

pub trait Variable<I, M>
where
    Self::Data: Field<I>,
    Self::Grid: Grid<I, M>,
    M: Mask,
{
    type Data;
    type Grid;

    fn zeros(grid: &Rc<Self::Grid>) -> Self;

    fn get_data(&self) -> &Self::Data;

    fn get_data_mut(&mut self) -> &mut Self::Data;

    fn get_grid(&self) -> &Rc<Self::Grid>;
}

#[derive(Debug)]
pub struct Var<CD, G> {
    data: CD,
    grid: Rc<G>,
}

impl<CD, G> Add for Var<CD, G>
where
    CD: Add<Output = CD>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data + rhs.data,
            grid: Rc::clone(&self.grid),
        }
    }
}

impl<CD, G> AddAssign for Var<CD, G>
where
    CD: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data;
    }
}

impl<'a, CD, G> AddAssign<&'a Var<CD, G>> for Var<CD, G>
where
    CD: AddAssign<&'a CD>,
{
    fn add_assign(&mut self, rhs: &'a Var<CD, G>) {
        self.data += &rhs.data
    }
}

impl<CD, G, I, M> Variable<I, M> for Var<CD, G>
where
    CD: Field<I>,
    G: Grid<I, M>,
    I: Numeric,
    M: Mask,
{
    type Data = CD;
    type Grid = G;

    fn zeros(grid: &Rc<Self::Grid>) -> Self {
        Self {
            data: Self::Data::full(I::zero(), grid.size()),
            grid: Rc::clone(grid),
        }
    }

    fn get_data(&self) -> &Self::Data {
        &self.data
    }

    fn get_data_mut(&mut self) -> &mut Self::Data {
        &mut self.data
    }

    fn get_grid(&self) -> &Rc<Self::Grid> {
        &self.grid
    }
}