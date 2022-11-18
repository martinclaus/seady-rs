use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::mask::Mask;
use crate::Numeric;
use crate::{field::Field, variable::Variable};

const AB2_FAC: [f64; 2] = [-1f64 / 2f64, 3f64 / 2f64];
const AB3_FAC: [f64; 3] = [5f64 / 12f64, -16f64 / 12f64, 23f64 / 12f64];

pub trait Integrator<V, I, M> {
    fn call(past_state: &VecDeque<&V>, step: I, idx: [usize; 2]) -> I;
}

pub struct Integrate<I> {
    _integrator_type: PhantomData<I>,
}

impl<IS> Integrate<IS> {
    pub fn compute_inc<V: Variable<I, M>, I, M: Mask>(
        past_state: VecDeque<&V>,
        step: I,
        out: &mut V,
    ) where
        IS: Integrator<V, I, M>,
        V: Variable<I, M>,
        I: Copy,
        M: Mask,
    {
        let d = out.get_data_mut();
        let (ny, nx) = d.size();

        for j in 0..ny {
            for i in 0..nx {
                d[[j, i]] = IS::call(&past_state, step, [j, i]);
            }
        }
    }
}

pub struct Ef;

impl<V, I, M> Integrator<V, I, M> for Ef
where
    V: Variable<I, M>,
    M: Mask,
    I: Numeric,
{
    fn call(past_state: &VecDeque<&V>, step: I, idx: [usize; 2]) -> I {
        past_state[0].get_data()[idx] * step
    }
}

pub struct Ab2;

impl<V, I, M> Integrator<V, I, M> for Ab2
where
    V: Variable<I, M>,
    M: Mask,
    I: Numeric,
{
    fn call(past_state: &VecDeque<&V>, step: I, idx: [usize; 2]) -> I {
        if past_state.len() < 2 {
            Ef::call(past_state, step, idx)
        } else {
            (past_state[0].get_data()[idx] * AB2_FAC[0]
                + past_state[1].get_data()[idx] * AB2_FAC[1])
                * step
        }
    }
}

pub struct Ab3;

impl<V, I, M> Integrator<V, I, M> for Ab3
where
    V: Variable<I, M>,
    M: Mask,
    I: Numeric,
{
    fn call(past_state: &VecDeque<&V>, step: I, idx: [usize; 2]) -> I {
        if past_state.len() < 3 {
            Ab2::call(past_state, step, idx)
        } else {
            (past_state[0].get_data()[idx] * AB3_FAC[0]
                + past_state[1].get_data()[idx] * AB3_FAC[1]
                + past_state[2].get_data()[idx] * AB3_FAC[2])
                * step
        }
    }
}
