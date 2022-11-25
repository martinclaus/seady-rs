//! Time integration methods.
//!
//! Time integration can be abstracted as a two step process. First, an increment is computed
//! which is then added to some previous state to update it to the next time step.
//! Often, both steps are combined, but we keep them separated to allow for mixing different integration
//! strategies, i.e. to use different strategies for different terms in the set of PDE.
//!
//! The strategy pattern is used to specialize about the particular integration scheme.
//! A strategy must implement the [Integrator] trait. Integration itself is then performed by the
//! [Integrate] struct which statically dispatches with respect to the strategy.

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::field::Ix;
use crate::mask::Mask;
use crate::Numeric;
use crate::{field::Field, variable::Variable};

/// Coefficients of the 2nd order Adams-Bashforth scheme
const AB2_FAC: [f64; 2] = [-1f64 / 2f64, 3f64 / 2f64];

/// Coefficients of the 3rd order Adams-Bashforth scheme
const AB3_FAC: [f64; 3] = [5f64 / 12f64, -16f64 / 12f64, 23f64 / 12f64];

/// Provides the necessary interface for integration strategies.
pub trait Integrator<const ND: usize, V, I, M> {
    /// Take (previous) evaluations of the right-hand side of the PDE and compute the
    /// numerical approximation of the derivative multiplied by the step size.
    /// The computation is at a single grid point.
    fn call(past_rhs: &VecDeque<&V>, step: I, idx: Ix<ND>) -> I;
}

/// Generic integrator that statically dispatch about the integration strategy.
pub struct Integrate<I> {
    _integrator_type: PhantomData<I>,
}

impl<IS> Integrate<IS> {
    //// Compute the change from one timestep to the next.
    // pub fn compute_inc<const ND: usize, V, I, M>(past_state: VecDeque<&V>, step: I, out: &mut V)
    // where
    //     IS: Integrator<ND, V, I, M>,
    //     // I: Copy,
    //     M: Mask,
    //     V: Variable<ND, I, M>,
    // {
    //     let d = out.get_data_mut();
    //     let shape = d.shape();

    //     for j in 0..ny {
    //         for i in 0..nx {
    //             d[[j, i]] = IS::call(&past_state, step, [j, i]);
    //         }
    //     }
    // }
}

/// Euler forward scheme.
pub struct Ef;

impl<const ND: usize, V, I, M> Integrator<ND, V, I, M> for Ef
where
    V: Variable<ND, I, M>,
    M: Mask,
    I: Numeric,
{
    fn call(past_state: &VecDeque<&V>, step: I, idx: Ix<ND>) -> I {
        past_state[0].get_data()[idx] * step
    }
}

/// Adams-Bashforth 2nd order scheme
pub struct Ab2;

impl<const ND: usize, V, I, M> Integrator<ND, V, I, M> for Ab2
where
    V: Variable<ND, I, M>,
    M: Mask,
    I: Numeric,
{
    fn call(past_state: &VecDeque<&V>, step: I, idx: Ix<ND>) -> I {
        if past_state.len() < 2 {
            Ef::call(past_state, step, idx)
        } else {
            (past_state[0].get_data()[idx] * AB2_FAC[0]
                + past_state[1].get_data()[idx] * AB2_FAC[1])
                * step
        }
    }
}

/// Adams-Bashforth 3rd order scheme
pub struct Ab3;

impl<const ND: usize, V, I, M> Integrator<ND, V, I, M> for Ab3
where
    V: Variable<ND, I, M>,
    M: Mask,
    I: Numeric,
{
    fn call(past_state: &VecDeque<&V>, step: I, idx: Ix<ND>) -> I {
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
