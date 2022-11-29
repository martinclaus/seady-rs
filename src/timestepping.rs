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

use crate::field::{Field, Ix};
use crate::variable::Variable;
use crate::Numeric;

/// Coefficients of the 2nd order Adams-Bashforth scheme
const AB2_FAC: [f64; 2] = [-1f64 / 2f64, 3f64 / 2f64];

/// Coefficients of the 3rd order Adams-Bashforth scheme
const AB3_FAC: [f64; 3] = [5f64 / 12f64, -16f64 / 12f64, 23f64 / 12f64];

/// Provides the necessary interface for integration strategies.
pub trait Integrator<const ND: usize, V>
where
    V: Variable<ND>,
{
    /// Take (previous) evaluations of the right-hand side of the PDE and compute the
    /// numerical approximation of the derivative multiplied by the step size.
    /// The computation is at a single grid point.
    fn call(
        past_rhs: &VecDeque<&V>,
        step: <<V as Variable<ND>>::Data as Field<ND>>::Item,
        idx: Ix<ND>,
    ) -> <<V as Variable<ND>>::Data as Field<ND>>::Item;
}

/// Generic integrator that statically dispatch about the integration strategy.
pub struct Integrate<I> {
    _integrator_type: PhantomData<I>,
}

impl<IS> Integrate<IS> {
    /// Compute the change from one timestep to the next.
    pub fn compute_inc<const ND: usize, V>(
        past_state: VecDeque<&V>,
        step: <<V as Variable<ND>>::Data as Field<ND>>::Item,
        out: &mut V,
    ) where
        IS: Integrator<ND, V>,
        V: Variable<ND>,
        <V::Data as Field<ND>>::Item: Copy,
    {
        let d = out.get_data_mut();
        let shape = d.shape();

        for idx in shape {
            d[idx] = IS::call(&past_state, step, idx);
        }
    }
}

/// Euler forward scheme.
pub struct Ef;

impl<const ND: usize, V> Integrator<ND, V> for Ef
where
    V: Variable<ND>,
    <V::Data as Field<ND>>::Item: Numeric,
{
    fn call(
        past_state: &VecDeque<&V>,
        step: <<V as Variable<ND>>::Data as Field<ND>>::Item,
        idx: Ix<ND>,
    ) -> <<V as Variable<ND>>::Data as Field<ND>>::Item {
        past_state[0].get_data()[idx] * step
    }
}

/// Adams-Bashforth 2nd order scheme
pub struct Ab2;

impl<const ND: usize, V> Integrator<ND, V> for Ab2
where
    V: Variable<ND>,
    <V::Data as Field<ND>>::Item: Numeric,
{
    fn call(
        past_state: &VecDeque<&V>,
        step: <<V as Variable<ND>>::Data as Field<ND>>::Item,
        idx: Ix<ND>,
    ) -> <<V as Variable<ND>>::Data as Field<ND>>::Item {
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

impl<const ND: usize, V> Integrator<ND, V> for Ab3
where
    V: Variable<ND>,
    <V::Data as Field<ND>>::Item: Numeric,
{
    fn call(
        past_state: &VecDeque<&V>,
        step: <<V as Variable<ND>>::Data as Field<ND>>::Item,
        idx: Ix<ND>,
    ) -> <<V as Variable<ND>>::Data as Field<ND>>::Item {
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
