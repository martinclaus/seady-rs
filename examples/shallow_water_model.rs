use fixed_map::Key;
use seady::{
    field::{cyclic_shift, ArrND, Field, IntoShape},
    grid::{FiniteVolumeGridBuilder, Grid, GridND, GridTopology},
    mask::{DomainMask, Mask},
    state::{State, StateDeque, StateFactory, VarKey},
    timestepping::{Ab3, Integrate},
    variable::{Var, Variable},
};

/// Prognostic variables of the shallow water equations
#[derive(Copy, Clone, Key, Debug)]
pub enum SwmVars {
    U,
    V,
    ETA,
}
impl VarKey for SwmVars {}

type S = State<SwmVars, Var<ArrND<2, f64>, GridND<2, f64, DomainMask>>>;

fn main() {
    let shape = [100, 100].into_shape();

    let dt = 0.05;

    let grid = FiniteVolumeGridBuilder::shape(shape)
        .cartesian_coordinates([0., 0.], [1., 1.])
        .mask(|idx| {
            if (idx[0] == 0) | (idx[1] == 0) | (idx[0] == shape[0] - 1) | (idx[1] == shape[1] - 1) {
                DomainMask::outside()
            } else {
                DomainMask::inside()
            }
        })
        .build(2);

    let mut state_factory = StateFactory::new([
        (SwmVars::ETA, grid.get_center()),
        (SwmVars::V, grid.get_face(0)),
        (SwmVars::U, grid.get_face(1)),
    ]);

    let mut state: S = state_factory.get();

    for idx in shape {
        state[SwmVars::U].get_data_mut()[idx] = 0.0;
        state[SwmVars::V].get_data_mut()[idx] = 0.0;
    }
    let eta = &mut state[SwmVars::ETA];
    {
        let eta_grid = eta.get_grid();
        let eta_x = eta_grid.get_coord(1);
        let eta_y = eta_grid.get_coord(1);
        for idx in shape {
            eta.get_data_mut()[idx] = ((-(eta_x[idx] - 99.0 / 2.0).powi(2)
                - (eta_y[idx] - 99.0 / 2.0).powi(2))
                / (20f64).powi(2))
            .exp()
        }
    }

    let mut rhs_eval = StateDeque::new(3);

    let now = std::time::Instant::now();
    for _ in 0..500 {
        rhs_eval.push(rhs(&state, state_factory.get()), Some(&mut state_factory));

        let mut inc = state_factory.get();
        for var in [SwmVars::U, SwmVars::V, SwmVars::ETA] {
            Integrate::<Ab3>::compute_inc(rhs_eval.get_var(var), dt, &mut inc[var]);
            state[var] += &inc[var];
        }
        state_factory.take(inc);
    }

    println!("Time: {} sec", now.elapsed().as_secs_f64());
}

fn rhs(state: &S, mut inc: S) -> S {
    pressure_gradient_i(state, &mut inc);
    pressure_gradient_j(state, &mut inc);
    divergence(state, &mut inc);
    inc
}

fn pressure_gradient_i(state: &S, inc: &mut S) {
    let u_grid = state[SwmVars::U].get_grid();
    let dx = u_grid.get_delta(1);
    let u_inc = inc[SwmVars::U].get_data_mut();
    let eta = state[SwmVars::ETA].get_data();
    let shape = eta.shape();
    for idx in shape {
        let ip1 = [idx[0], cyclic_shift(idx[1], 1, shape[1])];
        u_inc[idx] = (eta[ip1] - eta[idx]) / dx[idx];
    }
}

fn pressure_gradient_j(state: &S, inc: &mut S) {
    let grid_v = state[SwmVars::V].get_grid();
    let dy = grid_v.get_delta(0);
    let v_inc = inc[SwmVars::V].get_data_mut();
    let eta = state[SwmVars::ETA].get_data();
    let shape = eta.shape();
    for idx in shape {
        let jp1 = [cyclic_shift(idx[0], 1, shape[0]), idx[0]];
        v_inc[idx] = (eta[jp1] - eta[idx]) / dy[idx];
    }
}

fn divergence(state: &S, inc: &mut S) {
    let eta_grid = state[SwmVars::ETA].get_grid();
    let dx = eta_grid.get_delta(1);
    let dy = eta_grid.get_delta(0);
    let u = state[SwmVars::U].get_data();
    let v = state[SwmVars::V].get_data();
    let eta_inc = inc[SwmVars::ETA].get_data_mut();
    let shape = u.shape();
    for idx in shape {
        let im1 = [idx[0], cyclic_shift(idx[1], -1, shape[1])];
        let jm1 = [cyclic_shift(idx[0], -1, shape[0]), idx[0]];
        eta_inc[idx] = (u[idx] - u[im1]) / dx[idx] + (v[idx] - v[jm1]) / dy[idx];
    }
}
