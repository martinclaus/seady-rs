//! Solve the shallow water equations in 2D
//!
//! The Shallow water equations model the evolution of a shallow homogrneous layer of fluid.
//! In this example, we have solid boundaries enclosing the domain, an initial disturbance of
//! the fluid interface and zero visocsity.

use fixed_map::Key;
use seady::{
    field::{ArrND, Field, NumField},
    grid::{FiniteVolumeGridBuilder, Grid, GridND, GridTopology},
    mask::{DomainMask, Mask},
    state::{State, StateDeque, StateFactory, VarKey},
    timestepping::{Ab3, Integrate},
};

/// Prognostic variables of the shallow water equations
#[derive(Copy, Clone, Key, Debug)]
pub enum VAR {
    U,   // velocity along x-direction
    V,   // velocity along y-direction
    ETA, // interface displacement
}
impl VarKey for VAR {}

/// Type of the state object.
type S = State<VAR, ArrND<2, f64>, GridND<2, f64, DomainMask>>;

fn main() {
    // Shape of the domain
    let shape = [100, 100];

    // Time step
    let dt = 0.05;

    // Number of time steps to integrate
    let nt = 750;

    // Define cartesian finite volume grid enclosed by solid boundaries
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

    // Create state factory to produce states on an [Arakawa C-grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid)
    let mut state_factory = StateFactory::new([
        (VAR::ETA, grid.get_center()),
        (VAR::V, grid.get_face(0)),
        (VAR::U, grid.get_face(1)),
    ]);

    // Initialize a buffer of past evaluations of the right-hand side of the partial differential equations
    let mut rhs_eval = StateDeque::new(3);

    // Create initial state
    let mut state = state_factory.get();
    set_initial_conditions(&mut state);
    plot_state("examples/img/swm_initial_state.png", &state).expect("Plotting failed");

    // start time integration
    let now = std::time::Instant::now();
    for _ in 0..nt {
        // evaluate the right hand side of the equations and push it to the buffer.
        // Oldest buffer elements will be consumed by the state factory for latter reuse.
        rhs_eval.push(rhs(&state, state_factory.get()), Some(&mut state_factory));

        let mut inc = state_factory.get();
        for var in [VAR::U, VAR::V, VAR::ETA] {
            // compute increment of a state variable from one timestep to the next
            Integrate::<Ab3>::compute_inc(rhs_eval.get_data(var), dt, &mut inc[var]);
            // update the state variable
            state[var] += &inc[var];
        }
        state_factory.take(inc);
    }

    println!("Time: {} sec", now.elapsed().as_secs_f64());

    plot_state("examples/img/swm_final_state.png", &state).expect("Plotting failed");
}

fn set_initial_conditions(state: &mut S) {
    let shape = state[VAR::U].shape();
    for idx in shape {
        state[VAR::U][idx] = 0.0;
        state[VAR::V][idx] = 0.0;
    }
    let eta_grid = state.get_grid(VAR::ETA);
    let eta = &mut state[VAR::ETA];
    {
        let x = eta_grid.get_coord(1);
        let y = eta_grid.get_coord(0);
        for idx in shape {
            eta[idx] = ((-(x[idx] - 99.0 / 2.0).powi(2) - (y[idx] - 99.0 / 2.0).powi(2))
                / (20f64).powi(2))
            .exp()
        }
    }
}

fn rhs(state: &S, mut eval: S) -> S {
    pressure_gradient_i(state, &mut eval);
    pressure_gradient_j(state, &mut eval);
    divergence(state, &mut eval);
    eval
}

fn pressure_gradient_i(state: &S, eval: &mut S) {
    let u_grid = state.get_grid(VAR::U);
    let u_mask = u_grid.get_mask().clone();
    let dx = u_grid.get_delta(1).clone();
    let u_inc = &mut eval[VAR::U];
    let eta = state[VAR::ETA].clone();
    let shape = eta.shape();
    for idx in shape {
        u_inc[idx] = match u_mask[idx] {
            DomainMask::Outside => 0.0,
            DomainMask::Inside => {
                let ip1 = idx.cshift(1, 1, shape);
                (eta[ip1] - eta[idx]) / dx[idx]
            }
        }
    }
}

fn pressure_gradient_j(state: &S, eval: &mut S) {
    let v_grid = state.get_grid(VAR::V);
    let v_mask = v_grid.get_mask().clone();
    let dy = v_grid.get_delta(0).clone();
    let v_inc = &mut eval[VAR::V];
    let eta = state[VAR::ETA].clone();
    let shape = eta.shape();
    for idx in shape {
        v_inc[idx] = match v_mask[idx] {
            DomainMask::Outside => 0.0,
            DomainMask::Inside => {
                let jp1 = idx.cshift(0, 1, shape);
                (eta[jp1] - eta[idx]) / dy[idx]
            }
        }
    }
}

fn divergence(state: &S, eval: &mut S) {
    let eta_grid = state.get_grid(VAR::ETA);
    let eta_mask = eta_grid.get_mask().clone();
    let dx = eta_grid.get_delta(1).clone();
    let dy = eta_grid.get_delta(0).clone();
    let u = state[VAR::U].clone();
    let v = state[VAR::V].clone();
    let eta_inc = &mut eval[VAR::ETA];
    let shape = u.shape();
    for idx in shape {
        eta_inc[idx] = match eta_mask[idx] {
            DomainMask::Outside => 0.0,
            DomainMask::Inside => {
                let im1 = idx.cshift(1, -1, shape);
                let jm1 = idx.cshift(0, -1, shape);
                (u[idx] - u[im1]) / dx[idx] + (v[idx] - v[jm1]) / dy[idx]
            }
        }
    }
}

fn plot_state(file: &str, state: &S) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    let root = BitMapBackend::new(file, (1024, 360)).into_drawing_area();

    root.fill(&WHITE)?;
    let root = root.margin(5, 5, 5, 5);

    let subplots = root.split_evenly((1, 3));

    let _ = subplots
        .iter()
        .zip([VAR::ETA, VAR::U, VAR::V])
        .map(|(ax, var)| {
            let ax = ax.titled(
                match var {
                    VAR::U => "u",
                    VAR::V => "v",
                    VAR::ETA => "eta",
                },
                ("sans-serif", 22).into_font().color(&BLACK.mix(0.8)),
            )?;
            let x = state.get_grid(var).get_coord(1).clone();
            let y = state.get_grid(var).get_coord(0).clone();
            let dx = state.get_grid(var).get_delta(1).clone();
            let dy = state.get_grid(var).get_delta(0).clone();
            let mask = state.get_grid(var).get_mask().clone();
            let value = state[var].clone();
            let scale = value
                .min()
                .abs()
                .max(value.max().abs())
                .max(f64::EPSILON * 10.0);
            let mut chart = ChartBuilder::on(&ax)
                .margin(20)
                .x_label_area_size(10)
                .y_label_area_size(10)
                .build_cartesian_2d(x[[0, 0]]..x[[0, 99]], y[[0, 0]]..y[[99, 0]])?;

            chart
                .configure_mesh()
                .disable_x_mesh()
                .disable_y_mesh()
                .draw()?;

            chart.draw_series(mask.shape().iter().map(|idx| {
                Rectangle::new(
                    [
                        (x[idx] - dx[idx] * 0.5, y[idx] + dy[idx] * 0.5),
                        (x[idx] + dx[idx] * 0.5, y[idx] - dy[idx] * 0.5),
                    ],
                    {
                        let col = colorous::RED_BLUE
                            .eval_continuous(1.0 - (value[idx] + scale) / (2.0 * scale));
                        RGBColor(col.r, col.g, col.b).filled()
                    },
                )
            }))?;
            Ok(())
        })
        .collect::<Vec<Result<(), Box<dyn std::error::Error>>>>();

    root.present()?;
    Ok(())
}
