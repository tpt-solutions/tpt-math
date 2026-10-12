//! Failure paths of the units-aware solvers. The success paths are covered by
//! the doc examples in `src/solve.rs`.

#![cfg(all(feature = "ode", feature = "optimize", feature = "autodiff"))]

use tpt_math_optimize_general::Options;
use tpt_math_units_dyn::solve::{
    dopri5_dimensioned, gradient_dimensioned, minimize_dimensioned, SolveError,
};
use tpt_math_units_dyn::{Dimension, DynQuantity, UnitError};

fn q(value: f64, dim: Dimension) -> DynQuantity {
    DynQuantity::from_base(value, dim)
}

#[test]
fn ode_rejects_derivative_with_the_wrong_dimension() {
    // y is a velocity, so y′ must be an acceleration. Returning a length is wrong.
    let result = dopri5_dimensioned(
        |_t, _y| Ok([q(1.0, Dimension::LENGTH)]),
        q(0.0, Dimension::TIME),
        q(1.0, Dimension::TIME),
        [q(0.0, Dimension::VELOCITY)],
        1e-8,
        1e-10,
        1000,
    );
    assert!(matches!(
        result,
        Err(SolveError::Unit(UnitError::DimensionMismatch { .. }))
    ));
}

#[test]
fn ode_rejects_end_time_with_a_different_dimension() {
    let result = dopri5_dimensioned(
        |_t, _y| Ok([q(0.0, Dimension::ACCELERATION)]),
        q(0.0, Dimension::TIME),
        q(1.0, Dimension::LENGTH),
        [q(0.0, Dimension::VELOCITY)],
        1e-8,
        1e-10,
        1000,
    );
    assert!(matches!(
        result,
        Err(SolveError::Unit(UnitError::DimensionMismatch { .. }))
    ));
}

#[test]
fn ode_propagates_an_error_from_the_user_function() {
    let failure = UnitError::UnknownUnit {
        name: "furlong".to_string(),
    };
    let result = dopri5_dimensioned(
        |_t, _y| Err(failure.clone()),
        q(0.0, Dimension::TIME),
        q(1.0, Dimension::TIME),
        [q(0.0, Dimension::VELOCITY)],
        1e-8,
        1e-10,
        1000,
    );
    assert_eq!(result, Err(SolveError::Unit(failure)));
}

#[test]
fn minimise_rejects_a_cost_whose_dimension_changes_between_points() {
    // (p − 3)² in m² while p ≤ 1 m, and the same square as a mass-valued cost above
    // 1 m. The first evaluation fixes the cost dimension as m². The optimum at 3 m
    // lies beyond 1 m, so the solver has to evaluate the other branch.
    let cost = |p: &[DynQuantity; 1]| {
        let d = p[0].value() - 3.0;
        if p[0].value() > 1.0 {
            Ok(q(d * d, Dimension::MASS))
        } else {
            Ok(q(d * d, Dimension::LENGTH.checked_powi(2).unwrap()))
        }
    };
    let result = minimize_dimensioned(cost, [q(0.0, Dimension::LENGTH)], &Options::new(200));
    assert!(matches!(
        result,
        Err(SolveError::Unit(UnitError::DimensionMismatch { .. }))
    ));
}

#[test]
fn gradient_dimension_is_cost_over_parameter() {
    let point = [q(2.0, Dimension::MASS), q(3.0, Dimension::VELOCITY)];
    let g = gradient_dimensioned(|x| x[0] * x[1] * x[1] * 0.5, &point, Dimension::ENERGY).unwrap();
    assert_eq!(
        *g[0].dim(),
        Dimension::ENERGY.checked_div(Dimension::MASS).unwrap()
    );
    assert_eq!(
        *g[1].dim(),
        Dimension::ENERGY.checked_div(Dimension::VELOCITY).unwrap()
    );
}
