//! Units-aware solvers: ODE integration, minimisation and gradients on values
//! that carry runtime dimensions.
//!
//! The numerical cores in `tpt-math-numeric`, `tpt-math-optimize-general` and
//! `tpt-math-autodiff-rev` work on plain `f64`. This module puts a dimension
//! check at each boundary instead of inside the arithmetic:
//!
//! * the inputs are [`DynQuantity`] values, whose dimensions are fixed up front;
//! * each call to the user's function is checked against the dimension the
//!   result must have, and a mismatch stops the solve with a [`SolveError`];
//! * the solver itself runs on SI base-unit values, and the results come back
//!   as [`DynQuantity`] values with their dimensions attached.
//!
//! The derivative of a `Y`-dimensioned quantity with respect to a `T`-dimensioned
//! one has dimension `Y / T`, and the gradient of a cost with respect to a
//! parameter has dimension `cost / parameter`. These are the dimensions the
//! checks enforce.
//!
//! The checks run on the values the user's function returns. The autodiff path
//! attaches dimensions to gradients it computes from a tape, but it cannot check
//! the function written on the tape, so that check is the caller's job.

#[cfg(any(feature = "ode", feature = "optimize"))]
use core::cell::RefCell;

use crate::UnitError;
#[cfg(any(feature = "ode", feature = "optimize", feature = "autodiff"))]
use crate::{Dimension, DynQuantity};

/// Why a units-aware solve stopped.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    /// A dimension check failed, either on an input or on a value the user's
    /// function returned.
    Unit(UnitError),
    /// The ODE integrator failed.
    #[cfg(feature = "ode")]
    Ode(tpt_math_numeric::ode::OdeError),
    /// The optimiser rejected its input, or the cost was not finite.
    #[cfg(feature = "optimize")]
    Optimize(String),
}

impl From<UnitError> for SolveError {
    fn from(e: UnitError) -> Self {
        SolveError::Unit(e)
    }
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SolveError::Unit(e) => write!(f, "unit error: {e}"),
            #[cfg(feature = "ode")]
            SolveError::Ode(e) => write!(f, "ODE solver failed: {e}"),
            #[cfg(feature = "optimize")]
            SolveError::Optimize(msg) => write!(f, "optimiser failed: {msg}"),
        }
    }
}

impl std::error::Error for SolveError {}

#[cfg(any(feature = "ode", feature = "optimize"))]
/// Check that `actual` has the dimension `expected`, naming `context` on failure.
fn expect_dim(
    context: &'static str,
    expected: Dimension,
    actual: &DynQuantity,
) -> Result<(), UnitError> {
    if *actual.dim() == expected {
        Ok(())
    } else {
        Err(UnitError::DimensionMismatch {
            context,
            expected,
            actual: *actual.dim(),
        })
    }
}

#[cfg(any(feature = "ode", feature = "autodiff"))]
/// The dimension `a / b`, or an overflow error if an exponent leaves its range.
fn dim_div(context: &'static str, a: Dimension, b: Dimension) -> Result<Dimension, UnitError> {
    a.checked_div(b)
        .ok_or(UnitError::ExponentOverflow { context })
}

/// Integrate `y′ = f(t, y)` from `t0` to `t1` with adaptive Dormand–Prince 5(4),
/// with every quantity dimension-checked.
///
/// `t0` and `t1` must share a dimension `T`. Each `y0[i]` has dimension `Yᵢ`, and
/// `f` must return a derivative whose `i`th component has dimension `Yᵢ / T`.
/// The result has the dimensions of `y0`.
///
/// `rtol` and `atol` are SI-valued tolerances, as in the numeric crate. A
/// violated dimension, at the start or on any call to `f`, stops the solve with
/// [`SolveError::Unit`].
///
/// # Errors
///
/// Returns [`SolveError::Unit`] for a dimension mismatch, or [`SolveError::Ode`]
/// if the integrator runs out of steps or underflows.
///
/// # Examples
///
/// ```
/// use tpt_math_units_dyn::solve::dopri5_dimensioned;
/// use tpt_math_units_dyn::{DynQuantity, Dimension};
///
/// // Free fall: v′ = −g with g = 9.81 m/s². Start at rest, so v(2 s) = −19.62 m/s.
/// let t0 = DynQuantity::from_base(0.0, Dimension::TIME);
/// let t1 = DynQuantity::from_base(2.0, Dimension::TIME);
/// let v0 = [DynQuantity::from_base(0.0, Dimension::VELOCITY)];
/// let accel = Dimension::ACCELERATION;
/// let out = dopri5_dimensioned(
///     |_t, _y| Ok([DynQuantity::from_base(-9.81, accel)]),
///     t0,
///     t1,
///     v0,
///     1e-10,
///     1e-12,
///     10_000,
/// )
/// .unwrap();
/// assert!((out[0].value() + 19.62).abs() < 1e-6);
/// assert_eq!(*out[0].dim(), Dimension::VELOCITY);
/// ```
#[cfg(feature = "ode")]
pub fn dopri5_dimensioned<const N: usize, F>(
    f: F,
    t0: DynQuantity,
    t1: DynQuantity,
    y0: [DynQuantity; N],
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> Result<[DynQuantity; N], SolveError>
where
    F: Fn(&DynQuantity, &[DynQuantity; N]) -> Result<[DynQuantity; N], UnitError>,
{
    let time = *t0.dim();
    expect_dim("ode end time", time, &t1)?;
    let state_dims: [Dimension; N] = core::array::from_fn(|i| *y0[i].dim());
    let deriv_dims: [Dimension; N] = {
        let mut out = [Dimension::DIMENSIONLESS; N];
        for (slot, &d) in out.iter_mut().zip(&state_dims) {
            *slot = dim_div("ode derivative", d, time)?;
        }
        out
    };

    // The numeric solver takes plain closures, so a failed check is parked here
    // and the closure returns NaN until the solve finishes.
    let failure: RefCell<Option<UnitError>> = RefCell::new(None);
    let si_state = |values: &[f64; N]| -> [DynQuantity; N] {
        core::array::from_fn(|i| DynQuantity::from_base(values[i], state_dims[i]))
    };
    let raw = |t: f64, y: &[f64; N]| -> [f64; N] {
        if failure.borrow().is_some() {
            return [f64::NAN; N];
        }
        let t_q = DynQuantity::from_base(t, time);
        let y_q = si_state(y);
        match f(&t_q, &y_q) {
            Ok(dy) => {
                for i in 0..N {
                    if let Err(e) = expect_dim("ode derivative", deriv_dims[i], &dy[i]) {
                        *failure.borrow_mut() = Some(e);
                        return [f64::NAN; N];
                    }
                }
                core::array::from_fn(|i| dy[i].value())
            }
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                [f64::NAN; N]
            }
        }
    };

    let y0_si: [f64; N] = core::array::from_fn(|i| y0[i].value());
    let solved = tpt_math_numeric::ode::dopri5_integrate(
        raw,
        t0.value(),
        t1.value(),
        y0_si,
        rtol,
        atol,
        max_steps,
    );
    if let Some(e) = failure.into_inner() {
        return Err(SolveError::Unit(e));
    }
    let y1 = solved.map_err(SolveError::Ode)?;
    Ok(si_state(&y1))
}

/// Result of a units-aware minimisation.
#[cfg(feature = "optimize")]
#[derive(Clone, Debug, PartialEq)]
pub struct DimensionedSolution<const N: usize> {
    /// The parameters at the solution, with their dimensions.
    pub params: [DynQuantity; N],
    /// The cost at the solution. Its dimension is the one the cost function
    /// returned at the start.
    pub cost: DynQuantity,
    /// Iterations performed.
    pub iters: u64,
    /// `true` if the solver met its gradient tolerance.
    pub converged: bool,
    /// The solver's stop reason.
    pub termination: String,
}

/// Minimise a dimensioned cost over dimensioned parameters.
///
/// The cost must return the same dimension at every point it is evaluated, and
/// that dimension is fixed by the first evaluation at `init`. Gradients come from
/// central finite differences, so each component has dimension `cost / param`.
/// The solver is L-BFGS from `tpt-math-optimize-general`, run on SI values.
///
/// # Errors
///
/// Returns [`SolveError::Unit`] if the cost's dimension changes between points or
/// a value is not finite, and [`SolveError::Optimize`] if the solver rejects its
/// input.
///
/// # Examples
///
/// ```
/// use tpt_math_units_dyn::solve::minimize_dimensioned;
/// use tpt_math_units_dyn::{DynQuantity, Dimension};
/// use tpt_math_optimize_general::Options;
///
/// // Minimise (x − 3 m)² in metres. The optimum is 3 m, with cost in m².
/// let init = [DynQuantity::from_base(0.0, Dimension::LENGTH)];
/// let sol = minimize_dimensioned(
///     |p| {
///         let d = p[0].try_sub(&DynQuantity::from_base(3.0, Dimension::LENGTH))?;
///         d.try_mul(&d)
///     },
///     init,
///     &Options::new(200),
/// )
/// .unwrap();
/// assert!((sol.params[0].value() - 3.0).abs() < 1e-6);
/// assert_eq!(*sol.cost.dim(), Dimension::LENGTH.checked_powi(2).unwrap());
/// ```
#[cfg(feature = "optimize")]
pub fn minimize_dimensioned<const N: usize, F>(
    cost: F,
    init: [DynQuantity; N],
    options: &tpt_math_optimize_general::Options,
) -> Result<DimensionedSolution<N>, SolveError>
where
    F: Fn(&[DynQuantity; N]) -> Result<DynQuantity, UnitError>,
{
    use tpt_math_optimize_general::{
        minimize_lbfgs_with, numerical_gradient, tpt_math_linalg_dense::DVector,
    };

    let param_dims: [Dimension; N] = core::array::from_fn(|i| *init[i].dim());
    let to_quantities = |x: &DVector<f64>| -> [DynQuantity; N] {
        core::array::from_fn(|i| DynQuantity::from_base(x[i], param_dims[i]))
    };

    // The cost dimension is whatever the first evaluation returns.
    let first = cost(&init)?;
    let cost_dim = *first.dim();
    let failure: RefCell<Option<UnitError>> = RefCell::new(None);
    let eval = |x: &DVector<f64>| -> f64 {
        if failure.borrow().is_some() {
            return f64::NAN;
        }
        match cost(&to_quantities(x)) {
            Ok(c) => match expect_dim("cost", cost_dim, &c) {
                Ok(()) => c.value(),
                Err(e) => {
                    *failure.borrow_mut() = Some(e);
                    f64::NAN
                }
            },
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                f64::NAN
            }
        }
    };

    let x0 = DVector::from_fn(N, |i| init[i].value());
    let grad = |x: &DVector<f64>| numerical_gradient(&eval, x, 1e-6);
    let solved = minimize_lbfgs_with(eval, grad, x0, 5, options).map_err(SolveError::Optimize);
    if let Some(e) = failure.into_inner() {
        return Err(SolveError::Unit(e));
    }
    let sol = solved?;
    Ok(DimensionedSolution {
        params: to_quantities(&sol.param),
        cost: DynQuantity::from_base(sol.cost, cost_dim),
        iters: sol.iters,
        converged: sol.converged,
        termination: sol.termination,
    })
}

/// Gradient of a dimensioned function, computed by reverse-mode autodiff.
///
/// `f` is written on a tape with [`tpt_math_autodiff_rev::Variable`]s, one per
/// entry of `point`. Component `i` of the result has dimension
/// `cost_dim / point[i].dim()`, and its value is the SI derivative. Only the
/// dimensions of the output are checked, by construction. The function on the
/// tape is not checked, so a wrong unit inside `f` goes undetected.
///
/// # Errors
///
/// Returns [`SolveError::Unit`] if a dimension exponent overflows.
///
/// # Examples
///
/// ```
/// use tpt_math_units_dyn::solve::gradient_dimensioned;
/// use tpt_math_units_dyn::{DynQuantity, Dimension};
///
/// // Kinetic energy ½ m v² with mass in kg and speed in m/s: ∂E/∂v = m v.
/// let point = [
///     DynQuantity::from_base(2.0, Dimension::MASS),
///     DynQuantity::from_base(3.0, Dimension::VELOCITY),
/// ];
/// let g = gradient_dimensioned(|x| x[0] * x[1] * x[1] * 0.5, &point, Dimension::ENERGY).unwrap();
/// // ∂E/∂m = v²/2 = 4.5 J/kg, ∂E/∂v = m v = 6 N·s.
/// assert!((g[0].value() - 4.5).abs() < 1e-12);
/// assert!((g[1].value() - 6.0).abs() < 1e-12);
/// ```
#[cfg(feature = "autodiff")]
pub fn gradient_dimensioned<const N: usize, F>(
    f: F,
    point: &[DynQuantity; N],
    cost_dim: Dimension,
) -> Result<[DynQuantity; N], SolveError>
where
    F: for<'t> Fn(&[tpt_math_autodiff_rev::Variable<'t>]) -> tpt_math_autodiff_rev::Variable<'t>,
{
    let values: [f64; N] = core::array::from_fn(|i| point[i].value());
    let tape = tpt_math_autodiff_rev::GradientTape::new();
    let vars = tape.vars(&values);
    let out = f(&vars);
    let grads = tape.gradient(out, &vars);
    let mut result = Vec::with_capacity(N);
    for (i, g) in grads.iter().enumerate() {
        let dim = dim_div("gradient", cost_dim, *point[i].dim())?;
        result.push(DynQuantity::from_base(*g, dim));
    }
    let array: [DynQuantity; N] = result.try_into().map_err(|_| UnitError::ExponentOverflow {
        context: "gradient",
    })?;
    Ok(array)
}
