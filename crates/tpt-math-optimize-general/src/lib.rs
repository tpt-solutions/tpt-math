//! General numerical optimization — a thin wrapper around [`argmin`].
//!
//! [`argmin`] is a complete optimization framework: you implement traits such
//! as [`CostFunction`], [`Gradient`] and [`Hessian`] for a problem type, hand
//! that problem plus a solver to an [`Executor`], configure the initial state,
//! run it, and dig the answer back out of the final state. That is the right
//! amount of control when writing a bespoke solver — and far too much
//! ceremony when all you have is a closure and a starting point.
//!
//! This crate therefore does two things:
//!
//! 1. Re-exports argmin unchanged ([`argmin`], plus its [`core`] and
//!    [`solver`] modules and the [`argmin_math`] trait impls), so nothing is
//!    hidden behind the wrapper.
//! 2. Adds closure-driven entry points for the common smooth, unconstrained
//!    minimization cases:
//!
//! | Convenience function | argmin solver | Needs |
//! |---|---|---|
//! | [`minimize_gradient_descent`] | [`SteepestDescent`] + [`MoreThuenteLineSearch`] | cost, gradient |
//! | [`minimize_conjugate_gradient`] | [`NonlinearConjugateGradient`] (Polak–Ribière+) + [`MoreThuenteLineSearch`] | cost, gradient |
//! | [`minimize_newton`] | [`Newton`] | cost, gradient, Hessian |
//!
//! Parameters are plain [`DVector<f64>`](tpt_math_linalg_dense::DVector)s from the very
//! same `tpt-math-linalg-dense` (faer) backend that [`tpt_math_linalg`] wraps,
//! so unit-tagged
//! [`tpt_math_linalg::Vec`] values move in and out with
//! [`point_from_tagged`] / [`point_to_tagged`]. Optimization itself is
//! deliberately unit-less: a cost mixes every unit in the problem, so there is
//! no meaningful tag for a search direction or a step length.
//!
//! # Examples
//!
//! Minimize `f(x, y) = (x - 3)² + (y - 2)²` by gradient descent:
//!
//! ```
//! use tpt_math_optimize_general::{minimize_gradient_descent, tpt_math_linalg_dense::DVector};
//!
//! let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] - 2.0).powi(2);
//! let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 2.0)]);
//!
//! let best = minimize_gradient_descent(cost, grad, DVector::zeros(2), 100).unwrap();
//!
//! assert!((best[0] - 3.0).abs() < 1e-6);
//! assert!((best[1] - 2.0).abs() < 1e-6);
//! ```
//!
//! The `*_with` variants take [`Options`] and return a [`Solution`], which
//! also reports the cost, the iteration count and why the solver stopped:
//!
//! ```
//! use tpt_math_optimize_general::{minimize_newton, tpt_math_linalg_dense::{DMatrix, DVector}};
//!
//! // f(x) = exp(x) - x, minimized at x = 0 with f(0) = 1.
//! let cost = |p: &DVector<f64>| p[0].exp() - p[0];
//! let grad = |p: &DVector<f64>| DVector::from_vec(vec![p[0].exp() - 1.0]);
//! let hess = |p: &DVector<f64>| DMatrix::from_vec(1, 1, vec![p[0].exp()]);
//!
//! let best = minimize_newton(cost, grad, hess, DVector::from_vec(vec![1.0]), 50).unwrap();
//!
//! assert!(best[0].abs() < 1e-6);
//! ```
//!
//! # Conventions
//!
//! * Errors from argmin (a failed line search, a singular Hessian, ...) are
//!   flattened to `String` so callers need not depend on `argmin` to handle
//!   them. Use the re-exported [`argmin`] directly when a typed [`Error`] is
//!   needed.
//! * Every run stops at `max_iters`, or earlier once the gradient's L2 norm
//!   falls to [`Options::gradient_tolerance`]. The early stop matters: line
//!   searches fail outright once the gradient is (numerically) zero, so a
//!   solver iterated past its own optimum would otherwise report an error
//!   instead of the answer it had already found.
//! * The reported [`Solution::param`] is argmin's *best* parameter vector, and
//!   [`Solution::cost`] is the user cost function re-evaluated there.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub use argmin;
pub use argmin::core;
pub use argmin::solver;
pub use argmin_math;
pub use tpt_math_linalg;
pub use tpt_math_linalg::tpt_math_linalg_dense;

use argmin::core::{
    CostFunction, Error, Executor, Gradient, Hessian, IterState, Problem, Solver, State,
    TerminationReason, TerminationStatus, KV,
};
use argmin::solver::conjugategradient::beta::PolakRibierePlus;
use argmin::solver::conjugategradient::NonlinearConjugateGradient;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::newton::Newton;
use tpt_math_linalg_dense::{DMatrix, DVector};

/// Default iteration budget used by [`Options::default`].
pub const DEFAULT_MAX_ITERS: u64 = 100;

/// Default gradient L2 norm at or below which a run counts as converged.
pub const DEFAULT_GRADIENT_TOLERANCE: f64 = 1e-9;

/// Iterations between forced conjugate-gradient restarts (β reset to zero).
const RESTART_ITERS: u64 = 10;

/// Orthogonality threshold that triggers a conjugate-gradient restart.
const RESTART_ORTHOGONALITY: f64 = 0.1;

/// A unit-tagged vector of `f64`s, as used by [`tpt_math_linalg`].
pub type TaggedVec<U> = tpt_math_linalg::Vec<U, f64>;

/// The line search shared by the gradient-based convenience solvers.
type LineSearch = MoreThuenteLineSearch<DVector<f64>, DVector<f64>, f64>;

/// The argmin state shared by the convenience solvers, generic over the
/// Hessian slot (`()` when a solver does not use one).
type OptState<H> = IterState<DVector<f64>, DVector<f64>, (), H, (), f64>;

/// Knobs shared by every convenience minimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Hard cap on solver iterations.
    pub max_iters: u64,
    /// Stop as soon as the gradient's L2 norm is at or below this value.
    ///
    /// Setting it to `0.0` effectively runs the full iteration budget, since
    /// only an exactly zero gradient then counts as converged — but note that
    /// line searches error out on a zero gradient, so a solver that reaches its
    /// optimum early will fail rather than return.
    pub gradient_tolerance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_iters: DEFAULT_MAX_ITERS,
            gradient_tolerance: DEFAULT_GRADIENT_TOLERANCE,
        }
    }
}

impl Options {
    /// Options with the given iteration budget and the default gradient
    /// tolerance.
    ///
    /// ```
    /// use tpt_math_optimize_general::{Options, DEFAULT_GRADIENT_TOLERANCE};
    ///
    /// let opts = Options::new(250);
    /// assert_eq!(opts.max_iters, 250);
    /// assert_eq!(opts.gradient_tolerance, DEFAULT_GRADIENT_TOLERANCE);
    /// ```
    pub fn new(max_iters: u64) -> Self {
        Options {
            max_iters,
            ..Options::default()
        }
    }

    /// Set the gradient tolerance, returning the updated options.
    ///
    /// ```
    /// use tpt_math_optimize_general::Options;
    ///
    /// let opts = Options::new(10).with_gradient_tolerance(1e-6);
    /// assert_eq!(opts.gradient_tolerance, 1e-6);
    /// ```
    pub fn with_gradient_tolerance(mut self, tolerance: f64) -> Self {
        self.gradient_tolerance = tolerance;
        self
    }
}

/// The outcome of a minimization run.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    /// Best parameter vector found.
    pub param: DVector<f64>,
    /// The user cost function evaluated at [`Solution::param`].
    pub cost: f64,
    /// Number of iterations actually performed.
    pub iters: u64,
    /// `true` if the run stopped because the gradient tolerance was met,
    /// `false` if it merely ran out of iterations (or stopped for any other
    /// reason).
    pub converged: bool,
    /// Human-readable reason the solver stopped, straight from argmin.
    pub termination: String,
}

/// Copy a unit-tagged [`tpt_math_linalg::Vec`] into a raw parameter vector.
///
/// ```
/// use tpt_math_optimize_general::{point_from_tagged, TaggedVec, tpt_math_linalg_dense::DVector};
///
/// let tagged: TaggedVec<()> = TaggedVec::from_raw(DVector::from_vec(vec![1.0, 2.0]));
/// assert_eq!(point_from_tagged(&tagged), DVector::from_vec(vec![1.0, 2.0]));
/// ```
pub fn point_from_tagged<U>(v: &TaggedVec<U>) -> DVector<f64> {
    v.raw().clone()
}

/// Tag a raw parameter vector with the unit `U`, undoing [`point_from_tagged`].
///
/// ```
/// use tpt_math_optimize_general::{point_to_tagged, TaggedVec, tpt_math_linalg_dense::DVector};
///
/// let tagged: TaggedVec<()> = point_to_tagged(DVector::from_vec(vec![1.0, 2.0]));
/// assert_eq!(tagged.len(), 2);
/// ```
pub fn point_to_tagged<U>(point: DVector<f64>) -> TaggedVec<U> {
    TaggedVec::from_raw(point)
}

/// Minimize a smooth objective by gradient descent with a More-Thuente line
/// search ([`SteepestDescent`]).
///
/// `cost` and `grad` must describe the same function; `init` is the starting
/// point and `max_iters` the iteration budget. Returns the best parameter
/// vector found, or the flattened argmin error.
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::{minimize_gradient_descent, tpt_math_linalg_dense::DVector};
///
/// // f(x) = x², minimized at x = 0.
/// let cost = |p: &DVector<f64>| p[0] * p[0];
/// let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * p[0]]);
///
/// let best = minimize_gradient_descent(cost, grad, DVector::from_vec(vec![5.0]), 50).unwrap();
/// assert!(best[0].abs() < 1e-8);
/// ```
///
/// # Errors
///
/// Returns `Err` if `init` is empty or non-finite, or if argmin fails (most
/// commonly a line search that cannot find a descent direction).
pub fn minimize_gradient_descent<F, G>(
    cost: F,
    grad: G,
    init: DVector<f64>,
    max_iters: u64,
) -> Result<DVector<f64>, String>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    minimize_gradient_descent_with(cost, grad, init, &Options::new(max_iters)).map(|s| s.param)
}

/// [`minimize_gradient_descent`] with explicit [`Options`], reporting a full
/// [`Solution`].
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::{minimize_gradient_descent_with, Options, tpt_math_linalg_dense::DVector};
///
/// let cost = |p: &DVector<f64>| (p[0] - 1.0).powi(2);
/// let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 1.0)]);
///
/// let options = Options::new(100).with_gradient_tolerance(1e-10);
/// let solution = minimize_gradient_descent_with(cost, grad, DVector::zeros(1), &options).unwrap();
///
/// assert!(solution.converged);
/// assert!(solution.cost < 1e-12);
/// ```
///
/// # Errors
///
/// As [`minimize_gradient_descent`].
pub fn minimize_gradient_descent_with<F, G>(
    cost: F,
    grad: G,
    init: DVector<f64>,
    options: &Options,
) -> Result<Solution, String>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let problem = ClosureProblem {
        cost,
        gradient: grad,
        hessian: (),
    };
    let linesearch: LineSearch = MoreThuenteLineSearch::new();
    run(problem, SteepestDescent::new(linesearch), init, options)
}

/// Minimize a smooth objective with nonlinear conjugate gradients
/// ([`NonlinearConjugateGradient`], Polak–Ribière+ β update with restarts,
/// More-Thuente line search).
///
/// Usually a much better default than plain gradient descent on ill-conditioned
/// problems, at the same cost per iteration (one cost and one gradient
/// evaluation plus the line search).
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::{minimize_conjugate_gradient, tpt_math_linalg_dense::DVector};
///
/// // A stretched quadratic: gradient descent zig-zags here, CG does not.
/// let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + 50.0 * (p[1] - 2.0).powi(2);
/// let grad = |p: &DVector<f64>| {
///     DVector::from_vec(vec![2.0 * (p[0] - 3.0), 100.0 * (p[1] - 2.0)])
/// };
///
/// let best = minimize_conjugate_gradient(cost, grad, DVector::zeros(2), 100).unwrap();
/// assert!((best[0] - 3.0).abs() < 1e-5);
/// assert!((best[1] - 2.0).abs() < 1e-5);
/// ```
///
/// # Errors
///
/// As [`minimize_gradient_descent`].
pub fn minimize_conjugate_gradient<F, G>(
    cost: F,
    grad: G,
    init: DVector<f64>,
    max_iters: u64,
) -> Result<DVector<f64>, String>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    minimize_conjugate_gradient_with(cost, grad, init, &Options::new(max_iters)).map(|s| s.param)
}

/// [`minimize_conjugate_gradient`] with explicit [`Options`], reporting a full
/// [`Solution`].
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::{minimize_conjugate_gradient_with, Options, tpt_math_linalg_dense::DVector};
///
/// let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] + 1.0).powi(2);
/// let grad = |p: &DVector<f64>| {
///     DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 1.0)])
/// };
///
/// let solution =
///     minimize_conjugate_gradient_with(cost, grad, DVector::zeros(2), &Options::new(50)).unwrap();
/// assert!(solution.cost < 1e-12);
/// ```
///
/// # Errors
///
/// As [`minimize_gradient_descent`].
pub fn minimize_conjugate_gradient_with<F, G>(
    cost: F,
    grad: G,
    init: DVector<f64>,
    options: &Options,
) -> Result<Solution, String>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let problem = ClosureProblem {
        cost,
        gradient: grad,
        hessian: (),
    };
    let linesearch: LineSearch = MoreThuenteLineSearch::new();
    // Polak-Ribière+ clamps β at zero and the orthogonality test restarts the
    // recursion once successive gradients stop being orthogonal; together they
    // keep the search direction a descent direction on non-quadratic problems.
    let solver: NonlinearConjugateGradient<DVector<f64>, LineSearch, PolakRibierePlus, f64> =
        NonlinearConjugateGradient::new(linesearch, PolakRibierePlus::new())
            .restart_iters(RESTART_ITERS)
            .restart_orthogonality(RESTART_ORTHOGONALITY);
    run(problem, solver, init, options)
}

/// Minimize a smooth objective with [`Newton`]'s method, using the analytic
/// Hessian.
///
/// Newton converges quadratically near a minimum — a quadratic objective is
/// solved in a single iteration — but it needs a Hessian that is invertible
/// (and, to descend rather than ascend, positive definite) along the path. Fall
/// back to [`minimize_conjugate_gradient`] when that cannot be guaranteed.
///
/// Note that argmin 0.11's `Newton` takes no line search: the step is the full
/// Newton step scaled by a fixed `gamma` (1 by default). Build
/// [`Newton::with_gamma`] directly with the re-exported [`argmin`] for a damped
/// variant.
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::{minimize_newton, tpt_math_linalg_dense::{DMatrix, DVector}};
///
/// // f(x, y) = (x - 3)² + (y - 2)²: one Newton step lands exactly on (3, 2).
/// let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] - 2.0).powi(2);
/// let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 2.0)]);
/// let hess = |_: &DVector<f64>| DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 2.0]));
///
/// let best = minimize_newton(cost, grad, hess, DVector::zeros(2), 25).unwrap();
/// assert!((best[0] - 3.0).abs() < 1e-12);
/// assert!((best[1] - 2.0).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// As [`minimize_gradient_descent`], plus a non-invertible Hessian.
pub fn minimize_newton<F, G, H>(
    cost: F,
    grad: G,
    hessian: H,
    init: DVector<f64>,
    max_iters: u64,
) -> Result<DVector<f64>, String>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
    H: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    minimize_newton_with(cost, grad, hessian, init, &Options::new(max_iters)).map(|s| s.param)
}

/// [`minimize_newton`] with explicit [`Options`], reporting a full
/// [`Solution`].
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::{
///     minimize_newton_with, Options,
///     tpt_math_linalg_dense::{DMatrix, DVector},
/// };
///
/// let cost = |p: &DVector<f64>| p[0].powi(2);
/// let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * p[0]]);
/// let hess = |_: &DVector<f64>| DMatrix::from_vec(1, 1, vec![2.0]);
///
/// let solution =
///     minimize_newton_with(cost, grad, hess, DVector::from_vec(vec![9.0]), &Options::new(10))
///         .unwrap();
///
/// assert!(solution.converged);
/// assert!(solution.iters <= 2);
/// ```
///
/// # Errors
///
/// As [`minimize_newton`].
pub fn minimize_newton_with<F, G, H>(
    cost: F,
    grad: G,
    hessian: H,
    init: DVector<f64>,
    options: &Options,
) -> Result<Solution, String>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
    H: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    let problem = ClosureProblem {
        cost,
        gradient: grad,
        hessian,
    };
    let solver: Newton<f64> = Newton::new();
    run(problem, solver, init, options)
}

/// Drive an argmin [`Executor`] over one of the closure problems above and
/// repackage the final state as a [`Solution`].
fn run<O, S, H>(
    problem: O,
    solver: S,
    init: DVector<f64>,
    options: &Options,
) -> Result<Solution, String>
where
    O: CostFunction<Param = DVector<f64>, Output = f64>
        + Gradient<Param = DVector<f64>, Gradient = DVector<f64>>,
    S: Solver<O, OptState<H>>,
{
    validate(&init, options)?;

    let solver = WithGradientTolerance {
        inner: solver,
        tolerance: options.gradient_tolerance,
    };
    let executor: Executor<O, WithGradientTolerance<S>, OptState<H>> =
        Executor::new(problem, solver);
    let mut result = executor
        .configure(|state| state.param(init).max_iters(options.max_iters))
        // A library has no business installing a process-wide Ctrl-C handler.
        .ctrlc(false)
        .run()
        .map_err(stringify_error)?;

    let iters = result.state.get_iter();
    let status = result.state.get_termination_status();
    let converged = matches!(
        status,
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
    );
    let termination = status.to_string();

    let param = result
        .state
        .take_best_param()
        .or_else(|| result.state.take_param())
        .ok_or_else(|| "argmin returned no parameter vector".to_string())?;

    // Solvers that never evaluate the cost (Newton, for one) leave `best_cost`
    // at infinity, so report the cost freshly evaluated at the returned point.
    let problem = result
        .problem
        .take_problem()
        .ok_or_else(|| "argmin returned no problem".to_string())?;
    let cost = problem.cost(&param).map_err(stringify_error)?;

    Ok(Solution {
        param,
        cost,
        iters,
        converged,
        termination,
    })
}

/// Reject inputs that argmin would only reject later, with a worse message.
fn validate(init: &DVector<f64>, options: &Options) -> Result<(), String> {
    if init.is_empty() {
        return Err("initial parameter vector must not be empty".to_string());
    }
    if init.iter().any(|x| !x.is_finite()) {
        return Err("initial parameter vector must be finite".to_string());
    }
    if options.gradient_tolerance.is_nan() || options.gradient_tolerance < 0.0 {
        return Err("gradient tolerance must be finite and non-negative".to_string());
    }
    Ok(())
}

fn stringify_error(error: Error) -> String {
    error.to_string()
}

/// A user's cost / gradient / Hessian closures, wearing argmin's traits.
///
/// The Hessian slot is `()` for the first-order solvers; the [`Hessian`] impl
/// below then simply does not apply.
struct ClosureProblem<C, G, H> {
    cost: C,
    gradient: G,
    hessian: H,
}

impl<C, G, H> CostFunction for ClosureProblem<C, G, H>
where
    C: Fn(&DVector<f64>) -> f64,
{
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok((self.cost)(param))
    }
}

impl<C, G, H> Gradient for ClosureProblem<C, G, H>
where
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((self.gradient)(param))
    }
}

impl<C, G, H> Hessian for ClosureProblem<C, G, H>
where
    H: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    type Param = DVector<f64>;
    type Hessian = DMatrix<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok((self.hessian)(param))
    }
}

/// Adds a gradient-norm stopping criterion to any argmin solver.
///
/// argmin's built-in termination only covers the iteration budget and a target
/// cost. Without a first-order test, an already-converged run keeps iterating
/// and the line search eventually errors out on a zero gradient — turning a
/// perfectly good answer into an `Err`. The check happens *before* delegating
/// to the wrapped solver, so the failing step is never taken.
struct WithGradientTolerance<S> {
    inner: S,
    tolerance: f64,
}

impl<O, S, H> Solver<O, OptState<H>> for WithGradientTolerance<S>
where
    O: Gradient<Param = DVector<f64>, Gradient = DVector<f64>>,
    S: Solver<O, OptState<H>>,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: OptState<H>,
    ) -> Result<(OptState<H>, Option<KV>), Error> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: OptState<H>,
    ) -> Result<(OptState<H>, Option<KV>), Error> {
        let converged = match state.get_param() {
            Some(param) => problem.gradient(param)?.norm() <= self.tolerance,
            None => false,
        };
        if converged {
            return Ok((
                state.terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }
        self.inner.next_iter(problem, state)
    }

    fn terminate(&mut self, state: &OptState<H>) -> TerminationStatus {
        self.inner.terminate(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x, y) = (x - 3)² + (y - 2)², minimum 0 at (3, 2).
    fn bowl_cost(p: &DVector<f64>) -> f64 {
        (p[0] - 3.0).powi(2) + (p[1] - 2.0).powi(2)
    }

    fn bowl_grad(p: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 2.0)])
    }

    fn bowl_hess(_: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 2.0]))
    }

    /// f(x, y) = (x - 3)² + 50 (y - 2)²: same minimum, condition number 50.
    fn valley_cost(p: &DVector<f64>) -> f64 {
        (p[0] - 3.0).powi(2) + 50.0 * (p[1] - 2.0).powi(2)
    }

    fn valley_grad(p: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![2.0 * (p[0] - 3.0), 100.0 * (p[1] - 2.0)])
    }

    fn assert_close(actual: &DVector<f64>, expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (actual[i] - want).abs() < tol,
                "component {i}: {} is not within {tol} of {want}",
                actual[i]
            );
        }
    }

    #[test]
    fn gradient_descent_minimizes_scalar_square() {
        // f(x) = x², minimum 0 at x = 0.
        let best = minimize_gradient_descent(
            |p: &DVector<f64>| p[0] * p[0],
            |p: &DVector<f64>| DVector::from_vec(vec![2.0 * p[0]]),
            DVector::from_vec(vec![7.5]),
            50,
        )
        .unwrap();

        assert_close(&best, &[0.0], 1e-8);
    }

    #[test]
    fn gradient_descent_minimizes_bowl() {
        let best = minimize_gradient_descent(bowl_cost, bowl_grad, DVector::zeros(2), 100).unwrap();

        assert_close(&best, &[3.0, 2.0], 1e-6);
        assert!(bowl_cost(&best) < 1e-12);
    }

    #[test]
    fn gradient_descent_reports_convergence_and_cost() {
        let solution = minimize_gradient_descent_with(
            bowl_cost,
            bowl_grad,
            DVector::from_vec(vec![-4.0, 9.0]),
            &Options::new(200),
        )
        .unwrap();

        assert!(solution.converged, "termination: {}", solution.termination);
        assert!(solution.iters <= 200);
        assert!(!solution.termination.is_empty());
        assert!(solution.cost < 1e-12, "cost was {}", solution.cost);
        assert!((solution.cost - bowl_cost(&solution.param)).abs() < 1e-15);
    }

    #[test]
    fn gradient_descent_survives_starting_at_the_optimum() {
        // Zero gradient at the start: the line search would fail, the
        // convergence check must stop first.
        let solution = minimize_gradient_descent_with(
            bowl_cost,
            bowl_grad,
            DVector::from_vec(vec![3.0, 2.0]),
            &Options::new(25),
        )
        .unwrap();

        assert!(solution.converged);
        assert!(solution.iters <= 1);
        assert_close(&solution.param, &[3.0, 2.0], 1e-15);
    }

    #[test]
    fn conjugate_gradient_handles_ill_conditioned_valley() {
        let solution = minimize_conjugate_gradient_with(
            valley_cost,
            valley_grad,
            DVector::zeros(2),
            &Options::new(200),
        )
        .unwrap();

        assert_close(&solution.param, &[3.0, 2.0], 1e-5);
        assert!(solution.cost < 1e-9, "cost was {}", solution.cost);
    }

    #[test]
    fn conjugate_gradient_minimizes_rosenbrock() {
        // The classic banana valley: minimum 0 at (1, 1).
        let cost = |p: &DVector<f64>| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
        let grad = |p: &DVector<f64>| {
            DVector::from_vec(vec![
                -2.0 * (1.0 - p[0]) - 400.0 * p[0] * (p[1] - p[0] * p[0]),
                200.0 * (p[1] - p[0] * p[0]),
            ])
        };

        let best = minimize_conjugate_gradient(cost, grad, DVector::from_vec(vec![-1.2, 1.0]), 500)
            .unwrap();

        assert_close(&best, &[1.0, 1.0], 1e-3);
    }

    #[test]
    fn newton_solves_quadratic_in_one_step() {
        let solution = minimize_newton_with(
            bowl_cost,
            bowl_grad,
            bowl_hess,
            DVector::from_vec(vec![-10.0, 40.0]),
            &Options::new(25),
        )
        .unwrap();

        assert_close(&solution.param, &[3.0, 2.0], 1e-12);
        assert!(solution.converged);
        assert!(solution.iters <= 2, "took {} iterations", solution.iters);
        assert!(solution.cost < 1e-24);
    }

    #[test]
    fn newton_minimizes_non_quadratic() {
        // f(x) = exp(x) - x, minimum 1 at x = 0.
        let best = minimize_newton(
            |p: &DVector<f64>| p[0].exp() - p[0],
            |p: &DVector<f64>| DVector::from_vec(vec![p[0].exp() - 1.0]),
            |p: &DVector<f64>| DMatrix::from_vec(1, 1, vec![p[0].exp()]),
            DVector::from_vec(vec![1.5]),
            50,
        )
        .unwrap();

        assert_close(&best, &[0.0], 1e-7);
    }

    #[test]
    fn solvers_agree_on_the_same_problem() {
        let init = DVector::from_vec(vec![-2.0, 6.5]);
        let gd = minimize_gradient_descent(bowl_cost, bowl_grad, init.clone(), 100).unwrap();
        let cg = minimize_conjugate_gradient(bowl_cost, bowl_grad, init.clone(), 100).unwrap();
        let newton = minimize_newton(bowl_cost, bowl_grad, bowl_hess, init, 100).unwrap();

        assert_close(&gd, &[3.0, 2.0], 1e-6);
        assert_close(&cg, &[3.0, 2.0], 1e-6);
        assert_close(&newton, &[3.0, 2.0], 1e-12);
    }

    #[test]
    fn max_iters_caps_the_run() {
        // One iteration is nowhere near enough for this starting point.
        let solution = minimize_gradient_descent_with(
            valley_cost,
            valley_grad,
            DVector::from_vec(vec![-50.0, -50.0]),
            &Options::new(1),
        )
        .unwrap();

        assert_eq!(solution.iters, 1);
        assert!(!solution.converged);
        assert!(solution.cost > 0.0);
    }

    #[test]
    fn zero_iterations_returns_the_initial_point() {
        let init = DVector::from_vec(vec![1.0, -1.0]);
        let solution =
            minimize_gradient_descent_with(bowl_cost, bowl_grad, init.clone(), &Options::new(0))
                .unwrap();

        assert_eq!(solution.param, init);
        assert_eq!(solution.iters, 0);
        assert!((solution.cost - bowl_cost(&init)).abs() < 1e-15);
    }

    #[test]
    fn empty_initial_point_is_rejected() {
        let err =
            minimize_gradient_descent(bowl_cost, bowl_grad, DVector::zeros(0), 10).unwrap_err();
        assert!(err.contains("empty"), "unexpected message: {err}");
    }

    #[test]
    fn non_finite_initial_point_is_rejected() {
        let err = minimize_gradient_descent(
            bowl_cost,
            bowl_grad,
            DVector::from_vec(vec![f64::NAN, 0.0]),
            10,
        )
        .unwrap_err();
        assert!(err.contains("finite"), "unexpected message: {err}");
    }

    #[test]
    fn negative_gradient_tolerance_is_rejected() {
        let options = Options::new(10).with_gradient_tolerance(-1.0);
        let err = minimize_gradient_descent_with(bowl_cost, bowl_grad, DVector::zeros(2), &options)
            .unwrap_err();
        assert!(err.contains("tolerance"), "unexpected message: {err}");
    }

    #[test]
    fn singular_hessian_is_reported_as_an_error() {
        let err = minimize_newton(
            bowl_cost,
            bowl_grad,
            |_: &DVector<f64>| DMatrix::zeros(2, 2),
            DVector::zeros(2),
            10,
        )
        .unwrap_err();

        assert!(!err.is_empty());
    }

    #[test]
    fn tagged_vectors_round_trip() {
        let tagged: TaggedVec<()> = point_to_tagged(DVector::from_vec(vec![3.0, 4.0]));
        let raw = point_from_tagged(&tagged);

        let best = minimize_gradient_descent(bowl_cost, bowl_grad, raw, 100).unwrap();
        let back: TaggedVec<()> = point_to_tagged(best);

        assert_close(back.raw(), &[3.0, 2.0], 1e-6);
    }

    #[test]
    fn options_defaults_are_sane() {
        let options = Options::default();
        assert_eq!(options.max_iters, DEFAULT_MAX_ITERS);
        assert_eq!(options.gradient_tolerance, DEFAULT_GRADIENT_TOLERANCE);
        assert_eq!(Options::new(DEFAULT_MAX_ITERS), options);
    }
}
