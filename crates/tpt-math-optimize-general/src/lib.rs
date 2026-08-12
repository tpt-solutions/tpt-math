//! General numerical optimization — in-house smooth, unconstrained minimizers.
//!
//! This crate provides closure-driven entry points for the common smooth,
//! unconstrained minimization cases, built directly on
//! [`tpt_math_linalg_dense`]'s [`DVector`]/[`DMatrix`] (no `argmin`/`faer`
//! dependency):
//!
//! | Convenience function | Method |
//! |---|---|
//! | [`minimize_gradient_descent`] | steepest descent + More-Thuente line search |
//! | [`minimize_conjugate_gradient`] | nonlinear CG (Polak–Ribière+) + More-Thuente line search, periodic restarts |
//! | [`minimize_newton`] | Newton's method (full step, analytic Hessian) |
//!
//! Parameters are plain [`DVector<f64>`](tpt_math_linalg_dense::DVector)s; unit-tagged
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
//! * Every run stops at `max_iters`, or earlier once the gradient's L2 norm
//!   falls to [`Options::gradient_tolerance`]. The early stop matters: a solver
//!   that has already reached its optimum would otherwise keep iterating and
//!   report a worse (or non-finite) point.
//! * The reported [`Solution::param`] is the best parameter vector found, and
//!   [`Solution::cost`] is the user cost function re-evaluated there.
//! * A non-finite or empty initial point, or a non-invertible Hessian (Newton),
//!   is reported as an `Err` with a human-readable message.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub use tpt_math_linalg;
pub use tpt_math_linalg::tpt_math_linalg_dense;

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

/// Knobs shared by every convenience minimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Hard cap on solver iterations.
    pub max_iters: u64,
    /// Stop as soon as the gradient's L2 norm is at or below this value.
    ///
    /// Setting it to `0.0` effectively runs the full iteration budget, since
    /// only an exactly zero gradient then counts as converged.
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
    /// Human-readable reason the solver stopped.
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
/// search.
///
/// `cost` and `grad` must describe the same function; `init` is the starting
/// point and `max_iters` the iteration budget. Returns the best parameter
/// vector found, or a message string on bad input.
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
/// Returns `Err` if `init` is empty or non-finite.
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
    let mut state = OptimizerState::new(cost, grad, no_hessian, init, options)?;
    let mut iters = 0;
    loop {
        if state.grad_norm() <= options.gradient_tolerance {
            return Ok(state.finish(iters, true, "GradientTolerance"));
        }
        if iters >= options.max_iters {
            return Ok(state.finish(iters, false, "MaxIters"));
        }
        // Steepest descent: full negative gradient, scaled by the line search.
        let dir = -state.grad().clone();
        let alpha = line_search(state.cost_fn(), state.grad_fn(), &state.param, &dir);
        state.step(&(dir * alpha));
        iters += 1;
    }
}

/// Minimize a smooth objective with nonlinear conjugate gradients
/// (Polak–Ribière+ β update with periodic restarts, More-Thuente line search).
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
    let mut state = OptimizerState::new(cost, grad, no_hessian, init, options)?;
    let mut iters = 0;
    let mut prev_grad: Option<DVector<f64>> = None;
    let mut direction: Option<DVector<f64>> = None;
    loop {
        if state.grad_norm() <= options.gradient_tolerance {
            return Ok(state.finish(iters, true, "GradientTolerance"));
        }
        if iters >= options.max_iters {
            return Ok(state.finish(iters, false, "MaxIters"));
        }

        let dir = match (&prev_grad, &direction) {
            (None, None) => -state.grad().clone(),
            (Some(g_prev), Some(d_prev)) => {
                let denom = g_prev.dot(g_prev);
                let beta = if denom > 0.0 {
                    let num = state.grad().dot(&(state.grad().clone() - g_prev.clone()));
                    num / denom
                } else {
                    0.0
                };
                let beta = beta.max(0.0);
                let restart = (iters % RESTART_ITERS == 0)
                    || (denom > 0.0
                        && (state.grad().dot(g_prev)).abs() / denom >= RESTART_ORTHOGONALITY);
                // Fall back to steepest descent if β yields a non-descent dir.
                let cand = if restart {
                    -state.grad().clone()
                } else {
                    -state.grad().clone() + d_prev.clone() * beta
                };
                if cand.dot(state.grad()) >= 0.0 {
                    -state.grad().clone()
                } else {
                    cand
                }
            }
            _ => unreachable!("prev_grad and direction are set together"),
        };

        let alpha = line_search(state.cost_fn(), state.grad_fn(), &state.param, &dir);
        state.step(&(dir.clone() * alpha));
        prev_grad = Some(state.grad().clone());
        direction = Some(dir);
        iters += 1;
    }
}

/// Minimize a smooth objective with [`Newton`]'s method, using the analytic
/// Hessian.
///
/// Newton converges quadratically near a minimum — a quadratic objective is
/// solved in a single iteration — but it needs a Hessian that is invertible
/// (and, to descend rather than ascend, positive definite) along the path. Fall
/// back to [`minimize_conjugate_gradient`] when that cannot be guaranteed.
///
/// The step is the full Newton step `x - H⁻¹ ∇f` (no damping); a singular
/// Hessian is reported as an error.
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
    let mut state = OptimizerState::new(cost, grad, hessian, init, options)?;
    let mut iters = 0;
    loop {
        if state.grad_norm() <= options.gradient_tolerance {
            return Ok(state.finish(iters, true, "GradientTolerance"));
        }
        if iters >= options.max_iters {
            return Ok(state.finish(iters, false, "MaxIters"));
        }
        let neg_grad = -state.grad().clone();
        let step = state
            .hessian()?
            .solve(&neg_grad)
            .map_err(|_| "singular Hessian (Newton step undefined)".to_string())?;
        state.step(&step);
        iters += 1;
    }
}

/// A placeholder Hessian function for the first-order solvers (never called).
fn no_hessian(_: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::zeros(0, 0)
}

/// Reject inputs that the solvers would only reject later, with a worse message.
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

/// Mutable solver state shared by all three minimizers.
struct OptimizerState<F, G, H> {
    cost: F,
    grad: G,
    hessian: H,
    param: DVector<f64>,
    grad_cache: DVector<f64>,
    cost_cache: f64,
}

impl<F, G, H> OptimizerState<F, G, H>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
    H: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    fn new(cost: F, grad: G, hessian: H, init: DVector<f64>, options: &Options) -> Result<Self, String> {
        validate(&init, options)?;
        let grad_cache = grad(&init);
        if !grad_cache.iter().all(|x| x.is_finite()) {
            return Err("initial gradient is non-finite".to_string());
        }
        let cost_cache = cost(&init);
        Ok(OptimizerState {
            cost,
            grad,
            hessian,
            param: init,
            grad_cache,
            cost_cache,
        })
    }

    fn cost_fn(&self) -> &F {
        &self.cost
    }
    fn grad_fn(&self) -> &G {
        &self.grad
    }
    fn grad(&self) -> &DVector<f64> {
        &self.grad_cache
    }
    fn grad_norm(&self) -> f64 {
        self.grad_cache.norm()
    }
    fn hessian(&self) -> Result<DMatrix<f64>, String> {
        Ok((self.hessian)(&self.param))
    }

    /// Advance `param` by `step`, refreshing the cached gradient and cost.
    fn step(&mut self, step: &DVector<f64>) {
        self.param = self.param.clone() + step.clone();
        self.grad_cache = (self.grad)(&self.param);
        self.cost_cache = (self.cost)(&self.param);
    }

    fn finish(self, iters: u64, converged: bool, termination: &'static str) -> Solution {
        Solution {
            param: self.param,
            cost: self.cost_cache,
            iters,
            converged,
            termination: termination.to_string(),
        }
    }
}

/// More-Thuente (1994) cubic-interpolation line search along descent direction
/// `d` from `x`. Returns a non-negative step length `alpha`.
///
/// Uses the analytic gradient (no finite differences): `phi(alpha) =
/// cost(x + alpha*d)`, `phi'(alpha) = grad(x + alpha*d)·d`. Implements the
/// strong Wolfe conditions (sufficient decrease `c1 = 1e-4`, curvature `c2 =
/// 0.1`) with cubic interpolation between bracketed points (Nocedal & Wright
/// Algorithms 3.5/3.6).
fn line_search<F, G>(cost: &F, grad: &G, x: &DVector<f64>, d: &DVector<f64>) -> f64
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let c1 = 1e-4;
    let c2 = 0.1;
    let dphi0 = grad(x).dot(d);
    if dphi0 >= 0.0 {
        return 0.0; // not a descent direction
    }
    let phi0 = cost(x);

    let mut a_lo = 0.0;
    let mut phi_lo = phi0;
    let mut dphi_lo = dphi0;
    let mut a_hi = 1.0;
    let mut a = a_hi;

    for _ in 0..40 {
        let xa = x.clone() + d.clone() * a;
        let phi_a = cost(&xa);
        if phi_a > phi0 + c1 * a * dphi0 || (a_lo != 0.0 && phi_a >= phi_lo) {
            return zoom(cost, grad, x, d, a_lo, a_hi, phi_lo, dphi_lo, phi_a);
        }
        let dphi_a = grad(&xa).dot(d);
        if dphi_a.abs() <= -c2 * dphi0 {
            return a;
        }
        if dphi_a >= 0.0 {
            return zoom(cost, grad, x, d, a, a_lo, phi_a, dphi_a, phi_lo);
        }
        a_lo = a;
        phi_lo = phi_a;
        dphi_lo = dphi_a;
        a_hi = a;
        a = a * 2.0;
    }
    a
}

/// Zoom phase of the More-Thuente line search: narrow the bracket `[a_lo, a_hi]`
/// (where `a_lo` satisfies the sufficient-decrease condition and has a descent
/// gradient) by cubic interpolation until the strong Wolfe conditions hold.
#[allow(clippy::too_many_arguments)]
fn zoom<F, G>(
    cost: &F,
    grad: &G,
    x: &DVector<f64>,
    d: &DVector<f64>,
    mut a_lo: f64,
    mut a_hi: f64,
    mut phi_lo: f64,
    mut dphi_lo: f64,
    mut phi_hi: f64,
) -> f64
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let c1 = 1e-4;
    let c2 = 0.1;
    let dphi0 = grad(x).dot(d);

    for _ in 0..40 {
        let mut dphi_hi = grad(&(x.clone() + d.clone() * a_hi)).dot(d);
        let mut a = cubic_min(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi);
        if a <= a_lo || a >= a_hi {
            a = 0.5 * (a_lo + a_hi);
        }
        let xa = x.clone() + d.clone() * a;
        let phi_a = cost(&xa);
        if phi_a > cost(x) + c1 * a * dphi0 || phi_a >= phi_lo {
            a_hi = a;
            phi_hi = phi_a;
        } else {
            let dphi_a = grad(&xa).dot(d);
            if dphi_a * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
                phi_hi = phi_lo;
                dphi_hi = dphi_lo;
            }
            a_lo = a;
            phi_lo = phi_a;
            dphi_lo = dphi_a;
            if dphi_a.abs() <= -c2 * dphi0 {
                return a;
            }
        }
        if (a_hi - a_lo).abs() <= 1e-12 * (1.0 + a_hi.abs()) {
            return a;
        }
    }
    0.5 * (a_lo + a_hi)
}

/// Cubic-interpolation minimizer in `(al, ah)` given `phi`/`dphi` at both ends
/// (Nocedal & Wright, eq. 3.59).
fn cubic_min(al: f64, phi_al: f64, dphi_al: f64, ah: f64, phi_ah: f64, dphi_ah: f64) -> f64 {
    let d1 = dphi_al + dphi_ah - 3.0 * (phi_al - phi_ah) / (al - ah);
    let d2_sq = d1 * d1 - dphi_al * dphi_ah;
    if d2_sq < 0.0 {
        return 0.5 * (al + ah);
    }
    let d2 = (ah - al).signum() * d2_sq.sqrt();
    let denom = dphi_ah - dphi_al + 2.0 * d2;
    if denom.abs() < 1e-300 {
        return 0.5 * (al + ah);
    }
    ah - (ah - al) * (dphi_ah + d2 - d1) / denom
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
        // Zero gradient at the start: the run must stop immediately, converged.
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

        // Nonlinear CG on Rosenbrock converges slowly without preconditioning;
        // give it a generous iteration budget.
        let best =
            minimize_conjugate_gradient(cost, grad, DVector::from_vec(vec![-1.2, 1.0]), 5000)
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
