//! Nonlinear least squares: Levenberg–Marquardt, optionally with box bounds,
//! and a `curve_fit` convenience for models `y ≈ f(x; p)`.
//!
//! The problem is `min ½‖r(p)‖²` for a residual vector `r(p)`. Each step solves
//! the damped normal equations `(JᵀJ + μ·I) δ = −Jᵀr`, and the damping `μ` is
//! adapted from the ratio of actual to predicted cost reduction (Nielsen's
//! update). Large `μ` makes the step short and close to gradient descent, and
//! small `μ` makes it close to Gauss–Newton.
//!
//! With bounds, every trial point is projected onto the box. The convergence
//! test uses the projected gradient, which ignores components that point out of
//! the feasible set at an active bound. This is a projected method, not an
//! active-set one like L-BFGS-B, so it can take more iterations when many
//! bounds are active.

use std::vec::Vec;

use tpt_math_linalg_dense::{DMatrix, DVector, Lu};
use tpt_math_numeric::{SolveReport, SolveStatus};

/// Why a least-squares run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LsTermination {
    /// The (projected) gradient's infinity norm fell to the tolerance.
    GradientConverged,
    /// A step reduced the cost by less than the relative cost tolerance.
    CostConverged,
    /// The step became negligible relative to the parameters.
    StepConverged,
    /// The iteration budget ran out.
    MaxIterations,
    /// Every damping attempt was rejected, or the step was not finite.
    Stalled,
}

/// Settings for [`levenberg_marquardt`] and [`curve_fit`].
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresOptions {
    /// Cap on accepted iterations.
    pub max_iters: u64,
    /// Stop when the infinity norm of the (projected) gradient is at most this.
    pub gradient_tolerance: f64,
    /// Stop when a step changes the cost by at most `cost_tolerance × cost`.
    pub cost_tolerance: f64,
    /// Stop when `‖step‖ ≤ step_tolerance × (step_tolerance + ‖p‖)`.
    pub step_tolerance: f64,
    /// Componentwise lower bounds, if any.
    pub lower: Option<DVector<f64>>,
    /// Componentwise upper bounds, if any.
    pub upper: Option<DVector<f64>>,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        LeastSquaresOptions {
            max_iters: 200,
            gradient_tolerance: 1e-10,
            cost_tolerance: 1e-15,
            step_tolerance: 1e-12,
            lower: None,
            upper: None,
        }
    }
}

impl LeastSquaresOptions {
    /// Options with the given iteration budget and default tolerances.
    pub fn new(max_iters: u64) -> Self {
        LeastSquaresOptions {
            max_iters,
            ..Self::default()
        }
    }

    /// Restrict the parameters to `lower ≤ p ≤ upper` componentwise.
    #[must_use]
    pub fn with_bounds(mut self, lower: DVector<f64>, upper: DVector<f64>) -> Self {
        self.lower = Some(lower);
        self.upper = Some(upper);
        self
    }
}

/// The outcome of a least-squares run.
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresSolution {
    /// The best parameters found.
    pub param: DVector<f64>,
    /// `½‖r(param)‖²`.
    pub cost: f64,
    /// Infinity norm of the (projected) gradient at `param`.
    pub gradient_norm: f64,
    /// Number of accepted iterations.
    pub iters: u64,
    /// How the run stopped.
    pub termination: LsTermination,
}

impl LeastSquaresSolution {
    /// `true` when the run met a convergence test rather than running out of
    /// iterations or stalling.
    pub fn converged(&self) -> bool {
        !matches!(
            self.termination,
            LsTermination::MaxIterations | LsTermination::Stalled
        )
    }

    /// The outcome as a [`SolveReport`].
    #[must_use]
    pub fn report(&self) -> SolveReport<f64> {
        let status = match self.termination {
            LsTermination::MaxIterations => SolveStatus::NotConverged,
            LsTermination::Stalled => SolveStatus::Breakdown,
            _ => SolveStatus::Converged,
        };
        SolveReport {
            status,
            iterations: self.iters as usize,
            residual: self.gradient_norm,
        }
    }
}

/// Maximum rejected trial steps within one iteration before the run stalls.
const MAX_REJECTIONS: u32 = 60;

/// Minimize `½‖r(p)‖²` by Levenberg–Marquardt.
///
/// `residuals` returns the `m`-vector `r(p)`, and `jacobian` returns its
/// `m × n` derivative matrix `J(p)[i, j] = ∂r_i/∂p_j`.
///
/// # Errors
///
/// Returns an error if `init` is empty or not finite, if the bounds have the
/// wrong length or `init` lies outside them, if the residual or Jacobian sizes
/// are inconsistent, or if the residuals are not finite at the start.
pub fn levenberg_marquardt<R, J>(
    residuals: R,
    jacobian: J,
    init: DVector<f64>,
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresSolution, String>
where
    R: Fn(&DVector<f64>) -> DVector<f64>,
    J: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    let n = init.len();
    if n == 0 {
        return Err("initial parameter vector must not be empty".into());
    }
    if init.iter().any(|v| !v.is_finite()) {
        return Err("initial parameter vector must be finite".into());
    }
    let (lo, hi) = bounds_vectors(options, n)?;
    let mut x = init;
    for i in 0..n {
        if x[i] < lo[i] || x[i] > hi[i] {
            return Err(format!(
                "initial value {} for parameter {i} lies outside [{}, {}]",
                x[i], lo[i], hi[i]
            ));
        }
    }

    let mut r = residuals(&x);
    let m = r.len();
    if r.iter().any(|v| !v.is_finite()) {
        return Err("residuals are not finite at the initial point".into());
    }
    let mut jac = jacobian(&x);
    check_jacobian_shape(&jac, m, n)?;
    let mut cost = half_sum_squares(&r);
    let (mut grad, mut normal) = normal_equations(&jac, &r);

    // Initial damping: a fraction of the largest diagonal entry of JᵀJ.
    let diag_max = (0..n).map(|i| normal[(i, i)]).fold(0.0_f64, f64::max);
    let mut mu = if diag_max > 0.0 {
        1e-3 * diag_max
    } else {
        1e-3
    };
    let mut nu = 2.0_f64;

    let mut iters = 0u64;
    let mut termination = LsTermination::MaxIterations;
    let mut gradient_norm = projected_gradient_norm(&x, &grad, &lo, &hi);

    'outer: while iters < options.max_iters {
        if gradient_norm <= options.gradient_tolerance {
            termination = LsTermination::GradientConverged;
            break;
        }

        let mut rejections = 0u32;
        loop {
            if rejections >= MAX_REJECTIONS {
                termination = LsTermination::Stalled;
                break 'outer;
            }
            let damped =
                DMatrix::from_fn(n, n, |a, b| normal[(a, b)] + if a == b { mu } else { 0.0 });
            let rhs = DVector::from_fn(n, |i| -grad[i]);
            let delta = match Lu::new(&damped).and_then(|lu| lu.solve(&rhs)) {
                Ok(d) if d.iter().all(|v| v.is_finite()) => d,
                _ => {
                    mu *= nu;
                    nu *= 2.0;
                    rejections += 1;
                    continue;
                }
            };

            let trial = DVector::from_fn(n, |i| (x[i] + delta[i]).clamp(lo[i], hi[i]));
            let step = DVector::from_fn(n, |i| trial[i] - x[i]);
            let step_norm = norm2(&step);
            if step_norm == 0.0 {
                // Every component is pinned against a bound: nothing can move.
                termination = LsTermination::StepConverged;
                break 'outer;
            }

            let r_trial = residuals(&trial);
            let cost_trial = half_sum_squares(&r_trial);
            if !cost_trial.is_finite() {
                mu *= nu;
                nu *= 2.0;
                rejections += 1;
                continue;
            }

            if cost_trial < cost {
                // Predicted reduction from the quadratic model at x.
                let predicted = -(dot(&grad, &step) + 0.5 * quadratic_form(&normal, &step));
                let rho = if predicted > 0.0 {
                    (cost - cost_trial) / predicted
                } else {
                    1.0
                };
                let previous_cost = cost;
                x = trial;
                r = r_trial;
                cost = cost_trial;
                jac = jacobian(&x);
                check_jacobian_shape(&jac, m, n)?;
                let (g, a) = normal_equations(&jac, &r);
                grad = g;
                normal = a;
                gradient_norm = projected_gradient_norm(&x, &grad, &lo, &hi);
                mu *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
                nu = 2.0;
                iters += 1;

                if previous_cost - cost <= options.cost_tolerance * previous_cost {
                    termination = LsTermination::CostConverged;
                    break 'outer;
                }
                if step_norm <= options.step_tolerance * (options.step_tolerance + norm2(&x)) {
                    termination = LsTermination::StepConverged;
                    break 'outer;
                }
                break;
            }

            mu *= nu;
            nu *= 2.0;
            rejections += 1;
        }
    }

    Ok(LeastSquaresSolution {
        param: x,
        cost,
        gradient_norm,
        iters,
        termination,
    })
}

/// Fit `y ≈ model(x; p)` by least squares, with a finite-difference Jacobian.
///
/// `model(x, p)` is evaluated at each `(xs[i], p)` and compared with `ys[i]`.
/// The Jacobian is estimated by central differences with a step scaled to each
/// parameter.
///
/// # Errors
///
/// Returns an error if `xs` and `ys` differ in length, there are fewer data
/// points than parameters, or any of the errors from [`levenberg_marquardt`].
pub fn curve_fit<M>(
    model: M,
    xs: &DVector<f64>,
    ys: &DVector<f64>,
    init: DVector<f64>,
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresSolution, String>
where
    M: Fn(f64, &DVector<f64>) -> f64,
{
    let m = xs.len();
    if ys.len() != m {
        return Err(format!("xs has {m} points, ys has {}", ys.len()));
    }
    if m < init.len() {
        return Err(format!(
            "{} data points cannot determine {} parameters",
            m,
            init.len()
        ));
    }
    let residuals = |p: &DVector<f64>| DVector::from_fn(m, |i| model(xs[i], p) - ys[i]);
    let jacobian = |p: &DVector<f64>| {
        let n = p.len();
        // Central differences: error O(h²), with h ~ ε^(1/3) relative to scale.
        let h_rel = f64::EPSILON.cbrt();
        // One column per parameter, then assembled into the m × n matrix.
        let columns: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let h = h_rel * p[k].abs().max(1.0);
                let plus = DVector::from_fn(n, |i| if i == k { p[i] + h } else { p[i] });
                let minus = DVector::from_fn(n, |i| if i == k { p[i] - h } else { p[i] });
                let width = plus[k] - minus[k];
                (0..m)
                    .map(|i| (model(xs[i], &plus) - model(xs[i], &minus)) / width)
                    .collect()
            })
            .collect();
        DMatrix::from_fn(m, n, |i, k| columns[k][i])
    };
    levenberg_marquardt(residuals, jacobian, init, options)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The bound vectors, defaulting to ±∞ when unset.
fn bounds_vectors(options: &LeastSquaresOptions, n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let lo = match &options.lower {
        Some(v) if v.len() == n => (0..n).map(|i| v[i]).collect(),
        Some(v) => return Err(format!("lower bound has length {}, expected {n}", v.len())),
        None => vec![f64::NEG_INFINITY; n],
    };
    let hi = match &options.upper {
        Some(v) if v.len() == n => (0..n).map(|i| v[i]).collect(),
        Some(v) => return Err(format!("upper bound has length {}, expected {n}", v.len())),
        None => vec![f64::INFINITY; n],
    };
    if lo
        .iter()
        .zip(&hi)
        .any(|(l, h)| l > h || l.is_nan() || h.is_nan())
    {
        return Err("each lower bound must be ≤ its upper bound".into());
    }
    Ok((lo, hi))
}

fn check_jacobian_shape(jac: &DMatrix<f64>, m: usize, n: usize) -> Result<(), String> {
    if jac.nrows() == m && jac.ncols() == n {
        Ok(())
    } else {
        Err(format!(
            "Jacobian is {}×{}, expected {m}×{n}",
            jac.nrows(),
            jac.ncols()
        ))
    }
}

fn half_sum_squares(r: &DVector<f64>) -> f64 {
    0.5 * r.iter().map(|v| v * v).sum::<f64>()
}

/// `(Jᵀr, JᵀJ)`.
fn normal_equations(jac: &DMatrix<f64>, r: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let (m, n) = (jac.nrows(), jac.ncols());
    let grad = DVector::from_fn(n, |j| (0..m).map(|i| jac[(i, j)] * r[i]).sum());
    let normal = DMatrix::from_fn(n, n, |a, b| (0..m).map(|i| jac[(i, a)] * jac[(i, b)]).sum());
    (grad, normal)
}

/// Infinity norm of the gradient, with components that push against an active
/// bound removed.
fn projected_gradient_norm(x: &DVector<f64>, grad: &DVector<f64>, lo: &[f64], hi: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..x.len() {
        let blocked = (x[i] <= lo[i] && grad[i] > 0.0) || (x[i] >= hi[i] && grad[i] < 0.0);
        if !blocked {
            worst = worst.max(grad[i].abs());
        }
    }
    worst
}

fn norm2(v: &DVector<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `sᵀ A s`.
fn quadratic_form(a: &DMatrix<f64>, s: &DVector<f64>) -> f64 {
    let n = s.len();
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            total += s[i] * a[(i, j)] * s[j];
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-difference assertion; this crate has no `approx` dev-dependency.
    macro_rules! assert_close {
        ($a:expr, $b:expr, epsilon = $e:expr) => {
            assert!(
                ($a - $b).abs() <= $e,
                "{} is not within {} of {}",
                $a,
                $e,
                $b
            )
        };
    }

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_vec(v.to_vec())
    }

    #[test]
    fn fits_a_line_exactly() {
        // Residuals r_i = (a + b t_i) - y_i with data from a = 2, b = -3.
        let ts = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = ts.iter().map(|t| 2.0 - 3.0 * t).collect();
        let residuals =
            |p: &DVector<f64>| DVector::from_fn(ts.len(), |i| p[0] + p[1] * ts[i] - ys[i]);
        let jacobian = |_: &DVector<f64>| {
            DMatrix::from_fn(ts.len(), 2, |i, j| if j == 0 { 1.0 } else { ts[i] })
        };
        let sol = levenberg_marquardt(
            residuals,
            jacobian,
            dv(&[0.0, 0.0]),
            &LeastSquaresOptions::new(50),
        )
        .unwrap();
        assert_close!(sol.param[0], 2.0, epsilon = 1e-9);
        assert_close!(sol.param[1], -3.0, epsilon = 1e-9);
        assert!(sol.cost < 1e-18);
        assert!(sol.converged());
    }

    #[test]
    fn curve_fit_recovers_exponential_decay() {
        // y = 5 · exp(−0.7 x), noiseless.
        let xs: DVector<f64> = DVector::from_fn(12, |i| 0.25 * i as f64);
        let ys: DVector<f64> = DVector::from_fn(12, |i| 5.0 * (-0.7 * xs[i]).exp());
        let model = |x: f64, p: &DVector<f64>| p[0] * (-p[1] * x).exp();
        let sol = curve_fit(
            model,
            &xs,
            &ys,
            dv(&[1.0, 0.1]),
            &LeastSquaresOptions::new(200),
        )
        .unwrap();
        assert_close!(sol.param[0], 5.0, epsilon = 1e-6);
        assert_close!(sol.param[1], 0.7, epsilon = 1e-6);
    }

    #[test]
    fn curve_fit_handles_noisy_data_near_the_least_squares_optimum() {
        // Deterministic "noise" with zero mean over the sample.
        let xs: DVector<f64> = DVector::from_fn(40, |i| i as f64 / 10.0);
        let noise = |i: usize| if i % 2 == 0 { 0.05 } else { -0.05 };
        let ys: DVector<f64> = DVector::from_fn(40, |i| 1.5 * xs[i] + 0.5 + noise(i));
        let model = |x: f64, p: &DVector<f64>| p[0] * x + p[1];
        let sol = curve_fit(
            model,
            &xs,
            &ys,
            dv(&[0.0, 0.0]),
            &LeastSquaresOptions::new(100),
        )
        .unwrap();
        assert_close!(sol.param[0], 1.5, epsilon = 1e-2);
        assert_close!(sol.param[1], 0.5, epsilon = 1e-1);
        assert!(sol.converged());
    }

    #[test]
    fn bounds_keep_the_solution_feasible_and_pin_an_active_bound() {
        // Unconstrained minimum of (p - 3)² is p = 3; with p ≤ 2 the answer is 2.
        let residuals = |p: &DVector<f64>| dv(&[p[0] - 3.0]);
        let jacobian = |_: &DVector<f64>| DMatrix::from_vec(1, 1, vec![1.0]);
        let options = LeastSquaresOptions::new(100).with_bounds(dv(&[-10.0]), dv(&[2.0]));
        let sol = levenberg_marquardt(residuals, jacobian, dv(&[0.0]), &options).unwrap();
        assert_close!(sol.param[0], 2.0, epsilon = 1e-12);
        assert!(sol.converged());
    }

    #[test]
    fn interior_bounds_do_not_change_an_interior_optimum() {
        let residuals = |p: &DVector<f64>| dv(&[p[0] - 3.0]);
        let jacobian = |_: &DVector<f64>| DMatrix::from_vec(1, 1, vec![1.0]);
        let options = LeastSquaresOptions::new(100).with_bounds(dv(&[0.0]), dv(&[10.0]));
        let sol = levenberg_marquardt(residuals, jacobian, dv(&[1.0]), &options).unwrap();
        assert_close!(sol.param[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn rejects_bad_inputs() {
        let residuals = |p: &DVector<f64>| dv(&[p[0]]);
        let jacobian = |_: &DVector<f64>| DMatrix::from_vec(1, 1, vec![1.0]);
        let opts = LeastSquaresOptions::new(10);
        assert!(levenberg_marquardt(residuals, jacobian, dv(&[]), &opts).is_err());
        assert!(levenberg_marquardt(residuals, jacobian, dv(&[f64::NAN]), &opts).is_err());
        let outside = LeastSquaresOptions::new(10).with_bounds(dv(&[1.0]), dv(&[2.0]));
        assert!(levenberg_marquardt(residuals, jacobian, dv(&[0.0]), &outside).is_err());
        let wrong_len = LeastSquaresOptions::new(10).with_bounds(dv(&[0.0, 0.0]), dv(&[2.0, 2.0]));
        assert!(levenberg_marquardt(residuals, jacobian, dv(&[1.0]), &wrong_len).is_err());
    }

    #[test]
    fn curve_fit_rejects_underdetermined_and_mismatched_data() {
        let model = |x: f64, p: &DVector<f64>| p[0] + p[1] * x;
        let one = dv(&[1.0]);
        let opts = LeastSquaresOptions::new(10);
        assert!(curve_fit(model, &one, &one, dv(&[0.0, 0.0]), &opts).is_err());
        assert!(curve_fit(model, &dv(&[1.0, 2.0]), &one, dv(&[0.0, 0.0]), &opts).is_err());
    }

    #[test]
    fn report_maps_termination_to_status() {
        let mk = |t| LeastSquaresSolution {
            param: dv(&[0.0]),
            cost: 0.0,
            gradient_norm: 0.0,
            iters: 3,
            termination: t,
        };
        assert_eq!(
            mk(LsTermination::GradientConverged).report().status,
            SolveStatus::Converged
        );
        assert_eq!(
            mk(LsTermination::MaxIterations).report().status,
            SolveStatus::NotConverged
        );
        assert_eq!(
            mk(LsTermination::Stalled).report().status,
            SolveStatus::Breakdown
        );
        assert_eq!(mk(LsTermination::Stalled).report().iterations, 3);
    }
}
