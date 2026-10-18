//! Optimisation with gradients taken automatically by reverse-mode autodiff.
//!
//! The caller writes the cost once, as a function on the reverse-mode tape. The
//! gradient comes from `tpt-math-autodiff-rev`, so no derivative code is needed.

use core::cell::RefCell;

use tpt_math_autodiff_rev::{value_and_gradient, GradientTape, Variable};
use tpt_math_linalg_dense::DVector;

use crate::minimize_lbfgs;
/// A cached evaluation: the point, the cost there, and the gradient there.
type Evaluation = (Vec<f64>, f64, Vec<f64>);

/// Minimise a cost written on the autodiff tape, with the gradient from reverse
/// mode, by limited-memory BFGS.
///
/// `cost` receives the tape and the parameters as tape variables, and returns
/// the scalar cost. Each cost and gradient evaluation is one tape. Results are
/// cached by point, so a point the line search visits twice is only taped once.
///
/// `memory` and `max_iters` are as for [`minimize_lbfgs`].
///
/// # Errors
///
/// Returns the errors of [`minimize_lbfgs`].
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_dense::DVector;
/// use tpt_math_optimize_general::minimize_lbfgs_autodiff;
///
/// // f(x, y) = (x − 1)² + 10 (y + 2)², minimised at (1, −2).
/// let best = minimize_lbfgs_autodiff(
///     |_t, v| (v[0] - 1.0) * (v[0] - 1.0) + 10.0 * (v[1] + 2.0) * (v[1] + 2.0),
///     DVector::from_vec(vec![0.0, 0.0]),
///     5,
///     200,
/// )
/// .unwrap();
/// assert!((best[0] - 1.0).abs() < 1e-6 && (best[1] + 2.0).abs() < 1e-6);
/// ```
pub fn minimize_lbfgs_autodiff<F>(
    cost: F,
    init: DVector<f64>,
    memory: usize,
    max_iters: u64,
) -> Result<DVector<f64>, String>
where
    F: for<'t> Fn(&'t GradientTape, &[Variable<'t>]) -> Variable<'t>,
{
    // The last evaluated point and its value and gradient.
    let cache: RefCell<Option<Evaluation>> = RefCell::new(None);
    let evaluate = |p: &DVector<f64>| {
        let key: Vec<f64> = p.iter().copied().collect();
        if let Some((point, value, grad)) = cache.borrow().as_ref() {
            if *point == key {
                return (*value, grad.clone());
            }
        }
        let (value, grad) = value_and_gradient(&key, |t, v| cost(t, v));
        *cache.borrow_mut() = Some((key, value, grad.clone()));
        (value, grad)
    };
    minimize_lbfgs(
        |p| evaluate(p).0,
        |p| DVector::from_vec(evaluate(p).1),
        init,
        memory,
        max_iters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_bowl_is_minimised() {
        let best = minimize_lbfgs_autodiff(
            |_t, v| {
                let a = v[0] - 3.0;
                let b = v[1] + 1.0;
                a * a + 4.0 * b * b
            },
            DVector::from_vec(vec![0.0, 0.0]),
            4,
            200,
        )
        .unwrap();
        assert!((best[0] - 3.0).abs() < 1e-6 && (best[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn rosenbrock_reaches_the_valley_floor() {
        let best = minimize_lbfgs_autodiff(
            |_t, v| {
                let a = v[1] - v[0] * v[0];
                let b = 1.0 - v[0];
                a * a * 100.0 + b * b
            },
            DVector::from_vec(vec![-1.2, 1.0]),
            8,
            2000,
        )
        .unwrap();
        assert!(
            (best[0] - 1.0).abs() < 1e-4 && (best[1] - 1.0).abs() < 1e-4,
            "{best:?}"
        );
    }
}
