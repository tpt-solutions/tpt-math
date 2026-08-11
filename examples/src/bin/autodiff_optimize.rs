//! Cross-crate example: `tpt-math-autodiff` + `tpt-math-optimize`.
//!
//! Uses reverse-mode autodiff (`tpt-math-autodiff`) to obtain the gradient of a
//! scalar objective, then hands the cost + gradient closures to the gradient
//! descent minimizer in `tpt-math-optimize` (which wraps `argmin`).

use tpt_math_autodiff::rev::value_and_gradient;
use tpt_math_optimize::general::minimize_gradient_descent;
use tpt_math_optimize::general::tpt_math_linalg_dense::DVector;

fn main() {
    // f(x) = (x - 2)^2, minimized at x = 2 with f(2) = 0.
    let cost = |p: &DVector<f64>| {
        let (v, _) = value_and_gradient(&[p[0]], |_t, v| (v[0] - 2.0).powi(2));
        v
    };
    let grad = |p: &DVector<f64>| {
        let (_, g) = value_and_gradient(&[p[0]], |_t, v| (v[0] - 2.0).powi(2));
        DVector::from_vec(vec![g[0]])
    };

    let best = minimize_gradient_descent(cost, grad, DVector::from_vec(vec![0.0]), 200).unwrap();

    assert!(
        (best[0] - 2.0).abs() < 1e-6,
        "optimum not reached: {:?}",
        best
    );
    println!("autodiff+optimize: minimum at x = {:.6}", best[0]);
}
