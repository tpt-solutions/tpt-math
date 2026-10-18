//! Jacobians and Hessians built on the reverse-mode tape.
//!
//! The Jacobian is exact: each output costs one reverse pass over the same tape.
//! The Hessian is the central difference of the exact reverse-mode gradient, so it
//! carries a truncation error of order `h²` and a rounding error of order `ε/h`.
//! It is symmetrised afterwards.

use crate::rev::{GradientTape, Variable};

/// The gradient of a scalar function at `x`, from one reverse pass.
fn gradient_at<F>(x: &[f64], f: &F) -> Vec<f64>
where
    F: for<'t> Fn(&[Variable<'t>]) -> Variable<'t>,
{
    let tape = GradientTape::new();
    let vars: Vec<Variable<'_>> = x.iter().map(|&v| tape.var(v)).collect();
    let out = f(&vars);
    tape.gradient(out, &vars)
}

/// The Jacobian `J[i][j] = ∂fᵢ/∂xⱼ` of a vector function at `x`.
///
/// Returns one row per output and one column per input. A function with no
/// outputs gives an empty matrix.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff::derivatives::jacobian;
///
/// // f(x, y) = (x·y, x + sin y), so J = [[y, x], [1, cos y]].
/// let j = jacobian(&[2.0, 0.5], |v| vec![v[0] * v[1], v[0] + v[1].sin()]);
/// assert!((j[0][0] - 0.5).abs() < 1e-12 && (j[0][1] - 2.0).abs() < 1e-12);
/// assert!((j[1][1] - 0.5_f64.cos()).abs() < 1e-12);
/// ```
pub fn jacobian<F>(x: &[f64], f: F) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&[Variable<'t>]) -> Vec<Variable<'t>>,
{
    let tape = GradientTape::new();
    let vars: Vec<Variable<'_>> = x.iter().map(|&v| tape.var(v)).collect();
    let outputs = f(&vars);
    outputs
        .into_iter()
        .map(|out| tape.gradient(out, &vars))
        .collect()
}

/// The Hessian `H[i][j] = ∂²f/∂xᵢ∂xⱼ` of a scalar function at `x`.
///
/// Each step is `h = 10⁻⁵ · max(1, |xᵢ|)`. The result is symmetrised, so
/// `H[i][j] == H[j][i]` exactly.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff::derivatives::hessian;
///
/// // f(x, y) = x²y + sin y, so H = [[2y, 2x], [2x, −sin y]].
/// let h = hessian(&[1.5, 0.3], |v| v[0] * v[0] * v[1] + v[1].sin());
/// assert!((h[0][0] - 0.6).abs() < 1e-6);
/// assert!((h[0][1] - 3.0).abs() < 1e-6);
/// assert!((h[1][1] + 0.3_f64.sin()).abs() < 1e-6);
/// ```
pub fn hessian<F>(x: &[f64], f: F) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&[Variable<'t>]) -> Variable<'t>,
{
    let n = x.len();
    let mut h = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let step = 1e-5 * x[i].abs().max(1.0);
        let mut plus = x.to_vec();
        let mut minus = x.to_vec();
        plus[i] += step;
        minus[i] -= step;
        let gp = gradient_at(&plus, &f);
        let gm = gradient_at(&minus, &f);
        for (hij, (p, m)) in h[i].iter_mut().zip(gp.iter().zip(&gm)) {
            *hij = (p - m) / (2.0 * step);
        }
    }
    // Symmetrising reads and writes two entries at once, so index loops are the clear form.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobian_matches_the_analytic_matrix() {
        let (x, y) = (1.3_f64, -0.4_f64);
        let j = jacobian(&[x, y], |v| {
            vec![v[0] * v[1], v[0].sin() + v[1] * v[1], v[0] - v[1]]
        });
        let want = [[y, x], [x.cos(), 2.0 * y], [1.0, -1.0]];
        for (row, want_row) in j.iter().zip(want) {
            for (got, w) in row.iter().zip(want_row) {
                assert!((got - w).abs() < 1e-12, "{got} vs {w}");
            }
        }
    }

    #[test]
    fn jacobian_of_no_outputs_is_empty() {
        assert!(jacobian(&[1.0], |_| Vec::new()).is_empty());
    }

    #[test]
    fn hessian_matches_the_analytic_matrix() {
        let (x, y) = (0.7_f64, 1.9_f64);
        let h = hessian(&[x, y], |v| (v[0] * v[1]).tanh() + v[1].exp());
        // f = tanh(u) + eʸ with u = xy, and sech²u = 1 − tanh²u. Then
        // f_xx = −2y²·sech²u·tanh u, f_xy = sech²u − 2xy·sech²u·tanh u,
        // f_yy = eʸ − 2x²·sech²u·tanh u.
        let u = x * y;
        let sech2 = 1.0 - u.tanh() * u.tanh();
        let hxx = -2.0 * y * y * sech2 * u.tanh();
        let hxy = sech2 - 2.0 * u * sech2 * u.tanh();
        let hyy = y.exp() - 2.0 * x * x * sech2 * u.tanh();
        assert!((h[0][0] - hxx).abs() < 1e-6, "xx {} vs {hxx}", h[0][0]);
        assert!((h[0][1] - hxy).abs() < 1e-6, "xy {} vs {hxy}", h[0][1]);
        assert!((h[1][1] - hyy).abs() < 1e-6, "yy {} vs {hyy}", h[1][1]);
        assert_eq!(h[0][1], h[1][0]);
    }
}
