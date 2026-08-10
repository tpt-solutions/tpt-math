//! Reverse-mode (tape / Wengert-list) automatic differentiation.
//!
//! Where forward mode (see the `tpt-math-autodiff-fwd` crate) propagates one
//! derivative direction per evaluation, reverse mode records the computation
//! on a [`GradientTape`] and then replays it backwards, producing **all**
//! partial derivatives of a single scalar output in one pass. That makes it
//! the right tool for gradients of many-input, one-output functions such as
//! loss functions and objective functions.
//!
//! # How it works
//!
//! * [`GradientTape`] owns a Wengert list: an append-only sequence of nodes,
//!   each holding one primitive [`Op`] and the primal value it produced.
//! * [`Variable`] is a `Copy` handle to a node — its index, its value, and a
//!   back-pointer to the tape. Arithmetic operators and math methods evaluate
//!   the primal value, push a node, and return a handle to it.
//! * [`Op::diff`] supplies the local derivative rule for each primitive as a
//!   small list of `(operand, ∂out/∂operand)` [`Partials`].
//! * [`GradientTape::backward`] seeds the output node with an adjoint of `1`
//!   and scans the tape in reverse. Because operands are always recorded
//!   before the nodes that consume them, reverse index order *is* a reverse
//!   topological order, so each node's adjoint is complete by the time it is
//!   visited. Contributions are accumulated, so re-using a subexpression (a
//!   diamond in the DAG) is handled automatically.
//!
//! # Examples
//!
//! ```
//! use tpt_math_autodiff_rev::GradientTape;
//!
//! let tape = GradientTape::new();
//! let x = tape.var(2.0);
//! let y = tape.var(3.0);
//! let z = x * x + y.sin();
//! let g = tape.gradient(z, &[x, y]);
//!
//! assert!((g[0] - 4.0).abs() < 1e-9); // dz/dx = 2x
//! assert!((g[1] - 3.0_f64.cos()).abs() < 1e-9); // dz/dy = cos(y)
//! ```
//!
//! Supported primitives: `+`, `-`, `*`, `/` (between variables and with plain
//! `f64` on either side), unary `-`, [`sin`], [`cos`], [`tan`], [`exp`],
//! [`ln`], [`sqrt`], [`powi`], [`powf`] and [`recip`], plus constants that
//! never carry a gradient.
//!
//! [`sin`]: Variable::sin
//! [`cos`]: Variable::cos
//! [`tan`]: Variable::tan
//! [`exp`]: Variable::exp
//! [`ln`]: Variable::ln
//! [`sqrt`]: Variable::sqrt
//! [`powi`]: Variable::powi
//! [`powf`]: Variable::powf
//! [`recip`]: Variable::recip

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

mod op;
mod tape;
mod var;

pub use crate::op::{Op, Partials, MAX_OPERANDS};
pub use crate::tape::{Gradient, GradientTape};
pub use crate::var::Variable;

/// Everything needed to build and differentiate an expression.
pub mod prelude {
    pub use crate::{Gradient, GradientTape, Op, Partials, Variable};
}

/// Evaluate `f` at `inputs` and return `(value, gradient)`.
///
/// A convenience wrapper that creates a tape, seeds one input [`Variable`] per
/// entry of `inputs`, evaluates `f`, and runs the backward pass — useful when
/// the tape itself is not interesting to the caller.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff_rev::value_and_gradient;
///
/// // f(x, y) = x * y + sin(x)
/// let (value, grad) = value_and_gradient(&[2.0, 3.0], |_tape, v| v[0] * v[1] + v[0].sin());
///
/// assert!((value - (6.0 + 2.0_f64.sin())).abs() < 1e-9);
/// assert!((grad[0] - (3.0 + 2.0_f64.cos())).abs() < 1e-9);
/// assert!((grad[1] - 2.0).abs() < 1e-9);
/// ```
pub fn value_and_gradient<F>(inputs: &[f64], f: F) -> (f64, Vec<f64>)
where
    F: for<'t> FnOnce(&'t GradientTape, &[Variable<'t>]) -> Variable<'t>,
{
    let tape = GradientTape::with_capacity(inputs.len() * 4);
    let vars = tape.vars(inputs);
    let output = f(&tape, &vars);
    let gradient = tape.gradient(output, &vars);
    (output.value(), gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_autodiff_fwd::Dual;

    /// Tolerance for exact analytic comparisons.
    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < TOL,
            "expected {expected}, got {actual} (delta {})",
            (actual - expected).abs()
        );
    }

    /// f(x) = x^2 at x = 3 => f'(x) = 2x = 6.
    #[test]
    fn square_derivative() {
        let tape = GradientTape::new();
        let x = tape.var(3.0);
        let f = x * x;
        let g = tape.gradient(f, &[x]);
        assert_close(f.value(), 9.0);
        assert_close(g[0], 6.0);
    }

    /// f(x, y) = x*y + sin(x) at (2, 3)
    /// => df/dx = y + cos(x) = 3 + cos(2), df/dy = x = 2.
    #[test]
    fn product_plus_sine_gradient() {
        let tape = GradientTape::new();
        let x = tape.var(2.0);
        let y = tape.var(3.0);
        let f = x * y + x.sin();
        let g = tape.gradient(f, &[x, y]);

        assert_close(f.value(), 6.0 + 2.0_f64.sin());
        assert_close(g[0], 3.0 + 2.0_f64.cos());
        assert_close(g[1], 2.0);
    }

    /// A deep expression with a shared subexpression (a diamond in the DAG),
    /// which only differentiates correctly if the backward pass accumulates
    /// contributions in reverse topological order.
    ///
    /// u = x*y, f = sin(u)*u + u
    /// => df/du = u*cos(u) + sin(u) + 1, df/dx = y * df/du, df/dy = x * df/du.
    #[test]
    fn nested_shared_subexpression() {
        let tape = GradientTape::new();
        let (xv, yv) = (1.3_f64, 2.1_f64);
        let x = tape.var(xv);
        let y = tape.var(yv);

        let u = x * y;
        let f = u.sin() * u + u;

        let uv = xv * yv;
        let df_du = uv * uv.cos() + uv.sin() + 1.0;

        let g = tape.gradient(f, &[x, y]);
        assert_close(f.value(), uv.sin() * uv + uv);
        assert_close(g[0], yv * df_du);
        assert_close(g[1], xv * df_du);
    }

    /// A nested expression whose closed form collapses: ln(exp(x^2 + y)) is
    /// exactly x^2 + y, so the gradient must be (2x, 1) despite the long tape.
    #[test]
    fn nested_inverse_functions_collapse() {
        let tape = GradientTape::new();
        let x = tape.var(0.7);
        let y = tape.var(-0.3);
        let f = (x.powi(2) + y).exp().ln();
        let g = tape.gradient(f, &[x, y]);

        assert_close(f.value(), 0.7 * 0.7 - 0.3);
        assert_close(g[0], 1.4);
        assert_close(g[1], 1.0);
    }

    /// Division and integer powers: f = x^3 / y.
    #[test]
    fn quotient_and_power_gradient() {
        let tape = GradientTape::new();
        let (xv, yv) = (2.0_f64, 4.0_f64);
        let x = tape.var(xv);
        let y = tape.var(yv);
        let f = x.powi(3) / y;
        let g = tape.gradient(f, &[x, y]);

        assert_close(f.value(), 2.0);
        assert_close(g[0], 3.0 * xv * xv / yv);
        assert_close(g[1], -xv.powi(3) / (yv * yv));
    }

    /// Every primitive at once, checked against central finite differences.
    #[test]
    fn matches_central_finite_differences() {
        // f(x, y) = sin(x)*cos(y) + tan(x/4) + exp(x - y) + ln(x + 3)
        //           + sqrt(y) + x^3 + y^1.5 - x/y
        fn eval(tape: &GradientTape, x: Variable<'_>, y: Variable<'_>) -> f64 {
            let f = x.sin() * y.cos()
                + (x / 4.0).tan()
                + (x - y).exp()
                + (x + 3.0).ln()
                + y.sqrt()
                + x.powi(3)
                + y.powf(1.5)
                - x / y;
            let _ = tape;
            f.value()
        }

        let (xv, yv) = (0.9_f64, 1.7_f64);
        let tape = GradientTape::new();
        let x = tape.var(xv);
        let y = tape.var(yv);
        let f = x.sin() * y.cos()
            + (x / 4.0).tan()
            + (x - y).exp()
            + (x + 3.0).ln()
            + y.sqrt()
            + x.powi(3)
            + y.powf(1.5)
            - x / y;
        let g = tape.gradient(f, &[x, y]);

        let h = 1e-6;
        let fd = |dx: f64, dy: f64| {
            let t = GradientTape::new();
            let a = t.var(xv + dx);
            let b = t.var(yv + dy);
            eval(&t, a, b)
        };
        let fd_x = (fd(h, 0.0) - fd(-h, 0.0)) / (2.0 * h);
        let fd_y = (fd(0.0, h) - fd(0.0, -h)) / (2.0 * h);

        assert!((g[0] - fd_x).abs() < 1e-6, "d/dx: {} vs {fd_x}", g[0]);
        assert!((g[1] - fd_y).abs() < 1e-6, "d/dy: {} vs {fd_y}", g[1]);
    }

    /// Reverse mode must agree with the forward-mode dual numbers from
    /// `tpt-math-autodiff-fwd` on the same function.
    #[test]
    fn agrees_with_forward_mode() {
        let (xv, yv) = (1.3_f64, 2.7_f64);

        // Reverse mode: f = sin(x*y) + exp(x)/y - ln(x + y).
        let tape = GradientTape::new();
        let x = tape.var(xv);
        let y = tape.var(yv);
        let f = (x * y).sin() + x.exp() / y - (x + y).ln();
        let rev = tape.gradient(f, &[x, y]);

        // Forward mode: the same expression over `Dual<f64, 2>`.
        let dx = Dual::<f64, 2>::variable(xv, 0);
        let dy = Dual::<f64, 2>::variable(yv, 1);
        let df = (dx * dy).sin() + dx.exp() / dy - (dx + dy).ln();

        assert_close(f.value(), df.re());
        assert_close(rev[0], df.du(0));
        assert_close(rev[1], df.du(1));
    }

    /// The convenience wrapper produces the same answers as manual taping.
    #[test]
    fn value_and_gradient_helper() {
        let (value, grad) = value_and_gradient(&[2.0, 3.0], |_tape, v| v[0] * v[1] + v[0].sin());
        assert_close(value, 6.0 + 2.0_f64.sin());
        assert_close(grad[0], 3.0 + 2.0_f64.cos());
        assert_close(grad[1], 2.0);
    }

    /// A many-input function: reverse mode gets the whole gradient from one
    /// backward pass.
    #[test]
    fn many_inputs_single_backward_pass() {
        let values: Vec<f64> = (1..=8).map(|i| f64::from(i) / 4.0).collect();
        let tape = GradientTape::new();
        let vars = tape.vars(&values);

        // f = sum_i (x_i^2 * sin(x_0))
        let s = vars[0].sin();
        let mut f = tape.constant(0.0);
        for &v in &vars {
            f += v * v * s;
        }

        let sum_sq: f64 = values.iter().map(|v| v * v).sum();
        let grad = tape.backward(f);

        for (i, (&v, var)) in values.iter().zip(vars.iter()).enumerate() {
            let mut expected = 2.0 * v * values[0].sin();
            if i == 0 {
                expected += sum_sq * values[0].cos();
            }
            assert_close(grad.wrt(*var), expected);
        }
    }

    /// Gradient descent smoke test: the tape drives an optimiser to the
    /// minimum of the Rosenbrock-like bowl f = (x-1)^2 + 10*(y - x^2)^2.
    #[test]
    fn drives_gradient_descent() {
        let (mut x, mut y) = (-1.2_f64, 1.0_f64);
        for _ in 0..20_000 {
            let (_, g) = value_and_gradient(&[x, y], |_t, v| {
                let a = v[0] - 1.0;
                let b = v[1] - v[0] * v[0];
                a * a + 10.0 * b * b
            });
            x -= 1e-3 * g[0];
            y -= 1e-3 * g[1];
        }
        assert!((x - 1.0).abs() < 1e-3, "x = {x}");
        assert!((y - 1.0).abs() < 1e-3, "y = {y}");
    }
}
