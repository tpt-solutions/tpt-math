//! Fixed-step explicit Runge–Kutta integration of ODE systems.
//!
//! The state is a fixed-size array, so no allocation is needed and the
//! integrators work in `no_std` builds.

use crate::Scalar;

/// One classical fourth-order Runge–Kutta step for `y' = f(t, y)`.
///
/// Advances the state from `t` to `t + h`. The local error is `O(h⁵)`.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::ode::rk4_step;
///
/// // y' = −y, one step from y(0) = 1 with h = 0.1.
/// let y = rk4_step(|_t: f64, y: &[f64; 1]| [-y[0]], 0.0, [1.0], 0.1);
/// assert!((y[0] - (-0.1f64).exp()).abs() < 1e-7);
/// ```
pub fn rk4_step<T, const N: usize, F>(f: F, t: T, y: [T; N], h: T) -> [T; N]
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
{
    let two = T::one() + T::one();
    let six = two + two + two;
    let half_h = h / two;
    let k1 = f(t, &y);
    let y2 = core::array::from_fn(|i| y[i] + half_h * k1[i]);
    let k2 = f(t + half_h, &y2);
    let y3 = core::array::from_fn(|i| y[i] + half_h * k2[i]);
    let k3 = f(t + half_h, &y3);
    let y4 = core::array::from_fn(|i| y[i] + h * k3[i]);
    let k4 = f(t + h, &y4);
    core::array::from_fn(|i| y[i] + h * (k1[i] + two * k2[i] + two * k3[i] + k4[i]) / six)
}

/// Integrate `y' = f(t, y)` from `t0` to `t1` with `steps` equal RK4 steps and
/// return the state at `t1`.
///
/// `steps` is clamped to at least 1.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::ode::rk4_integrate;
///
/// // Harmonic oscillator as the system (y, v)' = (v, −y): after one full
/// // period 2π the state returns to (1, 0).
/// let f = |_t: f64, s: &[f64; 2]| [s[1], -s[0]];
/// let s = rk4_integrate(f, 0.0, 2.0 * core::f64::consts::PI, [1.0, 0.0], 1000);
/// assert!((s[0] - 1.0).abs() < 1e-9 && s[1].abs() < 1e-9);
/// ```
pub fn rk4_integrate<T, const N: usize, F>(f: F, t0: T, t1: T, y0: [T; N], steps: usize) -> [T; N]
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
{
    let steps = steps.max(1);
    let h = (t1 - t0) / T::from(steps).unwrap_or(T::one());
    let mut y = y0;
    let mut t = t0;
    for _ in 0..steps {
        y = rk4_step(&f, t, y, h);
        t = t + h;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay_is_accurate() {
        let y = rk4_integrate(|_t: f64, y: &[f64; 1]| [-y[0]], 0.0, 1.0, [1.0], 100);
        assert!((y[0] - (-1.0f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn error_shrinks_like_h_to_the_fourth() {
        let exact = (-1.0f64).exp();
        let err = |steps| {
            let y = rk4_integrate(|_t: f64, y: &[f64; 1]| [-y[0]], 0.0, 1.0, [1.0], steps);
            (y[0] - exact).abs()
        };
        // Halving h should cut the error by roughly 2⁴ = 16.
        let ratio = err(10) / err(20);
        assert!(ratio > 12.0 && ratio < 20.0, "ratio {ratio}");
    }
}
