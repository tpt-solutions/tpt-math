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

/// Failure modes of the adaptive integrator [`dopri5_integrate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeError {
    /// The integrator used its whole attempt budget before reaching `t1`.
    MaxStepsExceeded,
    /// The step size shrank below the resolution of `t`. This usually means the
    /// solution blows up or the tolerances are unattainable.
    StepSizeUnderflow,
}

impl core::fmt::Display for OdeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OdeError::MaxStepsExceeded => write!(f, "step budget exhausted before reaching t1"),
            OdeError::StepSizeUnderflow => write!(f, "step size underflow"),
        }
    }
}

impl core::error::Error for OdeError {}

/// Converts an `f64` coefficient into the scalar type `T`.
fn lit<T: Scalar>(x: f64) -> T {
    T::from(x).unwrap_or_else(T::zero)
}

/// `base + h · Σ cᵢ kᵢ`, applied component-wise.
fn weighted<T: Scalar, const N: usize>(base: &[T; N], h: T, terms: &[(f64, &[T; N])]) -> [T; N] {
    core::array::from_fn(|i| {
        let mut acc = base[i];
        for (c, k) in terms {
            acc = acc + h * lit::<T>(*c) * k[i];
        }
        acc
    })
}

/// Integrate `y' = f(t, y)` from `t0` to `t1` with the adaptive Dormand–Prince
/// 5(4) pair, returning the state at `t1`.
///
/// Each step advances with the fifth-order solution and estimates its error
/// from the embedded fourth-order solution. A step is accepted when the
/// weighted RMS error is at most 1, where each component is weighted by
/// `atol + rtol · max(|y|, |y_new|)`. The step size adapts by the standard
/// controller, with factors clamped to `[0.2, 5]`. The method is FSAL, so each
/// accepted step costs six new evaluations of `f`.
///
/// `t1 < t0` integrates backwards. `max_steps` bounds the total number of step
/// attempts, accepted or rejected.
///
/// # Errors
///
/// Returns [`OdeError::MaxStepsExceeded`] if the attempt budget runs out, and
/// [`OdeError::StepSizeUnderflow`] if the step size collapses.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::ode::dopri5_integrate;
///
/// // y' = y from y(0) = 1: at t = 1 the solution is e.
/// let y = dopri5_integrate(|_t: f64, y: &[f64; 1]| [y[0]], 0.0, 1.0, [1.0], 1e-10, 1e-12, 10_000)
///     .unwrap();
/// assert!((y[0] - core::f64::consts::E).abs() < 1e-8);
/// ```
pub fn dopri5_integrate<T, const N: usize, F>(
    f: F,
    t0: T,
    t1: T,
    y0: [T; N],
    rtol: T,
    atol: T,
    max_steps: usize,
) -> Result<[T; N], OdeError>
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
{
    let span = t1 - t0;
    if span == T::zero() {
        return Ok(y0);
    }
    let sign = if span < T::zero() {
        -T::one()
    } else {
        T::one()
    };
    let total = span.abs();
    let mut t = t0;
    let mut y = y0;
    let mut h = total / lit::<T>(100.0);
    let min_step = lit::<T>(16.0) * T::epsilon() * total.max(t0.abs());
    let exponent = lit::<T>(-0.2);

    for _ in 0..max_steps {
        let remaining = (t1 - t).abs();
        if remaining == T::zero() {
            return Ok(y);
        }
        let step = h.min(remaining);
        let hs = sign * step;

        // Dormand–Prince stages. k7 is the FSAL stage, used only for the error.
        let k1 = f(t, &y);
        let k2 = f(
            t + lit::<T>(1.0 / 5.0) * hs,
            &weighted(&y, hs, &[(1.0 / 5.0, &k1)]),
        );
        let k3 = f(
            t + lit::<T>(3.0 / 10.0) * hs,
            &weighted(&y, hs, &[(3.0 / 40.0, &k1), (9.0 / 40.0, &k2)]),
        );
        let k4 = f(
            t + lit::<T>(4.0 / 5.0) * hs,
            &weighted(
                &y,
                hs,
                &[(44.0 / 45.0, &k1), (-56.0 / 15.0, &k2), (32.0 / 9.0, &k3)],
            ),
        );
        let k5 = f(
            t + lit::<T>(8.0 / 9.0) * hs,
            &weighted(
                &y,
                hs,
                &[
                    (19372.0 / 6561.0, &k1),
                    (-25360.0 / 2187.0, &k2),
                    (64448.0 / 6561.0, &k3),
                    (-212.0 / 729.0, &k4),
                ],
            ),
        );
        let k6 = f(
            t + hs,
            &weighted(
                &y,
                hs,
                &[
                    (9017.0 / 3168.0, &k1),
                    (-355.0 / 33.0, &k2),
                    (46732.0 / 5247.0, &k3),
                    (49.0 / 176.0, &k4),
                    (-5103.0 / 18656.0, &k5),
                ],
            ),
        );
        let y_new = weighted(
            &y,
            hs,
            &[
                (35.0 / 384.0, &k1),
                (500.0 / 1113.0, &k3),
                (125.0 / 192.0, &k4),
                (-2187.0 / 6784.0, &k5),
                (11.0 / 84.0, &k6),
            ],
        );
        let k7 = f(t + hs, &y_new);
        let zero = [T::zero(); N];
        let err = weighted(
            &zero,
            hs,
            &[
                (71.0 / 57600.0, &k1),
                (-71.0 / 16695.0, &k3),
                (71.0 / 1920.0, &k4),
                (-17253.0 / 339200.0, &k5),
                (22.0 / 525.0, &k6),
                (-1.0 / 40.0, &k7),
            ],
        );

        // Weighted RMS error norm.
        let mut sum_sq = T::zero();
        for i in 0..N {
            let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
            let e = err[i] / scale;
            sum_sq = sum_sq + e * e;
        }
        let error = if N == 0 {
            T::zero()
        } else {
            (sum_sq / T::from(N).unwrap_or_else(T::one)).sqrt()
        };

        let accepted = error <= T::one();
        if accepted {
            t = if step == remaining { t1 } else { t + hs };
            y = y_new;
        }
        let factor = if error == T::zero() {
            lit::<T>(5.0)
        } else {
            (lit::<T>(0.9) * error.powf(exponent))
                .max(lit::<T>(0.2))
                .min(lit::<T>(5.0))
        };
        // A rejected step may only shrink.
        let factor = if accepted {
            factor
        } else {
            factor.min(T::one())
        };
        h = step * factor;
        if accepted && (t1 - t).abs() == T::zero() {
            return Ok(y);
        }
        if h < min_step {
            return Err(OdeError::StepSizeUnderflow);
        }
    }
    Err(OdeError::MaxStepsExceeded)
}

#[cfg(test)]
mod dopri5_tests {
    use super::*;

    #[test]
    fn exponential_growth_meets_tolerance() {
        let y = dopri5_integrate(
            |_t: f64, y: &[f64; 1]| [y[0]],
            0.0,
            1.0,
            [1.0],
            1e-10,
            1e-12,
            10_000,
        )
        .unwrap();
        assert!((y[0] - core::f64::consts::E).abs() < 1e-8, "{}", y[0]);
    }

    #[test]
    fn harmonic_oscillator_returns_after_one_period() {
        let f = |_t: f64, s: &[f64; 2]| [s[1], -s[0]];
        let y = dopri5_integrate(
            f,
            0.0,
            2.0 * core::f64::consts::PI,
            [1.0, 0.0],
            1e-10,
            1e-12,
            10_000,
        )
        .unwrap();
        assert!((y[0] - 1.0).abs() < 1e-7 && y[1].abs() < 1e-7, "{y:?}");
    }

    #[test]
    fn backward_integration_inverts_forward() {
        // y' = y from t = 1 (y = e) back to t = 0 should give y = 1.
        let y = dopri5_integrate(
            |_t: f64, y: &[f64; 1]| [y[0]],
            1.0,
            0.0,
            [core::f64::consts::E],
            1e-10,
            1e-12,
            10_000,
        )
        .unwrap();
        assert!((y[0] - 1.0).abs() < 1e-8, "{}", y[0]);
    }

    #[test]
    fn zero_span_returns_initial_state() {
        let y = dopri5_integrate(
            |_t: f64, y: &[f64; 1]| [y[0]],
            2.0,
            2.0,
            [3.0],
            1e-6,
            1e-9,
            10,
        )
        .unwrap();
        assert_eq!(y, [3.0]);
    }

    #[test]
    fn step_budget_is_enforced() {
        let f = |_t: f64, s: &[f64; 2]| [s[1], -s[0]];
        let result = dopri5_integrate(f, 0.0, 200.0, [1.0, 0.0], 1e-10, 1e-12, 5);
        assert_eq!(result, Err(OdeError::MaxStepsExceeded));
    }
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
