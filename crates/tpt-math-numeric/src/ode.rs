//! Explicit Runge–Kutta integration of ODE systems: fixed-step RK4, adaptive
//! Dormand–Prince 5(4), dense output and event location.
//!
//! The state is a fixed-size array, so the fixed-step and adaptive integrators
//! need no allocation and work in `no_std` builds. Dense output and event
//! location keep their samples in a `Vec` and are gated on the `alloc` feature.
//! For stiff problems see [`stiff`](crate::stiff).

use crate::Scalar;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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

/// Failure modes of the adaptive integrators in this module.
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
pub(crate) fn lit<T: Scalar>(x: f64) -> T {
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

/// One Dormand–Prince trial step of signed size `hs` from `(t, y)`, given the
/// slope `k1 = f(t, y)`.
///
/// Returns `(y_new, k7, err)`: the fifth-order result, the slope at `y_new`
/// (the FSAL stage, also used for dense output), and the error estimate
/// `y_new − ŷ`, where `ŷ` is the embedded fourth-order result.
fn dp5_trial<T, const N: usize, F>(
    f: &F,
    t: T,
    y: &[T; N],
    k1: &[T; N],
    hs: T,
) -> ([T; N], [T; N], [T; N])
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
{
    let k2 = f(
        t + lit::<T>(1.0 / 5.0) * hs,
        &weighted(y, hs, &[(1.0 / 5.0, k1)]),
    );
    let k3 = f(
        t + lit::<T>(3.0 / 10.0) * hs,
        &weighted(y, hs, &[(3.0 / 40.0, k1), (9.0 / 40.0, &k2)]),
    );
    let k4 = f(
        t + lit::<T>(4.0 / 5.0) * hs,
        &weighted(
            y,
            hs,
            &[(44.0 / 45.0, k1), (-56.0 / 15.0, &k2), (32.0 / 9.0, &k3)],
        ),
    );
    let k5 = f(
        t + lit::<T>(8.0 / 9.0) * hs,
        &weighted(
            y,
            hs,
            &[
                (19372.0 / 6561.0, k1),
                (-25360.0 / 2187.0, &k2),
                (64448.0 / 6561.0, &k3),
                (-212.0 / 729.0, &k4),
            ],
        ),
    );
    let k6 = f(
        t + hs,
        &weighted(
            y,
            hs,
            &[
                (9017.0 / 3168.0, k1),
                (-355.0 / 33.0, &k2),
                (46732.0 / 5247.0, &k3),
                (49.0 / 176.0, &k4),
                (-5103.0 / 18656.0, &k5),
            ],
        ),
    );
    let y_new = weighted(
        y,
        hs,
        &[
            (35.0 / 384.0, k1),
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
            (71.0 / 57600.0, k1),
            (-71.0 / 16695.0, &k3),
            (71.0 / 1920.0, &k4),
            (-17253.0 / 339200.0, &k5),
            (22.0 / 525.0, &k6),
            (-1.0 / 40.0, &k7),
        ],
    );
    (y_new, k7, err)
}

/// The result of driving the adaptive stepper.
#[cfg_attr(not(feature = "alloc"), allow(dead_code))]
struct Driven<T, const N: usize> {
    t: T,
    y: [T; N],
    /// `true` if the observer asked the run to stop before `t1`.
    stopped: bool,
}

/// Adaptive Dormand–Prince driver shared by every adaptive entry point.
///
/// `observe(t0, y0, f0, t1, y1, f1)` is called once per accepted step and
/// returns `false` to end the run early.
fn dp5_drive<T, const N: usize, F, S>(
    f: &F,
    t0: T,
    t1: T,
    y0: [T; N],
    tol: (T, T),
    max_steps: usize,
    mut observe: S,
) -> Result<Driven<T, N>, OdeError>
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
    S: FnMut(T, &[T; N], &[T; N], T, &[T; N], &[T; N]) -> bool,
{
    // Relative and absolute tolerances, passed together to keep the arity low.
    let (rtol, atol) = tol;
    let span = t1 - t0;
    if span == T::zero() {
        return Ok(Driven {
            t: t0,
            y: y0,
            stopped: false,
        });
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
            return Ok(Driven {
                t,
                y,
                stopped: false,
            });
        }
        let step = h.min(remaining);
        let hs = sign * step;

        let k1 = f(t, &y);
        let (y_new, k7, err) = dp5_trial(f, t, &y, &k1, hs);

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
            let t_new = if step == remaining { t1 } else { t + hs };
            let keep_going = observe(t, &y, &k1, t_new, &y_new, &k7);
            t = t_new;
            y = y_new;
            if !keep_going {
                return Ok(Driven {
                    t,
                    y,
                    stopped: true,
                });
            }
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
            return Ok(Driven {
                t,
                y,
                stopped: false,
            });
        }
        if h < min_step {
            return Err(OdeError::StepSizeUnderflow);
        }
    }
    Err(OdeError::MaxStepsExceeded)
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
    dp5_drive(
        &f,
        t0,
        t1,
        y0,
        (rtol, atol),
        max_steps,
        |_, _, _, _, _, _| true,
    )
    .map(|d| d.y)
}

/// A solution sampled at every accepted step, with cubic Hermite interpolation
/// between them (dense output).
///
/// Built by [`dopri5_dense`]. Between two accepted steps the interpolant is the
/// cubic Hermite polynomial through the endpoint states and slopes. It is
/// `O(h⁴)` accurate, which is one order below the integrator's own steps, so
/// it is intended for plotting, root location and event detection.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct DenseSolution<T, const N: usize> {
    /// `(t, y(t), f(t, y(t)))` for each sample, in integration order.
    samples: Vec<(T, [T; N], [T; N])>,
}

#[cfg(feature = "alloc")]
impl<T: Scalar, const N: usize> DenseSolution<T, N> {
    /// The time the solution starts at.
    pub fn t_start(&self) -> T {
        self.samples[0].0
    }

    /// The time the solution ends at.
    pub fn t_end(&self) -> T {
        self.samples[self.samples.len() - 1].0
    }

    /// The number of accepted steps (one fewer than the sample count).
    pub fn steps(&self) -> usize {
        self.samples.len() - 1
    }

    /// The accepted sample times, in integration order.
    pub fn times(&self) -> impl Iterator<Item = T> + '_ {
        self.samples.iter().map(|s| s.0)
    }

    /// The state at `t`, or `None` if `t` lies outside `[t_start, t_end]`
    /// (in either direction, depending on the integration direction).
    pub fn eval(&self, t: T) -> Option<[T; N]> {
        let forward = self.t_end() >= self.t_start();
        let inside = if forward {
            t >= self.t_start() && t <= self.t_end()
        } else {
            t <= self.t_start() && t >= self.t_end()
        };
        if !inside {
            return None;
        }
        // Index of the last sample not beyond `t` in the integration direction.
        let count = self
            .samples
            .partition_point(|s| if forward { s.0 <= t } else { s.0 >= t });
        let i = count.saturating_sub(1).min(self.samples.len() - 2);
        let (ta, ya, fa) = &self.samples[i];
        let (tb, yb, fb) = &self.samples[i + 1];
        let theta = (t - *ta) / (*tb - *ta);
        Some(hermite(*ta, ya, fa, *tb, yb, fb, theta))
    }
}

/// Cubic Hermite interpolant at `θ ∈ [0, 1]` between `(t0, y0, f0)` and
/// `(t1, y1, f1)`.
#[cfg(feature = "alloc")]
fn hermite<T: Scalar, const N: usize>(
    t0: T,
    y0: &[T; N],
    f0: &[T; N],
    t1: T,
    y1: &[T; N],
    f1: &[T; N],
    theta: T,
) -> [T; N] {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let h = t1 - t0;
    let s = one - theta;
    let h00 = (one + two * theta) * s * s;
    let h10 = theta * s * s;
    let h01 = theta * theta * (three - two * theta);
    let h11 = theta * theta * (theta - one);
    core::array::from_fn(|i| h00 * y0[i] + h * h10 * f0[i] + h01 * y1[i] + h * h11 * f1[i])
}

/// Integrate with the adaptive Dormand–Prince pair and keep dense output.
///
/// The same step controller as [`dopri5_integrate`]. The returned
/// [`DenseSolution`] holds every accepted step, so it can be evaluated at any
/// time within the span.
///
/// # Errors
///
/// As [`dopri5_integrate`].
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::ode::dopri5_dense;
///
/// let sol = dopri5_dense(|_t: f64, y: &[f64; 1]| [y[0]], 0.0, 2.0, [1.0], 1e-9, 1e-12, 10_000)
///     .unwrap();
/// let y_half = sol.eval(0.5).unwrap()[0];
/// assert!((y_half - 0.5f64.exp()).abs() < 1e-6);
/// ```
#[cfg(feature = "alloc")]
pub fn dopri5_dense<T, const N: usize, F>(
    f: F,
    t0: T,
    t1: T,
    y0: [T; N],
    rtol: T,
    atol: T,
    max_steps: usize,
) -> Result<DenseSolution<T, N>, OdeError>
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
{
    let mut samples = Vec::new();
    samples.push((t0, y0, f(t0, &y0)));
    dp5_drive(
        &f,
        t0,
        t1,
        y0,
        (rtol, atol),
        max_steps,
        |_, _, _, t_new, y_new, f_new| {
            samples.push((t_new, *y_new, *f_new));
            true
        },
    )?;
    Ok(DenseSolution { samples })
}

/// Which way a scalar event function crosses zero.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossing {
    /// The event function goes from negative to non-negative.
    Rising,
    /// The event function goes from positive to non-positive.
    Falling,
}

/// A located zero crossing of an event function.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct Event<T, const N: usize> {
    /// The time of the crossing.
    pub t: T,
    /// The state at the crossing.
    pub y: [T; N],
    /// The direction of the crossing.
    pub crossing: Crossing,
}

/// The outcome of [`dopri5_events`].
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct EventRun<T, const N: usize> {
    /// The time the run ended: `t1`, or the first terminal event.
    pub t_end: T,
    /// The state at `t_end`.
    pub y_end: [T; N],
    /// Every event located, in time order.
    pub events: Vec<Event<T, N>>,
    /// `true` if the run stopped at a terminal event before `t1`.
    pub terminated: bool,
}

/// Integrate with the adaptive Dormand–Prince pair and locate zero crossings of
/// an event function `g(t, y)`.
///
/// After each accepted step the sign of `g` is compared with its sign at the
/// previous accepted point. On a sign change, the crossing is located by
/// bisection on the dense-output interpolant to machine precision in time.
/// Events are reported in order. If `terminal` is `true`, the run stops at the
/// first event and reports it as the end point.
///
/// A crossing is detected only when `g` changes strict sign across an accepted
/// step, so two crossings inside one step are missed. Tighten the tolerances if
/// that matters.
///
/// # Errors
///
/// As [`dopri5_integrate`].
///
/// # Examples
///
/// A ball dropped from height 10 lands when its height crosses zero:
///
/// ```
/// use tpt_math_numeric::ode::{dopri5_events, Crossing};
///
/// // State (height, velocity) under gravity g = 9.81.
/// let fall = |_t: f64, s: &[f64; 2]| [s[1], -9.81];
/// let ground = |_t: f64, s: &[f64; 2]| s[0];
/// let run = dopri5_events(fall, ground, 0.0, 5.0, [10.0, 0.0], 1e-10, 1e-12, 10_000, true)
///     .unwrap();
/// assert!(run.terminated);
/// assert_eq!(run.events[0].crossing, Crossing::Falling);
/// // h(t) = 10 − 4.905 t² reaches zero at t = √(10 / 4.905).
/// assert!((run.t_end - (10.0f64 / 4.905).sqrt()).abs() < 1e-7);
/// ```
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
pub fn dopri5_events<T, const N: usize, F, G>(
    f: F,
    g: G,
    t0: T,
    t1: T,
    y0: [T; N],
    rtol: T,
    atol: T,
    max_steps: usize,
    terminal: bool,
) -> Result<EventRun<T, N>, OdeError>
where
    T: Scalar,
    F: Fn(T, &[T; N]) -> [T; N],
    G: Fn(T, &[T; N]) -> T,
{
    let mut events = Vec::new();
    let mut g_prev = g(t0, &y0);

    let driven = dp5_drive(
        &f,
        t0,
        t1,
        y0,
        (rtol, atol),
        max_steps,
        |ta, ya, fa, tb, yb, fb| {
            let g_next = g(tb, yb);
            let crossing = if g_prev < T::zero() && g_next >= T::zero() {
                Some(Crossing::Rising)
            } else if g_prev > T::zero() && g_next <= T::zero() {
                Some(Crossing::Falling)
            } else {
                None
            };
            let mut keep_going = true;
            if let Some(crossing) = crossing {
                let (te, ye) = locate_crossing(&g, ta, ya, fa, tb, yb, fb, g_prev);
                events.push(Event {
                    t: te,
                    y: ye,
                    crossing,
                });
                keep_going = !terminal;
            }
            g_prev = g_next;
            keep_going
        },
    )?;

    // A terminal event ends the run at the event itself, not at the end of the
    // accepted step that contained it.
    let (t_end, y_end) = match events.last() {
        Some(last) if driven.stopped => (last.t, last.y),
        _ => (driven.t, driven.y),
    };
    Ok(EventRun {
        t_end,
        y_end,
        events,
        terminated: driven.stopped,
    })
}

/// Bisection on the Hermite interpolant of one accepted step, for the time at
/// which `g` changes sign. `g_start` is `g` at the step's start.
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
fn locate_crossing<T, const N: usize, G>(
    g: &G,
    t0: T,
    y0: &[T; N],
    f0: &[T; N],
    t1: T,
    y1: &[T; N],
    f1: &[T; N],
    g_start: T,
) -> (T, [T; N])
where
    T: Scalar,
    G: Fn(T, &[T; N]) -> T,
{
    let two = T::one() + T::one();
    let start_negative = g_start < T::zero();
    let mut lo = T::zero();
    let mut hi = T::one();
    // Fixed iteration cap: each step halves the θ-interval, so 200 is ample.
    for _ in 0..200 {
        let mid = (lo + hi) / two;
        let t_mid = t0 + mid * (t1 - t0);
        let y_mid = hermite(t0, y0, f0, t1, y1, f1, mid);
        let g_mid = g(t_mid, &y_mid);
        // Keep the half that still brackets the sign change.
        if (g_mid < T::zero()) == start_negative {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= T::epsilon() * lit::<T>(4.0) {
            break;
        }
    }
    let theta = hi;
    (
        t0 + theta * (t1 - t0),
        hermite(t0, y0, f0, t1, y1, f1, theta),
    )
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

    #[cfg(feature = "alloc")]
    #[test]
    fn dense_output_matches_the_closed_form_between_steps() {
        let sol = dopri5_dense(
            |_t: f64, y: &[f64; 1]| [y[0]],
            0.0,
            3.0,
            [1.0],
            1e-10,
            1e-12,
            10_000,
        )
        .unwrap();
        // Interior points, not just the accepted times.
        for k in 0..=60 {
            let t = 3.0 * k as f64 / 60.0;
            let y = sol.eval(t).unwrap()[0];
            assert!((y - t.exp()).abs() < 1e-6, "t = {t}: {y}");
        }
        assert!(sol.steps() > 1);
        assert_eq!(sol.t_start(), 0.0);
        assert_eq!(sol.t_end(), 3.0);
        assert_eq!(sol.eval(3.5), None);
        assert_eq!(sol.eval(-0.1), None);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn dense_output_works_backwards() {
        let sol = dopri5_dense(
            |_t: f64, y: &[f64; 1]| [y[0]],
            1.0,
            0.0,
            [core::f64::consts::E],
            1e-10,
            1e-12,
            10_000,
        )
        .unwrap();
        let mid = sol.eval(0.5).unwrap()[0];
        assert!((mid - 0.5f64.exp()).abs() < 1e-6, "{mid}");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn event_locates_a_crossing_and_keeps_going_when_not_terminal() {
        // y(t) = (sin t, cos t) solves y0' = y1, y1' = −y0. Its first component
        // crosses zero at π (falling) and 2π (rising); t = 0 does not count.
        let sin_cos = |_t: f64, y: &[f64; 2]| [y[1], -y[0]];
        let run = dopri5_events(
            sin_cos,
            |_t: f64, y: &[f64; 2]| y[0],
            0.0,
            7.0,
            [0.0, 1.0],
            1e-10,
            1e-12,
            10_000,
            false,
        )
        .unwrap();
        assert!(!run.terminated);
        assert_eq!(run.events.len(), 2, "{:?}", run.events);
        let pi = core::f64::consts::PI;
        assert!((run.events[0].t - pi).abs() < 1e-8);
        assert_eq!(run.events[0].crossing, Crossing::Falling);
        assert!((run.events[1].t - 2.0 * pi).abs() < 1e-8);
        assert_eq!(run.events[1].crossing, Crossing::Rising);
        assert!((run.t_end - 7.0).abs() < 1e-12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn terminal_event_stops_the_run() {
        let fall = |_t: f64, s: &[f64; 2]| [s[1], -9.81];
        let ground = |_t: f64, s: &[f64; 2]| s[0];
        let run = dopri5_events(
            fall,
            ground,
            0.0,
            10.0,
            [10.0, 0.0],
            1e-10,
            1e-12,
            10_000,
            true,
        )
        .unwrap();
        assert!(run.terminated);
        assert_eq!(run.events.len(), 1);
        let expected = (10.0f64 / 4.905).sqrt();
        assert!((run.t_end - expected).abs() < 1e-7, "{}", run.t_end);
        // Velocity at impact: v = −9.81 t.
        assert!((run.y_end[1] + 9.81 * expected).abs() < 1e-5);
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

    /// Non-autonomous problems: the time argument must reach the right-hand side.
    /// y' = cos t has the solution y = sin t from y(0) = 0.
    #[test]
    fn time_dependent_right_hand_side_is_integrated_correctly() {
        let f = |t: f64, _y: &[f64; 1]| [t.cos()];
        let y = rk4_integrate(f, 0.0, 2.0, [0.0], 400);
        assert!((y[0] - 2.0f64.sin()).abs() < 1e-9, "rk4 {}", y[0]);
        let y = dopri5_integrate(f, 0.0, 2.0, [0.0], 1e-11, 1e-13, 100_000).unwrap();
        assert!((y[0] - 2.0f64.sin()).abs() < 1e-9, "dopri5 {}", y[0]);
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
