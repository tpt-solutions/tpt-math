//! Forward sensitivity analysis: `∂y/∂p` for an ODE `y' = f(t, y, p)`.
//!
//! The sensitivities satisfy the variational equations
//! `S' = J_y(t) S + J_p(t)`, with `S(t₀) = 0` when the initial state does not
//! depend on the parameters. They are integrated together with the state as one
//! augmented system, so the solver's error control covers both.
//!
//! This works on runtime-sized vectors, because the augmented size `N(1 + P)`
//! is not known at compile time. It uses explicit Jacobians. For a fixed, small
//! number of parameters, the fixed-size solvers can instead run on
//! `tpt_math_autodiff_fwd::Dual`, which implements `Float` and carries the
//! sensitivities through the arithmetic.

use alloc::vec;
use alloc::vec::Vec;

use crate::ode::OdeError;
use crate::Float;

/// Dormand–Prince 5(4) coefficients.
const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
const A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];
const B5: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];
const B4: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

/// Integrate `y' = f(t, y)` on runtime-sized vectors with adaptive Dormand–Prince
/// 5(4). Step control and tolerances are as in [`dopri5_integrate`](crate::ode::dopri5_integrate).
///
/// # Errors
///
/// As [`dopri5_integrate`](crate::ode::dopri5_integrate).
pub fn dopri5_vec<F>(
    f: F,
    t0: f64,
    t1: f64,
    y0: &[f64],
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> Result<Vec<f64>, OdeError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y0.len();
    let span = t1 - t0;
    if span == 0.0 {
        return Ok(y0.to_vec());
    }
    let sign = span.signum();
    let total = span.abs();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = total / 100.0;
    let min_step = 16.0 * f64::EPSILON * total.max(t0.abs());

    for _ in 0..max_steps {
        let remaining = (t1 - t).abs();
        if remaining == 0.0 {
            return Ok(y);
        }
        let step = h.min(remaining);
        let hs = sign * step;

        let mut k: Vec<Vec<f64>> = Vec::with_capacity(7);
        k.push(f(t, &y));
        for s in 0..6 {
            let stage: Vec<f64> = (0..n)
                .map(|i| {
                    let mut acc = y[i];
                    for (j, kj) in k.iter().enumerate() {
                        acc += hs * A[s][j] * kj[i];
                    }
                    acc
                })
                .collect();
            k.push(f(t + C[s + 1] * hs, &stage));
        }
        let y_new: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = y[i];
                for (j, kj) in k.iter().enumerate().take(6) {
                    acc += hs * B5[j] * kj[i];
                }
                acc
            })
            .collect();
        let err: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = 0.0;
                for (j, kj) in k.iter().enumerate() {
                    e += hs * (B5[j] - B4[j]) * kj[i];
                }
                e
            })
            .collect();

        let mut sum_sq = 0.0;
        for i in 0..n {
            let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
            let e = err[i] / scale;
            sum_sq += e * e;
        }
        let error = if n == 0 {
            0.0
        } else {
            Float::sqrt(sum_sq / n as f64)
        };
        let accepted = error <= 1.0;
        if accepted {
            t = if step == remaining { t1 } else { t + hs };
            y = y_new;
        }
        let factor = if error == 0.0 {
            5.0
        } else {
            Float::min(Float::max(0.9 * Float::powf(error, -0.2), 0.2), 5.0)
        };
        let factor = if accepted { factor } else { factor.min(1.0) };
        h = step * factor;
        if accepted && (t1 - t).abs() == 0.0 {
            return Ok(y);
        }
        if h < min_step {
            return Err(OdeError::StepSizeUnderflow);
        }
    }
    Err(OdeError::MaxStepsExceeded)
}

/// The state and parameter sensitivities at the end of a forward sensitivity run.
#[derive(Clone, Debug, PartialEq)]
pub struct Sensitivities {
    /// The state `y(t₁)`.
    pub y: Vec<f64>,
    /// `s[i][k] = ∂yᵢ/∂p_k` at `t₁`, with `N` rows and `P` columns.
    pub s: Vec<Vec<f64>>,
}

/// Integrate an ODE and its parameter sensitivities together.
///
/// `f(t, y, p)` is the right-hand side. `jac_y(t, y, p)` is the `N × N` Jacobian
/// `∂f/∂y` and `jac_p(t, y, p)` the `N × P` Jacobian `∂f/∂p`, both as row vectors.
/// The initial state is taken as independent of `p`, so `S(t₀) = 0`.
///
/// # Errors
///
/// As [`dopri5_integrate`](crate::ode::dopri5_integrate).
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::sensitivity::forward_sensitivity;
///
/// // y' = −k y with k = 0.5, y(0) = 2. Then ∂y/∂k = −t y₀ e^{−k t}.
/// let out = forward_sensitivity(
///     |_t, y, p| vec![-p[0] * y[0]],
///     |_t, y, p| vec![vec![-p[0]]],
///     |_t, y, _p| vec![vec![-y[0]]],
///     0.0,
///     2.0,
///     &[2.0],
///     &[0.5],
///     1e-10,
///     1e-12,
///     100_000,
/// )
/// .unwrap();
/// let want = -2.0 * 2.0 * (-0.5f64 * 2.0).exp();
/// assert!((out.s[0][0] - want).abs() < 1e-7);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn forward_sensitivity<F, Jy, Jp>(
    f: F,
    jac_y: Jy,
    jac_p: Jp,
    t0: f64,
    t1: f64,
    y0: &[f64],
    p: &[f64],
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> Result<Sensitivities, OdeError>
where
    F: Fn(f64, &[f64], &[f64]) -> Vec<f64>,
    Jy: Fn(f64, &[f64], &[f64]) -> Vec<Vec<f64>>,
    Jp: Fn(f64, &[f64], &[f64]) -> Vec<Vec<f64>>,
{
    let n = y0.len();
    let np = p.len();
    // Augmented state: z = [y (n entries), S (n·np entries, row-major)].
    let mut z0 = vec![0.0; n + n * np];
    z0[..n].copy_from_slice(y0);

    let rhs = |t: f64, z: &[f64]| -> Vec<f64> {
        let y = &z[..n];
        let dy = f(t, y, p);
        let jy = jac_y(t, y, p);
        let jp = jac_p(t, y, p);
        let mut dz = vec![0.0; n + n * np];
        dz[..n].copy_from_slice(&dy);
        // S' = J_y S + J_p.
        for i in 0..n {
            for k in 0..np {
                let mut acc = jp[i][k];
                for j in 0..n {
                    acc += jy[i][j] * z[n + j * np + k];
                }
                dz[n + i * np + k] = acc;
            }
        }
        dz
    };

    let z = dopri5_vec(rhs, t0, t1, &z0, rtol, atol, max_steps)?;
    let s = (0..n)
        .map(|i| z[n + i * np..n + (i + 1) * np].to_vec())
        .collect();
    Ok(Sensitivities {
        y: z[..n].to_vec(),
        s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay_sensitivity_matches_the_closed_form() {
        let (k, y0, t) = (0.5, 2.0, 2.0);
        let out = forward_sensitivity(
            |_t, y, p| vec![-p[0] * y[0]],
            |_t, _y, p| vec![vec![-p[0]]],
            |_t, y, _p| vec![vec![-y[0]]],
            0.0,
            t,
            &[y0],
            &[k],
            1e-11,
            1e-13,
            100_000,
        )
        .unwrap();
        let y_exact = y0 * (-k * t).exp();
        let s_exact = -t * y0 * (-k * t).exp();
        assert!((out.y[0] - y_exact).abs() < 1e-8);
        assert!((out.s[0][0] - s_exact).abs() < 1e-7, "{:?}", out.s);
    }

    #[test]
    fn two_parameter_oscillator_sensitivities_match_finite_differences() {
        // y0' = y1, y1' = −ω² y0 − c y1 with parameters p = (ω, c).
        let f = |_t: f64, y: &[f64], p: &[f64]| vec![y[1], -p[0] * p[0] * y[0] - p[1] * y[1]];
        let jy = |_t: f64, _y: &[f64], p: &[f64]| vec![vec![0.0, 1.0], vec![-p[0] * p[0], -p[1]]];
        let jp =
            |_t: f64, y: &[f64], p: &[f64]| vec![vec![0.0, 0.0], vec![-2.0 * p[0] * y[0], -y[1]]];
        let run = |p: &[f64]| {
            dopri5_vec(
                |t, y| f(t, y, p),
                0.0,
                3.0,
                &[1.0, 0.0],
                1e-12,
                1e-14,
                1_000_000,
            )
            .unwrap()
        };
        let p = [1.3, 0.2];
        let out = forward_sensitivity(
            f,
            jy,
            jp,
            0.0,
            3.0,
            &[1.0, 0.0],
            &p,
            1e-12,
            1e-14,
            1_000_000,
        )
        .unwrap();
        let h = 1e-6;
        for k in 0..2 {
            let mut hi = p;
            let mut lo = p;
            hi[k] += h;
            lo[k] -= h;
            let (yh, yl) = (run(&hi), run(&lo));
            for i in 0..2 {
                let fd = (yh[i] - yl[i]) / (2.0 * h);
                assert!(
                    (out.s[i][k] - fd).abs() < 1e-5,
                    "∂y{i}/∂p{k}: {} vs {fd}",
                    out.s[i][k]
                );
            }
        }
    }

    #[test]
    fn dopri5_vec_matches_the_fixed_array_solver() {
        let y = dopri5_vec(|_t, y| vec![y[0]], 0.0, 1.0, &[1.0], 1e-10, 1e-12, 10_000).unwrap();
        assert!((y[0] - core::f64::consts::E).abs() < 1e-8);
    }
}
