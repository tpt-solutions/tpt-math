//! A stiff ODE solver: the two-stage, second-order, L-stable Rosenbrock method
//! ROS2 with step-doubling error control.
//!
//! Explicit methods such as [`dopri5_integrate`](crate::ode::dopri5_integrate)
//! must take steps below the fastest decay rate of the system, however smooth
//! the solution is. Stiff systems (chemical kinetics, relaxation oscillators,
//! diffusion discretised on a fine grid) therefore need enormous numbers of
//! steps. Rosenbrock methods are linearly implicit: each step solves one linear
//! system with `I − γ·h·J`, where `J` is the Jacobian of `f`, and they stay
//! stable at step sizes far beyond the explicit stability limit.
//!
//! The solver handles autonomous systems `y' = f(y)`. A non-autonomous system
//! `y' = f(t, y)` is made autonomous by adding `t` as a state variable with
//! `t' = 1`.

use crate::ode::{lit, OdeError};
use crate::Scalar;

/// `γ = 1 + 1/√2`, the stage coefficient of ROS2 (L-stable).
fn gamma<T: Scalar>() -> T {
    T::one() + T::one() / (lit::<T>(2.0)).sqrt()
}

/// Solve `A x = b` for a fixed-size dense system by Gaussian elimination with
/// partial pivoting. Returns `None` if a pivot vanishes or is not finite.
fn solve_fixed<T: Scalar, const N: usize>(mut a: [[T; N]; N], mut b: [T; N]) -> Option<[T; N]> {
    for col in 0..N {
        // Pivot: the largest magnitude in this column at or below the diagonal.
        let mut pivot = col;
        for row in col + 1..N {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if !a[pivot][col].is_finite() || a[pivot][col] == T::zero() {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..N {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (x, p) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *x = *x - factor * p;
            }
            b[row] = b[row] - factor * b[col];
        }
    }
    let mut x = [T::zero(); N];
    for row in (0..N).rev() {
        let mut acc = b[row];
        for k in row + 1..N {
            acc = acc - a[row][k] * x[k];
        }
        x[row] = acc / a[row][row];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// One ROS2 step of size `h` from `y`. Returns `None` if the linear system is
/// singular.
///
/// ```text
/// (I − γh·J) k₁ = f(y)
/// (I − γh·J) k₂ = f(y + h·k₁) − 2·k₁
/// y⁺ = y + h·(3/2·k₁ + 1/2·k₂)
/// ```
fn ros2_step<T, const N: usize, F, J>(f: &F, jac: &J, y: &[T; N], h: T) -> Option<[T; N]>
where
    T: Scalar,
    F: Fn(&[T; N]) -> [T; N],
    J: Fn(&[T; N]) -> [[T; N]; N],
{
    let g = gamma::<T>();
    let jy = jac(y);
    let gh = g * h;
    // M = I − γh·J.
    let m: [[T; N]; N] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let identity = if i == j { T::one() } else { T::zero() };
            identity - gh * jy[i][j]
        })
    });

    let f0 = f(y);
    let k1 = solve_fixed(m, f0)?;
    let y_stage: [T; N] = core::array::from_fn(|i| y[i] + h * k1[i]);
    let f1 = f(&y_stage);
    let two = T::one() + T::one();
    let rhs2: [T; N] = core::array::from_fn(|i| f1[i] - two * k1[i]);
    let k2 = solve_fixed(m, rhs2)?;

    let three_halves = T::one() + T::one() / two;
    let half = T::one() / two;
    Some(core::array::from_fn(|i| {
        y[i] + h * (three_halves * k1[i] + half * k2[i])
    }))
}

/// Integrate the autonomous system `y' = f(y)` from `t0` to `t1` with the
/// stiff solver, returning the state at `t1`.
///
/// `jac(y)` is the Jacobian matrix `J[i][j] = ∂fᵢ/∂yⱼ` at `y`. Each step
/// takes two half-steps and one full step: the difference between the two
/// results estimates the local error, and the half-step result is kept. A step
/// is accepted when the weighted RMS error is at most 1, with weights
/// `atol + rtol · max(|y|, |y_new|)`, and the step size adapts by the
/// second-order controller.
///
/// `t1 < t0` integrates backwards. `max_steps` bounds the number of attempted
/// steps, accepted or rejected. The method is second order, so use tighter
/// tolerances than you would with the explicit solvers.
///
/// # Errors
///
/// Returns [`OdeError::MaxStepsExceeded`] if the attempt budget runs out,
/// [`OdeError::StepSizeUnderflow`] if the step collapses, or, when the linear
/// system `I − γhJ` is singular at every step size tried, the step underflow
/// error.
///
/// # Examples
///
/// Robertson's chemical kinetics problem, a standard stiff benchmark:
///
/// ```
/// use tpt_math_numeric::stiff::rosenbrock2_integrate;
///
/// let f = |y: &[f64; 3]| {
///     [
///         -0.04 * y[0] + 1e4 * y[1] * y[2],
///         0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
///         3e7 * y[1] * y[1],
///     ]
/// };
/// let jac = |y: &[f64; 3]| {
///     [
///         [-0.04, 1e4 * y[2], 1e4 * y[1]],
///         [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
///         [0.0, 6e7 * y[1], 0.0],
///     ]
/// };
/// let y = rosenbrock2_integrate(f, jac, 0.0, 40.0, [1.0, 0.0, 0.0], 1e-6, 1e-10, 100_000)
///     .unwrap();
/// assert!((y[0] - 0.7158).abs() < 1e-3);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn rosenbrock2_integrate<T, const N: usize, F, J>(
    f: F,
    jac: J,
    t0: T,
    t1: T,
    y0: [T; N],
    rtol: T,
    atol: T,
    max_steps: usize,
) -> Result<[T; N], OdeError>
where
    T: Scalar,
    F: Fn(&[T; N]) -> [T; N],
    J: Fn(&[T; N]) -> [[T; N]; N],
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
    let two = T::one() + T::one();
    let error_exponent = -T::one() / (lit::<T>(2.0) + T::one());
    let three = two + T::one();

    for _ in 0..max_steps {
        let remaining = (t1 - t).abs();
        if remaining == T::zero() {
            return Ok(y);
        }
        let step = h.min(remaining);
        let hs = sign * step;

        // Step doubling: one full step against two half steps.
        let Some(full) = ros2_step(&f, &jac, &y, hs) else {
            h = step / two;
            if h < min_step {
                return Err(OdeError::StepSizeUnderflow);
            }
            continue;
        };
        let half_result =
            ros2_step(&f, &jac, &y, hs / two).and_then(|mid| ros2_step(&f, &jac, &mid, hs / two));
        let Some(half) = half_result else {
            h = step / two;
            if h < min_step {
                return Err(OdeError::StepSizeUnderflow);
            }
            continue;
        };

        // Order-2 method: the half-step error is about (full − half) / (2² − 1).
        let mut sum_sq = T::zero();
        for i in 0..N {
            let err = (half[i] - full[i]) / three;
            let scale = atol + rtol * y[i].abs().max(half[i].abs());
            let e = err / scale;
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
            y = half;
        }
        let factor = if error == T::zero() {
            lit::<T>(5.0)
        } else {
            (lit::<T>(0.9) * error.powf(error_exponent))
                .max(lit::<T>(0.2))
                .min(lit::<T>(5.0))
        };
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
mod tests {
    use super::*;

    #[test]
    fn linear_solver_matches_a_known_system() {
        // [2 1; 1 3] x = [5; 10] has x = [1, 3].
        let x: [f64; 2] = solve_fixed([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn linear_solver_pivots_and_detects_singularity() {
        // First pivot is zero, so pivoting is required.
        let x: [f64; 2] = solve_fixed([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0]).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
        assert!(solve_fixed([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]).is_none());
    }

    #[test]
    fn exponential_decay_is_accurate() {
        // y' = −y is linear, so the Jacobian is constant.
        let y = rosenbrock2_integrate(
            |y: &[f64; 1]| [-y[0]],
            |_: &[f64; 1]| [[-1.0]],
            0.0,
            1.0,
            [1.0],
            1e-8,
            1e-10,
            10_000,
        )
        .unwrap();
        assert!((y[0] - (-1.0f64).exp()).abs() < 1e-5, "{}", y[0]);
    }

    #[test]
    fn stays_stable_at_steps_far_beyond_the_explicit_limit() {
        // y' = −10⁶ (y − 1). The explicit stability limit is about 3·10⁻⁶, so a
        // single step of size 1 would blow up for an explicit method. L-stability
        // makes the step land near the equilibrium y = 1 without oscillating.
        let k = 1e6;
        let y = rosenbrock2_integrate(
            move |y: &[f64; 1]| [-k * (y[0] - 1.0)],
            move |_: &[f64; 1]| [[-k]],
            0.0,
            1.0,
            [0.0],
            1e-3,
            1e-6,
            1_000,
        )
        .unwrap();
        assert!((y[0] - 1.0).abs() < 1e-4, "{}", y[0]);
    }

    #[test]
    fn robertson_kinetics_matches_the_reference_solution() {
        // Reference values at t = 40 (Hairer & Wanner, Vol. II).
        let f = |y: &[f64; 3]| {
            [
                -0.04 * y[0] + 1e4 * y[1] * y[2],
                0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                3e7 * y[1] * y[1],
            ]
        };
        let jac = |y: &[f64; 3]| {
            [
                [-0.04, 1e4 * y[2], 1e4 * y[1]],
                [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
                [0.0, 6e7 * y[1], 0.0],
            ]
        };
        let y = rosenbrock2_integrate(f, jac, 0.0, 40.0, [1.0, 0.0, 0.0], 1e-6, 1e-10, 100_000)
            .unwrap();
        assert!((y[0] - 0.7158270687).abs() < 2e-3, "y1 = {}", y[0]);
        assert!((y[1] - 9.185534764e-6).abs() < 1e-7, "y2 = {}", y[1]);
        assert!((y[2] - 0.2841637457).abs() < 2e-3, "y3 = {}", y[2]);
        // Mass is conserved by the kinetics.
        assert!((y[0] + y[1] + y[2] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn error_decreases_as_the_tolerance_tightens() {
        // y' = −2y has the exact solution e^{−2t}; the error is measured at t = 1.
        let exact = (-2.0f64).exp();
        let run = |rtol: f64| {
            let y = rosenbrock2_integrate(
                |y: &[f64; 1]| [-2.0 * y[0]],
                |_: &[f64; 1]| [[-2.0]],
                0.0,
                1.0,
                [1.0],
                rtol,
                rtol * 1e-2,
                1_000_000,
            )
            .unwrap();
            (y[0] - exact).abs()
        };
        let coarse = run(1e-3);
        let fine = run(1e-5);
        // The global error of a second-order method scales like tol^(2/3), so
        // tightening the tolerance must reduce it.
        assert!(fine < coarse, "coarse {coarse}, fine {fine}");
        assert!(fine < 1e-4, "fine error {fine}");
    }

    #[test]
    fn budget_is_enforced() {
        let result = rosenbrock2_integrate(
            |y: &[f64; 1]| [-y[0]],
            |_: &[f64; 1]| [[-1.0]],
            0.0,
            100.0,
            [1.0],
            1e-12,
            1e-14,
            3,
        );
        assert_eq!(result, Err(OdeError::MaxStepsExceeded));
    }

    #[test]
    fn integrates_backwards() {
        let y = rosenbrock2_integrate(
            |y: &[f64; 1]| [-y[0]],
            |_: &[f64; 1]| [[-1.0]],
            1.0,
            0.0,
            [(-1.0f64).exp()],
            1e-8,
            1e-10,
            10_000,
        )
        .unwrap();
        assert!((y[0] - 1.0).abs() < 1e-5, "{}", y[0]);
    }
}
