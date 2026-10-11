//! Hamiltonian Monte Carlo with a leapfrog integrator and a unit mass matrix.
//!
//! Each iteration draws a standard normal momentum `p`, simulates Hamiltonian
//! dynamics for `L` leapfrog steps of size `ε`, and accepts the endpoint with
//! the Metropolis probability `min(1, exp(H_current − H_proposed))`, where
//! `H(q, p) = −log π(q) + |p|²/2`.
//!
//! The target needs a gradient of its log-density. Unlike random-walk
//! Metropolis, proposals move coherently along the target's geometry, so the
//! chain mixes much faster in well-conditioned problems.

use tpt_math_prob_core::{Rng, SplitMix64};

/// Output of [`run_hmc`].
#[derive(Clone, Debug, PartialEq)]
pub struct HmcRun {
    /// The retained (post-burn-in) states, one vector per draw.
    pub samples: Vec<Vec<f64>>,
    /// Fraction of post-burn-in proposals that were accepted.
    pub acceptance_rate: f64,
}

/// One standard normal draw by the Box–Muller transform.
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Avoid ln(0): the uniform draw is in [0, 1), so retry on an exact zero.
    let mut u1 = rng.next_f64();
    while u1 <= 0.0 {
        u1 = rng.next_f64();
    }
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

/// Run Hamiltonian Monte Carlo from `initial`.
///
/// * `log_target` — the unnormalised log-density `log π(q)`.
/// * `grad_log_target` — its gradient `∇ log π(q)`, returned with the same
///   length as `q`.
/// * `step_size` — leapfrog step `ε`. Too large gives low acceptance; too small
///   gives slow exploration.
/// * `leapfrog_steps` — number of leapfrog steps `L` per proposal.
/// * `n_samples` — retained draws; `burn_in` iterations are discarded first.
/// * `seed` — seed for the momentum and acceptance randomness.
///
/// # Panics
///
/// Panics if `step_size` is not a positive finite number, if `leapfrog_steps`
/// is zero, if `initial` is empty or non-finite, or if `log_target(initial)` is
/// not finite.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::hmc::run_hmc;
///
/// // Standard normal target: log π(q) = −q²/2, ∇ = −q.
/// let run = run_hmc(|q: &[f64]| -0.5 * q[0] * q[0], |q: &[f64]| vec![-q[0]],
///                   &[0.0], 0.2, 10, 4000, 500, 7);
/// let mean = run.samples.iter().map(|s| s[0]).sum::<f64>() / run.samples.len() as f64;
/// assert!(mean.abs() < 0.1);
/// assert!(run.acceptance_rate > 0.5);
/// ```
// The parameters are the sampler's settings, one each. Bundling them into a
// struct would only add a call-site ceremony for the same information.
#[allow(clippy::too_many_arguments)]
pub fn run_hmc<F, G>(
    log_target: F,
    grad_log_target: G,
    initial: &[f64],
    step_size: f64,
    leapfrog_steps: usize,
    n_samples: usize,
    burn_in: usize,
    seed: u64,
) -> HmcRun
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    assert!(
        step_size > 0.0 && step_size.is_finite(),
        "HMC step size must be positive and finite, got {step_size}"
    );
    assert!(leapfrog_steps >= 1, "HMC needs at least one leapfrog step");
    assert!(
        !initial.is_empty() && initial.iter().all(|x| x.is_finite()),
        "HMC initial state must be non-empty and finite"
    );

    let dim = initial.len();
    let mut q = initial.to_vec();
    let mut log_p = log_target(&q);
    assert!(
        log_p.is_finite(),
        "log-density at the initial state must be finite"
    );

    let mut rng = SplitMix64::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n_samples);
    let mut accepted = 0usize;

    for iteration in 0..burn_in + n_samples {
        let momentum: Vec<f64> = (0..dim).map(|_| standard_normal(&mut rng)).collect();
        let kinetic = 0.5 * momentum.iter().map(|p| p * p).sum::<f64>();
        let h_current = -log_p + kinetic;

        // Leapfrog: half step in momentum, alternating full steps in position
        // and momentum, then a closing half step in momentum.
        let mut q_new = q.clone();
        let mut p = momentum;
        let mut grad = grad_log_target(&q_new);
        for (pi, gi) in p.iter_mut().zip(&grad) {
            *pi += 0.5 * step_size * gi;
        }
        for step in 0..leapfrog_steps {
            for (qi, pi) in q_new.iter_mut().zip(&p) {
                *qi += step_size * pi;
            }
            grad = grad_log_target(&q_new);
            let scale = if step + 1 == leapfrog_steps {
                0.5 * step_size
            } else {
                step_size
            };
            for (pi, gi) in p.iter_mut().zip(&grad) {
                *pi += scale * gi;
            }
        }

        let log_p_new = log_target(&q_new);
        let h_proposed = -log_p_new + 0.5 * p.iter().map(|x| x * x).sum::<f64>();
        // A non-finite proposal energy (for example outside the support) is
        // always rejected.
        let accept_probability = if h_proposed.is_finite() {
            (h_current - h_proposed).exp().min(1.0)
        } else {
            0.0
        };
        let accept = rng.next_f64() < accept_probability;
        if accept {
            q = q_new;
            log_p = log_p_new;
        }

        if iteration >= burn_in {
            if accept {
                accepted += 1;
            }
            samples.push(q.clone());
        }
    }

    let acceptance_rate = if n_samples == 0 {
        0.0
    } else {
        accepted as f64 / n_samples as f64
    };
    HmcRun {
        samples,
        acceptance_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_and_var(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var)
    }

    #[test]
    fn standard_normal_target_has_unit_moments() {
        let run = run_hmc(
            |q: &[f64]| -0.5 * q[0] * q[0],
            |q: &[f64]| vec![-q[0]],
            &[0.0],
            0.3,
            10,
            20_000,
            1_000,
            11,
        );
        let xs: Vec<f64> = run.samples.iter().map(|s| s[0]).collect();
        let (mean, var) = mean_and_var(&xs);
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
        assert!(
            run.acceptance_rate > 0.8,
            "acceptance {}",
            run.acceptance_rate
        );
    }

    #[test]
    fn recovers_different_scales_per_dimension() {
        // Independent N(0, 4) and N(0, 1): variances 4 and 1.
        let run = run_hmc(
            |q: &[f64]| -0.5 * q[0] * q[0] / 4.0 - 0.5 * q[1] * q[1],
            |q: &[f64]| vec![-q[0] / 4.0, -q[1]],
            &[0.0, 0.0],
            0.5,
            12,
            20_000,
            1_000,
            23,
        );
        let xs: Vec<f64> = run.samples.iter().map(|s| s[0]).collect();
        let ys: Vec<f64> = run.samples.iter().map(|s| s[1]).collect();
        let (_, var_x) = mean_and_var(&xs);
        let (_, var_y) = mean_and_var(&ys);
        assert!((var_x - 4.0).abs() < 0.5, "var_x {var_x}");
        assert!((var_y - 1.0).abs() < 0.15, "var_y {var_y}");
    }

    #[test]
    fn same_seed_gives_identical_chains() {
        let go = |seed| {
            run_hmc(
                |q: &[f64]| -0.5 * q[0] * q[0],
                |q: &[f64]| vec![-q[0]],
                &[1.0],
                0.2,
                5,
                200,
                50,
                seed,
            )
        };
        assert_eq!(go(4), go(4));
        assert_ne!(go(4).samples, go(5).samples);
    }

    #[test]
    fn support_boundary_is_respected() {
        // log π = −∞ for q < 0 (exponential target); every proposal below zero
        // must be rejected, so all samples stay non-negative.
        let run = run_hmc(
            |q: &[f64]| if q[0] < 0.0 { f64::NEG_INFINITY } else { -q[0] },
            |_q: &[f64]| vec![-1.0],
            &[1.0],
            0.4,
            8,
            2_000,
            200,
            9,
        );
        assert!(run.samples.iter().all(|s| s[0] >= 0.0));
    }

    #[test]
    #[should_panic(expected = "step size")]
    fn rejects_non_positive_step_size() {
        run_hmc(
            |_: &[f64]| 0.0,
            |_: &[f64]| vec![0.0],
            &[0.0],
            0.0,
            1,
            1,
            0,
            0,
        );
    }
}
