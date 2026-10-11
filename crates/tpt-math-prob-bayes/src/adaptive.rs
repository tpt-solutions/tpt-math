//! Adaptive random-walk Metropolis (Haario, Saksman and Tamminen, 2001).
//!
//! Proposals are Gaussian steps `y = x + L z` with `z ~ N(0, I)`. Before the
//! adaptation starts the Cholesky factor `L` is a fixed isotropic step. After
//! that it is recomputed from the chain's own history, as
//! `L Lᵀ = (2.38²/d)·(Σ̂ + ε I)`, where `Σ̂` is the empirical covariance of
//! every state so far. The factor `2.38²/d` is the classical optimal scaling for
//! Gaussian targets, and the `ε I` term keeps the proposal non-degenerate.
//!
//! The adaptation changes the transition kernel as it runs, so the chain is not
//! a Markov chain in the strict sense. The usual remedy is to adapt only during
//! burn-in, or to stop adapting once the covariance has settled. This module
//! takes an explicit `adapt_until` iteration for that reason.

use tpt_math_prob_core::{Rng, SplitMix64};

/// Output of [`run_adaptive_metropolis`].
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveRun {
    /// The retained (post-burn-in) states.
    pub samples: Vec<Vec<f64>>,
    /// Fraction of post-burn-in proposals accepted.
    pub acceptance_rate: f64,
}

/// Standard normal draw by the Box–Muller transform.
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let mut u1 = rng.next_f64();
    while u1 <= 0.0 {
        u1 = rng.next_f64();
    }
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

/// Lower-triangular Cholesky factor of a symmetric positive-definite matrix
/// given as rows. Returns `None` if the matrix is not positive definite.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - sum;
                if d.is_nan() || d <= 0.0 {
                    return None;
                }
                l[i][i] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Run adaptive random-walk Metropolis from `initial`.
///
/// * `log_target` — the unnormalised log-density.
/// * `initial_step` — the isotropic proposal standard deviation used until
///   adaptation starts.
/// * `adapt_until` — iterations (counting burn-in) during which the proposal
///   covariance is adapted. Afterwards it is frozen, which keeps the kernel
///   fixed for the retained draws.
/// * `n_samples`, `burn_in`, `seed` — as for [`crate::hmc::run_hmc`].
///
/// The covariance adaptation starts from the first iteration. A proposal
/// covariance that cannot be factorised leaves the previous factor in place.
///
/// # Panics
///
/// Panics if `initial` is empty or non-finite, if `log_target(initial)` is not
/// finite, or if `initial_step` is not positive and finite.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::adaptive::run_adaptive_metropolis;
///
/// // Standard normal target in one dimension.
/// let run = run_adaptive_metropolis(|q: &[f64]| -0.5 * q[0] * q[0], &[0.0],
///                                   0.5, 2_000, 4_000, 1_000, 3);
/// assert!(run.acceptance_rate > 0.1 && run.acceptance_rate < 0.6);
/// ```
pub fn run_adaptive_metropolis<F>(
    log_target: F,
    initial: &[f64],
    initial_step: f64,
    adapt_until: usize,
    n_samples: usize,
    burn_in: usize,
    seed: u64,
) -> AdaptiveRun
where
    F: Fn(&[f64]) -> f64,
{
    assert!(
        !initial.is_empty() && initial.iter().all(|x| x.is_finite()),
        "adaptive Metropolis initial state must be non-empty and finite"
    );
    assert!(
        initial_step > 0.0 && initial_step.is_finite(),
        "adaptive Metropolis initial step must be positive and finite, got {initial_step}"
    );
    let dim = initial.len();
    let mut x = initial.to_vec();
    let mut log_p = log_target(&x);
    assert!(
        log_p.is_finite(),
        "log-density at the initial state must be finite"
    );

    // Haario's scaling and a small regulariser so the proposal never collapses.
    let scale = 2.38 * 2.38 / dim as f64;
    let epsilon = 1e-6;
    let mut factor: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| if i == j { initial_step } else { 0.0 })
                .collect()
        })
        .collect();

    // Running sums for the empirical mean and covariance of all states so far.
    let mut count = 0.0;
    let mut sum = vec![0.0; dim];
    let mut cross = vec![vec![0.0; dim]; dim];

    let mut rng = SplitMix64::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n_samples);
    let mut accepted = 0usize;

    for iteration in 0..burn_in + n_samples {
        // Record the current state before proposing, so the history includes it.
        count += 1.0;
        for i in 0..dim {
            sum[i] += x[i];
            for j in 0..dim {
                cross[i][j] += x[i] * x[j];
            }
        }

        let z: Vec<f64> = (0..dim).map(|_| standard_normal(&mut rng)).collect();
        let proposal: Vec<f64> = (0..dim)
            .map(|i| x[i] + (0..=i).map(|k| factor[i][k] * z[k]).sum::<f64>())
            .collect();
        let log_p_new = log_target(&proposal);
        let log_alpha = if log_p_new.is_finite() {
            (log_p_new - log_p).min(0.0)
        } else {
            f64::NEG_INFINITY
        };
        let accept = rng.next_f64().ln() < log_alpha;
        if accept {
            x = proposal;
            log_p = log_p_new;
        }

        if iteration >= burn_in {
            if accept {
                accepted += 1;
            }
            samples.push(x.clone());
        }

        // Re-estimate the proposal factor from the history while adapting.
        if iteration + 1 < adapt_until && count >= 2.0 {
            let mut cov = vec![vec![0.0; dim]; dim];
            for i in 0..dim {
                for j in 0..dim {
                    let centred = cross[i][j] - sum[i] * sum[j] / count;
                    cov[i][j] = scale * centred / (count - 1.0);
                    if i == j {
                        cov[i][j] += scale * epsilon;
                    }
                }
            }
            if let Some(l) = cholesky(&cov) {
                factor = l;
            }
        }
    }

    let acceptance_rate = if n_samples == 0 {
        0.0
    } else {
        accepted as f64 / n_samples as f64
    };
    AdaptiveRun {
        samples,
        acceptance_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_reconstructs_matrix() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let rebuilt: f64 = (0..2).map(|k| l[i][k] * l[j][k]).sum();
                assert!((rebuilt - a[i][j]).abs() < 1e-12);
            }
        }
        assert!(cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
    }

    #[test]
    fn recovers_correlated_gaussian_structure() {
        // Target: bivariate normal with unit variances and correlation 0.9.
        let rho: f64 = 0.9;
        let det = 1.0 - rho * rho;
        let log_target =
            move |q: &[f64]| -(q[0] * q[0] - 2.0 * rho * q[0] * q[1] + q[1] * q[1]) / (2.0 * det);
        let run = run_adaptive_metropolis(log_target, &[0.0, 0.0], 0.1, 3_000, 30_000, 3_000, 17);
        let n = run.samples.len() as f64;
        let mx = run.samples.iter().map(|s| s[0]).sum::<f64>() / n;
        let my = run.samples.iter().map(|s| s[1]).sum::<f64>() / n;
        let vx = run.samples.iter().map(|s| (s[0] - mx).powi(2)).sum::<f64>() / n;
        let vy = run.samples.iter().map(|s| (s[1] - my).powi(2)).sum::<f64>() / n;
        let cov = run
            .samples
            .iter()
            .map(|s| (s[0] - mx) * (s[1] - my))
            .sum::<f64>()
            / n;
        let corr = cov / (vx * vy).sqrt();
        assert!(corr > 0.85 && corr < 0.95, "correlation {corr}");
        assert!(mx.abs() < 0.1 && my.abs() < 0.1, "mean ({mx}, {my})");
        assert!(
            run.acceptance_rate > 0.1 && run.acceptance_rate < 0.6,
            "acceptance {}",
            run.acceptance_rate
        );
    }

    #[test]
    fn same_seed_gives_identical_chains() {
        let target = |q: &[f64]| -0.5 * q[0] * q[0];
        let a = run_adaptive_metropolis(target, &[1.0], 0.5, 100, 200, 50, 8);
        let b = run_adaptive_metropolis(target, &[1.0], 0.5, 100, 200, 50, 8);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "initial step")]
    fn rejects_non_positive_step() {
        run_adaptive_metropolis(|_: &[f64]| 0.0, &[0.0], 0.0, 10, 10, 0, 1);
    }
}
