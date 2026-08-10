//! Monte Carlo methods.
//!
//! This crate provides a small, dependency-free toolkit of Monte Carlo
//! estimation routines built on top of the randomness traits from
//! [`tpt_math_prob_core`]:
//!
//! * plain (crude) Monte Carlo [`integrate`] with a standard-error estimate
//!   derived from the sample variance,
//! * bulk sampling helpers [`mean_and_var`] and [`estimate_mean`],
//! * [`ImportanceSampling`] for variance reduction via a change of measure.
//!
//! All routines are generic over [`Rng`] so they work with any generator,
//! including the deterministic [`SplitMix64`] used throughout the tests.

use std::vec::Vec;

use tpt_math_prob_core::{Distribution, Rng, Standard};

/// Compute the sample mean and (population) sample variance of `samples`.
///
/// The variance returned uses Bessel's correction (division by `n - 1`) when
/// there is more than one sample; for a single sample the variance is `0.0`.
///
/// # Examples
///
/// ```
/// let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let (mean, var) = tpt_math_prob_monte_carlo::mean_and_var(&xs);
/// assert!((mean - 3.0).abs() < 1e-12);
/// assert!((var - 2.5).abs() < 1e-12);
/// ```
pub fn mean_and_var(samples: &[f64]) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = samples.iter().copied().sum::<f64>() / (n as f64);
    if n == 1 {
        return (mean, 0.0);
    }
    let var = samples
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / ((n - 1) as f64);
    (mean, var)
}

/// Plain (crude) Monte Carlo estimate of the integral of `f` over `[a, b]`.
///
/// The estimator is `I = (b - a) * mean_i f(x_i)` with `x_i` drawn uniformly
/// from `[a, b]`. The returned standard error is `(b - a) * sqrt(var(f) / n)`,
/// where `var(f)` is the unbiased sample variance of `f(x_i)`.
///
/// Returns the tuple `(mean, stderr)`. With `n` samples the standard error
/// decays as `1 / sqrt(n)`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::{Rng, SplitMix64};
/// use tpt_math_prob_monte_carlo::integrate;
///
/// let mut rng = SplitMix64::seed_from_u64(0);
/// // ∫_0^1 x^2 dx = 1/3
/// let (mean, stderr) = integrate(&mut rng, |x| x * x, 0.0, 1.0, 100_000);
/// assert!((mean - 1.0 / 3.0).abs() < 10.0 * stderr);
/// ```
pub fn integrate<R, F>(rng: &mut R, f: F, a: f64, b: f64, n: usize) -> (f64, f64)
where
    R: Rng + ?Sized,
    F: Fn(f64) -> f64,
{
    let width = b - a;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n {
        let u: f64 = Standard.sample(rng);
        let x = a + width * u;
        let y = f(x);
        sum += y;
        sum_sq += y * y;
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let m = (n as f64).max(2.0);
    let mean_f = sum / (n as f64);
    let var_f = (sum_sq - mean_f * sum) / (m - 1.0).max(1.0);
    let var_f = if var_f < 0.0 { 0.0 } else { var_f };
    let mean = width * mean_f;
    let stderr = width * (var_f / (n as f64)).sqrt();
    (mean, stderr)
}

/// Draw `n` samples from distribution `d` and return `(mean, stderr)` of those
/// samples, where the standard error is the standard deviation divided by
/// `sqrt(n)`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::{Distribution, Standard, SplitMix64};
/// use tpt_math_prob_monte_carlo::estimate_mean;
///
/// let mut rng = SplitMix64::seed_from_u64(3);
/// // Uniform [0,1) has mean 0.5 and variance 1/12.
/// let (mean, stderr) = estimate_mean(&Standard, &mut rng, 200_000);
/// assert!((mean - 0.5).abs() < 10.0 * stderr);
/// ```
pub fn estimate_mean<R, D>(d: &D, rng: &mut R, n: usize) -> (f64, f64)
where
    R: Rng + ?Sized,
    D: Distribution<f64>,
{
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        buf.push(d.sample(rng));
    }
    let (mean, var) = mean_and_var(&buf);
    let stderr = if n > 0 {
        (var / (n as f64)).sqrt()
    } else {
        0.0
    };
    (mean, stderr)
}

/// Variance-reduction helper implementing importance sampling.
///
/// Given a target density `f` (the density we wish we were sampling from) and a
/// proposal density `g` (the one we can actually sample from), the standard
/// importance estimator for the expectation `E_f[h(X)]` is
///
/// ```text
/// (1/n) Σ_i h(x_i) * f(x_i) / g(x_i),   x_i ~ g.
/// ```
///
/// Supplying a proposal `g` that concentrates mass where `h` is large lowers
/// the variance compared to crude Monte Carlo.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::{Distribution, Standard, SplitMix64};
/// use tpt_math_prob_monte_carlo::ImportanceSampling;
///
/// let mut rng = SplitMix64::seed_from_u64(11);
/// // Estimate E[1] = 1 under the uniform target using the uniform proposal.
/// // Identical densities give a zero-variance estimator, so `est == 1.0`.
/// let is = ImportanceSampling::new(|_x| 1.0, |_x| 1.0);
/// let (est, _stderr) = is.estimate(&mut rng, &Standard, |_x| 1.0, 50_000);
/// assert!((est - 1.0).abs() < 1e-12);
/// ```
pub struct ImportanceSampling<F, G>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    /// Target density `f` evaluated at `x`.
    target: F,
    /// Proposal density `g` evaluated at `x`.
    proposal: G,
}

impl<F, G> ImportanceSampling<F, G>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    /// Build an importance sampler from the target density `f` and the proposal
    /// density `g`.
    pub fn new(target: F, proposal: G) -> Self {
        ImportanceSampling { target, proposal }
    }

    /// Estimate `E_f[h(X)]` by drawing `n` samples from `proposal` (which must
    /// be a [`Distribution`] over the same support) and reweighting with the
    /// likelihood ratio `f(x) / g(x)`.
    ///
    /// Returns `(estimate, stderr)` using the sample variance of the weighted
    /// terms.
    pub fn estimate<R, P, H>(&self, rng: &mut R, proposal: &P, h: H, n: usize) -> (f64, f64)
    where
        R: Rng + ?Sized,
        P: Distribution<f64>,
        H: Fn(f64) -> f64,
    {
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut buf = Vec::with_capacity(n);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = proposal.sample(rng);
            let w = (self.target)(x) / (self.proposal)(x);
            let term = h(x) * w;
            buf.push(term);
            sum += term;
            sum_sq += term * term;
        }
        let m = (n as f64).max(2.0);
        let mean = sum / (n as f64);
        let mut var = (sum_sq - mean * sum) / (m - 1.0).max(1.0);
        if var < 0.0 {
            var = 0.0;
        }
        let stderr = (var / (n as f64)).sqrt();
        (mean, stderr)
    }
}

/// Convenience importance-sampling estimator.
///
/// Draws `n` samples from `proposal` and estimates `E_f[h(X)]` using the
/// likelihood ratio `target(x) / proposal_density(x)`. Equivalent to
/// constructing an [`ImportanceSampling`] value and calling
/// [`ImportanceSampling::estimate`].
pub fn importance<R, P, F, G, H>(
    rng: &mut R,
    n: usize,
    proposal: &P,
    target: F,
    proposal_density: G,
    h: H,
) -> (f64, f64)
where
    R: Rng + ?Sized,
    P: Distribution<f64>,
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    H: Fn(f64) -> f64,
{
    ImportanceSampling::new(target, proposal_density).estimate(rng, proposal, h, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::SplitMix64;

    #[test]
    fn mean_and_var_basic() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var) = mean_and_var(&xs);
        assert!((mean - 3.0).abs() < 1e-12);
        assert!((var - 2.5).abs() < 1e-12);
    }

    #[test]
    fn mean_and_var_empty_and_single() {
        assert_eq!(mean_and_var(&[]), (0.0, 0.0));
        assert_eq!(mean_and_var(&[7.0]), (7.0, 0.0));
    }

    #[test]
    fn integrate_x_squared_equals_one_third() {
        let mut rng = SplitMix64::seed_from_u64(12345);
        let (mean, stderr) = integrate(&mut rng, |x| x * x, 0.0, 1.0, 200_000);
        assert!((mean - 1.0 / 3.0).abs() < 5.0 * stderr);
        assert!(stderr > 0.0);
        assert!(stderr < 1e-2);
    }

    #[test]
    fn integrate_constant_is_exact() {
        let mut rng = SplitMix64::seed_from_u64(1);
        let (mean, stderr) = integrate(&mut rng, |_x| 2.0, 0.0, 1.0, 10_000);
        assert!((mean - 2.0).abs() < 1e-12);
        assert!(stderr < 1e-12);
    }

    #[test]
    fn integrate_linear_is_exact_in_mean() {
        // ∫_0^2 x dx = 2; variance of x on [0,2] is 1/3 so stderr > 0.
        let mut rng = SplitMix64::seed_from_u64(2);
        let (mean, stderr) = integrate(&mut rng, |x| x, 0.0, 2.0, 100_000);
        assert!((mean - 2.0).abs() < 5.0 * stderr);
    }

    #[test]
    fn estimate_mean_uniform() {
        let mut rng = SplitMix64::seed_from_u64(7);
        let (mean, stderr) = estimate_mean(&Standard, &mut rng, 200_000);
        assert!((mean - 0.5).abs() < 10.0 * stderr);
        // std ~ sqrt(1/12) / sqrt(n) ~ 0.0289 / 447 ~ 6.5e-5
        assert!(stderr > 0.0);
        assert!(stderr < 1e-3);
    }

    #[test]
    fn estimate_mean_variance_sanity() {
        let n = 50_000usize;
        let mut rng = SplitMix64::seed_from_u64(8);
        let (mean, stderr) = estimate_mean(&Standard, &mut rng, n);
        // expected variance of uniform is 1/12; stderr^2 * n ~ var(f).
        let implied_var = stderr * stderr * (n as f64);
        assert!((implied_var - 1.0 / 12.0).abs() < 5e-3);
        assert!((mean - 0.5).abs() < 10.0 * stderr);
    }

    #[test]
    fn importance_sampling_matches_uniform_target() {
        // Target = uniform on [0,1) (density 1), proposal = uniform (density 1),
        // so E_f[x] = 0.5.
        let mut rng = SplitMix64::seed_from_u64(11);
        let is = ImportanceSampling::new(|_x| 1.0, |_x| 1.0);
        let (est, stderr) = is.estimate(&mut rng, &Standard, |x| x, 100_000);
        assert!((est - 0.5).abs() < 10.0 * stderr);
    }

    #[test]
    fn importance_sampling_reduces_to_expectation_of_one() {
        let mut rng = SplitMix64::seed_from_u64(13);
        // Identical target and proposal give a weight of exactly 1, so this is
        // a zero-variance estimator: `est` equals `1.0` and `stderr` is `0`.
        let (est, _stderr) = importance(&mut rng, 100_000, &Standard, |_x| 1.0, |_x| 1.0, |_x| 1.0);
        assert!((est - 1.0).abs() < 1e-12);
    }

    #[test]
    fn integrate_scales_with_width() {
        let mut rng = SplitMix64::seed_from_u64(21);
        let (mean, _) = integrate(&mut rng, |x| x, 0.0, 10.0, 200_000);
        assert!((mean - 50.0).abs() < 2.0);
    }
}
