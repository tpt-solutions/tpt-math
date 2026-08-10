//! Standalone log-likelihood helpers for the common likelihoods.
//!
//! Every function returns a value in log space, so terms compose by addition
//! and stay well behaved for long data sets. Parameters outside their valid
//! range yield `-∞` (an impossible model) rather than `NaN`.

use crate::special::{ln_binomial, ln_factorial, xlogy, LN_2PI};

/// Log-likelihood of `successes` and `failures` Bernoulli trials at rate `p`.
///
/// `s·ln p + f·ln(1 − p)`, using the `0 · ln 0 = 0` convention. Returns `-∞`
/// for `p` outside `[0, 1]`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::bernoulli_log_likelihood;
///
/// assert!((bernoulli_log_likelihood(0.5, 2, 2) - 0.0625f64.ln()).abs() < 1e-12);
/// ```
#[must_use]
pub fn bernoulli_log_likelihood(p: f64, successes: u64, failures: u64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NEG_INFINITY;
    }
    xlogy(successes as f64, p) + xlogy(failures as f64, 1.0 - p)
}

/// Log-likelihood of observing `successes` out of `trials` at rate `p`.
///
/// Same as [`bernoulli_log_likelihood`] plus the `ln C(n, k)` combinatorial
/// term. Returns `-∞` when `successes > trials` or `p` is out of range.
#[must_use]
pub fn binomial_log_likelihood(p: f64, successes: u64, trials: u64) -> f64 {
    if successes > trials {
        return f64::NEG_INFINITY;
    }
    ln_binomial(trials, successes) + bernoulli_log_likelihood(p, successes, trials - successes)
}

/// Log-likelihood of independent Gaussian `data` with mean `mean` and
/// standard deviation `std`.
///
/// Returns `-∞` for a non-positive or non-finite `std`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::{normal_log_likelihood, Gaussian};
///
/// let data = [0.2, -1.1, 0.7];
/// let direct = normal_log_likelihood(&data, 0.0, 1.0);
/// let via_dist = Gaussian::standard().log_likelihood(&data);
/// assert!((direct - via_dist).abs() < 1e-12);
/// ```
#[must_use]
pub fn normal_log_likelihood(data: &[f64], mean: f64, std: f64) -> f64 {
    if !std.is_finite() || std <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = data.len() as f64;
    let sum_sq: f64 = data
        .iter()
        .map(|x| {
            let z = (x - mean) / std;
            z * z
        })
        .sum();
    -0.5 * sum_sq - n * (std.ln() + 0.5 * LN_2PI)
}

/// Log-likelihood of Poisson `counts` at rate `rate`.
///
/// `Σ (kᵢ ln λ − λ − ln kᵢ!)`. Returns `-∞` for a negative or non-finite
/// rate, and handles `λ = 0` (only all-zero counts are possible).
#[must_use]
pub fn poisson_log_likelihood(rate: f64, counts: &[u64]) -> f64 {
    if rate < 0.0 || !rate.is_finite() {
        return f64::NEG_INFINITY;
    }
    let n = counts.len() as f64;
    counts
        .iter()
        .map(|&k| xlogy(k as f64, rate) - ln_factorial(k))
        .sum::<f64>()
        - n * rate
}

/// Log-likelihood of exponential `data` at rate `rate`.
///
/// `n ln λ − λ Σxᵢ`. Returns `-∞` for a non-positive rate or negative data.
#[must_use]
pub fn exponential_log_likelihood(rate: f64, data: &[f64]) -> f64 {
    if !rate.is_finite() || rate <= 0.0 || data.iter().any(|x| *x < 0.0 || x.is_nan()) {
        return f64::NEG_INFINITY;
    }
    data.len() as f64 * rate.ln() - rate * data.iter().sum::<f64>()
}

/// Numerically stable `ln Σ exp(xᵢ)`.
///
/// Useful for normalising a vector of log-weights (importance weights, model
/// evidences, mixture components). Returns `-∞` for an empty slice.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::log_sum_exp;
///
/// let logs = [-1000.0, -1000.0];
/// assert!((log_sum_exp(&logs) - (-1000.0 + 2f64.ln())).abs() < 1e-12);
/// ```
#[must_use]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gaussian;

    #[test]
    fn bernoulli_matches_direct_product() {
        assert!((bernoulli_log_likelihood(0.5, 2, 2) - 0.0625f64.ln()).abs() < 1e-12);
        assert!((bernoulli_log_likelihood(0.3, 1, 1) - 0.21f64.ln()).abs() < 1e-12);
        assert_eq!(bernoulli_log_likelihood(0.0, 0, 0), 0.0);
        assert_eq!(bernoulli_log_likelihood(1.0, 3, 0), 0.0);
        assert_eq!(bernoulli_log_likelihood(1.0, 3, 1), f64::NEG_INFINITY);
        assert_eq!(bernoulli_log_likelihood(1.5, 1, 1), f64::NEG_INFINITY);
        assert_eq!(bernoulli_log_likelihood(f64::NAN, 1, 1), f64::NEG_INFINITY);
    }

    #[test]
    fn binomial_sums_to_one_over_all_counts() {
        let p = 0.37;
        let n = 9;
        let total: f64 = (0..=n)
            .map(|k| binomial_log_likelihood(p, k, n).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert_eq!(binomial_log_likelihood(0.5, 4, 3), f64::NEG_INFINITY);
    }

    #[test]
    fn normal_matches_distribution_impl() {
        let data = [0.2, -1.1, 0.7, 2.4];
        let direct = normal_log_likelihood(&data, 0.5, 1.25);
        let via_dist = Gaussian::new(0.5, 1.25).log_likelihood(&data);
        assert!((direct - via_dist).abs() < 1e-12);
        assert_eq!(normal_log_likelihood(&data, 0.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(normal_log_likelihood(&[], 0.0, 1.0), 0.0);
    }

    #[test]
    fn normal_likelihood_is_maximised_at_the_sample_mean() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let best = normal_log_likelihood(&data, 2.5, 1.0);
        for &m in &[2.0, 2.4, 2.6, 3.0] {
            assert!(normal_log_likelihood(&data, m, 1.0) < best);
        }
    }

    #[test]
    fn poisson_probabilities_sum_to_one() {
        let rate = 2.5;
        let total: f64 = (0..40u64)
            .map(|k| poisson_log_likelihood(rate, &[k]).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert_eq!(poisson_log_likelihood(0.0, &[0, 0]), 0.0);
        assert_eq!(poisson_log_likelihood(0.0, &[1]), f64::NEG_INFINITY);
        assert_eq!(poisson_log_likelihood(-1.0, &[1]), f64::NEG_INFINITY);
    }

    #[test]
    fn exponential_likelihood_is_maximised_at_the_inverse_mean() {
        let data = [0.5, 1.5, 2.0, 1.0];
        let mle = data.len() as f64 / data.iter().sum::<f64>();
        let best = exponential_log_likelihood(mle, &data);
        for delta in [-0.2, -0.05, 0.05, 0.2] {
            assert!(exponential_log_likelihood(mle + delta, &data) < best);
        }
        assert_eq!(exponential_log_likelihood(1.0, &[-1.0]), f64::NEG_INFINITY);
        assert_eq!(exponential_log_likelihood(0.0, &[1.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_is_stable() {
        assert!((log_sum_exp(&[-1000.0, -1000.0]) - (-1000.0 + 2f64.ln())).abs() < 1e-12);
        assert!((log_sum_exp(&[0.0, 0.0, 0.0]) - 3f64.ln()).abs() < 1e-12);
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }
}
