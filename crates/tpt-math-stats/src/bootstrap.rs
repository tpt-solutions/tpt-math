//! Nonparametric bootstrap confidence intervals.
//!
//! The percentile method: resample the data with replacement, evaluate the
//! statistic on each resample, and take the empirical quantiles of those
//! values. Resampling uses [`tpt_math_prob_core::SplitMix64`] seeded by the
//! caller, so a given seed always gives the same interval.

use crate::descriptive::try_quantile;
use crate::error::{check_all_finite, check_min_len, StatsError};
use tpt_math_prob_core::{Rng, SplitMix64};

/// Percentile bootstrap confidence interval for `statistic(samples)`.
///
/// Draws `resamples` bootstrap samples of the same size as `samples`, evaluates
/// `statistic` on each, and returns the `(1 − c)/2` and `(1 + c)/2` empirical
/// quantiles of those values, where `c` is the confidence level.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if `samples` is empty or `resamples`
/// is zero, [`StatsError::NotFinite`] for a non-finite observation, and
/// [`StatsError::ProbabilityOutOfRange`] unless `confidence` lies strictly
/// between 0 and 1.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_bootstrap_ci;
/// # use tpt_math_stats::try_mean;
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let (lo, hi) = try_bootstrap_ci(&data, |s| try_mean(s).unwrap(), 2000, 0.95, 42).unwrap();
/// assert!(lo < 5.5 && 5.5 < hi);
/// ```
pub fn try_bootstrap_ci<F>(
    samples: &[f64],
    statistic: F,
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> Result<(f64, f64), StatsError>
where
    F: Fn(&[f64]) -> f64,
{
    check_min_len("samples", samples, 1)?;
    check_all_finite("samples", samples)?;
    if resamples == 0 {
        return Err(StatsError::NotEnoughData {
            sample: "resamples",
            required: 1,
            found: 0,
        });
    }
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }

    let n = samples.len();
    let mut rng = SplitMix64::seed_from_u64(seed);
    let mut resample = vec![0.0; n];
    let mut stats = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        for slot in resample.iter_mut() {
            // next_f64 is in [0, 1), so the index is always below n.
            let index = ((rng.next_f64() * n as f64) as usize).min(n - 1);
            *slot = samples[index];
        }
        stats.push(statistic(&resample));
    }

    let alpha = (1.0 - confidence) / 2.0;
    let lower = try_quantile(&stats, alpha)?;
    let upper = try_quantile(&stats, 1.0 - alpha)?;
    Ok((lower, upper))
}

/// Percentile bootstrap confidence interval. See [`try_bootstrap_ci`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_bootstrap_ci`].
pub fn bootstrap_ci<F>(
    samples: &[f64],
    statistic: F,
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    try_bootstrap_ci(samples, statistic, resamples, confidence, seed)
        .unwrap_or_else(|e| panic!("bootstrap_ci: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::mean;

    #[test]
    fn constant_sample_gives_a_point_interval() {
        let (lo, hi) = try_bootstrap_ci(&[2.5; 6], mean, 200, 0.9, 1).unwrap();
        assert_eq!(lo, 2.5);
        assert_eq!(hi, 2.5);
    }

    #[test]
    fn same_seed_gives_the_same_interval() {
        let data = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0];
        let a = try_bootstrap_ci(&data, mean, 500, 0.95, 7).unwrap();
        let b = try_bootstrap_ci(&data, mean, 500, 0.95, 7).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn interval_brackets_the_sample_mean_and_lies_within_data_range() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (lo, hi) = try_bootstrap_ci(&data, mean, 4000, 0.95, 99).unwrap();
        assert!(lo < 5.5 && 5.5 < hi, "({lo}, {hi})");
        // Every resampled mean lies within the data range.
        assert!(lo >= 1.0 && hi <= 10.0);
        // The standard error of the mean is about 0.91, so a 95% interval is
        // roughly ±1.8 wide. Allow generous slack for Monte Carlo error.
        assert!(hi - lo > 2.5 && hi - lo < 5.5, "width {}", hi - lo);
    }

    #[test]
    fn higher_confidence_gives_a_wider_interval() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let narrow = try_bootstrap_ci(&data, mean, 3000, 0.5, 3).unwrap();
        let wide = try_bootstrap_ci(&data, mean, 3000, 0.99, 3).unwrap();
        assert!(wide.1 - wide.0 > narrow.1 - narrow.0);
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(matches!(
            try_bootstrap_ci(&[], mean, 10, 0.95, 0),
            Err(StatsError::NotEnoughData { .. })
        ));
        assert!(matches!(
            try_bootstrap_ci(&[1.0], mean, 0, 0.95, 0),
            Err(StatsError::NotEnoughData { .. })
        ));
        assert!(matches!(
            try_bootstrap_ci(&[1.0], mean, 10, 1.0, 0),
            Err(StatsError::ProbabilityOutOfRange { .. })
        ));
        assert!(matches!(
            try_bootstrap_ci(&[f64::NAN], mean, 10, 0.95, 0),
            Err(StatsError::NotFinite { .. })
        ));
    }
}
