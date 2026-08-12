//! Descriptive statistics for `f64` samples.
//!
//! Each statistic comes in two flavours:
//!
//! * a checked `try_*` function returning [`Result<_, StatsError>`], and
//! * a panicking convenience wrapper with the short, obvious name.
//!
//! All inputs must be finite; `NaN` and infinities are rejected up front so
//! that the results (and any ordering performed internally) are well defined.
//!
//! The sums below are computed with [Neumaier][neumaier] compensated
//! summation, and the variance uses the corrected two-pass formula, so long or
//! badly scaled samples do not lose precision the way a naive
//! `sum(x^2) - n * mean^2` would.
//!
//! [neumaier]: https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

use crate::error::{check_all_finite, check_min_len, StatsError};

/// Neumaier-compensated summation: an accurate `Σ values`.
pub(crate) fn compensated_sum<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for value in values {
        let total = sum + value;
        compensation += if sum.abs() >= value.abs() {
            (sum - total) + value
        } else {
            (value - total) + sum
        };
        sum = total;
    }
    sum + compensation
}

/// Arithmetic mean of a non-empty, all-finite sample.
pub(crate) fn mean_unchecked(samples: &[f64]) -> f64 {
    compensated_sum(samples.iter().copied()) / samples.len() as f64
}

/// Unbiased (`n - 1`) sample variance of an all-finite sample of length >= 2.
pub(crate) fn variance_unchecked(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = mean_unchecked(samples);

    // Corrected two-pass algorithm: the second term cancels the error left in
    // `mean` by floating-point rounding.
    let sum_sq = compensated_sum(samples.iter().map(|&x| (x - mean) * (x - mean)));
    let sum_dev = compensated_sum(samples.iter().map(|&x| x - mean));

    let variance = (sum_sq - sum_dev * sum_dev / n) / (n - 1.0);

    // Rounding can push an exactly-constant sample a hair below zero.
    variance.max(0.0)
}

/// Arithmetic mean of `samples`.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if `samples` is empty, or
/// [`StatsError::NotFinite`] if any observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_mean;
/// assert_eq!(try_mean(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 2.5);
/// assert!(try_mean(&[]).is_err());
/// ```
pub fn try_mean(samples: &[f64]) -> Result<f64, StatsError> {
    check_min_len("samples", samples, 1)?;
    check_all_finite("samples", samples)?;
    Ok(mean_unchecked(samples))
}

/// Arithmetic mean of `samples`.
///
/// # Panics
///
/// Panics if `samples` is empty or contains a non-finite value; see
/// [`try_mean`] for the checked version.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::mean;
/// assert_eq!(mean(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 5.0);
/// ```
pub fn mean(samples: &[f64]) -> f64 {
    try_mean(samples).unwrap_or_else(|e| panic!("mean: {e}"))
}

/// Unbiased sample variance of `samples` (Bessel-corrected, `n - 1` divisor).
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if `samples` holds fewer than two
/// observations, or [`StatsError::NotFinite`] if any observation is `NaN` or
/// infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_variance;
/// // Σ(x - 5)² = 32 over 8 observations => 32 / 7.
/// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert!((try_variance(&data).unwrap() - 32.0 / 7.0).abs() < 1e-12);
/// ```
pub fn try_variance(samples: &[f64]) -> Result<f64, StatsError> {
    check_min_len("samples", samples, 2)?;
    check_all_finite("samples", samples)?;
    Ok(variance_unchecked(samples))
}

/// Unbiased sample variance of `samples` (Bessel-corrected, `n - 1` divisor).
///
/// # Panics
///
/// Panics if `samples` holds fewer than two observations or contains a
/// non-finite value; see [`try_variance`] for the checked version.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::variance;
/// assert_eq!(variance(&[1.0, 3.0]), 2.0);
/// ```
pub fn variance(samples: &[f64]) -> f64 {
    try_variance(samples).unwrap_or_else(|e| panic!("variance: {e}"))
}

/// Sample standard deviation: the square root of [`try_variance`].
///
/// # Errors
///
/// Same conditions as [`try_variance`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_std_dev;
/// assert_eq!(try_std_dev(&[1.0, 5.0]).unwrap(), 8.0f64.sqrt());
/// ```
pub fn try_std_dev(samples: &[f64]) -> Result<f64, StatsError> {
    try_variance(samples).map(f64::sqrt)
}

/// Sample standard deviation: the square root of [`variance`].
///
/// # Panics
///
/// Same conditions as [`variance`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::std_dev;
/// assert!((std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) - 2.13808993529939).abs() < 1e-12);
/// ```
pub fn std_dev(samples: &[f64]) -> f64 {
    try_std_dev(samples).unwrap_or_else(|e| panic!("std_dev: {e}"))
}

/// Smallest observation in `samples`.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if `samples` is empty, or
/// [`StatsError::NotFinite`] if any observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_min;
/// assert_eq!(try_min(&[3.0, -1.0, 2.0]).unwrap(), -1.0);
/// ```
pub fn try_min(samples: &[f64]) -> Result<f64, StatsError> {
    check_min_len("samples", samples, 1)?;
    check_all_finite("samples", samples)?;
    Ok(samples.iter().copied().fold(f64::INFINITY, f64::min))
}

/// Smallest observation in `samples`.
///
/// # Panics
///
/// Panics if `samples` is empty or contains a non-finite value; see
/// [`try_min`] for the checked version.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::min;
/// assert_eq!(min(&[3.0, -1.0, 2.0]), -1.0);
/// ```
pub fn min(samples: &[f64]) -> f64 {
    try_min(samples).unwrap_or_else(|e| panic!("min: {e}"))
}

/// Largest observation in `samples`.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if `samples` is empty, or
/// [`StatsError::NotFinite`] if any observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_max;
/// assert_eq!(try_max(&[3.0, -1.0, 2.0]).unwrap(), 3.0);
/// ```
pub fn try_max(samples: &[f64]) -> Result<f64, StatsError> {
    check_min_len("samples", samples, 1)?;
    check_all_finite("samples", samples)?;
    Ok(samples.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

/// Largest observation in `samples`.
///
/// # Panics
///
/// Panics if `samples` is empty or contains a non-finite value; see
/// [`try_max`] for the checked version.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::max;
/// assert_eq!(max(&[3.0, -1.0, 2.0]), 3.0);
/// ```
pub fn max(samples: &[f64]) -> f64 {
    try_max(samples).unwrap_or_else(|e| panic!("max: {e}"))
}

/// Median of `samples`: the middle observation, or the mean of the two middle
/// observations when the sample size is even.
///
/// The input is left untouched; sorting happens on an internal copy.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if `samples` is empty, or
/// [`StatsError::NotFinite`] if any observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_median;
/// assert_eq!(try_median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
/// assert_eq!(try_median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
/// ```
pub fn try_median(samples: &[f64]) -> Result<f64, StatsError> {
    check_min_len("samples", samples, 1)?;
    check_all_finite("samples", samples)?;

    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);

    let mid = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    })
}

/// Median of `samples`.
///
/// # Panics
///
/// Panics if `samples` is empty or contains a non-finite value; see
/// [`try_median`] for the checked version.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::median;
/// assert_eq!(median(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 4.5);
/// ```
pub fn median(samples: &[f64]) -> f64 {
    try_median(samples).unwrap_or_else(|e| panic!("median: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATA: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    #[test]
    fn known_dataset_matches_hand_computed_values() {
        // Σx = 40 over 8 observations, Σ(x - 5)² = 32.
        assert_eq!(mean(&DATA), 5.0);
        assert!((variance(&DATA) - 32.0 / 7.0).abs() < 1e-15);
        assert!((std_dev(&DATA) - (32.0f64 / 7.0).sqrt()).abs() < 1e-15);
        assert_eq!(min(&DATA), 2.0);
        assert_eq!(max(&DATA), 9.0);
        assert_eq!(median(&DATA), 4.5);
    }

    #[test]
    fn median_handles_odd_and_unsorted_input() {
        let unsorted = [5.0, -2.0, 11.0, 0.5, 3.0];
        assert_eq!(median(&unsorted), 3.0);
        // The caller's slice is untouched.
        assert_eq!(unsorted[0], 5.0);

        assert_eq!(median(&[42.0]), 42.0);
        assert_eq!(median(&[1.0, 2.0]), 1.5);
    }

    #[test]
    fn variance_matches_population_identity() {
        // Bessel-corrected variance: s² = (Σx² - n·x̄²) / (n - 1).
        let data = [1.5, -3.25, 8.0, 0.0, 4.75, 4.75, -1.0];
        let n = data.len() as f64;
        let m = mean(&data);
        let sum_sq = compensated_sum(data.iter().map(|&x| x * x));
        let reference = (sum_sq - n * m * m) / (n - 1.0);
        assert!((variance(&data) - reference).abs() < 1e-12);
    }

    #[test]
    fn constant_sample_has_zero_variance() {
        let flat = [2.5; 6];
        assert_eq!(variance(&flat), 0.0);
        assert_eq!(std_dev(&flat), 0.0);
        assert_eq!(median(&flat), 2.5);
    }

    #[test]
    fn compensated_sum_beats_naive_summation() {
        // 1.0 followed by 10_000 copies of 1e-13: the naive running sum loses
        // most of the small terms, the compensated one does not.
        let mut data = vec![1e-13; 10_000];
        data.push(1.0);
        let expected = 1.0 + 1e-13 * 10_000.0;
        assert!((compensated_sum(data.iter().copied()) - expected).abs() < 1e-16);
    }

    #[test]
    fn variance_is_shift_invariant_for_large_offsets() {
        let base = [1.0, 2.0, 3.0, 4.0, 5.0];
        let shifted: Vec<f64> = base.iter().map(|x| x + 1e9).collect();
        assert!((variance(&base) - variance(&shifted)).abs() < 1e-12);
        assert_eq!(variance(&base), 2.5);
    }

    #[test]
    fn checked_variants_report_bad_input() {
        assert!(matches!(
            try_mean(&[]),
            Err(StatsError::NotEnoughData {
                required: 1,
                found: 0,
                ..
            })
        ));
        assert!(matches!(
            try_variance(&[1.0]),
            Err(StatsError::NotEnoughData {
                required: 2,
                found: 1,
                ..
            })
        ));
        assert!(matches!(
            try_std_dev(&[1.0]),
            Err(StatsError::NotEnoughData { .. })
        ));
        assert!(matches!(
            try_median(&[1.0, f64::NAN]),
            Err(StatsError::NotFinite { index: 1, .. })
        ));
        assert!(matches!(
            try_min(&[f64::INFINITY, 1.0]),
            Err(StatsError::NotFinite { index: 0, .. })
        ));
        assert!(matches!(
            try_max(&[]),
            Err(StatsError::NotEnoughData { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "mean")]
    fn mean_panics_on_empty_input() {
        let _ = mean(&[]);
    }

    #[test]
    #[should_panic(expected = "variance")]
    fn variance_panics_on_single_observation() {
        let _ = variance(&[1.0]);
    }
}
