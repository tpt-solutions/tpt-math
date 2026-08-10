//! Correlation and ordinary least-squares regression for paired samples.
//!
//! Both routines share the same centred sums of squares and cross-products,
//! computed with compensated summation about the sample means so that data
//! sitting far from the origin (timestamps, absolute temperatures, …) does not
//! destroy the result the way the textbook "computational" formula would.

use crate::descriptive::{compensated_sum, mean_unchecked};
use crate::error::{check_all_finite, check_equal_len, check_min_len, StatsError};

/// Centred sums `(Sxx, Syy, Sxy)` and the two means, for equal-length,
/// all-finite samples.
fn centred_sums(x: &[f64], y: &[f64]) -> CentredSums {
    let mean_x = mean_unchecked(x);
    let mean_y = mean_unchecked(y);

    let sxx = compensated_sum(x.iter().map(|&xi| (xi - mean_x) * (xi - mean_x)));
    let syy = compensated_sum(y.iter().map(|&yi| (yi - mean_y) * (yi - mean_y)));
    let sxy = compensated_sum(
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)),
    );

    CentredSums {
        mean_x,
        mean_y,
        sxx: sxx.max(0.0),
        syy: syy.max(0.0),
        sxy,
    }
}

/// Intermediate quantities shared by correlation and regression.
struct CentredSums {
    mean_x: f64,
    mean_y: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

/// Pearson product-moment correlation coefficient of `x` and `y`.
///
/// ```text
/// r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √( Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)² )
/// ```
///
/// The result is clamped to `[-1, 1]`, so a perfect linear relationship
/// returns exactly `1.0` (or `-1.0`) instead of `1.0 + 2 ulp`.
///
/// If either sample is constant the correlation is undefined (a zero variance
/// divides the definition) and `NaN` is returned; this is a documented result,
/// not an error.
///
/// # Errors
///
/// * [`StatsError::LengthMismatch`] — the two slices differ in length.
/// * [`StatsError::NotEnoughData`] — fewer than two paired observations.
/// * [`StatsError::NotFinite`] — an observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_pearson_correlation;
/// let x = [1.0, 2.0, 3.0, 4.0];
/// let y = [2.0, 4.0, 6.0, 8.0];
/// assert_eq!(try_pearson_correlation(&x, &y).unwrap(), 1.0);
///
/// let flat = [7.0; 4];
/// assert!(try_pearson_correlation(&x, &flat).unwrap().is_nan());
/// ```
pub fn try_pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, StatsError> {
    check_equal_len(x.len(), y.len())?;
    check_min_len("x", x, 2)?;
    check_all_finite("x", x)?;
    check_all_finite("y", y)?;

    let sums = centred_sums(x, y);
    if sums.sxx == 0.0 || sums.syy == 0.0 {
        return Ok(f64::NAN);
    }

    Ok((sums.sxy / (sums.sxx * sums.syy).sqrt()).clamp(-1.0, 1.0))
}

/// Pearson product-moment correlation coefficient of `x` and `y`.
///
/// # Panics
///
/// Panics on the conditions listed for [`try_pearson_correlation`]. A constant
/// sample is *not* one of them: it yields `NaN`.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::pearson_correlation;
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [5.0, 4.0, 3.0, 2.0, 1.0];
/// assert_eq!(pearson_correlation(&x, &y), -1.0);
/// ```
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    try_pearson_correlation(x, y).unwrap_or_else(|e| panic!("pearson_correlation: {e}"))
}

/// Ordinary least-squares fit of `y = slope · x + intercept`.
///
/// ```text
/// slope     = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
/// intercept = ȳ - slope · x̄
/// ```
///
/// If `x` is constant the normal equations are singular — infinitely many
/// lines fit equally well — and `(NaN, NaN)` is returned; this is a documented
/// result, not an error.
///
/// # Errors
///
/// * [`StatsError::LengthMismatch`] — the two slices differ in length.
/// * [`StatsError::NotEnoughData`] — fewer than two paired observations.
/// * [`StatsError::NotFinite`] — an observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_linear_regression;
/// let x = [0.0, 1.0, 2.0, 3.0];
/// let y = [7.0, 10.0, 13.0, 16.0]; // exactly 3x + 7
/// let (slope, intercept) = try_linear_regression(&x, &y).unwrap();
/// assert!((slope - 3.0).abs() < 1e-12);
/// assert!((intercept - 7.0).abs() < 1e-12);
/// ```
pub fn try_linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64), StatsError> {
    check_equal_len(x.len(), y.len())?;
    check_min_len("x", x, 2)?;
    check_all_finite("x", x)?;
    check_all_finite("y", y)?;

    let sums = centred_sums(x, y);
    if sums.sxx == 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let slope = sums.sxy / sums.sxx;
    Ok((slope, sums.mean_y - slope * sums.mean_x))
}

/// Ordinary least-squares fit of `y = slope · x + intercept`.
///
/// # Panics
///
/// Panics on the conditions listed for [`try_linear_regression`]. A constant
/// predictor is *not* one of them: it yields `(NaN, NaN)`.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::linear_regression;
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.1, 3.9, 6.2, 7.8, 10.1];
/// let (slope, intercept) = linear_regression(&x, &y);
/// assert!((slope - 2.0).abs() < 0.1);
/// assert!(intercept.abs() < 0.2);
/// ```
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    try_linear_regression(x, y).unwrap_or_else(|e| panic!("linear_regression: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{std_dev, Distribution as _, SplitMix64, Standard};

    #[test]
    fn perfectly_linear_data_correlates_exactly() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let increasing: Vec<f64> = x.iter().map(|xi| 3.0 * xi + 7.0).collect();
        let decreasing: Vec<f64> = x.iter().map(|xi| -0.5 * xi + 2.0).collect();

        assert_eq!(pearson_correlation(&x, &increasing), 1.0);
        assert_eq!(pearson_correlation(&x, &decreasing), -1.0);
        assert_eq!(pearson_correlation(&x, &x), 1.0);
    }

    #[test]
    fn regression_recovers_an_exact_line() {
        let x: Vec<f64> = (0..25).map(|i| i as f64 * 0.5 - 3.0).collect();
        let y: Vec<f64> = x.iter().map(|xi| -1.25 * xi + 4.75).collect();

        let (slope, intercept) = linear_regression(&x, &y);
        assert!((slope + 1.25).abs() < 1e-12, "slope = {slope}");
        assert!((intercept - 4.75).abs() < 1e-12, "intercept = {intercept}");
    }

    #[test]
    fn regression_recovers_a_line_hidden_under_noise() {
        // Uniform(-0.5, 0.5) noise from the workspace's deterministic PRNG.
        let mut rng = SplitMix64::seed_from_u64(20_260_810);
        let (true_slope, true_intercept) = (2.5, -1.0);

        let x: Vec<f64> = (0..2_000).map(|i| i as f64 * 0.01).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|xi| {
                let noise: f64 = Standard.sample(&mut rng);
                true_slope * xi + true_intercept + (noise - 0.5)
            })
            .collect();

        let (slope, intercept) = linear_regression(&x, &y);
        assert!((slope - true_slope).abs() < 0.01, "slope = {slope}");
        assert!(
            (intercept - true_intercept).abs() < 0.05,
            "intercept = {intercept}"
        );

        // Strong but imperfect linear relationship.
        let r = pearson_correlation(&x, &y);
        assert!(r > 0.99 && r < 1.0, "r = {r}");
    }

    #[test]
    fn regression_is_stable_far_from_the_origin() {
        // Years as absolute offsets: the naive Σx² - nx̄² formula loses ~9
        // significant digits here.
        let x: Vec<f64> = (0..50).map(|i| 1.0e9 + i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * (xi - 1.0e9) + 11.0).collect();

        let (slope, _) = linear_regression(&x, &y);
        assert!((slope - 3.0).abs() < 1e-9, "slope = {slope}");
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correlation_is_symmetric_and_scale_invariant() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        let r = pearson_correlation(&x, &y);
        assert!((r - pearson_correlation(&y, &x)).abs() < 1e-15);

        let scaled: Vec<f64> = x.iter().map(|xi| 100.0 * xi + 17.0).collect();
        assert!((r - pearson_correlation(&scaled, &y)).abs() < 1e-14);

        // Negating one variable flips the sign.
        let negated: Vec<f64> = y.iter().map(|yi| -yi).collect();
        assert!((r + pearson_correlation(&x, &negated)).abs() < 1e-15);
        assert!((-1.0..=1.0).contains(&r));
    }

    #[test]
    fn slope_matches_the_correlation_identity() {
        // slope = r · s_y / s_x
        let x = [1.0, 3.0, 4.0, 6.0, 8.0, 9.0, 11.0, 14.0];
        let y = [1.0, 2.0, 4.0, 4.0, 5.0, 7.0, 8.0, 9.0];

        let (slope, intercept) = linear_regression(&x, &y);
        let identity = pearson_correlation(&x, &y) * std_dev(&y) / std_dev(&x);
        assert!((slope - identity).abs() < 1e-12, "slope = {slope}");

        // The fitted line passes through the centroid.
        let (mean_x, mean_y) = (crate::mean(&x), crate::mean(&y));
        assert!((slope * mean_x + intercept - mean_y).abs() < 1e-12);

        // Hand-checked values: Sxy = 84, Sxx = 132, x̄ = 7, ȳ = 5, so the fit
        // is exactly y = (7/11)x + 6/11.
        assert!((slope - 7.0 / 11.0).abs() < 1e-12, "slope = {slope}");
        assert!(
            (intercept - 6.0 / 11.0).abs() < 1e-12,
            "intercept = {intercept}"
        );
    }

    #[test]
    fn residuals_are_orthogonal_to_the_predictor() {
        let x = [0.5, 1.5, 2.5, 4.0, 6.0, 7.5];
        let y = [1.1, 2.3, 2.2, 4.9, 5.5, 7.7];
        let (slope, intercept) = linear_regression(&x, &y);

        let residuals: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| yi - (slope * xi + intercept))
            .collect();

        // Least squares forces Σe = 0 and Σxe = 0.
        assert!(compensated_sum(residuals.iter().copied()).abs() < 1e-12);
        let weighted = compensated_sum(x.iter().zip(residuals.iter()).map(|(&xi, &ei)| xi * ei));
        assert!(weighted.abs() < 1e-12, "Σxe = {weighted}");
    }

    #[test]
    fn constant_inputs_give_nan_rather_than_an_error() {
        let x = [1.0, 2.0, 3.0];
        let flat = [4.0, 4.0, 4.0];

        assert!(pearson_correlation(&x, &flat).is_nan());
        assert!(pearson_correlation(&flat, &x).is_nan());

        let (slope, intercept) = linear_regression(&flat, &x);
        assert!(slope.is_nan() && intercept.is_nan());

        // A constant *response* is still a legitimate fit: the zero line.
        let (slope, intercept) = linear_regression(&x, &flat);
        assert_eq!(slope, 0.0);
        assert!((intercept - 4.0).abs() < 1e-12);
    }

    #[test]
    fn checked_variants_report_bad_input() {
        assert!(matches!(
            try_pearson_correlation(&[1.0, 2.0], &[1.0]),
            Err(StatsError::LengthMismatch {
                first: 2,
                second: 1
            })
        ));
        assert!(matches!(
            try_pearson_correlation(&[1.0], &[1.0]),
            Err(StatsError::NotEnoughData {
                required: 2,
                found: 1,
                ..
            })
        ));
        assert!(matches!(
            try_linear_regression(&[1.0, 2.0], &[1.0, f64::NAN]),
            Err(StatsError::NotFinite {
                sample: "y",
                index: 1,
                ..
            })
        ));
        assert!(matches!(
            try_linear_regression(&[], &[]),
            Err(StatsError::NotEnoughData { found: 0, .. })
        ));
    }

    #[test]
    #[should_panic(expected = "linear_regression")]
    fn linear_regression_panics_on_mismatched_lengths() {
        let _ = linear_regression(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
    }
}
