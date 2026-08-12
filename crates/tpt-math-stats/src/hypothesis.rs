//! Classical hypothesis tests built on the in-house [`crate::dist`] distributions.
//!
//! Every test returns the pair `(statistic, p_value)`. The p-values are
//! *two-sided* for the t-tests and *upper-tail* for the chi-squared test, and
//! are always evaluated through the survival function
//! ([`ContinuousCDF::sf`]) rather than `1 - cdf`, so that very small p-values
//! keep their significant digits instead of being annihilated by
//! floating-point cancellation.
//!
//! # Degenerate inputs
//!
//! When the standard error of a t-test is exactly zero (every observation
//! identical) no t-distribution applies. Rather than returning `NaN`, these
//! functions use the limiting convention:
//!
//! * means equal as well: `(0.0, 1.0)` — no evidence at all against `H₀`;
//! * means differ: `(±∞, 0.0)` — the difference is infinitely many standard
//!   errors wide.

use crate::dist::{ChiSquared, ContinuousCDF, StudentsT};

use crate::descriptive::{compensated_sum, mean_unchecked, variance_unchecked};
use crate::error::{
    check_all_finite, check_equal_len, check_min_len, check_parameter_finite, StatsError,
};

/// Two-sided p-value of `statistic` under `StudentsT(0, 1, freedom)`.
fn two_sided_t_p_value(statistic: f64, freedom: f64) -> Result<f64, StatsError> {
    if statistic.is_nan() {
        return Ok(f64::NAN);
    }
    if statistic.is_infinite() {
        return Ok(0.0);
    }

    let t = StudentsT::new(0.0, 1.0, freedom)
        .map_err(|_| StatsError::InvalidDegreesOfFreedom { freedom })?;

    // `sf` is the exact upper tail, so `2 * sf(|t|)` stays accurate deep into
    // the tails where `2 * (1 - cdf(|t|))` would round to zero.
    Ok((2.0 * t.sf(statistic.abs())).clamp(0.0, 1.0))
}

/// Upper-tail p-value of `statistic` under `ChiSquared(freedom)`.
fn upper_tail_chi_squared_p_value(statistic: f64, freedom: f64) -> Result<f64, StatsError> {
    if statistic.is_nan() {
        return Ok(f64::NAN);
    }
    if statistic.is_infinite() {
        return Ok(0.0);
    }

    let chi2 =
        ChiSquared::new(freedom).map_err(|_| StatsError::InvalidDegreesOfFreedom { freedom })?;

    Ok(chi2.sf(statistic).clamp(0.0, 1.0))
}

/// Student's one-sample t-test of `H₀: μ = mu0` against `H₁: μ ≠ mu0`.
///
/// Returns `(statistic, p_value)` where
///
/// ```text
/// t  = (x̄ - μ₀) / (s / √n)      with s the Bessel-corrected sample std dev
/// df = n - 1
/// p  = 2 · P(T_df > |t|)
/// ```
///
/// # Errors
///
/// * [`StatsError::NotEnoughData`] — fewer than two observations.
/// * [`StatsError::NotFinite`] — an observation is `NaN` or infinite.
/// * [`StatsError::ParameterNotFinite`] — `mu0` is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_one_sample_t_test;
/// // x̄ = 2, s = 1, n = 3  =>  t = 2 / (1 / √3) = 2√3.
/// let (t, p) = try_one_sample_t_test(&[1.0, 2.0, 3.0], 0.0).unwrap();
/// assert!((t - 2.0 * 3.0f64.sqrt()).abs() < 1e-12);
/// assert!((p - 0.07417990022744847).abs() < 1e-12);
/// ```
pub fn try_one_sample_t_test(samples: &[f64], mu0: f64) -> Result<(f64, f64), StatsError> {
    check_min_len("samples", samples, 2)?;
    check_all_finite("samples", samples)?;
    check_parameter_finite("mu0", mu0)?;

    let n = samples.len() as f64;
    let difference = mean_unchecked(samples) - mu0;
    let standard_error = (variance_unchecked(samples) / n).sqrt();

    if standard_error == 0.0 {
        return Ok(degenerate_t_result(difference));
    }

    let statistic = difference / standard_error;
    let p_value = two_sided_t_p_value(statistic, n - 1.0)?;
    Ok((statistic, p_value))
}

/// Student's one-sample t-test of `H₀: μ = mu0` against `H₁: μ ≠ mu0`.
///
/// # Panics
///
/// Panics on the conditions listed for [`try_one_sample_t_test`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::one_sample_t_test;
/// let heights = [5.1, 4.9, 5.6, 5.2, 5.0, 4.8, 5.3];
/// let (t, p) = one_sample_t_test(&heights, 5.0);
/// assert!(t > 0.0 && p > 0.05); // no significant departure from 5.0
/// ```
pub fn one_sample_t_test(samples: &[f64], mu0: f64) -> (f64, f64) {
    try_one_sample_t_test(samples, mu0).unwrap_or_else(|e| panic!("one_sample_t_test: {e}"))
}

/// Welch's unequal-variances two-sample t-test of `H₀: μ_a = μ_b`.
///
/// Unlike the pooled-variance Student test, Welch's test does not assume the
/// two samples share a variance; it is the safer default whenever the group
/// sizes or spreads differ.
///
/// Returns `(statistic, p_value)` where
///
/// ```text
/// t  = (ā - b̄) / √(s²ₐ/nₐ + s²_b/n_b)
/// df = (s²ₐ/nₐ + s²_b/n_b)² / [ (s²ₐ/nₐ)²/(nₐ-1) + (s²_b/n_b)²/(n_b-1) ]
/// p  = 2 · P(T_df > |t|)
/// ```
///
/// The Welch–Satterthwaite `df` is generally fractional, which the in-house
/// [`StudentsT`](crate::dist::StudentsT) distribution handles natively.
///
/// # Errors
///
/// * [`StatsError::NotEnoughData`] — either sample has fewer than two
///   observations.
/// * [`StatsError::NotFinite`] — an observation is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_two_sample_t_test;
/// let (t, p) = try_two_sample_t_test(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
/// assert!((t + 3.674234614174767).abs() < 1e-12); // df = 4 exactly here
/// assert!(p < 0.05);
/// ```
pub fn try_two_sample_t_test(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    check_min_len("a", a, 2)?;
    check_min_len("b", b, 2)?;
    check_all_finite("a", a)?;
    check_all_finite("b", b)?;

    let (na, nb) = (a.len() as f64, b.len() as f64);
    let difference = mean_unchecked(a) - mean_unchecked(b);

    // Squared standard error of each group mean.
    let sa = variance_unchecked(a) / na;
    let sb = variance_unchecked(b) / nb;
    let squared_standard_error = sa + sb;

    if squared_standard_error == 0.0 {
        return Ok(degenerate_t_result(difference));
    }

    let statistic = difference / squared_standard_error.sqrt();
    let freedom = squared_standard_error * squared_standard_error
        / (sa * sa / (na - 1.0) + sb * sb / (nb - 1.0));

    let p_value = two_sided_t_p_value(statistic, freedom)?;
    Ok((statistic, p_value))
}

/// Welch's unequal-variances two-sample t-test of `H₀: μ_a = μ_b`.
///
/// # Panics
///
/// Panics on the conditions listed for [`try_two_sample_t_test`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::two_sample_t_test;
/// let control = [10.1, 9.8, 10.4, 10.0, 9.9];
/// let treated = [12.2, 11.8, 12.5, 12.1, 11.9];
/// let (t, p) = two_sample_t_test(&control, &treated);
/// assert!(t < 0.0 && p < 0.001);
/// ```
pub fn two_sample_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    try_two_sample_t_test(a, b).unwrap_or_else(|e| panic!("two_sample_t_test: {e}"))
}

/// Pearson's chi-squared goodness-of-fit test.
///
/// Returns `(statistic, p_value)` where
///
/// ```text
/// X² = Σ (Oᵢ - Eᵢ)² / Eᵢ
/// df = k - 1          (k = number of categories)
/// p  = P(χ²_df > X²)
/// ```
///
/// The `k - 1` degrees of freedom assume `expected` was specified up front. If
/// the expected counts were themselves fitted from the data, subtract one
/// further degree of freedom per estimated parameter by evaluating the tail
/// yourself, e.g. `tpt_math_stats::dist::ChiSquared::new(df).unwrap().sf(statistic)`.
///
/// The caller owns the (conventional) requirement that `expected` sums to the
/// same total as `observed`; it is not enforced.
///
/// # Errors
///
/// * [`StatsError::LengthMismatch`] — the two slices differ in length.
/// * [`StatsError::NotEnoughData`] — fewer than two categories, which would
///   leave zero degrees of freedom.
/// * [`StatsError::NotFinite`] — an expected count is `NaN` or infinite.
/// * [`StatsError::NonPositiveExpected`] — an expected count is `<= 0`.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_chi_squared_goodness_of_fit;
/// // A die rolled 60 times, expected 10 per face.
/// let observed = [10u64, 12, 9, 11, 8, 10];
/// let expected = [10.0; 6];
/// let (x2, p) = try_chi_squared_goodness_of_fit(&observed, &expected).unwrap();
/// assert!((x2 - 1.0).abs() < 1e-12);
/// assert!(p > 0.9); // an excellent fit
/// ```
pub fn try_chi_squared_goodness_of_fit(
    observed: &[u64],
    expected: &[f64],
) -> Result<(f64, f64), StatsError> {
    check_equal_len(observed.len(), expected.len())?;
    check_all_finite("expected", expected)?;

    if observed.len() < 2 {
        return Err(StatsError::NotEnoughData {
            sample: "observed",
            required: 2,
            found: observed.len(),
        });
    }

    if let Some(index) = expected.iter().position(|&e| e <= 0.0) {
        return Err(StatsError::NonPositiveExpected {
            index,
            value: expected[index],
        });
    }

    let statistic = compensated_sum(observed.iter().zip(expected.iter()).map(|(&o, &e)| {
        let residual = o as f64 - e;
        residual * residual / e
    }));

    let freedom = observed.len() as f64 - 1.0;
    let p_value = upper_tail_chi_squared_p_value(statistic, freedom)?;
    Ok((statistic, p_value))
}

/// Pearson's chi-squared goodness-of-fit test.
///
/// # Panics
///
/// Panics on the conditions listed for [`try_chi_squared_goodness_of_fit`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::chi_squared_goodness_of_fit;
/// // Clearly not uniform: X² = (10² + 0 + 10²) / 20 = 10 on 2 df.
/// let (x2, p) = chi_squared_goodness_of_fit(&[10, 20, 30], &[20.0, 20.0, 20.0]);
/// assert_eq!(x2, 10.0);
/// assert!((p - (-5.0f64).exp()).abs() < 1e-12);
/// ```
pub fn chi_squared_goodness_of_fit(observed: &[u64], expected: &[f64]) -> (f64, f64) {
    try_chi_squared_goodness_of_fit(observed, expected)
        .unwrap_or_else(|e| panic!("chi_squared_goodness_of_fit: {e}"))
}

/// Limiting `(statistic, p_value)` for a t-test whose standard error is zero.
fn degenerate_t_result(difference: f64) -> (f64, f64) {
    if difference == 0.0 {
        (0.0, 1.0)
    } else {
        (f64::INFINITY.copysign(difference), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact two-sided p-value of Student's t with 1 df (a Cauchy variate).
    fn exact_p_df1(t: f64) -> f64 {
        1.0 - 2.0 * t.abs().atan() / std::f64::consts::PI
    }

    /// Exact two-sided p-value of Student's t with 2 df.
    fn exact_p_df2(t: f64) -> f64 {
        1.0 - t.abs() / (t * t + 2.0).sqrt()
    }

    /// Exact two-sided p-value of Student's t with 4 df.
    ///
    /// `F(t) = 1/2 + (3/4)(s - s³/3)` with `s = t / √(t² + 4)`.
    fn exact_p_df4(t: f64) -> f64 {
        let s = t.abs() / (t * t + 4.0).sqrt();
        1.0 - 1.5 * s + 0.5 * s * s * s
    }

    #[test]
    fn one_sample_t_matches_closed_form_with_one_degree_of_freedom() {
        // x̄ = 2, s = √2, n = 2 => se = 1, t = 2 on 1 df.
        let (t, p) = one_sample_t_test(&[1.0, 3.0], 0.0);
        assert!((t - 2.0).abs() < 1e-14, "t = {t}");
        assert!((p - exact_p_df1(2.0)).abs() < 1e-12, "p = {p}");
        assert!((p - 0.2951672353008665).abs() < 1e-12, "p = {p}");
    }

    #[test]
    fn one_sample_t_matches_closed_form_with_two_degrees_of_freedom() {
        // x̄ = 2, s = 1, n = 3 => t = 2√3 on 2 df.
        let (t, p) = one_sample_t_test(&[1.0, 2.0, 3.0], 0.0);
        assert!((t - 2.0 * 3.0f64.sqrt()).abs() < 1e-13, "t = {t}");
        assert!((p - exact_p_df2(t)).abs() < 1e-12, "p = {p}");
        // Same value scipy's `ttest_1samp([1, 2, 3], 0)` reports.
        assert!((p - 0.07417990022744847).abs() < 1e-12, "p = {p}");
    }

    #[test]
    fn one_sample_t_is_zero_at_the_sample_mean() {
        let data = [3.0, 7.0, 11.0, 5.0];
        let (t, p) = one_sample_t_test(&data, crate::mean(&data));
        assert!(t.abs() < 1e-14, "t = {t}");
        assert!((p - 1.0).abs() < 1e-12, "p = {p}");
    }

    #[test]
    fn one_sample_t_is_antisymmetric_in_the_null_mean() {
        let data = [1.0, 2.0, 4.0, 8.0];
        let mean = crate::mean(&data);
        let (t_lo, p_lo) = one_sample_t_test(&data, mean - 1.5);
        let (t_hi, p_hi) = one_sample_t_test(&data, mean + 1.5);
        assert!((t_lo + t_hi).abs() < 1e-12);
        assert!((p_lo - p_hi).abs() < 1e-14);
        assert!(t_lo > 0.0 && t_hi < 0.0);
    }

    #[test]
    fn welch_matches_closed_form_when_the_degrees_of_freedom_are_integral() {
        // Equal sizes and equal variances make Welch's df exactly 2n - 2 = 4.
        let (t, p) = two_sample_t_test(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((t + 3.674234614174767).abs() < 1e-12, "t = {t}");
        assert!((p - exact_p_df4(t)).abs() < 1e-12, "p = {p}");
        assert!((p - 0.02131164112875677).abs() < 1e-12, "p = {p}");
    }

    #[test]
    fn welch_is_symmetric_under_swapping_the_groups() {
        let a = [12.9, 13.5, 12.8, 15.6, 17.2, 19.2, 12.6, 15.3, 14.4, 11.3];
        let b = [12.7, 13.6, 12.0, 15.2, 16.8, 20.0, 12.0, 15.9, 16.0, 11.1];

        let (t_ab, p_ab) = two_sample_t_test(&a, &b);
        let (t_ba, p_ba) = two_sample_t_test(&b, &a);

        assert!((t_ab + t_ba).abs() < 1e-14);
        assert!((p_ab - p_ba).abs() < 1e-14);
        assert!(p_ab > 0.05, "these paired-looking groups are not different");
    }

    #[test]
    fn welch_detects_a_clear_location_shift() {
        let control: Vec<f64> = (0..40).map(|i| 10.0 + (i % 5) as f64 * 0.1).collect();
        let treated: Vec<f64> = control.iter().map(|x| x + 2.0).collect();

        let (t, p) = two_sample_t_test(&control, &treated);
        assert!(t < -50.0, "t = {t}");
        assert!(p < 1e-12, "p = {p}");
    }

    #[test]
    fn identical_samples_give_a_p_value_of_one() {
        let data = [4.0, 4.5, 5.5, 6.0];
        let (t, p) = two_sample_t_test(&data, &data);
        assert_eq!(t, 0.0);
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_variance_samples_use_the_limiting_convention() {
        // Every observation identical and equal to the null mean.
        assert_eq!(one_sample_t_test(&[2.0, 2.0, 2.0], 2.0), (0.0, 1.0));
        // Every observation identical but shifted away from it.
        assert_eq!(
            one_sample_t_test(&[2.0, 2.0, 2.0], 1.0),
            (f64::INFINITY, 0.0)
        );
        assert_eq!(
            one_sample_t_test(&[2.0, 2.0, 2.0], 3.0),
            (f64::NEG_INFINITY, 0.0)
        );
        assert_eq!(
            two_sample_t_test(&[1.0, 1.0], &[5.0, 5.0]),
            (f64::NEG_INFINITY, 0.0)
        );
        assert_eq!(two_sample_t_test(&[1.0, 1.0], &[1.0, 1.0]), (0.0, 1.0));
    }

    #[test]
    fn chi_squared_matches_the_two_degree_of_freedom_closed_form() {
        // With df = 2 the survival function is exactly exp(-x / 2).
        let (x2, p) = chi_squared_goodness_of_fit(&[10, 20, 30], &[20.0, 20.0, 20.0]);
        assert_eq!(x2, 10.0);
        assert!((p - (-5.0f64).exp()).abs() < 1e-14, "p = {p}");
        assert!((p - 0.006737946999085467).abs() < 1e-14, "p = {p}");
    }

    #[test]
    fn chi_squared_matches_the_one_degree_of_freedom_closed_form() {
        // With df = 1 the survival function is erfc(√(x / 2)).
        let (x2, p) = chi_squared_goodness_of_fit(&[30, 20], &[25.0, 25.0]);
        assert_eq!(x2, 2.0);
        // The in-house `erfc` is exact to double precision here.
        let expected = crate::special::erfc((x2 / 2.0).sqrt());
        assert!((p - expected).abs() < 1e-14, "p = {p}, erfc = {expected}");
        // erfc(1) to full double precision.
        assert!((p - 0.15729920705028513).abs() < 1e-14, "p = {p}");
    }

    #[test]
    fn chi_squared_is_zero_for_a_perfect_fit() {
        let (x2, p) = chi_squared_goodness_of_fit(&[25, 25, 25, 25], &[25.0; 4]);
        assert_eq!(x2, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn chi_squared_rejects_a_badly_loaded_die() {
        let observed = [5u64, 8, 9, 8, 10, 20];
        let expected = [10.0; 6];
        let (x2, p) = chi_squared_goodness_of_fit(&observed, &expected);
        // (25 + 4 + 1 + 4 + 0 + 100) / 10 = 13.4 on 5 df.
        assert!((x2 - 13.4).abs() < 1e-12, "x2 = {x2}");
        assert!(p < 0.05, "p = {p}");
    }

    #[test]
    fn chi_squared_accepts_non_integral_expected_counts() {
        // Expected counts need not be whole numbers.
        let (x2, p) = chi_squared_goodness_of_fit(&[7, 13], &[8.5, 11.5]);
        let hand = 1.5f64 * 1.5 / 8.5 + 1.5 * 1.5 / 11.5;
        assert!((x2 - hand).abs() < 1e-12, "x2 = {x2}");
        assert!(p > 0.4, "p = {p}");
    }

    #[test]
    fn tail_helpers_stay_accurate_far_out_in_the_tail() {
        // 2 * (1 - cdf) would collapse to 0 here; 2 * sf does not.
        let p = two_sided_t_p_value(60.0, 30.0).unwrap();
        assert!(p > 0.0 && p < 1e-30, "p = {p}");

        let p = upper_tail_chi_squared_p_value(400.0, 3.0).unwrap();
        assert!(p > 0.0 && p < 1e-80, "p = {p}");
    }

    #[test]
    fn p_values_are_confined_to_the_unit_interval() {
        for &t in &[-1e6, -3.5, -0.25, 0.0, 0.25, 3.5, 1e6] {
            for &df in &[1.0, 2.5, 7.0, 1e4] {
                let p = two_sided_t_p_value(t, df).unwrap();
                assert!((0.0..=1.0).contains(&p), "t = {t}, df = {df}, p = {p}");
            }
        }
        assert_eq!(two_sided_t_p_value(0.0, 5.0).unwrap(), 1.0);
    }

    #[test]
    fn invalid_degrees_of_freedom_are_reported() {
        assert!(matches!(
            two_sided_t_p_value(1.0, 0.0),
            Err(StatsError::InvalidDegreesOfFreedom { .. })
        ));
        assert!(matches!(
            upper_tail_chi_squared_p_value(1.0, -1.0),
            Err(StatsError::InvalidDegreesOfFreedom { .. })
        ));
    }

    #[test]
    fn checked_variants_report_bad_input() {
        assert!(matches!(
            try_one_sample_t_test(&[1.0], 0.0),
            Err(StatsError::NotEnoughData { required: 2, .. })
        ));
        assert!(matches!(
            try_one_sample_t_test(&[1.0, f64::NAN], 0.0),
            Err(StatsError::NotFinite { index: 1, .. })
        ));
        assert!(matches!(
            try_one_sample_t_test(&[1.0, 2.0], f64::NAN),
            Err(StatsError::ParameterNotFinite {
                parameter: "mu0",
                ..
            })
        ));
        assert!(matches!(
            try_two_sample_t_test(&[1.0, 2.0], &[3.0]),
            Err(StatsError::NotEnoughData { sample: "b", .. })
        ));
        assert!(matches!(
            try_chi_squared_goodness_of_fit(&[1, 2, 3], &[1.0, 2.0]),
            Err(StatsError::LengthMismatch {
                first: 3,
                second: 2
            })
        ));
        assert!(matches!(
            try_chi_squared_goodness_of_fit(&[4], &[4.0]),
            Err(StatsError::NotEnoughData { required: 2, .. })
        ));
        assert!(matches!(
            try_chi_squared_goodness_of_fit(&[4, 5], &[4.0, 0.0]),
            Err(StatsError::NonPositiveExpected { index: 1, .. })
        ));
        assert!(matches!(
            try_chi_squared_goodness_of_fit(&[4, 5], &[4.0, f64::INFINITY]),
            Err(StatsError::NotFinite {
                sample: "expected",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    #[should_panic(expected = "two_sample_t_test")]
    fn two_sample_t_test_panics_on_a_degenerate_group() {
        let _ = two_sample_t_test(&[1.0, 2.0], &[]);
    }
}
