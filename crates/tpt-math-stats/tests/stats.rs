//! End-to-end exercises of the public `tpt-math-stats` surface.
//!
//! These run against the crate exactly as a downstream user sees it: only
//! re-exported items, no `pub(crate)` shortcuts.

use tpt_math_stats::{
    chi_squared_goodness_of_fit, linear_regression, max, mean, median, min, one_sample_t_test,
    pearson_correlation, std_dev, try_chi_squared_goodness_of_fit, try_linear_regression, try_mean,
    two_sample_t_test, variance, Distribution, Rng, SplitMix64, Standard, StatsError, StudentsT,
};

/// Anscombe's quartet, dataset I — the canonical regression sanity check.
const ANSCOMBE_X: [f64; 11] = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
const ANSCOMBE_Y: [f64; 11] = [
    8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68,
];

#[test]
fn anscombe_dataset_one_reproduces_the_published_fit() {
    // Published values: y = 3.00009 + 0.50009x, r = 0.81642.
    let (slope, intercept) = linear_regression(&ANSCOMBE_X, &ANSCOMBE_Y);
    assert!((slope - 0.50009).abs() < 1e-4, "slope = {slope}");
    assert!(
        (intercept - 3.00009).abs() < 1e-4,
        "intercept = {intercept}"
    );

    let r = pearson_correlation(&ANSCOMBE_X, &ANSCOMBE_Y);
    assert!((r - 0.81642).abs() < 1e-4, "r = {r}");

    // The quartet is built around x̄ = 9, s²ₓ = 11, ȳ ≈ 7.5, s²_y ≈ 4.127.
    assert_eq!(mean(&ANSCOMBE_X), 9.0);
    assert!((variance(&ANSCOMBE_X) - 11.0).abs() < 1e-12);
    assert!((mean(&ANSCOMBE_Y) - 7.500909090909091).abs() < 1e-12);
    assert!((variance(&ANSCOMBE_Y) - 4.127269090909091).abs() < 1e-12);
    assert_eq!(min(&ANSCOMBE_X), 4.0);
    assert_eq!(max(&ANSCOMBE_X), 14.0);
    assert_eq!(median(&ANSCOMBE_X), 9.0);
}

#[test]
fn a_descriptive_summary_hangs_together() {
    let data = [12.0, 15.0, 11.0, 19.0, 14.0, 16.0, 13.0, 18.0];

    assert_eq!(mean(&data), 14.75);
    assert_eq!(median(&data), 14.5);
    assert_eq!(min(&data), 11.0);
    assert_eq!(max(&data), 19.0);
    assert!((std_dev(&data) * std_dev(&data) - variance(&data)).abs() < 1e-12);
    assert!(min(&data) <= median(&data) && median(&data) <= max(&data));
}

#[test]
fn welch_agrees_with_the_pooled_test_for_balanced_equal_variance_groups() {
    // With equal n and equal variances Welch's t is algebraically identical to
    // Student's pooled t, and its df collapses to 2n - 2.
    let a = [3.0, 5.0, 7.0, 9.0, 11.0];
    let b = [6.0, 8.0, 10.0, 12.0, 14.0];

    let (t, p) = two_sample_t_test(&a, &b);

    let n = a.len() as f64;
    let pooled = 0.5 * (variance(&a) + variance(&b));
    let expected_t = (mean(&a) - mean(&b)) / (2.0 * pooled / n).sqrt();
    assert!((t - expected_t).abs() < 1e-12, "t = {t}");

    // Cross-check the p-value against the in-house Student's t distribution,
    // on 2n - 2 = 8 df.
    use tpt_math_stats::{ContinuousCDF, StudentsT};
    let reference = 2.0 * StudentsT::new(0.0, 1.0, 2.0 * n - 2.0).unwrap().sf(t.abs());
    assert!((p - reference).abs() < 1e-12, "p = {p}");
    assert!(p > 0.05, "a 3-unit shift is not detectable at n = 5");
}

#[test]
fn a_one_sample_test_grows_more_significant_as_the_null_moves_away() {
    let data = [20.4, 21.1, 19.8, 20.9, 20.2, 21.4, 20.7, 20.1];

    let mut previous = 1.0;
    for offset in [0.0, 0.25, 0.5, 1.0, 2.0] {
        let (_t, p) = one_sample_t_test(&data, mean(&data) + offset);
        assert!(p <= previous, "p rose from {previous} to {p}");
        previous = p;
    }
    assert!(previous < 1e-4, "a 2-unit shift should be overwhelming");
}

#[test]
fn chi_squared_accepts_a_uniform_prng_and_rejects_a_skewed_one() {
    // Bin 20 000 uniforms into deciles; a sound PRNG should pass comfortably.
    let mut rng = SplitMix64::seed_from_u64(0xC0FFEE);
    let mut observed = [0u64; 10];
    for _ in 0..20_000 {
        let u: f64 = Standard.sample(&mut rng);
        let bin = ((u * 10.0) as usize).min(9);
        observed[bin] += 1;
    }
    let expected = [2_000.0; 10];
    let (x2, p) = chi_squared_goodness_of_fit(&observed, &expected);
    assert!(
        p > 0.01,
        "SplitMix64 failed a decile test: X² = {x2}, p = {p}"
    );

    // Now square the draws, which piles mass into the low bins.
    let mut rng = SplitMix64::seed_from_u64(0xC0FFEE);
    let mut skewed = [0u64; 10];
    for _ in 0..20_000 {
        let u = rng.next_f64() * rng.next_f64();
        let bin = ((u * 10.0) as usize).min(9);
        skewed[bin] += 1;
    }
    let (x2, p) = chi_squared_goodness_of_fit(&skewed, &expected);
    assert!(x2 > 100.0 && p < 1e-10, "X² = {x2}, p = {p}");
}

#[test]
fn simulated_group_difference_is_detected() {
    let mut rng = SplitMix64::seed_from_u64(31_337);
    let mut draw = |shift: f64| -> Vec<f64> {
        (0..60)
            .map(|_| {
                // Irwin–Hall(12) - 6 approximates a standard normal.
                let z: f64 = (0..12).map(|_| rng.next_f64()).sum::<f64>() - 6.0;
                shift + z
            })
            .collect()
    };

    let control = draw(0.0);
    let treated = draw(1.0);

    let (t, p) = two_sample_t_test(&control, &treated);
    assert!(t < 0.0, "t = {t}");
    assert!(
        p < 0.01,
        "a one-sigma shift at n = 60 should be clear, p = {p}"
    );

    // The same data seen as a regression on a 0/1 group indicator: the slope
    // is the difference in means.
    let x: Vec<f64> = std::iter::repeat_n(0.0, control.len())
        .chain(std::iter::repeat_n(1.0, treated.len()))
        .collect();
    let y: Vec<f64> = control.iter().chain(treated.iter()).copied().collect();
    let (slope, intercept) = linear_regression(&x, &y);

    assert!((slope - (mean(&treated) - mean(&control))).abs() < 1e-12);
    assert!((intercept - mean(&control)).abs() < 1e-12);
}

#[test]
fn errors_surface_with_useful_messages() {
    let err = try_mean(&[]).unwrap_err();
    assert!(matches!(err, StatsError::NotEnoughData { found: 0, .. }));
    assert!(err.to_string().contains("at least 1"), "{err}");

    let err = try_linear_regression(&[1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap_err();
    assert_eq!(
        err,
        StatsError::LengthMismatch {
            first: 3,
            second: 2
        }
    );

    let err = try_chi_squared_goodness_of_fit(&[1, 2], &[1.0, -1.0]).unwrap_err();
    assert!(matches!(
        err,
        StatsError::NonPositiveExpected { index: 1, .. }
    ));
    assert!(err.to_string().contains("expected[1]"), "{err}");

    // `StatsError` is a normal `std::error::Error`, so `?` works upstream.
    fn fallible() -> Result<f64, Box<dyn std::error::Error>> {
        Ok(try_mean(&[1.0, 2.0, 3.0])?)
    }
    assert_eq!(fallible().unwrap(), 2.0);
}

#[test]
fn distributions_are_reachable_from_the_public_api() {
    use tpt_math_stats::{ChiSquared, ContinuousCDF};

    // Our chi-squared p-value equals the upper regularized incomplete gamma,
    // which is exactly what `ChiSquared::sf` computes in-house.
    let (x2, p) = chi_squared_goodness_of_fit(&[18, 22, 30, 30], &[25.0; 4]);
    let reference = ChiSquared::new(3.0).unwrap().sf(x2);
    assert!((p - reference).abs() < 1e-15);
    assert!(ChiSquared::new(3.0).is_ok());
    assert!(StudentsT::new(0.0, 1.0, 3.0).is_ok());
}
