#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::all)]

//! Descriptive statistics, hypothesis tests, and least-squares regression.
//!
//! This crate is the workspace's home for "classical" statistics. It wraps
//! [`statrs`] — whose distributions, special functions, and `Statistics`
//! traits are battle-tested — behind a small, slice-oriented API that speaks
//! plain `&[f64]` and returns plain tuples, so callers never have to build a
//! distribution object just to get a p-value.
//!
//! # What's here
//!
//! | Area | Functions |
//! |------|-----------|
//! | Descriptive | [`mean`], [`variance`], [`std_dev`], [`min`], [`max`], [`median`] |
//! | Hypothesis tests | [`one_sample_t_test`], [`two_sample_t_test`], [`chi_squared_goodness_of_fit`] |
//! | Regression | [`pearson_correlation`], [`linear_regression`] |
//!
//! # Conventions
//!
//! * Every function has a panicking short-named form and a checked `try_*`
//!   twin returning [`StatsError`]; the former is exactly the latter unwrapped.
//! * Inputs must be finite. `NaN`s and infinities are rejected rather than
//!   silently poisoning a result.
//! * Sample variance is Bessel-corrected (`n - 1`).
//! * Tests return `(statistic, p_value)`. t-test p-values are two-sided;
//!   chi-squared p-values are upper-tail. Both come from `statrs` survival
//!   functions, so tiny p-values keep their precision.
//! * Sums use compensated (Neumaier) accumulation and variances use the
//!   corrected two-pass formula, which keeps results accurate for long or
//!   badly offset samples.
//!
//! # Examples
//!
//! Is this sample drawn from a population with mean 100?
//!
//! ```
//! use tpt_math_stats::{mean, one_sample_t_test, std_dev};
//!
//! let iq = [105.0, 98.0, 110.0, 102.0, 99.0, 107.0, 101.0, 103.0];
//! let (t, p) = one_sample_t_test(&iq, 100.0);
//!
//! assert!(mean(&iq) > 100.0 && std_dev(&iq) > 0.0);
//! assert!(t > 0.0);
//! assert!(p < 0.10, "mildly suggestive, p = {p}");
//! ```
//!
//! Compare two groups, then fit a line:
//!
//! ```
//! use tpt_math_stats::{linear_regression, pearson_correlation, two_sample_t_test};
//!
//! let control = [4.1, 3.9, 4.4, 4.0, 3.8];
//! let treated = [5.2, 5.5, 5.1, 5.4, 5.3];
//! let (_t, p) = two_sample_t_test(&control, &treated);
//! assert!(p < 0.001);
//!
//! let dose = [0.0, 1.0, 2.0, 3.0, 4.0];
//! let response = [1.0, 3.0, 5.0, 7.0, 9.0];
//! assert_eq!(linear_regression(&dose, &response), (2.0, 1.0));
//! assert_eq!(pearson_correlation(&dose, &response), 1.0);
//! ```
//!
//! Goodness of fit for a categorical model:
//!
//! ```
//! use tpt_math_stats::chi_squared_goodness_of_fit;
//!
//! // 100 draws that should have been split 25/25/25/25.
//! let observed = [22u64, 27, 24, 27];
//! let expected = [25.0; 4];
//! let (x2, p) = chi_squared_goodness_of_fit(&observed, &expected);
//! assert!(x2 < 1.0 && p > 0.5);
//! ```
//!
//! # Reaching through to `statrs`
//!
//! Anything this crate does not wrap is one path away: `statrs` is re-exported
//! verbatim, so distributions, special functions, and its own `Statistics`
//! traits stay available without a second dependency declaration.
//!
//! ```
//! use tpt_math_stats::statrs::distribution::{ContinuousCDF, FisherSnedecor};
//!
//! let f = FisherSnedecor::new(3.0, 16.0).unwrap();
//! let p = f.sf(5.29);
//! assert!(p < 0.05);
//! ```
//!
//! The randomness traits from `tpt-math-prob-core` are re-exported too, which
//! makes simulation studies against these tests a two-liner:
//!
//! ```
//! use tpt_math_stats::{mean, Distribution, SplitMix64, Standard};
//!
//! let mut rng = SplitMix64::seed_from_u64(7);
//! let sample: Vec<f64> = (0..10_000).map(|_| Standard.sample(&mut rng)).collect();
//! assert!((mean(&sample) - 0.5).abs() < 0.02);
//! ```

mod descriptive;
mod error;
mod hypothesis;
mod regression;

pub use descriptive::{
    max, mean, median, min, std_dev, try_max, try_mean, try_median, try_min, try_std_dev,
    try_variance, variance,
};
pub use error::StatsError;
pub use hypothesis::{
    chi_squared_goodness_of_fit, one_sample_t_test, try_chi_squared_goodness_of_fit,
    try_one_sample_t_test, try_two_sample_t_test, two_sample_t_test,
};
pub use regression::{
    linear_regression, pearson_correlation, try_linear_regression, try_pearson_correlation,
};

/// The wrapped statistics library, re-exported in full.
///
/// Use it for anything outside this crate's surface — other distributions,
/// `statrs::function::{beta, erf, gamma}`, or the `statrs::statistics` traits.
pub use statrs;

/// The workspace's randomness traits, re-exported for convenience.
pub use tpt_math_prob_core as prob_core;

pub use tpt_math_prob_core::{Distribution, Rng, Sampler, SplitMix64, Standard};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statrs_reexport_is_usable() {
        use statrs::distribution::{Continuous, ContinuousCDF, Normal};
        use statrs::statistics::Distribution as _;

        let normal = Normal::new(0.0, 1.0).unwrap();
        assert!((normal.cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((normal.pdf(0.0) - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-15);
        assert_eq!(normal.mean(), Some(0.0));
        assert_eq!(normal.variance(), Some(1.0));

        // Special functions come along for the ride.
        assert!((statrs::function::gamma::gamma(5.0) - 24.0).abs() < 1e-10);
        assert!((statrs::function::erf::erf(0.0)).abs() < 1e-15);
        assert!((statrs::function::beta::beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn prob_core_reexport_drives_a_simulation() {
        // A deterministic uniform sample should look uniform to our helpers.
        let mut rng = SplitMix64::seed_from_u64(1_234_567);
        let sample: Vec<f64> = (0..20_000).map(|_| Standard.sample(&mut rng)).collect();

        assert!((mean(&sample) - 0.5).abs() < 0.01);
        assert!((variance(&sample) - 1.0 / 12.0).abs() < 0.005);
        assert!(min(&sample) < 0.001 && max(&sample) > 0.999);
        assert!((median(&sample) - 0.5).abs() < 0.02);

        // The mean of a uniform sample is not significantly different from 0.5.
        let (_t, p) = one_sample_t_test(&sample, 0.5);
        assert!(p > 0.01, "p = {p}");
    }

    #[test]
    fn t_test_false_positive_rate_is_calibrated() {
        // Under H0 the p-value is uniform, so ~5% of runs fall below 0.05.
        let mut rng = SplitMix64::seed_from_u64(2026);
        let mut rejections = 0usize;
        let trials = 400;

        for _ in 0..trials {
            // Sum of 12 uniforms - 6 is a decent standard normal.
            let sample: Vec<f64> = (0..30)
                .map(|_| (0..12).map(|_| rng.next_f64()).sum::<f64>() - 6.0)
                .collect();
            let (_t, p) = one_sample_t_test(&sample, 0.0);
            if p < 0.05 {
                rejections += 1;
            }
        }

        // Binomial(400, 0.05) has sd ~4.4; allow a wide, flake-proof band.
        assert!(rejections < 40, "rejected {rejections} of {trials}");
    }
}
