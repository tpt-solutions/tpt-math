//! Densities, probability mass functions, CDFs and quantiles.
//!
//! The sampling side of this crate ([`rand_distr`] wrappers) says nothing about
//! the shape of a distribution. These free functions supply it, built on the
//! special functions in [`tpt_math_prob_core::special`].
//!
//! Each function takes the distribution's parameters first, then the point.
//! Invalid parameters (for example a non-positive scale) return `NaN`. Densities
//! are `0` outside the support, and a `NaN` point returns `NaN`.
//!
//! # Examples
//!
//! ```
//! use tpt_math_prob_dist::density::{normal_cdf, normal_quantile};
//!
//! assert!((normal_cdf(0.0, 1.0, 1.96) - 0.975_002_104_851_779_5).abs() < 1e-12);
//! assert!((normal_quantile(0.0, 1.0, 0.975) - 1.959_963_984_540_054).abs() < 1e-9);
//! ```

use tpt_math_prob_core::special::{
    gamma_p, gamma_q, ln_beta, ln_gamma, regularized_incomplete_beta, standard_normal_cdf,
    standard_normal_pdf, standard_normal_quantile,
};

/// `true` for a strictly positive parameter. NaN is rejected.
fn is_positive(x: f64) -> bool {
    x > 0.0
}

/// `true` for a non-negative parameter. NaN is rejected.
fn is_nonnegative(x: f64) -> bool {
    x >= 0.0
}

/// `true` when `low < high`. NaN is rejected.
fn is_ordered(low: f64, high: f64) -> bool {
    low < high
}

/// Normal distribution with the given mean and standard deviation.
pub fn normal_pdf(mean: f64, sd: f64, x: f64) -> f64 {
    if !is_positive(sd) || !mean.is_finite() {
        return f64::NAN;
    }
    standard_normal_pdf((x - mean) / sd) / sd
}

/// Cumulative distribution function of the normal distribution.
pub fn normal_cdf(mean: f64, sd: f64, x: f64) -> f64 {
    if !is_positive(sd) || !mean.is_finite() {
        return f64::NAN;
    }
    standard_normal_cdf((x - mean) / sd)
}

/// Quantile (inverse CDF) of the normal distribution.
pub fn normal_quantile(mean: f64, sd: f64, p: f64) -> f64 {
    if !is_positive(sd) || !mean.is_finite() {
        return f64::NAN;
    }
    mean + sd * standard_normal_quantile(p)
}

/// Exponential distribution with the given rate `λ`.
pub fn exponential_pdf(rate: f64, x: f64) -> f64 {
    if !is_positive(rate) {
        return f64::NAN;
    }
    if x < 0.0 {
        return 0.0;
    }
    rate * (-rate * x).exp()
}

/// Cumulative distribution function of the exponential distribution.
pub fn exponential_cdf(rate: f64, x: f64) -> f64 {
    if !is_positive(rate) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    -(-rate * x).exp_m1()
}

/// Quantile of the exponential distribution, `-ln(1 - p) / λ`.
pub fn exponential_quantile(rate: f64, p: f64) -> f64 {
    if !is_positive(rate) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    -(-p).ln_1p() / rate
}

/// Gamma distribution with the given shape `k` and scale `θ`.
pub fn gamma_pdf(shape: f64, scale: f64, x: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) {
        return f64::NAN;
    }
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        // The density at the origin is finite only for shape >= 1.
        return if shape < 1.0 {
            f64::INFINITY
        } else if shape == 1.0 {
            1.0 / scale
        } else {
            0.0
        };
    }
    ((shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - ln_gamma(shape)).exp()
}

/// Cumulative distribution function of the gamma distribution.
pub fn gamma_cdf(shape: f64, scale: f64, x: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) {
        return f64::NAN;
    }
    gamma_p(shape, x / scale)
}

/// Upper-tail probability `P(X > x)` of the gamma distribution.
pub fn gamma_sf(shape: f64, scale: f64, x: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) {
        return f64::NAN;
    }
    gamma_q(shape, x / scale)
}

/// Beta distribution on `[0, 1]` with shape parameters `a` and `b`.
pub fn beta_pdf(a: f64, b: f64, x: f64) -> f64 {
    if !is_positive(a) || !is_positive(b) {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return if x.is_nan() { f64::NAN } else { 0.0 };
    }
    // The density is unbounded at an endpoint whose exponent is below one.
    // Report zero at the endpoints for the regular cases, which is the usual
    // convention for a density evaluated at a single point.
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - ln_beta(a, b)).exp()
}

/// Cumulative distribution function of the beta distribution.
pub fn beta_cdf(a: f64, b: f64, x: f64) -> f64 {
    regularized_incomplete_beta(a, b, x)
}

/// Uniform distribution on `[low, high]`.
pub fn uniform_pdf(low: f64, high: f64, x: f64) -> f64 {
    if !is_ordered(low, high) {
        return f64::NAN;
    }
    if (low..=high).contains(&x) {
        1.0 / (high - low)
    } else if x.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// Cumulative distribution function of the uniform distribution.
pub fn uniform_cdf(low: f64, high: f64, x: f64) -> f64 {
    if !is_ordered(low, high) {
        return f64::NAN;
    }
    ((x - low) / (high - low)).clamp(0.0, 1.0)
}

/// Quantile of the uniform distribution.
pub fn uniform_quantile(low: f64, high: f64, p: f64) -> f64 {
    if !is_ordered(low, high) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    low + p * (high - low)
}

/// Poisson probability mass at `k` for rate `λ`.
pub fn poisson_pmf(lambda: f64, k: u64) -> f64 {
    if !is_nonnegative(lambda) {
        return f64::NAN;
    }
    if lambda == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    (k as f64 * lambda.ln() - lambda - ln_gamma(k as f64 + 1.0)).exp()
}

/// `P(X ≤ k)` for a Poisson variable with rate `λ`.
pub fn poisson_cdf(lambda: f64, k: u64) -> f64 {
    if !is_nonnegative(lambda) {
        return f64::NAN;
    }
    // P(X ≤ k) = Q(k + 1, λ), the regularized upper incomplete gamma.
    if lambda == 0.0 {
        return 1.0;
    }
    gamma_q(k as f64 + 1.0, lambda)
}

/// Density of Student's t distribution with `nu` degrees of freedom.
pub fn student_t_pdf(nu: f64, x: f64) -> f64 {
    if !is_positive(nu) {
        return f64::NAN;
    }
    let log_norm =
        ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0) - 0.5 * (nu * core::f64::consts::PI).ln();
    (log_norm - (nu + 1.0) / 2.0 * (x * x / nu).ln_1p()).exp()
}

/// Cumulative distribution function of Student's t distribution.
///
/// `P(T ≤ x) = 1 − ½·I_{ν/(ν+x²)}(ν/2, ½)` for `x ≥ 0`, by symmetry otherwise.
/// The incomplete beta keeps the tail accurate.
pub fn student_t_cdf(nu: f64, x: f64) -> f64 {
    if !is_positive(nu) || x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let tail = 0.5 * regularized_incomplete_beta(nu / 2.0, 0.5, nu / (nu + x * x));
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Density of the Cauchy distribution with location `x0` and scale `s`.
pub fn cauchy_pdf(x0: f64, s: f64, x: f64) -> f64 {
    if !is_positive(s) {
        return f64::NAN;
    }
    let z = (x - x0) / s;
    1.0 / (core::f64::consts::PI * s * (1.0 + z * z))
}

/// Cumulative distribution function of the Cauchy distribution,
/// `½ + arctan((x − x₀)/s)/π`.
pub fn cauchy_cdf(x0: f64, s: f64, x: f64) -> f64 {
    if !is_positive(s) {
        return f64::NAN;
    }
    0.5 + ((x - x0) / s).atan() / core::f64::consts::PI
}

/// Density of the chi-squared distribution with `k` degrees of freedom.
///
/// This is the gamma distribution with shape `k/2` and scale `2`.
pub fn chi_squared_pdf(k: f64, x: f64) -> f64 {
    gamma_pdf(k / 2.0, 2.0, x)
}

/// Cumulative distribution function of the chi-squared distribution.
pub fn chi_squared_cdf(k: f64, x: f64) -> f64 {
    gamma_cdf(k / 2.0, 2.0, x)
}

/// Mean and variance of Student's t, `(0, ν/(ν−2))`.
///
/// The mean is `NaN` for `ν ≤ 1`, where it does not exist. The variance is
/// infinite for `1 < ν ≤ 2`, and `NaN` for `ν ≤ 1`.
pub fn student_t_moments(nu: f64) -> (f64, f64) {
    if !is_positive(nu) {
        return (f64::NAN, f64::NAN);
    }
    if nu <= 1.0 {
        return (f64::NAN, f64::NAN);
    }
    let variance = if nu > 2.0 {
        nu / (nu - 2.0)
    } else {
        f64::INFINITY
    };
    (0.0, variance)
}

/// The Cauchy distribution has no finite mean or variance, so both are `NaN`.
pub fn cauchy_moments() -> (f64, f64) {
    (f64::NAN, f64::NAN)
}

/// Mean and variance of the chi-squared distribution, `(k, 2k)`.
pub fn chi_squared_moments(k: f64) -> (f64, f64) {
    if !is_positive(k) {
        return (f64::NAN, f64::NAN);
    }
    (k, 2.0 * k)
}

/// Mean and variance of the normal distribution, `(μ, σ²)`.
///
/// Returns `(NaN, NaN)` for invalid parameters.
pub fn normal_moments(mean: f64, sd: f64) -> (f64, f64) {
    if !is_positive(sd) || !mean.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    (mean, sd * sd)
}

/// Mean and variance of the exponential distribution, `(1/λ, 1/λ²)`.
pub fn exponential_moments(rate: f64) -> (f64, f64) {
    if !is_positive(rate) {
        return (f64::NAN, f64::NAN);
    }
    (1.0 / rate, 1.0 / (rate * rate))
}

/// Mean and variance of the gamma distribution, `(kθ, kθ²)`.
pub fn gamma_moments(shape: f64, scale: f64) -> (f64, f64) {
    if !is_positive(shape) || !is_positive(scale) {
        return (f64::NAN, f64::NAN);
    }
    (shape * scale, shape * scale * scale)
}

/// Mean and variance of the beta distribution,
/// `(a/(a+b), ab/((a+b)²(a+b+1)))`.
pub fn beta_moments(a: f64, b: f64) -> (f64, f64) {
    if !is_positive(a) || !is_positive(b) {
        return (f64::NAN, f64::NAN);
    }
    let s = a + b;
    (a / s, a * b / (s * s * (s + 1.0)))
}

/// Mean and variance of the uniform distribution on `[low, high]`.
pub fn uniform_moments(low: f64, high: f64) -> (f64, f64) {
    if !is_ordered(low, high) {
        return (f64::NAN, f64::NAN);
    }
    let width = high - low;
    (0.5 * (low + high), width * width / 12.0)
}

/// Mean and variance of the Poisson distribution, both `λ`.
pub fn poisson_moments(lambda: f64) -> (f64, f64) {
    if !is_nonnegative(lambda) {
        return (f64::NAN, f64::NAN);
    }
    (lambda, lambda)
}

#[cfg(test)]
mod family_tests {
    use super::*;

    #[test]
    fn student_t_with_one_df_is_cauchy() {
        // t(1) is the standard Cauchy: CDF(x) = ½ + atan(x)/π.
        for &x in &[-3.0_f64, -0.5, 0.0, 0.8, 4.0] {
            let want = 0.5 + x.atan() / core::f64::consts::PI;
            assert!((student_t_cdf(1.0, x) - want).abs() < 1e-12, "x = {x}");
        }
    }

    #[test]
    fn student_t_known_values() {
        // t(2) density at 0 is 1/(2√2).
        assert!((student_t_pdf(2.0, 0.0) - 1.0 / (2.0 * 2f64.sqrt())).abs() < 1e-13);
        assert!((student_t_cdf(4.0, 0.0) - 0.5).abs() < 1e-14);
        // Two-sided 5% critical value with 4 df is 2.776445105.
        assert!((student_t_cdf(4.0, 2.776_445_105) - 0.975).abs() < 1e-7);
        assert!((student_t_cdf(4.0, -2.776_445_105) - 0.025).abs() < 1e-7);
    }

    #[test]
    fn student_t_pdf_integrates_to_its_cdf() {
        // Midpoint-rule integral of the density over [-2, 1.5] matches the CDF
        // difference.
        let (a, b, n) = (-2.0, 1.5, 4000);
        let h = (b - a) / n as f64;
        let integral: f64 = (0..n)
            .map(|i| student_t_pdf(5.0, a + (i as f64 + 0.5) * h) * h)
            .sum();
        let diff = student_t_cdf(5.0, b) - student_t_cdf(5.0, a);
        assert!((integral - diff).abs() < 1e-6, "{integral} vs {diff}");
    }

    #[test]
    fn cauchy_closed_forms() {
        assert!((cauchy_pdf(0.0, 1.0, 0.0) - 1.0 / core::f64::consts::PI).abs() < 1e-15);
        assert!((cauchy_cdf(0.0, 1.0, 1.0) - 0.75).abs() < 1e-15);
        assert!((cauchy_cdf(5.0, 2.0, 5.0) - 0.5).abs() < 1e-15);
        assert!(cauchy_moments().0.is_nan());
    }

    #[test]
    fn chi_squared_two_df_is_exponential() {
        // χ²(2) is exponential with rate 1/2: CDF(x) = 1 − e^{−x/2}.
        for &x in &[0.5, 2.0, 7.0] {
            let want = 1.0 - (-x / 2.0f64).exp();
            assert!((chi_squared_cdf(2.0, x) - want).abs() < 1e-13, "x = {x}");
        }
        assert_eq!(chi_squared_moments(3.0), (3.0, 6.0));
    }

    #[test]
    fn student_t_moments_respect_existence() {
        assert!(student_t_moments(0.5).0.is_nan());
        assert!(student_t_moments(1.5).1.is_infinite());
        assert_eq!(student_t_moments(5.0), (0.0, 5.0 / 3.0));
    }
}

#[cfg(test)]
mod moment_tests {
    use super::*;

    #[test]
    fn moments_match_closed_forms() {
        assert_eq!(normal_moments(1.0, 2.0), (1.0, 4.0));
        assert_eq!(exponential_moments(2.0), (0.5, 0.25));
        assert_eq!(gamma_moments(3.0, 2.0), (6.0, 12.0));
        // Beta(2, 2): mean 1/2, variance 4/(16·5) = 1/20.
        let (m, v) = beta_moments(2.0, 2.0);
        assert!((m - 0.5).abs() < 1e-15 && (v - 0.05).abs() < 1e-15);
        assert_eq!(uniform_moments(0.0, 2.0), (1.0, 1.0 / 3.0));
        assert_eq!(poisson_moments(4.0), (4.0, 4.0));
    }

    #[test]
    fn invalid_parameters_give_nan_moments() {
        let (m, v) = gamma_moments(-1.0, 1.0);
        assert!(m.is_nan() && v.is_nan());
        assert!(uniform_moments(1.0, 1.0).0.is_nan());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn normal_matches_reference_values() {
        assert!(close(
            normal_pdf(0.0, 1.0, 0.0),
            0.398_942_280_401_432_7,
            1e-15
        ));
        assert!(close(
            normal_cdf(0.0, 1.0, 1.96),
            0.975_002_104_851_779_5,
            1e-12
        ));
        assert!(close(normal_cdf(10.0, 2.0, 10.0), 0.5, 1e-15));
        assert!(close(
            normal_quantile(0.0, 1.0, 0.975),
            1.959_963_984_540_054,
            1e-9
        ));
        assert!(close(normal_quantile(5.0, 2.0, 0.5), 5.0, 1e-12));
        assert!(normal_pdf(0.0, 0.0, 0.0).is_nan());
    }

    #[test]
    fn exponential_closed_forms() {
        assert!(close(exponential_pdf(1.0, 2.0), (-2.0f64).exp(), 1e-15));
        assert!(close(
            exponential_cdf(2.0, 1.0),
            1.0 - (-2.0f64).exp(),
            1e-15
        ));
        assert!(close(exponential_quantile(1.0, 0.5), 2f64.ln(), 1e-15));
        assert_eq!(exponential_pdf(1.0, -1.0), 0.0);
        assert!(exponential_quantile(1.0, 1.5).is_nan());
    }

    #[test]
    fn gamma_reduces_to_exponential_and_known_densities() {
        // Shape 1 is the exponential distribution with mean `scale`.
        assert!(close(
            gamma_cdf(1.0, 1.0, 2.0),
            1.0 - (-2.0f64).exp(),
            1e-13
        ));
        // Shape 2, scale 1: pdf(x) = x e^{-x}.
        assert!(close(
            gamma_pdf(2.0, 1.0, 3.0),
            3.0 * (-3.0f64).exp(),
            1e-14
        ));
        assert!(close(
            gamma_cdf(2.0, 1.0, 3.0) + gamma_sf(2.0, 1.0, 3.0),
            1.0,
            1e-13
        ));
        assert!(gamma_pdf(0.0, 1.0, 1.0).is_nan());
    }

    #[test]
    fn beta_known_values() {
        // Beta(2, 2): pdf(x) = 6x(1-x), so pdf(0.5) = 1.5 and cdf(0.5) = 0.5.
        assert!(close(beta_pdf(2.0, 2.0, 0.5), 1.5, 1e-13));
        assert!(close(beta_cdf(2.0, 2.0, 0.5), 0.5, 1e-13));
        // Beta(1, 1) is uniform on [0, 1].
        assert!(close(beta_pdf(1.0, 1.0, 0.3), 1.0, 1e-13));
        assert_eq!(beta_pdf(2.0, 2.0, 1.5), 0.0);
    }

    #[test]
    fn uniform_closed_forms() {
        assert!(close(uniform_pdf(0.0, 2.0, 1.0), 0.5, 1e-15));
        assert_eq!(uniform_pdf(0.0, 2.0, 3.0), 0.0);
        assert!(close(uniform_cdf(0.0, 2.0, 0.5), 0.25, 1e-15));
        assert_eq!(uniform_cdf(0.0, 2.0, 5.0), 1.0);
        assert!(close(uniform_quantile(0.0, 2.0, 0.5), 1.0, 1e-15));
        assert!(uniform_pdf(2.0, 1.0, 1.5).is_nan());
    }

    #[test]
    fn poisson_pmf_and_cdf() {
        let e2 = (-2.0f64).exp();
        assert!(close(poisson_pmf(2.0, 0), e2, 1e-15));
        assert!(close(poisson_pmf(2.0, 2), 2.0 * e2, 1e-14));
        assert!(close(poisson_cdf(2.0, 1), 3.0 * e2, 1e-13));
        // The pmf sums to one: check the first terms reach the cdf.
        let sum: f64 = (0..=20).map(|k| poisson_pmf(2.0, k)).sum();
        assert!(close(sum, 1.0, 1e-12));
        assert_eq!(poisson_pmf(0.0, 0), 1.0);
        assert!(poisson_pmf(-1.0, 0).is_nan());
    }
}
