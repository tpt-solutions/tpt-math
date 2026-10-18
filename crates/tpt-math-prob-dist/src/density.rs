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

use core::f64::consts::PI;
use tpt_math_prob_core::special::{
    gamma_p, gamma_q, ln_beta, ln_binomial, ln_gamma, regularized_incomplete_beta,
    standard_normal_cdf, standard_normal_pdf, standard_normal_quantile,
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
/// `P(T ≤ x) = ½ + sign(x)·½·I_w(½, ν/2)` with `w = x²/(x² + ν)`.
///
/// This uses `P(|T| ≤ |x|) = I_w(½, ν/2)`, so the small-`x` region, which is
/// the centre of the distribution, is evaluated where the incomplete beta is
/// small and accurate. The form `1 − ½·I_{ν/(ν+x²)}(ν/2, ½)` instead needs the
/// argument `ν/(ν+x²)`, which rounds to 1 for small `x` and loses the centre.
pub fn student_t_cdf(nu: f64, x: f64) -> f64 {
    if !is_positive(nu) || x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    // Two-sided tail probability P(|T| > |x|), evaluated in whichever form is
    // accurate for this x: the direct one near the centre, and the complement
    // (argument ν/(x²+ν) < ½) out in the tails.
    let w = x * x / (x * x + nu);
    let two_sided_tail = if w < 0.5 {
        1.0 - regularized_incomplete_beta(0.5, nu / 2.0, w)
    } else {
        regularized_incomplete_beta(nu / 2.0, 0.5, nu / (x * x + nu))
    };
    if x >= 0.0 {
        1.0 - 0.5 * two_sided_tail
    } else {
        0.5 * two_sided_tail
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

/// Bisection for `x` with `cdf(x) = p`, given a bracket `lo ≤ x ≤ hi` with
/// `cdf(lo) ≤ p ≤ cdf(hi)`. Runs until the midpoint no longer separates the
/// bracket, so the result is accurate to the last bit of the representable
/// interval, for any continuous CDF.
fn bisect_cdf<F: Fn(f64) -> f64>(cdf: F, p: f64, mut lo: f64, mut hi: f64) -> f64 {
    loop {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break;
        }
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Quantile of a distribution supported on `[0, ∞)`, by bisection on its CDF.
/// `p` must lie in `(0, 1)`.
fn quantile_nonnegative<F: Fn(f64) -> f64>(cdf: F, p: f64) -> f64 {
    // Double the upper bracket until it passes p. The CDF tends to 1, so this
    // terminates for any p < 1; the cap guards against a non-convergent CDF.
    let mut hi = 1.0;
    for _ in 0..2000 {
        if cdf(hi) >= p {
            return bisect_cdf(&cdf, p, 0.0, hi);
        }
        hi *= 2.0;
    }
    f64::NAN
}

/// Quantile of a distribution on the whole real line, by bisection on its CDF.
/// `p` must lie in `(0, 1)`.
fn quantile_real<F: Fn(f64) -> f64>(cdf: F, p: f64) -> f64 {
    // Expand the bracket outward on both sides until it straddles p.
    let mut lo = -1.0;
    let mut hi = 1.0;
    for _ in 0..2000 {
        if cdf(lo) <= p && cdf(hi) >= p {
            return bisect_cdf(&cdf, p, lo, hi);
        }
        if cdf(lo) > p {
            lo *= 2.0;
        }
        if cdf(hi) < p {
            hi *= 2.0;
        }
    }
    f64::NAN
}

/// Quantile of the gamma distribution with shape `k` and scale `θ`.
///
/// Computed by bisection on [`gamma_cdf`]. `p = 0` gives `0` and `p = 1` gives
/// `∞`.
pub fn gamma_quantile(shape: f64, scale: f64, p: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    scale * quantile_nonnegative(|x| gamma_p(shape, x), p)
}

/// Quantile of the beta distribution on `[0, 1]` with shape parameters `a`
/// and `b`, by bisection on [`beta_cdf`].
pub fn beta_quantile(a: f64, b: f64, p: f64) -> f64 {
    if !is_positive(a) || !is_positive(b) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    bisect_cdf(|x| beta_cdf(a, b, x), p, 0.0, 1.0)
}

/// Quantile of the chi-squared distribution with `k` degrees of freedom.
pub fn chi_squared_quantile(k: f64, p: f64) -> f64 {
    gamma_quantile(k / 2.0, 2.0, p)
}

/// Quantile of Student's t distribution with `nu` degrees of freedom, by
/// bisection on [`student_t_cdf`]. `p = 0` gives `−∞` and `p = 1` gives `∞`.
pub fn student_t_quantile(nu: f64, p: f64) -> f64 {
    if !is_positive(nu) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    quantile_real(|x| student_t_cdf(nu, x), p)
}

/// Quantile of the Cauchy distribution, `x₀ + s·tan(π(p − ½))` (closed form).
pub fn cauchy_quantile(x0: f64, s: f64, p: f64) -> f64 {
    if !is_positive(s) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    x0 + s * (core::f64::consts::PI * (p - 0.5)).tan()
}

/// Smallest integer `k` with `P(X ≤ k) ≥ p` for a Poisson variable with rate
/// `λ`, returned as a float so that `NaN` can mark invalid input.
///
/// Searches by doubling and then bisecting over the integers, using
/// [`poisson_cdf`]. `p = 1` gives `∞`.
pub fn poisson_quantile(lambda: f64, p: f64) -> f64 {
    if !is_nonnegative(lambda) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut hi: u64 = 1;
    while poisson_cdf(lambda, hi) < p {
        match hi.checked_mul(2) {
            Some(next) => hi = next,
            None => return f64::INFINITY,
        }
    }
    let mut lo: u64 = 0;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if poisson_cdf(lambda, mid) >= p {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo as f64
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

/// Lognormal density: `ln X ~ N(μ, σ²)`.
pub fn lognormal_pdf(mu: f64, sigma: f64, x: f64) -> f64 {
    if !is_positive(sigma) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = (x.ln() - mu) / sigma;
    (-0.5 * z * z).exp() / (x * sigma * (2.0 * PI).sqrt())
}

/// Lognormal cumulative distribution function.
pub fn lognormal_cdf(mu: f64, sigma: f64, x: f64) -> f64 {
    if !is_positive(sigma) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    standard_normal_cdf((x.ln() - mu) / sigma)
}

/// Lognormal quantile, `exp(μ + σ Φ⁻¹(p))`.
pub fn lognormal_quantile(mu: f64, sigma: f64, p: f64) -> f64 {
    if !is_positive(sigma) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    (mu + sigma * standard_normal_quantile(p)).exp()
}

/// Mean and variance of the lognormal distribution,
/// `(e^{μ + σ²/2}, (e^{σ²} − 1) e^{2μ + σ²})`.
pub fn lognormal_moments(mu: f64, sigma: f64) -> (f64, f64) {
    if !is_positive(sigma) {
        return (f64::NAN, f64::NAN);
    }
    let s2 = sigma * sigma;
    let mean = (mu + 0.5 * s2).exp();
    let var = s2.exp_m1() * (2.0 * mu + s2).exp();
    (mean, var)
}

/// Weibull density with shape `k` and scale `λ`.
///
/// At `x = 0` the density is `∞` for `k < 1`, `1/λ` for `k = 1` and `0` for `k > 1`.
pub fn weibull_pdf(shape: f64, scale: f64, x: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) {
        return f64::NAN;
    }
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return match shape.partial_cmp(&1.0) {
            Some(core::cmp::Ordering::Less) => f64::INFINITY,
            Some(core::cmp::Ordering::Equal) => 1.0 / scale,
            _ => 0.0,
        };
    }
    let z = x / scale;
    (shape / scale) * z.powf(shape - 1.0) * (-z.powf(shape)).exp()
}

/// Weibull cumulative distribution function, `1 − e^{−(x/λ)^k}`.
pub fn weibull_cdf(shape: f64, scale: f64, x: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    -(-(x / scale).powf(shape)).exp_m1()
}

/// Weibull quantile, `λ (−ln(1 − p))^{1/k}`.
pub fn weibull_quantile(shape: f64, scale: f64, p: f64) -> f64 {
    if !is_positive(shape) || !is_positive(scale) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    scale * (-(-p).ln_1p()).powf(1.0 / shape)
}

/// Mean and variance of the Weibull distribution,
/// `(λ Γ(1 + 1/k), λ² (Γ(1 + 2/k) − Γ(1 + 1/k)²))`.
pub fn weibull_moments(shape: f64, scale: f64) -> (f64, f64) {
    if !is_positive(shape) || !is_positive(scale) {
        return (f64::NAN, f64::NAN);
    }
    let g1 = ln_gamma(1.0 + 1.0 / shape).exp();
    let g2 = ln_gamma(1.0 + 2.0 / shape).exp();
    (scale * g1, scale * scale * (g2 - g1 * g1))
}

/// Binomial probability mass at `k` for `n` trials with success probability `p`.
pub fn binomial_pmf(n: u64, p: f64, k: u64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k > n {
        return 0.0;
    }
    if p == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    (ln_binomial(n, k) + k as f64 * p.ln() + (n - k) as f64 * (-p).ln_1p()).exp()
}

/// `P(X ≤ k)` for a binomial variable, `I_{1−p}(n − k, k + 1)`.
pub fn binomial_cdf(n: u64, p: f64, k: u64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k >= n {
        return 1.0;
    }
    regularized_incomplete_beta((n - k) as f64, (k + 1) as f64, 1.0 - p)
}

/// Mean and variance of the binomial distribution, `(np, np(1 − p))`.
pub fn binomial_moments(n: u64, p: f64) -> (f64, f64) {
    if !(0.0..=1.0).contains(&p) {
        return (f64::NAN, f64::NAN);
    }
    let n = n as f64;
    (n * p, n * p * (1.0 - p))
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

#[cfg(test)]
mod quantile_tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gamma_with_shape_one_is_exponential() {
        // Gamma(1, θ) is Exponential(rate 1/θ): quantile = −θ·ln(1 − p).
        for &p in &[0.01, 0.3, 0.5, 0.9, 0.999] {
            let q = gamma_quantile(1.0, 2.5, p);
            assert!(close(q, -2.5 * (-p).ln_1p(), 1e-9), "p={p}: {q}");
        }
    }

    #[test]
    fn chi_squared_two_dof_is_exponential_with_mean_two() {
        for &p in &[0.05, 0.5, 0.95] {
            let q = chi_squared_quantile(2.0, p);
            assert!(close(q, -2.0 * (-p).ln_1p(), 1e-9), "p={p}: {q}");
        }
    }

    #[test]
    fn beta_one_one_is_uniform() {
        for &p in &[0.1, 0.5, 0.77] {
            assert!(close(beta_quantile(1.0, 1.0, p), p, 1e-12));
        }
    }

    #[test]
    fn student_t_one_dof_is_cauchy() {
        for &p in &[0.1, 0.5, 0.9, 0.975] {
            let q = student_t_quantile(1.0, p);
            assert!(
                close(q, (core::f64::consts::PI * (p - 0.5)).tan(), 1e-8),
                "p={p}: {q}"
            );
        }
    }

    #[test]
    fn student_t_cdf_keeps_centre_and_left_tail_precise() {
        // ν = 1 is Cauchy: F(x) = ½ + atan(x)/π. Check the centre, where the
        // CDF is 0.5 + small, and the far left tail, where it is small.
        let cauchy = |x: f64| 0.5 + x.atan() / core::f64::consts::PI;
        for &x in &[1e-8, -1e-8, 0.3, -0.3, -1e4, -1e8] {
            let got = student_t_cdf(1.0, x);
            // For x < 0 the left tail is atan(1/|x|)/π; the sum ½ + atan(x)/π would
            // cancel there and be the inaccurate side of the comparison.
            let want = if x < 0.0 {
                (1.0 / -x).atan() / core::f64::consts::PI
            } else {
                cauchy(x)
            };
            let rel = ((got - want) / want).abs();
            assert!(rel < 1e-9, "x={x}: got {got}, want {want}, rel {rel}");
        }
    }

    #[test]
    fn student_t_matches_known_critical_value() {
        // t₀.₉₇₅ with 4 dof is 2.776445 (two-sided 5 % critical value).
        assert!(close(student_t_quantile(4.0, 0.975), 2.776_445, 1e-6));
    }

    #[test]
    fn cauchy_quantile_is_closed_form_and_inverts_cdf() {
        let q = cauchy_quantile(1.0, 2.0, 0.75);
        assert!(close(q, 3.0, 1e-12));
        assert!(close(cauchy_cdf(1.0, 2.0, q), 0.75, 1e-12));
    }

    #[test]
    fn continuous_quantiles_invert_their_cdfs() {
        for &p in &[1e-6, 0.02, 0.4, 0.8, 1.0 - 1e-6] {
            let g = gamma_quantile(3.7, 0.8, p);
            assert!(close(gamma_cdf(3.7, 0.8, g), p, 1e-9), "gamma p={p}");
            let b = beta_quantile(0.6, 4.2, p);
            assert!(close(beta_cdf(0.6, 4.2, b), p, 1e-9), "beta p={p}");
            let t = student_t_quantile(7.0, p);
            assert!(close(student_t_cdf(7.0, t), p, 1e-9), "t p={p}");
        }
    }

    #[test]
    fn poisson_quantile_is_smallest_k_reaching_p() {
        // λ = 2: P(X ≤ 0) ≈ 0.1353, P(X ≤ 1) ≈ 0.4060, P(X ≤ 2) ≈ 0.6767.
        assert_eq!(poisson_quantile(2.0, 0.1), 0.0);
        assert_eq!(poisson_quantile(2.0, 0.2), 1.0);
        assert_eq!(poisson_quantile(2.0, 0.5), 2.0);
        // The defining property: the CDF at the quantile reaches p, and the
        // CDF one step below does not.
        for &p in &[0.01, 0.3, 0.9, 0.999] {
            let k = poisson_quantile(12.5, p) as u64;
            assert!(poisson_cdf(12.5, k) >= p);
            if k > 0 {
                assert!(poisson_cdf(12.5, k - 1) < p);
            }
        }
    }

    #[test]
    fn quantiles_reject_invalid_input() {
        assert!(gamma_quantile(-1.0, 1.0, 0.5).is_nan());
        assert!(gamma_quantile(1.0, 1.0, 1.5).is_nan());
        assert!(beta_quantile(1.0, 1.0, f64::NAN).is_nan());
        assert!(student_t_quantile(0.0, 0.5).is_nan());
        assert!(cauchy_quantile(0.0, 0.0, 0.5).is_nan());
        assert!(poisson_quantile(-1.0, 0.5).is_nan());
        assert_eq!(gamma_quantile(2.0, 1.0, 1.0), f64::INFINITY);
        assert_eq!(student_t_quantile(3.0, 0.0), f64::NEG_INFINITY);
    }
}

#[cfg(test)]
mod new_distribution_tests {
    use super::*;

    #[test]
    fn lognormal_known_values() {
        // Median is e^μ; standard lognormal has mean √e.
        assert!((lognormal_cdf(0.0, 1.0, 1.0) - 0.5).abs() < 1e-14);
        assert!((lognormal_quantile(0.0, 1.0, 0.5) - 1.0).abs() < 1e-14);
        let (mean, var) = lognormal_moments(0.0, 1.0);
        assert!((mean - 0.5_f64.exp()).abs() < 1e-12);
        assert!((var - (1.0_f64.exp() - 1.0) * 1.0_f64.exp()).abs() < 1e-12);
        // Density integrates to the CDF (checked by a midpoint rule).
        let (a, b, n) = (1e-9, 3.0, 200_000);
        let h = (b - a) / n as f64;
        let integral: f64 = (0..n)
            .map(|i| lognormal_pdf(0.2, 0.7, a + (i as f64 + 0.5) * h) * h)
            .sum();
        assert!((integral - lognormal_cdf(0.2, 0.7, b)).abs() < 1e-6);
        assert!(lognormal_pdf(0.0, 0.0, 1.0).is_nan());
        assert_eq!(lognormal_cdf(0.0, 1.0, -1.0), 0.0);
    }

    #[test]
    fn weibull_with_unit_shape_is_exponential() {
        let (scale, rate) = (2.0, 0.5);
        for x in [0.1, 1.0, 3.7] {
            assert!((weibull_pdf(1.0, scale, x) - exponential_pdf(rate, x)).abs() < 1e-14);
            assert!((weibull_cdf(1.0, scale, x) - exponential_cdf(rate, x)).abs() < 1e-14);
        }
        let (mean, var) = weibull_moments(1.0, scale);
        assert!((mean - scale).abs() < 1e-12);
        assert!((var - scale * scale).abs() < 1e-10);
        assert_eq!(weibull_pdf(1.0, scale, 0.0), 1.0 / scale);
        assert_eq!(weibull_pdf(2.0, scale, 0.0), 0.0);
        assert_eq!(weibull_pdf(0.5, scale, 0.0), f64::INFINITY);
    }

    #[test]
    fn weibull_quantile_inverts_cdf() {
        for p in [0.01, 0.3, 0.9, 0.999] {
            let x = weibull_quantile(1.7, 3.0, p);
            assert!((weibull_cdf(1.7, 3.0, x) - p).abs() < 1e-12);
        }
    }

    #[test]
    fn binomial_known_values_and_normalisation() {
        // C(10, 5) / 2^10 = 252 / 1024; the cumulative sum to 5 is 638 / 1024.
        assert!((binomial_pmf(10, 0.5, 5) - 252.0 / 1024.0).abs() < 1e-14);
        assert!((binomial_cdf(10, 0.5, 5) - 638.0 / 1024.0).abs() < 1e-12);
        let total: f64 = (0..=12).map(|k| binomial_pmf(12, 0.3, k)).sum();
        assert!((total - 1.0).abs() < 1e-12);
        let direct: f64 = (0..=4).map(|k| binomial_pmf(12, 0.3, k)).sum();
        assert!((binomial_cdf(12, 0.3, 4) - direct).abs() < 1e-12);
        let (mean, var) = binomial_moments(12, 0.3);
        assert!((mean - 3.6).abs() < 1e-12 && (var - 2.52).abs() < 1e-12);
        assert_eq!(binomial_pmf(5, 0.0, 0), 1.0);
        assert_eq!(binomial_pmf(5, 1.0, 4), 0.0);
        assert!(binomial_pmf(5, 1.5, 2).is_nan());
    }
}

/// Negative binomial pmf: the probability of `k` failures before the `r`-th success
/// with success probability `p`, for a real `r > 0`.
///
/// `P(X = k) = Γ(k + r) / (Γ(r) k!) · p^r (1 − p)^k`, evaluated in log space.
pub fn negative_binomial_pmf(r: f64, p: f64, k: u64) -> f64 {
    if !is_positive(r) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let k = k as f64;
    let ln_pmf = ln_gamma(k + r) - ln_gamma(r) - ln_gamma(k + 1.0) + r * p.ln() + k * (-p).ln_1p();
    ln_pmf.exp()
}

/// `P(X ≤ k)` for the negative binomial, `I_p(r, k + 1)`.
pub fn negative_binomial_cdf(r: f64, p: f64, k: u64) -> f64 {
    if !is_positive(r) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    regularized_incomplete_beta(r, k as f64 + 1.0, p)
}

/// Mean and variance of the negative binomial, `(r(1 − p)/p, r(1 − p)/p²)`.
pub fn negative_binomial_moments(r: f64, p: f64) -> (f64, f64) {
    if !is_positive(r) || !(0.0 < p && p <= 1.0) {
        return (f64::NAN, f64::NAN);
    }
    let q = 1.0 - p;
    (r * q / p, r * q / (p * p))
}

#[cfg(test)]
mod negative_binomial_tests {
    use super::*;

    #[test]
    fn pmf_sums_to_one_and_matches_the_geometric_case() {
        // r = 1 is the geometric distribution: P(X = k) = p (1 − p)^k.
        for k in 0..6 {
            let want = 0.3 * 0.7_f64.powi(k);
            assert!((negative_binomial_pmf(1.0, 0.3, k as u64) - want).abs() < 1e-14);
        }
        let total: f64 = (0..400).map(|k| negative_binomial_pmf(2.5, 0.4, k)).sum();
        assert!((total - 1.0).abs() < 1e-10, "total {total}");
    }

    #[test]
    fn cdf_matches_the_partial_pmf_sum_and_moments_are_right() {
        let (r, p) = (3.5, 0.45);
        let partial: f64 = (0..=7).map(|k| negative_binomial_pmf(r, p, k)).sum();
        assert!((negative_binomial_cdf(r, p, 7) - partial).abs() < 1e-12);
        let (mean, var) = negative_binomial_moments(r, p);
        assert!((mean - r * (1.0 - p) / p).abs() < 1e-12);
        assert!((var - r * (1.0 - p) / (p * p)).abs() < 1e-12);
        assert!(negative_binomial_pmf(0.0, 0.5, 1).is_nan());
    }
}
