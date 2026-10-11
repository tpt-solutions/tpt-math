//! Special functions shared by the `tpt-math-prob-*` and `tpt-math-stats`
//! crates.
//!
//! Implemented from standard numerical recipes with no external dependency,
//! so no Apache-2.0-only crate enters the workspace through them:
//!
//! * log-gamma ([`ln_gamma`]) via a 9-term Lanczos approximation, with the
//!   reflection formula for arguments below one half;
//! * gamma and beta ([`gamma`], [`beta`], [`ln_beta`]) and their regularized
//!   incomplete forms ([`gamma_p`], [`gamma_q`], [`regularized_incomplete_beta`]);
//! * the error function ([`erf`], [`erfc`]) built on the incomplete gamma
//!   function, so both tails keep relative precision;
//! * the standard normal CDF, PDF and quantile;
//! * combinatorial helpers ([`ln_binomial`], [`ln_factorial`]) and [`xlogy`].
//!
//! The series and continued fractions stop once they converge. If they exhaust
//! their iteration budget they return `NaN` rather than an unconverged value.
//!
//! This module requires `std` (it uses `f64` transcendental methods).

use core::f64::consts::{FRAC_1_SQRT_2, PI};

/// Convergence tolerance for the series and continued fractions.
const EPS: f64 = 3.0e-15;
/// Iteration budget for the series and continued fractions. Large shape
/// parameters need on the order of `sqrt(a)` terms, so this is generous.
const MAX_ITER: usize = 10_000;
/// Guard against division by (near) zero inside the continued fractions.
const TINY: f64 = 1.0e-300;

/// Lanczos `g` parameter.
const LANCZOS_G: f64 = 7.0;
/// Lanczos coefficients (g = 7, n = 9).
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln(2π)`, used by the Lanczos and Gaussian density formulas.
pub const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// `√(2π)`, the normalising constant of the standard normal density.
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

// ---------------------------------------------------------------------------
// Gamma and beta
// ---------------------------------------------------------------------------

/// Natural logarithm of the absolute value of the gamma function, `ln|Γ(x)|`.
///
/// Uses the Lanczos approximation, with the reflection formula for `x < 0.5`.
/// Returns `+∞` at the poles (zero and the negative integers) and `NaN` for a
/// `NaN` input.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::ln_gamma;
///
/// // Γ(5) = 4! = 24
/// assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
/// // Γ(1/2) = √π
/// assert!((ln_gamma(0.5) - 0.572_364_942_924_700_1).abs() < 1e-12);
/// ```
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    // Γ has poles at zero and the negative integers.
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }
    // Exact values, so factorial-based terms cancel cleanly.
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    if x < 0.5 {
        // Reflection: |Γ(x)Γ(1 - x)| = π / |sin(πx)|.
        let sin = (PI * x).sin();
        return (PI / sin.abs()).ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let t = z + LANCZOS_G + 0.5;
    let mut series = LANCZOS_COEFFS[0];
    for (i, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        series += coeff / (z + i as f64);
    }
    0.5 * LN_2PI + (z + 0.5) * t.ln() - t + series.ln()
}

/// The gamma function `Γ(x)`, including its sign for negative non-integers.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::gamma;
///
/// assert!((gamma(5.0) - 24.0).abs() < 1e-10);
/// // Γ(-1/2) = -2√π is negative.
/// assert!((gamma(-0.5) + 2.0 * std::f64::consts::PI.sqrt()).abs() < 1e-10);
/// ```
pub fn gamma(x: f64) -> f64 {
    let magnitude = ln_gamma(x).exp();
    // Γ is negative on the intervals (-1, 0), (-3, -2), (-5, -4), ...
    // i.e. when floor(x) is odd.
    let negative = x < 0.0 && x.fract() != 0.0 && x.floor() % 2.0 != 0.0;
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// Natural logarithm of the beta function, `ln B(a, b)`, where
/// `B(a, b) = Γ(a)Γ(b) / Γ(a + b)`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::ln_beta;
///
/// // B(2, 3) = 1/12
/// assert!((ln_beta(2.0, 3.0) - (1.0f64 / 12.0).ln()).abs() < 1e-12);
/// ```
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// The beta function `B(a, b) = Γ(a)Γ(b) / Γ(a + b)`.
pub fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// Natural logarithm of the binomial coefficient `ln C(n, k)`.
///
/// Returns `-∞` when `k > n`.
pub fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Natural logarithm of the factorial, `ln(n!)`.
pub fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// `x * ln(y)`, defined as `0` whenever `x == 0` (even for `y == 0`).
///
/// This keeps log-densities finite at the boundary of their support, where
/// `0 * ln 0` would otherwise evaluate to `NaN`.
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

// ---------------------------------------------------------------------------
// Incomplete gamma and beta
// ---------------------------------------------------------------------------

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`.
///
/// Returns `NaN` for `a <= 0` or a `NaN` argument.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::gamma_p;
///
/// // P(1, x) = 1 - e^{-x}
/// assert!((gamma_p(1.0, 2.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-14);
/// ```
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

/// Regularized upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
///
/// Returns `NaN` for `a <= 0` or a `NaN` argument.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

/// Series evaluation of `P(a, x)` (Numerical Recipes `gser`).
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            return (-x + a * x.ln() - ln_gamma(a)).exp() * sum;
        }
    }
    f64::NAN
}

/// Continued-fraction evaluation of `Q(a, x)` (Numerical Recipes `gcf`).
///
/// Only called for `x >= a + 1`, where the fraction converges quickly.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return (-x + a * x.ln() - ln_gamma(a)).exp() * h;
        }
    }
    f64::NAN
}

/// Regularized incomplete beta `I_x(a, b)`: the CDF of `Beta(a, b)` at `x`.
///
/// Computed with the modified Lentz continued fraction. Returns `NaN` for
/// non-positive shape parameters or a `NaN` argument.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::regularized_incomplete_beta;
///
/// // CDF of Beta(2, 3) at 0.5 is 11/16.
/// let v = regularized_incomplete_beta(2.0, 3.0, 0.5);
/// assert!((v - 0.6875).abs() < 1e-12);
/// ```
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // ln(1 - x) via ln_1p keeps precision when x is small.
    let front = (a * x.ln() + b * (-x).ln_1p() - ln_beta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Lentz continued fraction for the incomplete beta (Numerical Recipes
/// `betacf`). Seeds `c = 1`; seeding it with `1/TINY` bakes a wrong first
/// factor into the product and skews the result.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;

        // Even step.
        let num = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + num * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + num / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step.
        let num = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + num * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + num / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            return h;
        }
    }
    f64::NAN
}

// ---------------------------------------------------------------------------
// Error function and the standard normal distribution
// ---------------------------------------------------------------------------

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Computed as `Q(1/2, x²)`, so it keeps relative precision deep into the
/// tail, where `erf(x)` is indistinguishable from 1.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::erfc;
///
/// assert!((erfc(1.0) - 0.157_299_207_050_285_1).abs() < 1e-12);
/// ```
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    // exp(-x²) underflows past ~27, so erfc is 0 or 2 to double precision.
    if ax > 27.0 {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    let q = gamma_q(0.5, ax * ax);
    if x > 0.0 {
        q
    } else {
        2.0 - q
    }
}

/// The error function `erf(x) = 2/√π ∫₀ˣ e^{-t²} dt`.
///
/// Computed as `P(1/2, x²)` with the sign of `x`. Small arguments keep
/// relative precision, which `1 - erfc(x)` would lose.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::erf;
///
/// assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
/// ```
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let p = gamma_p(0.5, x * x);
    if x >= 0.0 {
        p
    } else {
        -p
    }
}

/// Cumulative distribution function of the standard normal, `Φ(x)`.
///
/// Computed as `Φ(x) = erfc(-x/√2) / 2`, so both tails keep relative
/// precision. For example `Φ(-10) ≈ 7.62e-24` rather than 0.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::standard_normal_cdf;
///
/// assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-15);
/// assert!((standard_normal_cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-13);
/// ```
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Probability density of the standard normal, `φ(x)`.
pub fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// Inverse CDF (quantile) of the standard normal, `Φ⁻¹(p)`.
///
/// Acklam's rational approximation followed by one Halley refinement against
/// [`standard_normal_cdf`], which brings the result to near machine precision.
/// Returns `∓∞` for `p ≤ 0` / `p ≥ 1` and `NaN` for a `NaN` input.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::special::standard_normal_quantile;
///
/// assert!((standard_normal_quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-9);
/// ```
pub fn standard_normal_quantile(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley step against the (much more accurate) CDF.
    let err = standard_normal_cdf(x) - p;
    let density = standard_normal_pdf(x);
    if density > 0.0 {
        let u = err / density;
        x -= u / (1.0 + 0.5 * x * u);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sqrt_pi() -> f64 {
        PI.sqrt()
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        assert_eq!(ln_gamma(1.0), 0.0);
        assert_eq!(ln_gamma(2.0), 0.0);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(11.0) - 3_628_800f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn ln_gamma_half_integers_and_reflection() {
        assert!((ln_gamma(0.5) - sqrt_pi().ln()).abs() < 1e-13);
        assert!((ln_gamma(1.5) - (sqrt_pi() / 2.0).ln()).abs() < 1e-13);
        // Γ(1/4)Γ(3/4) = π / sin(π/4)
        let lhs = ln_gamma(0.25) + ln_gamma(0.75);
        let rhs = (PI / (PI * 0.25).sin()).ln();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn ln_gamma_of_negative_non_integer() {
        // |Γ(-1/2)| = 2√π.
        let want = (2.0 * sqrt_pi()).ln();
        assert!((ln_gamma(-0.5) - want).abs() < 1e-10);
    }

    #[test]
    fn ln_gamma_poles_and_nan() {
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-3.0).is_infinite());
        assert!(ln_gamma(f64::NAN).is_nan());
        assert!(ln_gamma(f64::INFINITY).is_infinite());
    }

    #[test]
    fn gamma_keeps_sign_on_negative_intervals() {
        assert!((gamma(5.0) - 24.0).abs() < 1e-10);
        assert!((gamma(0.5) - sqrt_pi()).abs() < 1e-12);
        // Γ(-1/2) = -2√π (negative); Γ(-3/2) = 4√π/3 (positive).
        assert!((gamma(-0.5) + 2.0 * sqrt_pi()).abs() < 1e-10);
        assert!((gamma(-1.5) - 4.0 * sqrt_pi() / 3.0).abs() < 1e-10);
    }

    #[test]
    fn beta_and_binomial_helpers() {
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-12);
        assert!((beta(0.5, 0.5) - PI).abs() < 1e-12);
        assert!((ln_beta(1.0, 1.0)).abs() < 1e-13);
        assert!((ln_binomial(5, 2) - 10f64.ln()).abs() < 1e-12);
        assert!((ln_binomial(52, 5) - 2_598_960f64.ln()).abs() < 1e-9);
        assert!(ln_binomial(3, 4).is_infinite());
        assert!((ln_factorial(6) - 720f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn xlogy_handles_zero_weight() {
        assert_eq!(xlogy(0.0, 0.0), 0.0);
        assert_eq!(xlogy(0.0, 1.0), 0.0);
        assert!((xlogy(2.0, core::f64::consts::E) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn incomplete_gamma_sums_to_one() {
        for (a, x) in [(1.0, 1.0), (2.5, 0.7), (0.5, 3.0), (4.0, 10.0)] {
            let p = gamma_p(a, x);
            let q = gamma_q(a, x);
            assert!((p + q - 1.0).abs() < 1e-13, "a={a}, x={x}, p={p}, q={q}");
        }
    }

    #[test]
    fn incomplete_gamma_chi_squared_identity() {
        // Chi-squared(df=1) survival at x = erfc(sqrt(x/2)).
        let x = 2.0;
        let sf = gamma_q(0.5, x / 2.0);
        let reference = erfc((x / 2.0).sqrt());
        assert!((sf - reference).abs() < 1e-14, "sf={sf}, ref={reference}");
    }

    #[test]
    fn incomplete_gamma_rejects_bad_shape_and_reports_nonconvergence() {
        assert!(gamma_p(0.0, 1.0).is_nan());
        assert!(gamma_q(-1.0, 1.0).is_nan());
        // The series needs ~sqrt(a) terms here, far beyond the budget, so the
        // result is NaN rather than a silently unconverged value.
        assert!(gamma_p(1e12, 1e12).is_nan());
    }

    #[test]
    fn error_function_known_values() {
        assert!(erf(0.0).abs() < 1e-15);
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-15);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-14);
        assert!((erfc(1.0) - 0.157_299_207_050_285_1).abs() < 1e-14);
        // erfc(5) ≈ 1.5374597944280349e-12 keeps relative precision in the tail.
        let want = 1.537_459_794_428_035e-12;
        assert!(((erfc(5.0) - want) / want).abs() < 1e-8);
    }

    #[test]
    fn erf_is_relatively_accurate_near_zero() {
        // erf(x) ≈ 2x/√π for small x. Computing it as 1 - erfc(x) loses all
        // relative precision here.
        let x = 1e-10_f64;
        let want = 2.0 * x / sqrt_pi();
        assert!(((erf(x) - want) / want).abs() < 1e-10, "erf = {}", erf(x));
        assert!(((erf(-x) + want) / want).abs() < 1e-10);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((standard_normal_cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-13);
        assert!((standard_normal_cdf(-1.96) - 0.024_997_895_148_220_435).abs() < 1e-13);
        assert!((standard_normal_cdf(2.0) - 0.977_249_868_051_820_8).abs() < 1e-13);
        assert_eq!(standard_normal_cdf(-40.0), 0.0);
        assert_eq!(standard_normal_cdf(40.0), 1.0);
    }

    #[test]
    fn normal_cdf_keeps_left_tail() {
        // Φ(-10) = 7.619853024160527e-24.
        let want = 7.619_853_024_160_527e-24;
        let got = standard_normal_cdf(-10.0);
        assert!(((got - want) / want).abs() < 1e-8, "Φ(-10) = {got}");
    }

    #[test]
    fn normal_quantile_inverts_cdf() {
        for &p in &[1e-8, 0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999] {
            let x = standard_normal_quantile(p);
            assert!((standard_normal_cdf(x) - p).abs() < 1e-12, "p = {p}");
        }
        assert!((standard_normal_quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-10);
        assert!(standard_normal_quantile(0.0).is_infinite());
        assert!(standard_normal_quantile(1.0).is_infinite());
    }

    #[test]
    fn incomplete_beta_known_values() {
        assert!((regularized_incomplete_beta(2.0, 3.0, 0.5) - 0.6875).abs() < 1e-12);
        // Beta(1, 1) is uniform.
        assert!((regularized_incomplete_beta(1.0, 1.0, 0.3) - 0.3).abs() < 1e-12);
        // Symmetry: I_x(a, a) = 1 - I_{1-x}(a, a).
        let lhs = regularized_incomplete_beta(2.5, 2.5, 0.3);
        let rhs = 1.0 - regularized_incomplete_beta(2.5, 2.5, 0.7);
        assert!((lhs - rhs).abs() < 1e-12);
        assert_eq!(regularized_incomplete_beta(2.0, 2.0, 0.0), 0.0);
        assert_eq!(regularized_incomplete_beta(2.0, 2.0, 1.0), 1.0);
        assert!(regularized_incomplete_beta(0.0, 2.0, 0.5).is_nan());
    }
}
