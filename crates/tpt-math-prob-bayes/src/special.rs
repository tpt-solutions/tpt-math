//! Special functions backing the conjugate-prior distributions.
//!
//! These are plain `f64` routines with no external dependencies, chosen for a
//! good accuracy/size trade-off:
//!
//! * [`ln_gamma`] — Lanczos approximation (`g = 7`, 9 coefficients), accurate
//!   to roughly 15 significant digits over the positive reals.
//! * [`standard_normal_cdf`] — Hart's rational approximation, accurate to
//!   about 1e-15 absolute; [`erf`] and [`erfc`] are derived from it.
//! * [`standard_normal_quantile`] — Acklam's inverse-normal approximation
//!   refined by one Halley step, giving near machine precision.
//! * [`regularized_incomplete_beta`] — modified Lentz continued fraction.

use core::f64::consts::PI;

/// `ln(2π)`, used by the Lanczos and Gaussian density formulas.
pub const LN_2PI: f64 = 1.837_877_066_409_345_3;

const LANCZOS_G: f64 = 7.0;
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

/// Natural logarithm of the gamma function, `ln Γ(x)`.
///
/// Uses the Lanczos approximation, with the reflection formula for `x < 0.5`.
/// Returns `+∞` at the poles (zero and the negative integers) and `NaN` for a
/// `NaN` input.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::special::ln_gamma;
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
    // Γ has poles at zero and the negative integers.
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }
    // Exact values, so factorial-based terms cancel cleanly.
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1 - x) = π / sin(πx).
        let sin = (PI * x).sin();
        if sin == 0.0 {
            return f64::INFINITY;
        }
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

/// Natural logarithm of the beta function, `ln B(a, b)`.
///
/// `B(a, b) = Γ(a)Γ(b) / Γ(a + b)`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::special::ln_beta;
///
/// // B(2, 3) = 1/12
/// assert!((ln_beta(2.0, 3.0) - (1.0f64 / 12.0).ln()).abs() < 1e-12);
/// ```
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
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
/// This is the convention that keeps log-densities finite at the boundary of
/// their support, where `0 * ln 0` would otherwise evaluate to `NaN`.
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

/// Cumulative distribution function of the standard normal, `Φ(x)`.
///
/// Hart's rational approximation as popularised by West; the absolute error is
/// on the order of 1e-15.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::special::standard_normal_cdf;
///
/// assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-15);
/// assert!((standard_normal_cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-12);
/// ```
pub fn standard_normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let abs = x.abs();
    let tail = if abs > 37.0 {
        0.0
    } else {
        let exponential = (-0.5 * abs * abs).exp();
        if abs < 7.071_067_811_865_47 {
            let mut num = 3.526_249_659_989_11e-2 * abs + 0.700_383_064_443_688;
            num = num * abs + 6.373_962_203_531_65;
            num = num * abs + 33.912_866_078_383;
            num = num * abs + 112.079_291_497_871;
            num = num * abs + 221.213_596_169_931;
            num = num * abs + 220.206_867_912_376;
            let mut den = 8.838_834_764_831_84e-2 * abs + 1.755_667_163_182_64;
            den = den * abs + 16.064_177_579_207;
            den = den * abs + 86.780_732_202_946_1;
            den = den * abs + 296.564_248_779_674;
            den = den * abs + 637.333_633_378_831;
            den = den * abs + 793.826_512_519_948;
            den = den * abs + 440.413_735_824_752;
            exponential * num / den
        } else {
            let mut cf = abs + 0.65;
            cf = abs + 4.0 / cf;
            cf = abs + 3.0 / cf;
            cf = abs + 2.0 / cf;
            cf = abs + 1.0 / cf;
            exponential / (cf * 2.506_628_274_631)
        }
    };
    if x > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// The error function, `erf(x)`.
///
/// Derived from [`standard_normal_cdf`] via `erf(x) = 2Φ(x√2) − 1`.
pub fn erf(x: f64) -> f64 {
    2.0 * standard_normal_cdf(x * core::f64::consts::SQRT_2) - 1.0
}

/// The complementary error function, `erfc(x) = 1 − erf(x)`.
pub fn erfc(x: f64) -> f64 {
    2.0 * standard_normal_cdf(-x * core::f64::consts::SQRT_2)
}

/// Probability density of the standard normal, `φ(x)`.
pub fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Inverse CDF (quantile) of the standard normal, `Φ⁻¹(p)`.
///
/// Acklam's rational approximation followed by a single Halley refinement,
/// which brings the result to near machine precision. Returns `∓∞` for
/// `p ≤ 0` / `p ≥ 1` and `NaN` for a `NaN` input.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::special::standard_normal_quantile;
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

/// The regularized incomplete beta function `I_x(a, b)`.
///
/// This is the CDF of `Beta(a, b)` evaluated at `x`. Computed with the
/// modified Lentz continued fraction, mirroring the classic *Numerical
/// Recipes* formulation.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::special::regularized_incomplete_beta;
///
/// // CDF of Beta(2, 3) at 0.5 is 11/16.
/// let v = regularized_incomplete_beta(2.0, 3.0, 0.5);
/// assert!((v - 0.6875).abs() < 1e-12);
/// ```
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Continued-fraction expansion used by [`regularized_incomplete_beta`].
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 300;
    const EPS: f64 = 1e-15;
    const TINY: f64 = 1e-300;

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
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        assert_eq!(ln_gamma(1.0), 0.0);
        assert_eq!(ln_gamma(2.0), 0.0);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(11.0) - 3_628_800f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn ln_gamma_half_integers_and_reflection() {
        // Γ(1/2) = √π
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-13);
        // Γ(3/2) = √π / 2
        assert!((ln_gamma(1.5) - (PI.sqrt() / 2.0).ln()).abs() < 1e-13);
        // Γ(1/4)Γ(3/4) = π / sin(π/4)
        let lhs = ln_gamma(0.25) + ln_gamma(0.75);
        let rhs = (PI / (PI * 0.25).sin()).ln();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn ln_gamma_poles_are_infinite() {
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-3.0).is_infinite());
        assert!(ln_gamma(f64::NAN).is_nan());
    }

    #[test]
    fn ln_beta_and_binomial() {
        assert!((ln_beta(2.0, 3.0) - (1.0f64 / 12.0).ln()).abs() < 1e-13);
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
    fn normal_cdf_known_values() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((standard_normal_cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-13);
        assert!((standard_normal_cdf(-1.96) - 0.024_997_895_148_220_435).abs() < 1e-13);
        assert!((standard_normal_cdf(2.0) - 0.977_249_868_051_820_8).abs() < 1e-13);
        assert_eq!(standard_normal_cdf(-40.0), 0.0);
        assert_eq!(standard_normal_cdf(40.0), 1.0);
    }

    #[test]
    fn erf_matches_reference() {
        assert!(erf(0.0).abs() < 1e-15);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
        assert!((erfc(1.0) - 0.157_299_207_050_285_1).abs() < 1e-12);
        assert!((erf(-0.5) + erf(0.5)).abs() < 1e-15);
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
    }
}
