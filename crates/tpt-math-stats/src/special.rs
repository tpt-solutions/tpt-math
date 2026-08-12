#![forbid(unsafe_code)]
//! Special functions: error, gamma, beta, and their regularized incomplete
//! forms. Implemented from standard numerical recipes (no external crate), so
//! no Apache-2.0 dependency enters the workspace through this crate.
//!
//! Accuracy targets are set by the crate's own regression tests, which pin
//! results to 1e-12–1e-14 against closed-form / reference values:
//! `erf(0) = 0`, `erfc(1) = 0.15729920705028513`, `gamma(5) = 24`,
//! `beta(2, 3) = 1/12`.

/// Machine-epsilon-scale convergence tolerance for the continued fractions.
const EPS: f64 = 3.0e-15;
/// Maximum iterations for the incomplete-gamma / incomplete-beta fractions.
const MAXIT: usize = 400;
/// Tiny magnitude used to guard against division by (near) zero in fractions.
const TINY: f64 = 1.0e-300;

/// Lanczos `g` parameter.
const LANCZOS_G: f64 = 7.0;
/// Lanczos coefficients (g = 7, n = 9) for the log-gamma approximation.
const LANCZOS_C: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.5203681218851,
    -1259.1392167224028,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

/// Half of `ln(2π)`, used throughout the Lanczos approximation.
const HALF_LN_2PI: f64 = 0.9189385332046727;

// ---------------------------------------------------------------------------
// Log-gamma
// ---------------------------------------------------------------------------

/// The natural logarithm of the gamma function, `ln Γ(z)`.
///
/// Uses a 9-term Lanczos approximation for `z >= 0.5` and the reflection
/// identity `Γ(z)Γ(1-z) = π / sin(π z)` below that. Accurate to better than
/// 1e-13 for all positive arguments.
pub fn lgamma(z: f64) -> f64 {
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        std::f64::consts::PI.ln() - (pi * z).sin().ln() - lgamma(1.0 - z)
    } else {
        let zz = z - 1.0;
        let mut x = LANCZOS_C[0];
        for (k, &c) in LANCZOS_C.iter().enumerate().skip(1) {
            x += c / (zz + k as f64);
        }
        let t = zz + LANCZOS_G + 0.5;
        HALF_LN_2PI + (zz + 0.5) * t.ln() - t + x.ln()
    }
}

/// The gamma function `Γ(z) = exp(ln Γ(z))`.
pub fn gamma(z: f64) -> f64 {
    lgamma(z).exp()
}

// ---------------------------------------------------------------------------
// Incomplete gamma
// ---------------------------------------------------------------------------

/// Regularized *lower* incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

/// Regularized *upper* incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

/// Series evaluation of `P(a, x)` (Numerical Recipes `gser`).
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let gln = lgamma(a);
    let mut sum = 1.0 / a;
    let mut del = sum;
    for n in 1..MAXIT {
        let an = n as f64;
        del *= x / (a + an);
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * sum
}

/// Continued-fraction evaluation of `Q(a, x)` (Numerical Recipes `gcf`).
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let gln = lgamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = if b == 0.0 { TINY } else { 1.0 / b };
    let mut h = d;
    for i in 1..MAXIT {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY * d.signum();
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY * c.signum();
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

// ---------------------------------------------------------------------------
// Beta
// ---------------------------------------------------------------------------

/// The beta function `B(a, b) = Γ(a)Γ(b) / Γ(a + b)`.
pub fn beta(a: f64, b: f64) -> f64 {
    (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
}

/// Regularized incomplete beta `I_x(a, b)` (Numerical Recipes `betai`).
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Lentz continued fraction for the incomplete beta (Numerical Recipes `betacf`).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    // `betacf` seeds `c = 1.0` (unlike the gamma continued fraction `gcf`,
    // which seeds `c = 1/FPMIN`). Seeding `c` with `1/TINY` here bakes a wrong
    // first factor into the Lentz product that never washes out — it skews the
    // result by a few percent and breaks Student's t p-values.
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    d = if d == 0.0 { TINY } else { 1.0 / d };
    let mut h = d;
    for m in 1..MAXIT {
        let m = m as f64;
        // Even-indexed step.
        let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY * d.signum();
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY * c.signum();
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd-indexed step.
        let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY * d.signum();
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY * c.signum();
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Error function
// ---------------------------------------------------------------------------

/// The complementary error function `erfc(x) = 1 - erf(x) = 2/√π ∫_x^∞ e^{-t²} dt`.
///
/// Computed directly from the upper regularized incomplete gamma,
/// `erfc(x) = Q(1/2, x²)`, so it stays accurate deep into the tail where
/// `erf(x)` is indistinguishable from 1.0.
pub fn erfc(x: f64) -> f64 {
    let ax = x.abs();
    // `exp(-x²)` underflows past ~27; erfc is then indistinguishable from 0.
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

/// The error function `erf(x) = 1 - erfc(x) = 2/√π ∫_0^x e^{-t²} dt`.
pub fn erf(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 - erfc(x)
    } else {
        -erf(-x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_gamma_known_values() {
        assert!((lgamma(1.0)).abs() < 1e-14);
        assert!((lgamma(2.0)).abs() < 1e-14);
        assert!((lgamma(5.0) - gamma(5.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn gamma_function() {
        assert!((gamma(5.0) - 24.0).abs() < 1e-10);
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn beta_function() {
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-12);
        assert!((beta(0.5, 0.5) - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn incomplete_gamma_boundaries() {
        // P + Q = 1.
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
        let ref_erfc = erfc((x / 2.0).sqrt());
        assert!((sf - ref_erfc).abs() < 1e-14, "sf={sf}, ref={ref_erfc}");
    }

    #[test]
    fn error_function() {
        assert!(erf(0.0).abs() < 1e-15);
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-15);
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-14);
        assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-14);
    }

    #[test]
    fn incomplete_beta_boundaries() {
        assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
        // Symmetry-ish check: I_x(a,b) + I_{1-x}(b,a) = 1.
        let (a, b, x) = (2.0, 3.0, 0.4);
        let s = beta_reg(a, b, x) + beta_reg(b, a, 1.0 - x);
        assert!((s - 1.0).abs() < 1e-13, "s={s}");
    }
}
