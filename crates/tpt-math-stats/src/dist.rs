#![forbid(unsafe_code)]
//! Probability distributions used by the hypothesis tests, implemented in-house
//! (no `statrs` / Apache-2.0 dependency).
//!
//! Each continuous distribution exposes [`ContinuousCDF`] so call sites that
//! need a cumulative (`cdf`) or survival (`sf`) probability share one trait,
//! mirroring the surface the rest of the crate was written against.

/// Cumulative and survival probabilities of a continuous distribution.
pub trait ContinuousCDF {
    /// Cumulative distribution function `P(X <= x)`.
    fn cdf(&self, x: f64) -> f64;
    /// Survival function `P(X > x) = 1 - cdf(x)`.
    fn sf(&self, x: f64) -> f64;
}

// ---------------------------------------------------------------------------
// Chi-squared
// ---------------------------------------------------------------------------

/// A chi-squared distribution with `k` degrees of freedom.
///
/// `cdf(x) = P(k/2, x/2)` and `sf(x) = Q(k/2, x/2)`, the regularized
/// incomplete gamma functions.
#[derive(Debug)]
pub struct ChiSquared {
    /// Degrees of freedom (must be positive).
    k: f64,
}

impl ChiSquared {
    /// Construct a chi-squared distribution with `k` degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns an error string if `k <= 0`.
    pub fn new(k: f64) -> Result<Self, &'static str> {
        if k <= 0.0 {
            return Err("degrees of freedom must be positive");
        }
        Ok(ChiSquared { k })
    }
}

impl ContinuousCDF for ChiSquared {
    fn cdf(&self, x: f64) -> f64 {
        crate::special::gamma_p(self.k / 2.0, x / 2.0)
    }

    fn sf(&self, x: f64) -> f64 {
        crate::special::gamma_q(self.k / 2.0, x / 2.0)
    }
}

// ---------------------------------------------------------------------------
// Student's t
// ---------------------------------------------------------------------------

/// Student's t-distribution with location `loc`, scale `scale`, and `df`
/// degrees of freedom.
///
/// The survival/cumulative probabilities follow from the regularized incomplete
/// beta: for the standardized value `u = (x - loc) / scale`,
/// `sf(u) = 0.5 * I_{df/(df + u²)}(df/2, 1/2)`.
#[derive(Debug)]
pub struct StudentsT {
    loc: f64,
    scale: f64,
    df: f64,
}

impl StudentsT {
    /// Construct a Student's t-distribution with location `loc`, scale
    /// `scale`, and `df` degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns an error string if `df <= 0` or `scale <= 0`.
    pub fn new(loc: f64, scale: f64, df: f64) -> Result<Self, &'static str> {
        if df <= 0.0 {
            return Err("degrees of freedom must be positive");
        }
        if scale <= 0.0 {
            return Err("scale must be positive");
        }
        Ok(StudentsT { loc, scale, df })
    }
}

impl ContinuousCDF for StudentsT {
    fn cdf(&self, x: f64) -> f64 {
        let u = (x - self.loc) / self.scale;
        let z = self.df / (self.df + u * u);
        let ib = crate::special::beta_reg(self.df / 2.0, 0.5, z);
        if u >= 0.0 {
            1.0 - 0.5 * ib
        } else {
            0.5 * ib
        }
    }

    fn sf(&self, x: f64) -> f64 {
        // Evaluate the upper tail directly from the regularized incomplete beta
        // rather than `1 - cdf(x)`: for large `|x|` the cdf rounds to 1.0 and
        // `1 - cdf` annihilates the tiny, significant tail probability.
        let u = (x - self.loc) / self.scale;
        let z = self.df / (self.df + u * u);
        let ib = crate::special::beta_reg(self.df / 2.0, 0.5, z);
        if u >= 0.0 {
            0.5 * ib
        } else {
            1.0 - 0.5 * ib
        }
    }
}

// ---------------------------------------------------------------------------
// Normal
// ---------------------------------------------------------------------------

/// A normal (Gaussian) distribution with mean `mu` and standard deviation
/// `sigma`.
#[derive(Debug)]
pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    /// Construct a normal distribution with mean `mu` and standard deviation
    /// `sigma`.
    ///
    /// # Errors
    ///
    /// Returns an error string if `sigma <= 0`.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, &'static str> {
        if sigma <= 0.0 {
            return Err("standard deviation must be positive");
        }
        Ok(Normal { mu, sigma })
    }

    /// The mean `mu`.
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// The variance `sigma²`.
    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * std::f64::consts::PI).sqrt())
    }
}

impl ContinuousCDF for Normal {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / (self.sigma * std::f64::consts::SQRT_2);
        0.5 * (1.0 + crate::special::erf(z))
    }

    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / (self.sigma * std::f64::consts::SQRT_2);
        0.5 * crate::special::erfc(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_squared_survival_matches_erfc_df1() {
        // df = 1: sf(x) = erfc(sqrt(x/2)).
        let chi = ChiSquared::new(1.0).unwrap();
        let p = chi.sf(2.0);
        assert!((p - crate::special::erfc(1.0)).abs() < 1e-14);
    }

    #[test]
    fn chi_squared_df2_is_exponential() {
        // df = 2: sf(x) = exp(-x/2).
        let chi = ChiSquared::new(2.0).unwrap();
        assert!((chi.sf(10.0) - (-5.0f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn students_t_cdf_boundaries() {
        let t = StudentsT::new(0.0, 1.0, 4.0).unwrap();
        // Symmetric about zero.
        assert!((t.cdf(1.0) - t.sf(-1.0)).abs() < 1e-14);
        assert!((t.cdf(0.0) - 0.5).abs() < 1e-14);
        // cdf + sf = 1.
        assert!((t.cdf(3.0) + t.sf(3.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn students_t_closed_form_df1() {
        // df = 1 (Cauchy): two-sided p = 1 - 2/π * atan(|t|).
        let t = StudentsT::new(0.0, 1.0, 1.0).unwrap();
        let p = 2.0 * t.sf(2.0);
        let exact = 1.0 - 2.0 * 2.0f64.atan() / std::f64::consts::PI;
        assert!((p - exact).abs() < 1e-12, "p={p}, exact={exact}");
    }

    #[test]
    fn students_t_critical_value_df4() {
        // For df = 4 the two-sided 5% critical value is t = 2.776445.
        let t = StudentsT::new(0.0, 1.0, 4.0).unwrap();
        let p = 2.0 * t.sf(2.776445);
        assert!((p - 0.05).abs() < 1e-6, "two-sided p={p}, expected 0.05");
    }

    #[test]
    fn normal_basics() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((n.pdf(0.0) - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-15);
        assert!((n.mean() - 0.0).abs() < 1e-15);
        assert!((n.variance() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
        assert!(StudentsT::new(0.0, 1.0, 0.0).is_err());
        assert!(StudentsT::new(0.0, 0.0, 1.0).is_err());
        assert!(Normal::new(0.0, 0.0).is_err());
    }
}
