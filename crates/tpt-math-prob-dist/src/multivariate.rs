//! The multivariate normal distribution.
//!
//! The covariance is factored once, with a Cholesky factor from
//! `tpt-math-linalg-dense`. The density then needs one triangular solve per point
//! and the log-determinant is read off the diagonal of the factor.

use alloc::vec;
use alloc::vec::Vec;

use core::f64::consts::TAU;

use tpt_math_linalg_dense::{Cholesky, DMatrix};

/// A multivariate normal `N(μ, Σ)` with a positive-definite covariance `Σ`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_dist::multivariate::MultivariateNormal;
/// use tpt_math_linalg_dense::DMatrix;
///
/// let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
/// let mvn = MultivariateNormal::new(&[0.0, 0.0], &cov).unwrap();
/// // Standard bivariate normal at the origin: 1 / (2π).
/// assert!((mvn.pdf(&[0.0, 0.0]) - 1.0 / (2.0 * std::f64::consts::PI)).abs() < 1e-14);
/// ```
#[derive(Clone, Debug)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    chol: Cholesky,
}

impl MultivariateNormal {
    /// Build the distribution from a mean and a covariance matrix.
    ///
    /// # Errors
    ///
    /// Returns a message if the covariance is not square, its size does not match
    /// the mean, or it is not positive definite.
    pub fn new(mean: &[f64], cov: &DMatrix<f64>) -> Result<Self, &'static str> {
        let k = mean.len();
        if cov.nrows() != k || cov.ncols() != k {
            return Err("covariance must be square with one row per mean component");
        }
        if mean.iter().any(|v| !v.is_finite()) {
            return Err("mean must be finite");
        }
        let chol = Cholesky::new(cov).map_err(|_| "covariance is not positive definite")?;
        Ok(Self {
            mean: mean.to_vec(),
            chol,
        })
    }

    /// The dimension `k` of the distribution.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// The mean vector `μ`.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Log-density at `x`:
    /// `−(k/2) ln 2π − ½ ln det Σ − ½ (x − μ)ᵀ Σ⁻¹ (x − μ)`.
    ///
    /// Returns `NaN` if `x` has the wrong length.
    pub fn log_pdf(&self, x: &[f64]) -> f64 {
        let k = self.dim();
        if x.len() != k {
            return f64::NAN;
        }
        // With Σ = L Lᵀ, the quadratic form is |L⁻¹ (x − μ)|².
        let l = self.chol.factor();
        let diff: Vec<f64> = x.iter().zip(&self.mean).map(|(a, m)| a - m).collect();
        let mut y = vec![0.0_f64; k];
        for i in 0..k {
            let dot: f64 = (0..i).map(|j| l[(i, j)] * y[j]).sum();
            y[i] = (diff[i] - dot) / l[(i, i)];
        }
        let mahalanobis: f64 = y.iter().map(|v| v * v).sum();
        let ln_det: f64 = (0..k).map(|i| 2.0 * l[(i, i)].ln()).sum();
        -0.5 * (k as f64 * TAU.ln() + ln_det + mahalanobis)
    }

    /// Density at `x`, `exp(log_pdf(x))`.
    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.log_pdf(x).exp()
    }

    /// The covariance `Σ = L Lᵀ` rebuilt from the factor.
    pub fn covariance(&self) -> DMatrix<f64> {
        let l = self.chol.factor();
        let k = self.dim();
        DMatrix::from_fn(k, k, |i, j| (0..k).map(|t| l[(i, t)] * l[(j, t)]).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bivariate_density_matches_the_closed_form() {
        // Σ = [[2, 0.6], [0.6, 1]], det Σ = 1.64, Σ⁻¹ = (1/1.64) [[1, −0.6], [−0.6, 2]].
        let cov = DMatrix::from_row_slice(2, 2, &[2.0, 0.6, 0.6, 1.0]);
        let mvn = MultivariateNormal::new(&[0.0, 0.0], &cov).unwrap();
        let (x0, x1) = (0.5_f64, -0.2_f64);
        let quad = (1.0 * x0 * x0 - 2.0 * 0.6 * x0 * x1 + 2.0 * x1 * x1) / 1.64;
        let want = -TAU.ln() - 0.5 * 1.64_f64.ln() - 0.5 * quad;
        assert!((mvn.log_pdf(&[x0, x1]) - want).abs() < 1e-12);
    }

    #[test]
    fn one_dimension_matches_the_normal_density() {
        let cov = DMatrix::from_row_slice(1, 1, &[2.25]);
        let mvn = MultivariateNormal::new(&[1.0], &cov).unwrap();
        let x = 2.4_f64;
        let sd = 1.5_f64;
        let z = (x - 1.0) / sd;
        let want = (-0.5 * z * z).exp() / (sd * TAU.sqrt());
        assert!((mvn.pdf(&[x]) - want).abs() < 1e-14);
    }

    #[test]
    fn covariance_round_trips_and_bad_inputs_are_rejected() {
        let cov = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let mvn = MultivariateNormal::new(&[0.0, 0.0], &cov).unwrap();
        let back = mvn.covariance();
        for i in 0..2 {
            for j in 0..2 {
                assert!((back[(i, j)] - cov[(i, j)]).abs() < 1e-12);
            }
        }
        assert!(MultivariateNormal::new(&[0.0], &cov).is_err());
        let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(MultivariateNormal::new(&[0.0, 0.0], &indefinite).is_err());
        assert!(mvn.log_pdf(&[1.0]).is_nan());
    }
}
