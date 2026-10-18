//! A dense solve that also reports how trustworthy its answer is.

use alloc::format;

use tpt_math_numeric::{SolveReport, SolveStatus};

use crate::{DMatrix, DVector, DenseError};

/// Quality measures for a dense linear solve, from [`DMatrix::solve_with_report`].
#[derive(Clone, Debug, PartialEq)]
pub struct SolveDiagnostics {
    /// `Converged` when the backward error is at the level of rounding, and
    /// `NotConverged` otherwise. There are no iterations, so `iterations` is zero
    /// and `residual` is the Euclidean norm of `b − A x`.
    pub report: SolveReport<f64>,
    /// Normwise backward error `‖b − A x‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞)`. A value near
    /// machine epsilon means `x` solves a system close to `A x = b`.
    pub backward_error: f64,
    /// Condition number estimate `‖A‖₁ · ‖A⁻¹‖₁` in the 1-norm. It is exact in
    /// the 1-norm, not an estimator, because the inverse is formed.
    pub condition_estimate: f64,
    /// `true` when `condition_estimate · ε` exceeds `1e-8`, meaning roughly eight
    /// digits of the answer may be lost to rounding in `A`.
    pub ill_conditioned: bool,
}

/// The threshold on `cond · ε` above which a solve is flagged.
pub const ILL_CONDITIONED_THRESHOLD: f64 = 1e-8;

/// The backward error below which a solve counts as converged.
const BACKWARD_TOLERANCE: f64 = 1e-12;

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

impl DMatrix<f64> {
    /// Solve `A x = b` by LU, and report the residual, backward error and
    /// condition number.
    ///
    /// The condition number needs the inverse, so this costs about `n³` more than
    /// [`solve`](Self::solve). Use it where you want to check an answer, not in a
    /// loop.
    ///
    /// # Errors
    ///
    /// As [`solve`](Self::solve), and [`DenseError::Singular`] if the inverse used
    /// for the condition number does not exist. A singular matrix never reaches
    /// the report.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_linalg_dense::{DMatrix, DVector};
    ///
    /// let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 2.0, 3.0]);
    /// let b = DVector::from_vec(vec![1.0, 2.0]);
    /// let (x, diag) = a.solve_with_report(&b).unwrap();
    /// assert!((x[0] - 0.1).abs() < 1e-12);
    /// assert!(diag.backward_error < 1e-14 && !diag.ill_conditioned);
    /// ```
    pub fn solve_with_report(
        &self,
        b: &DVector<f64>,
    ) -> Result<(DVector<f64>, SolveDiagnostics), DenseError> {
        let n = self.nrows();
        if self.ncols() != n || b.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!(
                    "solve_with_report needs a square matrix and a matching rhs, got {}x{} and {}",
                    self.nrows(),
                    self.ncols(),
                    b.len()
                ),
            });
        }
        let x = self.solve(b)?;
        let ax = DVector::from_fn(n, |i| (0..n).map(|j| self[(i, j)] * x[j]).sum::<f64>());
        let r = DVector::from_fn(n, |i| b[i] - ax[i]);

        let a_inf = (0..n)
            .map(|i| (0..n).map(|j| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let denom = a_inf * norm_inf(x.as_slice()) + norm_inf(b.as_slice());
        let backward_error = if denom > 0.0 {
            norm_inf(r.as_slice()) / denom
        } else {
            0.0
        };

        let a_one = (0..n)
            .map(|j| (0..n).map(|i| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let inv = self.inverse()?;
        let inv_one = (0..n)
            .map(|j| (0..n).map(|i| inv[(i, j)].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let condition_estimate = a_one * inv_one;
        let ill_conditioned = condition_estimate * f64::EPSILON > ILL_CONDITIONED_THRESHOLD;

        let residual = r.iter().map(|v| v * v).sum::<f64>();
        let residual = crate::sqrt_f64(residual);
        let status = if backward_error <= BACKWARD_TOLERANCE {
            SolveStatus::Converged
        } else {
            SolveStatus::NotConverged
        };
        Ok((
            x,
            SolveDiagnostics {
                report: SolveReport {
                    status,
                    iterations: 0,
                    residual,
                },
                backward_error,
                condition_estimate,
                ill_conditioned,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn well_conditioned_system_is_clean() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let (x, d) = a.solve_with_report(&b).unwrap();
        assert_eq!(d.report.status, SolveStatus::Converged);
        assert!(!d.ill_conditioned);
        assert!(d.condition_estimate < 100.0);
        let ax = DVector::from_fn(3, |i| (0..3).map(|j| a[(i, j)] * x[j]).sum::<f64>());
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn hilbert_matrix_is_flagged_as_ill_conditioned() {
        let n = 7;
        let h = DMatrix::from_fn(n, n, |i, j| 1.0 / (i + j + 1) as f64);
        let b = DVector::from_fn(n, |_| 1.0);
        let (_, d) = h.solve_with_report(&b).unwrap();
        // The Hilbert matrix of order 7 has a condition number near 5·10⁸.
        assert!(d.condition_estimate > 1e8, "cond {}", d.condition_estimate);
        assert!(d.ill_conditioned);
    }

    #[test]
    fn rejects_mismatched_shapes_and_singular_input() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        assert!(a.solve_with_report(&b).is_err());
        let short = DVector::from_vec(vec![1.0]);
        assert!(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])
            .solve_with_report(&short)
            .is_err());
    }
}
