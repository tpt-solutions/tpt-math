//! Reusable factorisations: a Cholesky factor that solves many right-hand
//! sides, and a tridiagonal solver.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{DMatrix, DVector, DenseError};

/// A Cholesky factorisation `A = L Lᵀ` of a symmetric positive-definite matrix,
/// kept for repeated solves.
///
/// Factor once with [`Cholesky::new`], then call [`Cholesky::solve`] for each
/// right-hand side at O(n²) per solve.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_dense::{Cholesky, DMatrix, DVector};
///
/// let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
/// let chol = Cholesky::new(&a).unwrap();
/// let x = chol.solve(&DVector::from_vec(vec![2.0, 1.0])).unwrap();
/// assert!((x[0] - 0.5).abs() < 1e-12 && (x[1] - 0.0).abs() < 1e-12);
/// assert!((chol.det() - 8.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct Cholesky {
    l: DMatrix<f64>,
}

impl Cholesky {
    /// Factor the symmetric positive-definite matrix `a`. Only its lower triangle
    /// is read.
    ///
    /// # Errors
    ///
    /// As [`DMatrix::cholesky`].
    pub fn new(a: &DMatrix<f64>) -> Result<Self, DenseError> {
        Ok(Cholesky { l: a.cholesky()? })
    }

    /// The lower-triangular factor `L`.
    pub fn factor(&self) -> &DMatrix<f64> {
        &self.l
    }

    /// The dimension `n` of the factored matrix.
    pub fn dim(&self) -> usize {
        self.l.nrows()
    }

    /// Solve `A x = b` by forward substitution with `L`, then back substitution
    /// with `Lᵀ`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `b` does not have `dim()`
    /// entries.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, DenseError> {
        let n = self.dim();
        if b.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("rhs has length {}, expected {n}", b.len()),
            });
        }
        let l = &self.l;
        let mut y = vec![0.0; n];
        for i in 0..n {
            let s: f64 = (0..i).map(|k| l[(i, k)] * y[k]).sum();
            y[i] = (b[i] - s) / l[(i, i)];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let s: f64 = ((i + 1)..n).map(|k| l[(k, i)] * x[k]).sum();
            x[i] = (y[i] - s) / l[(i, i)];
        }
        Ok(DVector::from_vec(x))
    }

    /// Determinant `det A = (∏ L_ii)²`.
    pub fn det(&self) -> f64 {
        let prod: f64 = (0..self.dim()).map(|i| self.l[(i, i)]).product();
        prod * prod
    }
}

/// Solve a tridiagonal system `T x = rhs` with the Thomas algorithm in O(n).
///
/// `sub` is the subdiagonal (length `n − 1`), `diag` the main diagonal (length
/// `n`), and `sup` the superdiagonal (length `n − 1`). No pivoting is done, so
/// the method is reliable for diagonally dominant or symmetric positive-definite
/// systems, which is the usual case for discretised differential operators.
///
/// # Errors
///
/// Returns [`DenseError::DimensionMismatch`] for inconsistent lengths, and
/// [`DenseError::Singular`] when a pivot becomes zero or non-finite.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_dense::solve_tridiagonal;
///
/// // [2 1 0; 1 2 1; 0 1 2] x = [4, 8, 8] has solution x = [1, 2, 3].
/// let x = solve_tridiagonal(&[1.0, 1.0], &[2.0, 2.0, 2.0], &[1.0, 1.0], &[4.0, 8.0, 8.0])
///     .unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12 && (x[2] - 3.0).abs() < 1e-12);
/// ```
pub fn solve_tridiagonal(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, DenseError> {
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if sub.len() != n - 1 || sup.len() != n - 1 || rhs.len() != n {
        return Err(DenseError::DimensionMismatch {
            what: format!(
                "tridiagonal system of size {n} needs {} sub/super entries and {n} rhs entries",
                n - 1
            ),
        });
    }
    let singular = || DenseError::Singular {
        what: "solve_tridiagonal",
    };
    // Forward sweep: c' and d' replace the superdiagonal and right-hand side.
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut denom = diag[0];
    if denom == 0.0 || !denom.is_finite() {
        return Err(singular());
    }
    if n > 1 {
        c_prime[0] = sup[0] / denom;
    }
    d_prime[0] = rhs[0] / denom;
    for i in 1..n {
        denom = diag[i] - sub[i - 1] * c_prime[i - 1];
        if denom == 0.0 || !denom.is_finite() {
            return Err(singular());
        }
        if i < n - 1 {
            c_prime[i] = sup[i] / denom;
        }
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }
    // Back substitution.
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_and_matches_lu() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0]);
        let chol = Cholesky::new(&a).unwrap();
        let b = DVector::from_vec(vec![1.0, -2.0, 0.5]);
        let x = chol.solve(&b).unwrap();
        let want = a.solve(&b).unwrap();
        for i in 0..3 {
            assert!((x[i] - want[i]).abs() < 1e-12);
        }
        assert!((chol.det() - a.det().unwrap()).abs() < 1e-10);
        assert!(chol.solve(&DVector::from_vec(vec![1.0])).is_err());
        // Reusing the factor for a second right-hand side.
        let y = chol.solve(&DVector::from_vec(vec![0.0, 1.0, 0.0])).unwrap();
        let back = a.clone() * y.clone();
        assert!((back[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(Cholesky::new(&a).is_err());
    }

    #[test]
    fn tridiagonal_matches_dense_solve() {
        let n = 6;
        let sub: Vec<f64> = (0..n - 1).map(|i| -1.0 + 0.1 * i as f64).collect();
        let sup: Vec<f64> = (0..n - 1).map(|i| -0.5 - 0.05 * i as f64).collect();
        let diag: Vec<f64> = (0..n).map(|i| 3.0 + i as f64).collect();
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let x = solve_tridiagonal(&sub, &diag, &sup, &rhs).unwrap();
        let dense = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                diag[i]
            } else if i == j + 1 {
                sub[j]
            } else if j == i + 1 {
                sup[i]
            } else {
                0.0
            }
        });
        let want = dense.solve(&DVector::from_vec(rhs)).unwrap();
        for i in 0..n {
            assert!((x[i] - want[i]).abs() < 1e-12, "component {i}");
        }
    }

    #[test]
    fn tridiagonal_edge_cases() {
        assert_eq!(
            solve_tridiagonal(&[], &[], &[], &[]).unwrap(),
            Vec::<f64>::new()
        );
        assert_eq!(
            solve_tridiagonal(&[], &[4.0], &[], &[2.0]).unwrap(),
            vec![0.5]
        );
        assert!(matches!(
            solve_tridiagonal(&[1.0], &[0.0, 1.0], &[1.0], &[1.0, 1.0]),
            Err(DenseError::Singular { .. })
        ));
        assert!(matches!(
            solve_tridiagonal(&[1.0, 1.0], &[2.0, 2.0], &[1.0], &[1.0, 1.0]),
            Err(DenseError::DimensionMismatch { .. })
        ));
    }
}
