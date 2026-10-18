//! Direct solver for banded systems with partial pivoting.
//!
//! A matrix with `kl` nonzero sub-diagonals and `ku` super-diagonals has
//! `(kl + ku + 1)·n` nonzeros, but a general LU factorisation fills the whole
//! triangle. [`solve_banded`] keeps elimination inside the band, so its cost is
//! `O(n·kl·(kl + ku))` rather than `O(n³)`. Row interchanges widen the upper
//! band to `kl + ku`, which is accounted for below.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{DMatrix, DVector, DenseError};

/// Solve `A x = b` for a banded `A` with `kl` sub-diagonals and `ku`
/// super-diagonals, by Gaussian elimination with partial pivoting.
///
/// `a` is stored as a dense matrix, but only its band is read. Entries outside
/// the declared band must be zero, so a wrongly declared band is reported rather
/// than silently ignored.
///
/// # Errors
///
/// * [`DenseError::DimensionMismatch`] if `a` is not square, `b` has the wrong
///   length, or `a` has a nonzero entry outside the declared band.
/// * [`DenseError::Singular`] if a pivot vanishes or is not finite.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_dense::{solve_banded, DMatrix, DVector};
///
/// // Tridiagonal (kl = ku = 1): [2 1 0; 1 3 1; 0 1 4] x = [1, 2, 2].
/// let a = DMatrix::from_row_slice(3, 3, &[2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]);
/// let b = DVector::from_vec(vec![4.0, 9.0, 10.0]);
/// let x = solve_banded(&a, 1, 1, &b).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12 && (x[2] - 2.0).abs() < 1e-12);
/// ```
pub fn solve_banded(
    a: &DMatrix<f64>,
    kl: usize,
    ku: usize,
    b: &DVector<f64>,
) -> Result<DVector<f64>, DenseError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(DenseError::DimensionMismatch {
            what: format!(
                "solve_banded needs a square matrix and a matching rhs, got {}x{} and {}",
                a.nrows(),
                a.ncols(),
                b.len()
            ),
        });
    }
    // Reject entries outside the declared band instead of silently dropping them.
    for i in 0..n {
        for j in 0..n {
            let inside = j + kl >= i && j <= i + ku;
            if !inside && a[(i, j)] != 0.0 {
                return Err(DenseError::DimensionMismatch {
                    what: format!("entry ({i}, {j}) lies outside the declared band"),
                });
            }
        }
    }
    if n == 0 {
        return Ok(DVector::zeros(0));
    }

    // Working copy in row-major order. Row interchanges can push fill up to
    // `kl + ku` columns right of the diagonal, so the row length is widened.
    let mut w: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[(i, j)]).collect())
        .collect();
    let mut rhs: Vec<f64> = b.iter().copied().collect();

    for k in 0..n {
        // Pivot among the rows that can have a nonzero in column k.
        let last_row = (k + kl).min(n - 1);
        let mut pivot = k;
        for r in k + 1..=last_row {
            if w[r][k].abs() > w[pivot][k].abs() {
                pivot = r;
            }
        }
        if !w[pivot][k].is_finite() || w[pivot][k] == 0.0 {
            return Err(DenseError::Singular {
                what: "solve_banded",
            });
        }
        if pivot != k {
            w.swap(k, pivot);
            rhs.swap(k, pivot);
        }
        let last_col = (k + kl + ku).min(n - 1);
        let pivot_row = w[k].clone();
        for r in k + 1..=last_row {
            let factor = w[r][k] / pivot_row[k];
            if factor == 0.0 {
                continue;
            }
            for c in k..=last_col {
                w[r][c] -= factor * pivot_row[c];
            }
            rhs[r] -= factor * rhs[k];
        }
    }

    // Back substitution over the widened upper band.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let last_col = (i + kl + ku).min(n - 1);
        let mut acc = rhs[i];
        for c in i + 1..=last_col {
            acc -= w[i][c] * x[c];
        }
        if !acc.is_finite() || w[i][i] == 0.0 {
            return Err(DenseError::Singular {
                what: "solve_banded",
            });
        }
        x[i] = acc / w[i][i];
    }
    Ok(DVector::from_vec(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense matrix with the given band, filled with deterministic values
    /// and a dominant diagonal so it is nonsingular.
    fn banded_matrix(n: usize, kl: usize, ku: usize) -> DMatrix<f64> {
        DMatrix::from_fn(n, n, |i, j| {
            let inside = j + kl >= i && j <= i + ku;
            if !inside {
                0.0
            } else if i == j {
                4.0 + i as f64 * 0.1
            } else {
                -1.0 + 0.25 * ((i + 2 * j) % 3) as f64
            }
        })
    }

    fn multiply(a: &DMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(a.nrows(), |i| {
            (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum()
        })
    }

    #[test]
    fn agrees_with_the_dense_solver_for_several_bands() {
        for &(kl, ku) in &[(0usize, 0usize), (1, 1), (2, 1), (1, 3), (4, 4)] {
            let n = 15;
            let a = banded_matrix(n, kl, ku);
            let x_true = DVector::from_fn(n, |i| (i as f64 * 0.7).cos());
            let b = multiply(&a, &x_true);
            let x = solve_banded(&a, kl, ku, &b).unwrap();
            for i in 0..n {
                assert!((x[i] - x_true[i]).abs() < 1e-10, "band ({kl},{ku}) at {i}");
            }
            let dense = a.solve(&b).unwrap();
            for i in 0..n {
                assert!((x[i] - dense[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn pivoting_handles_a_zero_leading_diagonal() {
        // [0 1; 1 0] x = [2, 3] needs a row swap; the band is kl = ku = 1.
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let x = solve_banded(&a, 1, 1, &DVector::from_vec(vec![2.0, 3.0])).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn singular_banded_matrix_is_reported() {
        // Row 3 = row 1 + row 2, so the matrix is singular. Its (2, 0) entry
        // makes the lower bandwidth 2.
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        assert!(matches!(
            solve_banded(&a, 2, 1, &b),
            Err(DenseError::Singular { .. })
        ));
    }

    #[test]
    fn entries_outside_the_band_are_rejected() {
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        // (0, 2) lies two columns right of the diagonal, outside ku = 1.
        assert!(matches!(
            solve_banded(&a, 1, 1, &b),
            Err(DenseError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_mismatched_rhs() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!(solve_banded(&a, 0, 0, &DVector::from_vec(vec![1.0])).is_err());
    }

    #[test]
    fn eigenvector_by_inverse_iteration_for_a_nonsymmetric_matrix() {
        // Eigenvalues 1 and 3 with eigenvectors (1, −1) and (1, 1) respectively.
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (v, r) = a.eigenvector_for_eigenvalue(1.0).unwrap();
        assert!(r < 1e-9);
        assert!((v[0] + v[1]).abs() < 1e-8, "{v:?}");
        assert!(((v[0] * v[0] + v[1] * v[1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_wrong_eigenvalue_is_reported_as_no_convergence() {
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        assert!(matches!(
            a.eigenvector_for_eigenvalue(5.0),
            Err(DenseError::NoConvergence { .. })
        ));
    }
}
