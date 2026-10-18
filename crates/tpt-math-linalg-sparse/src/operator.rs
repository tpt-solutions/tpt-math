//! Matrix-free linear operators, a conjugate-gradient entry point that takes
//! one, and a Lanczos eigensolver for symmetric operators.
//!
//! A [`LinearOperator`] only has to apply itself to a vector. That lets a
//! caller solve or diagnose a system whose matrix is never stored, such as a
//! discretised operator applied through a stencil, or a product of factors.
//! [`CsrMatrix`] implements the trait, so the same code works for stored
//! matrices.

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_numeric::{Float, Scalar, SolveReport};

use crate::{cg_core, CsrMatrix, SparseError};

/// A linear map `y = A x` on vectors of length [`ncols`](Self::ncols), giving
/// vectors of length [`nrows`](Self::nrows).
pub trait LinearOperator<T> {
    /// The number of rows of the operator (the length of each output).
    fn nrows(&self) -> usize;
    /// The number of columns of the operator (the length of each input).
    fn ncols(&self) -> usize;
    /// Write `A·x` into `y`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `x` or `y` has the wrong
    /// length.
    fn apply(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError>;
}

impl<T: Scalar> LinearOperator<T> for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        CsrMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        CsrMatrix::ncols(self)
    }

    fn apply(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        self.matvec_into(x, y)
    }
}

/// A square operator defined by a closure, with no stored matrix.
///
/// Build one with [`matrix_free`].
pub struct MatrixFree<T, F> {
    n: usize,
    apply: F,
    _scalar: core::marker::PhantomData<fn(T)>,
}

impl<T, F> core::fmt::Debug for MatrixFree<T, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MatrixFree")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

/// Wrap a closure `apply(x, y)`, which writes `A·x` into `y`, as an `n × n`
/// [`LinearOperator`].
///
/// The closure must write every entry of `y`. The operator is assumed to be
/// square, since the Krylov solvers in this crate need that.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_dense::DVector;
/// use tpt_math_linalg_sparse::{conjugate_gradient_op, matrix_free};
///
/// // The 1-D Laplacian stencil [-1, 2, -1], applied without storing a matrix.
/// let n = 20;
/// let op = matrix_free(n, |x: &[f64], y: &mut [f64]| {
///     let m = x.len();
///     for i in 0..m {
///         let left = if i > 0 { x[i - 1] } else { 0.0 };
///         let right = if i + 1 < m { x[i + 1] } else { 0.0 };
///         y[i] = 2.0 * x[i] - left - right;
///     }
/// });
/// let b = DVector::from_fn(n, |_| 1.0);
/// let (x, _) = conjugate_gradient_op(&op, &b, None, 1e-12, 200).unwrap();
/// // Check the residual of the returned solution.
/// let mut ax = vec![0.0; n];
/// use tpt_math_linalg_sparse::LinearOperator;
/// op.apply(x.as_slice(), &mut ax).unwrap();
/// assert!((0..n).all(|i| (ax[i] - b[i]).abs() < 1e-8));
/// ```
pub fn matrix_free<T, F>(n: usize, apply: F) -> MatrixFree<T, F>
where
    F: Fn(&[T], &mut [T]),
{
    MatrixFree {
        n,
        apply,
        _scalar: core::marker::PhantomData,
    }
}

impl<T, F> LinearOperator<T> for MatrixFree<T, F>
where
    F: Fn(&[T], &mut [T]),
{
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(SparseError::DimensionMismatch);
        }
        (self.apply)(x, y);
        Ok(())
    }
}

/// Conjugate gradients for a symmetric positive-definite operator.
///
/// The same method as [`conjugate_gradient_with_report`](crate::conjugate_gradient_with_report),
/// but `A` is any [`LinearOperator`]. The tolerance is relative, as there:
/// the iteration stops when `‖r‖ ≤ tol·‖b‖`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the operator is not square, if
/// `b` does not match it, or if `x0` has the wrong length. A solver that fails
/// to converge returns its last iterate with a non-converged report.
pub fn conjugate_gradient_op<T, Op>(
    op: &Op,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<(DVector<T>, SolveReport<T>), SparseError>
where
    T: Scalar + Copy + 'static,
    Op: LinearOperator<T>,
{
    let n = op.nrows();
    if op.ncols() != n || b.len() != n || x0.as_ref().is_some_and(|v| v.len() != n) {
        return Err(SparseError::DimensionMismatch);
    }
    cg_core(n, |p, ap| op.apply(p, ap), b, x0, tol, max_iter)
}

/// BiCGSTAB for a general square operator.
///
/// The same method as [`bicgstab_with_report`](crate::bicgstab_with_report),
/// with `A` given as a [`LinearOperator`].
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the operator is not square, if
/// `b` does not match it, or if `x0` has the wrong length.
pub fn bicgstab_op<T, Op>(
    op: &Op,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<(DVector<T>, SolveReport<T>), SparseError>
where
    T: Scalar + Copy + 'static,
    Op: LinearOperator<T>,
{
    let n = op.nrows();
    if op.ncols() != n || b.len() != n || x0.as_ref().is_some_and(|v| v.len() != n) {
        return Err(SparseError::DimensionMismatch);
    }
    crate::bicgstab_core(n, |p, out| op.apply(p, out), b, x0, tol, max_iter)
}

/// Restarted, right-preconditioned GMRES for a general square operator.
///
/// The same method as [`gmres_right_preconditioned_with_report`](crate::gmres_right_preconditioned_with_report),
/// with `A` given as a [`LinearOperator`].
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] for incompatible dimensions, and
/// propagates any error from `precond`.
#[allow(clippy::too_many_arguments)]
pub fn gmres_op<T, Op, P>(
    op: &Op,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    restart: usize,
    max_iter: usize,
    precond: P,
) -> Result<(DVector<T>, SolveReport<T>), SparseError>
where
    T: Scalar + Copy + 'static,
    Op: LinearOperator<T>,
    P: Fn(&[T]) -> Result<Vec<T>, SparseError>,
{
    let n = op.nrows();
    if op.ncols() != n || b.len() != n {
        return Err(SparseError::DimensionMismatch);
    }
    crate::gmres_core(
        n,
        |p, out| op.apply(p, out),
        b,
        x0,
        tol,
        restart,
        max_iter,
        precond,
    )
}

/// Which end of the spectrum [`lanczos`] should return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    /// The algebraically largest eigenvalues.
    Largest,
    /// The algebraically smallest eigenvalues.
    Smallest,
}

/// Approximate eigenpairs of a symmetric operator from a Lanczos run.
#[derive(Clone, Debug, PartialEq)]
pub struct LanczosEigenpairs {
    /// Ritz values, in ascending order.
    pub values: Vec<f64>,
    /// Ritz vectors, one per value, unit length and in the same order.
    pub vectors: Vec<DVector<f64>>,
    /// Residual estimates `|β_m · s_{m,i}|` for each Ritz pair. A value is
    /// accurate to roughly this size when the Krylov space has converged to it.
    pub residuals: Vec<f64>,
    /// The Krylov dimension actually built. It is smaller than requested when
    /// the space became invariant early (an exact eigen-subspace was found).
    pub krylov_dim: usize,
}

/// Approximate `k` eigenpairs of a symmetric operator by the Lanczos method.
///
/// Builds a Krylov subspace of dimension `steps` (capped at the operator size)
/// with full reorthogonalization, so the basis stays orthogonal in floating
/// point. The eigenvalues of the resulting tridiagonal matrix give the Ritz
/// values. The extreme ones converge first, so `which` picks the end you need.
///
/// The starting vector is deterministic, so repeated calls give identical
/// results. Choose `steps` a few times larger than `k`. When `steps` equals the
/// operator size, the eigenvalues are exact up to rounding.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the operator is not square, or
/// if `k` is zero or larger than the operator size.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{lanczos, matrix_free, Which};
///
/// // Eigenvalues of the 1-D Laplacian: 2 − 2 cos(kπ/(n+1)).
/// let n = 30;
/// let op = matrix_free(n, |x: &[f64], y: &mut [f64]| {
///     let m = x.len();
///     for i in 0..m {
///         let left = if i > 0 { x[i - 1] } else { 0.0 };
///         let right = if i + 1 < m { x[i + 1] } else { 0.0 };
///         y[i] = 2.0 * x[i] - left - right;
///     }
/// });
/// let pairs = lanczos(&op, 2, n, Which::Smallest).unwrap();
/// let exact = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
/// assert!((pairs.values[0] - exact).abs() < 1e-10);
/// ```
pub fn lanczos<Op>(
    op: &Op,
    k: usize,
    steps: usize,
    which: Which,
) -> Result<LanczosEigenpairs, SparseError>
where
    Op: LinearOperator<f64>,
{
    let n = op.nrows();
    if op.ncols() != n || k == 0 || k > n {
        return Err(SparseError::DimensionMismatch);
    }
    let m_max = steps.clamp(k, n);

    // Deterministic start: a positive ramp, so it has a component along most
    // eigenvectors and the result does not depend on a random seed.
    let mut v = DVector::from_fn(n, |i| 1.0 + i as f64 / n as f64);
    let norm0 = norm(&v);
    v = DVector::from_fn(n, |i| v[i] / norm0);

    let mut basis: Vec<DVector<f64>> = Vec::with_capacity(m_max);
    let mut alpha: Vec<f64> = Vec::with_capacity(m_max);
    // beta[j] couples basis vectors j and j+1.
    let mut beta: Vec<f64> = Vec::with_capacity(m_max);
    let mut w = DVector::zeros(n);
    // β_m: the coupling beyond the last basis vector, used by the residual estimate.
    let mut residual_beta = 0.0;

    for j in 0..m_max {
        basis.push(v.clone());
        op.apply(v.as_slice(), w.as_mut_slice())?;
        let a_j = dot(&v, &w);
        alpha.push(a_j);

        // Three-term recurrence, then two rounds of full reorthogonalization.
        let mut r = DVector::from_fn(n, |i| w[i] - a_j * v[i]);
        for _ in 0..2 {
            for q in &basis {
                let c = dot(q, &r);
                r = DVector::from_fn(n, |i| r[i] - c * q[i]);
            }
        }

        let b_j = norm(&r);
        residual_beta = b_j;
        if j + 1 == m_max {
            break;
        }
        // An (essentially) zero coupling means the space is invariant: stop.
        if b_j <= 1e-12 * (a_j.abs() + 1.0) {
            break;
        }
        beta.push(b_j);
        v = DVector::from_fn(n, |i| r[i] / b_j);
    }

    let m = alpha.len();
    // Symmetric tridiagonal: diagonal α, off-diagonals β.
    let t = DMatrix::from_fn(m, m, |i, j| {
        if i == j {
            alpha[i]
        } else if j == i + 1 {
            beta[i]
        } else if i == j + 1 {
            beta[j]
        } else {
            0.0
        }
    });
    let (ritz, s) = t
        .symmetric_eigen()
        .map_err(|_| SparseError::DimensionMismatch)?;

    // Sort ascending, then take the requested end. Fewer than `k` pairs come
    // back only when the Krylov space became invariant before reaching `k`.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| ritz[a].total_cmp(&ritz[b]));
    let take = k.min(m);
    let chosen: Vec<usize> = match which {
        Which::Smallest => order[..take].to_vec(),
        Which::Largest => order[m - take..].to_vec(),
    };

    // β_m is the coupling beyond the last basis vector; the residual estimate
    // uses it. When the space became invariant it is zero, so the Ritz pairs are
    // exact.
    let mut values = Vec::with_capacity(chosen.len());
    let mut vectors = Vec::with_capacity(chosen.len());
    let mut residuals = Vec::with_capacity(chosen.len());
    for &idx in &chosen {
        // Ritz vector: Σ_j s_{j,idx} · basis_j.
        let mut x = DVector::zeros(n);
        for (j, q) in basis.iter().enumerate().take(m) {
            let c = s[(j, idx)];
            x = DVector::from_fn(n, |i| x[i] + c * q[i]);
        }
        let len = norm(&x);
        let x = if len > 0.0 {
            DVector::from_fn(n, |i| x[i] / len)
        } else {
            x
        };
        values.push(ritz[idx]);
        vectors.push(x);
        residuals.push((residual_beta * s[(m - 1, idx)]).abs());
    }

    Ok(LanczosEigenpairs {
        values,
        vectors,
        residuals,
        krylov_dim: m,
    })
}

fn dot(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &DVector<f64>) -> f64 {
    Float::sqrt(dot(a, a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooMatrix;
    use alloc::vec;

    /// The n × n 1-D Laplacian stencil as a matrix-free operator.
    fn laplacian(n: usize) -> impl LinearOperator<f64> {
        matrix_free(n, |x: &[f64], y: &mut [f64]| {
            let m = x.len();
            for i in 0..m {
                let left = if i > 0 { x[i - 1] } else { 0.0 };
                let right = if i + 1 < m { x[i + 1] } else { 0.0 };
                y[i] = 2.0 * x[i] - left - right;
            }
        })
    }

    fn exact_laplacian_eigenvalue(n: usize, k: usize) -> f64 {
        2.0 - 2.0 * (k as f64 * core::f64::consts::PI / (n as f64 + 1.0)).cos()
    }

    #[test]
    fn full_krylov_space_gives_every_eigenvalue() {
        let n = 12;
        let pairs = lanczos(&laplacian(n), n, n, Which::Smallest).unwrap();
        assert_eq!(pairs.values.len(), n);
        for (k, &lambda) in pairs.values.iter().enumerate() {
            let want = exact_laplacian_eigenvalue(n, k + 1);
            assert!((lambda - want).abs() < 1e-10, "λ_{k}: {lambda} vs {want}");
        }
    }

    #[test]
    fn largest_end_is_returned_in_ascending_order() {
        let n = 12;
        let pairs = lanczos(&laplacian(n), 3, n, Which::Largest).unwrap();
        assert_eq!(pairs.values.len(), 3);
        assert!(pairs.values.windows(2).all(|w| w[0] < w[1]));
        let top = exact_laplacian_eigenvalue(n, n);
        assert!((pairs.values[2] - top).abs() < 1e-10);
    }

    #[test]
    fn ritz_vectors_are_unit_eigenvectors_when_converged() {
        let n = 20;
        let op = laplacian(n);
        let pairs = lanczos(&op, 2, n, Which::Smallest).unwrap();
        for (lambda, v) in pairs.values.iter().zip(&pairs.vectors) {
            let mut av = vec![0.0; n];
            op.apply(v.as_slice(), &mut av).unwrap();
            let residual: f64 = (0..n)
                .map(|i| (av[i] - lambda * v[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual < 1e-9, "residual {residual}");
            let len: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((len - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn residual_estimates_bound_the_error_of_a_partial_run() {
        // With fewer steps than the dimension the extreme value is approximate,
        // and the reported residual should be an honest estimate of its error.
        let n = 60;
        let pairs = lanczos(&laplacian(n), 1, 25, Which::Smallest).unwrap();
        let err = (pairs.values[0] - exact_laplacian_eigenvalue(n, 1)).abs();
        assert!(
            err <= pairs.residuals[0].max(1e-12) * 10.0 + 1e-10,
            "err {err}"
        );
        assert_eq!(pairs.krylov_dim, 25);
    }

    #[test]
    fn invalid_requests_are_rejected() {
        let op = laplacian(5);
        assert!(lanczos(&op, 0, 5, Which::Smallest).is_err());
        assert!(lanczos(&op, 6, 6, Which::Smallest).is_err());
    }

    #[test]
    fn operator_cg_matches_the_csr_path() {
        let n = 8;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        let csr = coo.to_csr();
        let b = DVector::from_fn(n, |i| i as f64 - 3.0);
        let (via_op, _) = conjugate_gradient_op(&csr, &b, None, 1e-12, 500).unwrap();
        let via_csr = crate::conjugate_gradient(&csr, &b, None, 1e-12, 500).unwrap();
        for i in 0..n {
            assert!((via_op[i] - via_csr[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn operator_cg_rejects_non_square_and_bad_lengths() {
        let op = matrix_free(3, |x: &[f64], y: &mut [f64]| y.copy_from_slice(x));
        let short = DVector::from_fn(2, |_| 1.0);
        assert!(conjugate_gradient_op(&op, &short, None, 1e-8, 10).is_err());
        let b = DVector::from_fn(3, |_| 1.0);
        assert!(conjugate_gradient_op(&op, &b, Some(short), 1e-8, 10).is_err());
    }
}

/// Approximate eigenvalues of a general operator from an Arnoldi run.
#[derive(Clone, Debug, PartialEq)]
pub struct ArnoldiRitz {
    /// Ritz values sorted by decreasing modulus. Complex values come in
    /// conjugate pairs.
    pub values: Vec<tpt_math_linalg_dense::Eigenvalue>,
    /// Residual estimates `|h_{m+1,m}·y_m|` for the real Ritz values, `None`
    /// for complex ones (their eigenvectors are not formed here).
    pub residuals: Vec<Option<f64>>,
    /// The Krylov dimension actually built.
    pub krylov_dim: usize,
}

/// Approximate the `k` eigenvalues of largest modulus of a general operator by
/// the Arnoldi method.
///
/// Builds an orthonormal Krylov basis of dimension `steps` (capped at the
/// operator size) with full reorthogonalization, and takes the eigenvalues of
/// the resulting upper Hessenberg matrix as Ritz values. The start vector is
/// deterministic.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the operator is not square, or if
/// `k` is zero or larger than the operator size, and
/// [`SparseError::NotConverged`] if the Hessenberg eigenvalue iteration fails.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{arnoldi, matrix_free};
///
/// // A non-symmetric upper bidiagonal operator, diagonal 1..=n: the spectrum
/// // is {1, …, n}, so the largest-modulus Ritz value is n.
/// let n = 8;
/// let op = matrix_free(n, |x: &[f64], y: &mut [f64]| {
///     for i in 0..x.len() {
///         y[i] = (i as f64 + 1.0) * x[i] + if i + 1 < x.len() { 0.5 * x[i + 1] } else { 0.0 };
///     }
/// });
/// let ritz = arnoldi(&op, 1, n).unwrap();
/// assert!((ritz.values[0].re - n as f64).abs() < 1e-8);
/// ```
pub fn arnoldi<Op>(op: &Op, k: usize, steps: usize) -> Result<ArnoldiRitz, SparseError>
where
    Op: LinearOperator<f64>,
{
    let n = op.nrows();
    if op.ncols() != n || k == 0 || k > n {
        return Err(SparseError::DimensionMismatch);
    }
    let m_max = steps.clamp(k, n);

    let mut v = DVector::from_fn(n, |i| 1.0 + i as f64 / n as f64);
    let norm0 = norm(&v);
    v = DVector::from_fn(n, |i| v[i] / norm0);

    let mut basis: Vec<DVector<f64>> = Vec::with_capacity(m_max);
    // Upper Hessenberg matrix, row-major: h[i][j] for i <= j + 1.
    let mut h = vec![vec![0.0_f64; m_max]; m_max];
    let mut w = DVector::zeros(n);
    // Subdiagonal entry h[m][m-1] that couples to the next (unbuilt) vector.
    let mut last_sub = 0.0_f64;
    let mut m = m_max;

    for j in 0..m_max {
        basis.push(v.clone());
        op.apply(v.as_slice(), w.as_mut_slice())?;
        let mut r = w.clone();
        // Modified Gram–Schmidt against the whole basis, twice for stability.
        // The first pass defines column j of the Hessenberg matrix.
        for pass in 0..2 {
            for (i, q) in basis.iter().enumerate() {
                let c = dot(q, &r);
                if pass == 0 {
                    h[i][j] += c;
                }
                r = DVector::from_fn(n, |t| r[t] - c * q[t]);
            }
        }
        let beta = norm(&r);
        if j + 1 == m_max {
            last_sub = beta;
            break;
        }
        if beta <= 1e-12 * (1.0 + norm(&w)) {
            // Invariant subspace: the Ritz values are exact eigenvalues.
            m = j + 1;
            break;
        }
        h[j + 1][j] = beta;
        v = DVector::from_fn(n, |t| r[t] / beta);
    }

    let hm = DMatrix::from_fn(m, m, |i, j| h[i][j]);
    let values_raw = hm
        .eigenvalues()
        .map_err(|_| SparseError::NotConverged { iterations: m })?;
    let mut ranked: Vec<tpt_math_linalg_dense::Eigenvalue> = values_raw;
    ranked.sort_by(|a, b| b.re.hypot(b.im).total_cmp(&a.re.hypot(a.im)));
    ranked.truncate(k.min(m));

    let mut residuals = Vec::with_capacity(ranked.len());
    for e in &ranked {
        let residual = if !e.is_real() {
            None
        } else if last_sub == 0.0 {
            Some(0.0)
        } else {
            hm.eigenvector_for_eigenvalue(e.re)
                .ok()
                .map(|(y, _)| (last_sub * y[m - 1]).abs())
        };
        residuals.push(residual);
    }
    Ok(ArnoldiRitz {
        values: ranked,
        residuals,
        krylov_dim: m,
    })
}

#[cfg(test)]
mod arnoldi_tests {
    use super::*;
    use crate::CooMatrix;

    #[test]
    fn arnoldi_full_dimension_recovers_a_nonsymmetric_spectrum() {
        // Upper bidiagonal with diagonal 1..=n: the spectrum is 1..=n.
        let n = 7;
        let op = matrix_free(n, |x: &[f64], y: &mut [f64]| {
            for i in 0..x.len() {
                y[i] = (i as f64 + 1.0) * x[i] + if i + 1 < x.len() { 0.5 * x[i + 1] } else { 0.0 };
            }
        });
        let ritz = arnoldi(&op, n, n).unwrap();
        assert_eq!(ritz.values.len(), n);
        let mut re: Vec<f64> = ritz.values.iter().map(|e| e.re).collect();
        re.sort_by(|a, b| a.total_cmp(b));
        for (k, r) in re.iter().enumerate() {
            assert!((r - (k as f64 + 1.0)).abs() < 1e-8, "{re:?}");
        }
        assert!(ritz.values.iter().all(|e| e.im.abs() < 1e-8));
    }

    #[test]
    fn arnoldi_finds_the_complex_pair_of_largest_modulus() {
        // Block diagonal of rotation-scaled 2×2 blocks: moduli 3, 2 and 1, with
        // the modulus-3 pair 3·e^{±iπ/4} as the dominant one.
        let scale = [3.0, 2.0, 1.0];
        let c = core::f64::consts::FRAC_1_SQRT_2;
        let op = matrix_free(6, move |x: &[f64], y: &mut [f64]| {
            for b in 0..3 {
                let (p, q) = (x[2 * b], x[2 * b + 1]);
                let s = scale[b];
                // [[c, −c], [c, c]]·s has eigenvalues s·c·(1 ± i).
                y[2 * b] = s * c * (p - q);
                y[2 * b + 1] = s * c * (p + q);
            }
        });
        let ritz = arnoldi(&op, 2, 6).unwrap();
        let top = &ritz.values[0];
        assert!((top.re.hypot(top.im) - 3.0).abs() < 1e-8, "{top:?}");
        assert!((top.re - 3.0 * c).abs() < 1e-8 && (top.im.abs() - 3.0 * c).abs() < 1e-8);
        assert!(
            ritz.residuals[0].is_none(),
            "complex Ritz values carry no residual"
        );
    }

    #[test]
    fn arnoldi_stops_early_when_the_krylov_space_is_invariant() {
        // The identity: every Krylov space is one-dimensional.
        let op = matrix_free(4, |x: &[f64], y: &mut [f64]| y.copy_from_slice(x));
        let ritz = arnoldi(&op, 1, 4).unwrap();
        assert_eq!(ritz.krylov_dim, 1);
        assert!((ritz.values[0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn arnoldi_rejects_invalid_requests() {
        let op = matrix_free(3, |x: &[f64], y: &mut [f64]| y.copy_from_slice(x));
        assert!(arnoldi(&op, 0, 3).is_err());
        assert!(arnoldi(&op, 4, 4).is_err());
    }

    #[test]
    fn operator_bicgstab_and_gmres_match_their_csr_forms() {
        // Non-symmetric, diagonally dominant tridiagonal system.
        let n = 10;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.5);
                coo.push(i + 1, i, -0.5);
            }
        }
        let csr = coo.to_csr();
        let b = DVector::from_fn(n, |i| 1.0 + i as f64);
        let (x_op, rep_op) = bicgstab_op(&csr, &b, None, 1e-12, 200).unwrap();
        let (x_csr, _) = crate::bicgstab_with_report(&csr, &b, None, 1e-12, 200).unwrap();
        assert!(rep_op.status.is_converged());
        for i in 0..n {
            assert!((x_op[i] - x_csr[i]).abs() < 1e-9, "bicgstab {i}");
        }
        let identity = |v: &[f64]| Ok(v.to_vec());
        let (x_g, rep_g) = gmres_op(&csr, &b, None, 1e-12, 10, 200, identity).unwrap();
        assert!(rep_g.status.is_converged());
        let ax = csr.matvec(&x_g).unwrap();
        for i in 0..n {
            assert!((ax[i] - b[i]).abs() < 1e-9, "gmres row {i}");
        }
    }

    #[test]
    fn operator_forms_reject_bad_dimensions() {
        let op = matrix_free(3, |x: &[f64], y: &mut [f64]| y.copy_from_slice(x));
        let b = DVector::from_fn(2, |_| 1.0);
        assert!(bicgstab_op(&op, &b, None, 1e-8, 10).is_err());
        let identity = |v: &[f64]| Ok(v.to_vec());
        assert!(gmres_op(&op, &b, None, 1e-8, 3, 10, identity).is_err());
    }
}
