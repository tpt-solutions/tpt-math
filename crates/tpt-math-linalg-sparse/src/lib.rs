#![no_std]
#![forbid(unsafe_code)]
//! Sparse linear algebra implemented entirely in-house (no external backend).
//!
//! This crate provides general-purpose sparse matrix storage — coordinate
//! (triplet, [`CooMatrix`]), compressed-sparse-row ([`CsrMatrix`]) and
//! compressed-sparse-column ([`CscMatrix`]) — plus iterative solvers, preconditioners
//! and direct LU / Cholesky factorisations:
//! [`conjugate_gradient`] for symmetric positive-definite systems and
//! [`bicgstab`] for general (non-symmetric) systems.
//!
//! Everything is hand-rolled on top of [`tpt_math_linalg_dense::DVector`] (the
//! dense RHS/solution type) and [`tpt_math_numeric::Scalar`], so there is no
//! `nalgebra`/`faer`/`sprs` dependency and no license exposure. The design
//! deliberately mirrors the *storage* portion of a sparse format only; it does
//! not try to replace [`tpt-fem-sparse`](https://crates.io/crates/tpt-fem-sparse),
//! which is a separate FEM-assembly adapter (element scatter + duplicate-summing
//! triplet accumulation) living in the `tpt-fem` repo.
//!
//! # Features
//!
//! * `std` (default) — enable the allocator and the `std` support of deps.
//! * `alloc` — signal allocator availability (sparse containers need it).
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg_dense::DVector;
//! use tpt_math_linalg_sparse::{CooMatrix, conjugate_gradient, SparseError};
//!
//! // Build the 2x2 SPD system  A x = b  with  A = [[4,1],[1,3]], b = [1,2].
//! let mut coo = CooMatrix::<f64>::new(2, 2);
//! coo.push(0, 0, 4.0);
//! coo.push(0, 1, 1.0);
//! coo.push(1, 0, 1.0);
//! coo.push(1, 1, 3.0);
//!
//! let a = coo.to_csr();
//! let b = DVector::from_vec(vec![1.0, 2.0]);
//! let x = conjugate_gradient(&a, &b, None, 1e-12, 100).unwrap();
//! let expected = DVector::from_vec(vec![1.0 / 11.0, 7.0 / 11.0]);
//! assert!((x[0] - expected[0]).abs() < 1e-10);
//! assert!((x[1] - expected[1]).abs() < 1e-10);
//! # let _ = SparseError::DimensionMismatch;
//! ```

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg_dense::DVector;
use tpt_math_numeric::{Scalar, SolveReport, SolveStatus};

/// Errors returned by the sparse solvers and operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseError {
    /// A matrix/vector dimension pairing was incompatible (e.g. a matrix with
    /// `ncols` columns multiplied by a vector whose length differs).
    DimensionMismatch,
    /// The iterative solver reached `max_iter` without meeting the tolerance.
    /// `iterations` reports how many iterations were run.
    NotConverged {
        /// Number of iterations performed before giving up.
        iterations: usize,
    },
    /// The method hit a numerical breakdown (a zero or negative inner product
    /// such as `pᵀAp` for CG, or a non-finite value) and cannot continue.
    Breakdown {
        /// Iteration at which the breakdown occurred.
        iterations: usize,
    },
}

impl core::fmt::Display for SparseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SparseError::DimensionMismatch => {
                write!(f, "sparse matrix/vector dimension mismatch")
            }
            SparseError::NotConverged { iterations } => {
                write!(
                    f,
                    "iterative solver did not converge in {iterations} iterations"
                )
            }
            SparseError::Breakdown { iterations } => {
                write!(f, "iterative solver broke down at iteration {iterations}")
            }
        }
    }
}

impl core::error::Error for SparseError {}

// ---------------------------------------------------------------------------
// COO (triplet) storage
// ---------------------------------------------------------------------------

/// A sparse matrix in coordinate (COO / triplet) form.
///
/// Entries are stored as an unordered list of `(row, col, value)` triplets.
/// Duplicates are permitted and are summed together on conversion to a
/// compressed format ([`CooMatrix::to_csr`] / [`CooMatrix::to_csc`]), matching
/// the duplicate-summing semantics `tpt-fem-sparse` relies on for element
/// scatter + accumulation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CooMatrix<T> {
    nrows: usize,
    ncols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<T>,
}

impl<T> CooMatrix<T> {
    /// Create an empty `nrows` × `ncols` matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        CooMatrix {
            nrows,
            ncols,
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored entries (including any pending duplicates).
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Push a single entry. Duplicates are summed on conversion to CSR/CSC.
    pub fn push(&mut self, row: usize, col: usize, val: T) {
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }
}

impl<T: Scalar + Copy> CooMatrix<T> {
    /// Build from parallel `(rows, cols, vals)` triplets.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        vals: Vec<T>,
    ) -> Self {
        assert_eq!(rows.len(), cols.len());
        assert_eq!(rows.len(), vals.len());
        CooMatrix {
            nrows,
            ncols,
            rows,
            cols,
            vals,
        }
    }

    /// Convert to compressed-sparse-row form, summing any duplicate entries.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        to_csr_impl(self.nrows, self.ncols, &self.rows, &self.cols, &self.vals)
    }

    /// Convert to compressed-sparse-column form, summing any duplicate entries.
    pub fn to_csc(&self) -> CscMatrix<T> {
        to_csc_impl(self.nrows, self.ncols, &self.rows, &self.cols, &self.vals)
    }
}

// ---------------------------------------------------------------------------
// CSR storage
// ---------------------------------------------------------------------------

/// A sparse matrix in compressed-sparse-row (CSR) form.
///
/// `row_ptr` has length `nrows + 1`; `row_ptr[i]` .. `row_ptr[i + 1]` indexes
/// the stored entries of row `i` inside `col_idx` / `values`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrMatrix<T> {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T> CsrMatrix<T> {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored (post-merge) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate over the stored entries as `(row, col, &value)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        let nrows = self.nrows;
        (0..nrows).flat_map(move |i| {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            (start..end).map(move |k| (i, self.col_idx[k], &self.values[k]))
        })
    }
}

impl<T: Scalar + Copy> CsrMatrix<T> {
    /// Transpose, producing a [`CscMatrix`].
    pub fn transpose(&self) -> CscMatrix<T> {
        // New matrix is `ncols x nrows`; swap row/col roles.
        let rows: Vec<usize> = self.iter().map(|(_, c, _)| c).collect();
        let cols: Vec<usize> = self.iter().map(|(r, _, _)| r).collect();
        let vals: Vec<T> = self.iter().map(|(_, _, v)| *v).collect();
        to_csc_impl(self.ncols, self.nrows, &rows, &cols, &vals)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> CsrMatrix<T> {
    /// Sparse matrix–vector product `y = A x`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `x.len() != ncols()`.
    pub fn matvec(&self, x: &DVector<T>) -> Result<DVector<T>, SparseError> {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch);
        }
        let mut y = vec![T::zero(); self.nrows];
        self.matvec_into(x.as_slice(), &mut y)?;
        Ok(DVector::from_vec(y))
    }

    /// Sparse matrix–vector product `y = A x` written into a caller-owned
    /// buffer.
    ///
    /// Reusing `y` avoids a fresh allocation on every call, which matters in
    /// loops that apply the same matrix repeatedly.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `x.len() != ncols()` or
    /// `y.len() != nrows()`.
    pub fn matvec_into(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        if x.len() != self.ncols || y.len() != self.nrows {
            return Err(SparseError::DimensionMismatch);
        }
        for (i, yi) in y.iter_mut().enumerate() {
            let mut s = T::zero();
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                s = s + self.values[k] * x[self.col_idx[k]];
            }
            *yi = s;
        }
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> core::ops::Mul<&DVector<T>> for &CsrMatrix<T> {
    type Output = DVector<T>;
    /// # Panics
    ///
    /// Panics (via a `DimensionMismatch` message) if `rhs.len() != ncols()`.
    fn mul(self, rhs: &DVector<T>) -> DVector<T> {
        self.matvec(rhs)
            .unwrap_or_else(|_| panic!("CsrMatrix * DVector: dimension mismatch"))
    }
}

// ---------------------------------------------------------------------------
// CSC storage
// ---------------------------------------------------------------------------

/// A sparse matrix in compressed-sparse-column (CSC) form.
///
/// `col_ptr` has length `ncols + 1`; `col_ptr[j]` .. `col_ptr[j + 1]` indexes
/// the stored entries of column `j` inside `row_idx` / `values`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CscMatrix<T> {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T> CscMatrix<T> {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored (post-merge) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate over the stored entries as `(row, col, &value)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        let ncols = self.ncols;
        (0..ncols).flat_map(move |j| {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            (start..end).map(move |k| (self.row_idx[k], j, &self.values[k]))
        })
    }
}

impl<T: Scalar + Copy> CscMatrix<T> {
    /// Transpose, producing a [`CsrMatrix`].
    pub fn transpose(&self) -> CsrMatrix<T> {
        // New matrix is `ncols x nrows`; swap row/col roles.
        let rows: Vec<usize> = self.iter().map(|(_, c, _)| c).collect();
        let cols: Vec<usize> = self.iter().map(|(r, _, _)| r).collect();
        let vals: Vec<T> = self.iter().map(|(_, _, v)| *v).collect();
        to_csr_impl(self.ncols, self.nrows, &rows, &cols, &vals)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> CscMatrix<T> {
    /// Sparse matrix–vector product `y = A x`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `x.len() != ncols()`.
    pub fn matvec(&self, x: &DVector<T>) -> Result<DVector<T>, SparseError> {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch);
        }
        let nrows = self.nrows;
        let mut y_data = vec![T::zero(); nrows];
        for j in 0..self.ncols {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            let xj = x[j];
            for k in start..end {
                y_data[self.row_idx[k]] = y_data[self.row_idx[k]] + self.values[k] * xj;
            }
        }
        Ok(DVector::from_vec(y_data))
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers (duplicate-summing)
// ---------------------------------------------------------------------------

/// Stable counting sort of `(row, col, value)` triplets by one of their
/// indices: `key` selects the row (`0`) or column (`1`). Runs in `O(nnz + n)`.
#[cfg(feature = "alloc")]
fn bucket_by<T: Copy>(
    entries: &[(usize, usize, T)],
    key: impl Fn(&(usize, usize, T)) -> usize,
    nkeys: usize,
) -> Vec<(usize, usize, T)> {
    let mut offsets = vec![0usize; nkeys + 1];
    for e in entries {
        offsets[key(e) + 1] += 1;
    }
    for k in 0..nkeys {
        offsets[k + 1] += offsets[k];
    }
    // Slots are pre-filled with a copy of the first entry (or left empty) and
    // then overwritten, each at its bucket's running offset.
    let mut out = match entries.first() {
        Some(&first) => vec![first; entries.len()],
        None => Vec::new(),
    };
    for e in entries {
        let k = key(e);
        out[offsets[k]] = *e;
        offsets[k] += 1;
    }
    out
}

/// Sum adjacent entries with the same `(row, col)` in an input already grouped
/// by `(outer, inner)` order. Returns the merged entries and, per outer index,
/// how many survive.
#[cfg(feature = "alloc")]
fn merge_sorted<T: Scalar + Copy>(
    sorted: &[(usize, usize, T)],
    outer: impl Fn(&(usize, usize, T)) -> usize,
    nouter: usize,
) -> (Vec<(usize, usize, T)>, Vec<usize>) {
    let mut merged: Vec<(usize, usize, T)> = Vec::with_capacity(sorted.len());
    let mut counts = vec![0usize; nouter];
    for &(r, c, v) in sorted {
        match merged.last_mut() {
            Some(last) if last.0 == r && last.1 == c => last.2 = last.2 + v,
            _ => {
                merged.push((r, c, v));
                counts[outer(&(r, c, v))] += 1;
            }
        }
    }
    (merged, counts)
}

#[cfg(feature = "alloc")]
fn zip_triplets<T: Copy>(rows: &[usize], cols: &[usize], vals: &[T]) -> Vec<(usize, usize, T)> {
    rows.iter()
        .zip(cols)
        .zip(vals)
        .map(|((&r, &c), &v)| (r, c, v))
        .collect()
}

#[cfg(feature = "alloc")]
fn to_csr_impl<T: Scalar + Copy>(
    nrows: usize,
    ncols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> CsrMatrix<T> {
    // Column-bucket first, then stable row-bucket: the result is row-major with
    // columns ascending inside each row, so duplicates are adjacent.
    let entries = zip_triplets(rows, cols, vals);
    let by_col = bucket_by(&entries, |e| e.1, ncols);
    let by_row = bucket_by(&by_col, |e| e.0, nrows);
    let (merged, counts) = merge_sorted(&by_row, |e| e.0, nrows);
    let mut row_ptr = vec![0usize; nrows + 1];
    for i in 0..nrows {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }
    let mut col_idx = Vec::with_capacity(merged.len());
    let mut values = Vec::with_capacity(merged.len());
    for (_, c, v) in merged {
        col_idx.push(c);
        values.push(v);
    }
    CsrMatrix {
        nrows,
        ncols,
        row_ptr,
        col_idx,
        values,
    }
}

#[cfg(feature = "alloc")]
fn to_csc_impl<T: Scalar + Copy>(
    nrows: usize,
    ncols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> CscMatrix<T> {
    // Mirror image of `to_csr_impl`: row-bucket first, then stable column-bucket.
    let entries = zip_triplets(rows, cols, vals);
    let by_row = bucket_by(&entries, |e| e.0, nrows);
    let by_col = bucket_by(&by_row, |e| e.1, ncols);
    let (merged, counts) = merge_sorted(&by_col, |e| e.1, ncols);
    let mut col_ptr = vec![0usize; ncols + 1];
    for j in 0..ncols {
        col_ptr[j + 1] = col_ptr[j] + counts[j];
    }
    let mut row_idx = Vec::with_capacity(merged.len());
    let mut values = Vec::with_capacity(merged.len());
    for (r, _, v) in merged {
        row_idx.push(r);
        values.push(v);
    }
    CscMatrix {
        nrows,
        ncols,
        col_ptr,
        row_idx,
        values,
    }
}

// ---------------------------------------------------------------------------
// Iterative solvers
// ---------------------------------------------------------------------------

/// Solve `A x = b` with the conjugate gradient method for symmetric
/// positive-definite `A`.
///
/// * `a` — system matrix in CSR form (must be square and SPD).
/// * `b` — right-hand side.
/// * `x0` — optional starting guess; zeros are used if `None`.
/// * `tol` — absolute residual-norm tolerance.
/// * `max_iter` — iteration cap.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square or `b`'s
/// length does not match `a.nrows()`. Returns [`SparseError::NotConverged`] if
/// the method fails to reach `tol` within `max_iter` steps (or breaks down).
#[cfg(feature = "alloc")]
pub fn conjugate_gradient<T: Scalar + Copy + 'static>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<DVector<T>, SparseError> {
    let (x, report) = conjugate_gradient_with_report(a, b, x0, tol, max_iter)?;
    report_to_result(x, report)
}

/// [`conjugate_gradient`] with a [`SolveReport`] instead of an error for
/// breakdown or non-convergence.
///
/// Returns the last iterate together with its report, so a caller can inspect
/// how far the solve got. Only a dimension mismatch is an `Err`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square or `b`'s
/// length does not match `a.nrows()`, or if a supplied `x0` has the wrong length.
#[cfg(feature = "alloc")]
pub fn conjugate_gradient_with_report<T: Scalar + Copy + 'static>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<(DVector<T>, SolveReport<T>), SparseError> {
    if a.nrows() != a.ncols() || b.len() != a.nrows() {
        return Err(SparseError::DimensionMismatch);
    }
    let n = b.len();
    let mut x = x0.unwrap_or_else(|| DVector::zeros(n));
    let mut r = b.clone() - a.matvec(&x)?;
    // Relative tolerance: stop when ‖r‖ ≤ tol·‖b‖ (absolute tol if b = 0).
    let b_norm = b.norm();
    let target = if b_norm > T::zero() {
        tol * b_norm
    } else {
        tol
    };
    if r.norm() <= target {
        return Ok(report(x, SolveStatus::Converged, 0, r.norm()));
    }
    let mut p = r.clone();
    // Work buffer reused across iterations: the loop allocates nothing.
    let mut ap = DVector::zeros(n);
    let mut rs_old = r.dot(&r);
    for it in 0..max_iter {
        a.matvec_into(p.as_slice(), ap.as_mut_slice())?;
        let p_ap = p.dot(&ap);
        // For SPD A, pᵀAp > 0 until convergence; anything else (including NaN)
        // means the method cannot continue.
        if p_ap.is_nan() || p_ap <= T::zero() {
            return Ok(report(x, SolveStatus::Breakdown, it, r.norm()));
        }
        let alpha = rs_old / p_ap;
        axpy_slice(x.as_mut_slice(), alpha, p.as_slice());
        axpy_slice(r.as_mut_slice(), -alpha, ap.as_slice());
        if r.norm() <= target {
            return Ok(report(x, SolveStatus::Converged, it + 1, r.norm()));
        }
        let rs_new = r.dot(&r);
        if !rs_new.is_finite() {
            return Ok(report(x, SolveStatus::Breakdown, it, r.norm()));
        }
        let beta = rs_new / rs_old;
        xpay_slice(p.as_mut_slice(), beta, r.as_slice());
        rs_old = rs_new;
    }
    Ok(report(x, SolveStatus::NotConverged, max_iter, r.norm()))
}

/// Build the `(iterate, report)` pair returned by the `*_with_report` solvers.
#[cfg(feature = "alloc")]
fn report<T>(
    x: DVector<T>,
    status: SolveStatus,
    iterations: usize,
    residual: T,
) -> (DVector<T>, SolveReport<T>) {
    (
        x,
        SolveReport {
            status,
            iterations,
            residual,
        },
    )
}

/// Map a report back to the error-returning convention of the plain solvers.
#[cfg(feature = "alloc")]
fn report_to_result<T>(x: DVector<T>, report: SolveReport<T>) -> Result<DVector<T>, SparseError> {
    match report.status {
        SolveStatus::Converged => Ok(x),
        SolveStatus::NotConverged => Err(SparseError::NotConverged {
            iterations: report.iterations,
        }),
        SolveStatus::Breakdown => Err(SparseError::Breakdown {
            iterations: report.iterations,
        }),
    }
}

/// Solve `A x = b` with the BiCGSTAB method for general (non-symmetric)
/// systems.
///
/// Arguments mirror [`conjugate_gradient`]. BiCGSTAB is less robust than CG on
/// SPD systems but converges on many indefinite / non-symmetric problems where
/// CG would fail.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the dimensions are
/// incompatible, and [`SparseError::NotConverged`] if the method does not reach
/// `tol` within `max_iter` steps or encounters a breakdown.
#[cfg(feature = "alloc")]
pub fn bicgstab<T: Scalar + Copy + 'static>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<DVector<T>, SparseError> {
    let (x, report) = bicgstab_with_report(a, b, x0, tol, max_iter)?;
    report_to_result(x, report)
}

/// [`bicgstab`] with a [`SolveReport`] instead of an error for breakdown or
/// non-convergence. Returns the last iterate with its report.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] for incompatible dimensions only.
#[cfg(feature = "alloc")]
pub fn bicgstab_with_report<T: Scalar + Copy + 'static>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<(DVector<T>, SolveReport<T>), SparseError> {
    if a.nrows() != a.ncols() || b.len() != a.nrows() {
        return Err(SparseError::DimensionMismatch);
    }
    let n = b.len();
    let mut x = x0.unwrap_or_else(|| DVector::zeros(n));
    let mut r = b.clone() - a.matvec(&x)?;
    // Relative tolerance: stop when ‖r‖ ≤ tol·‖b‖ (absolute tol if b = 0).
    let b_norm = b.norm();
    let target = if b_norm > T::zero() {
        tol * b_norm
    } else {
        tol
    };
    if r.norm() <= target {
        return Ok(report(x, SolveStatus::Converged, 0, r.norm()));
    }
    let r0 = r.clone();
    // Work buffers reused across iterations: the loop allocates nothing.
    let mut p = DVector::zeros(n);
    let mut v = DVector::zeros(n);
    let mut s = DVector::zeros(n);
    let mut t = DVector::zeros(n);
    let mut rho = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    for it in 0..max_iter {
        let rho_new = r0.dot(&r);
        if rho_new == T::zero() || !rho_new.is_finite() {
            return Ok(report(x, SolveStatus::Breakdown, it, r.norm()));
        }
        let beta = (rho_new / rho) * (alpha / omega);
        // p = r + beta * (p - omega * v), using the previous v.
        for ((pi, &vi), &ri) in p
            .as_mut_slice()
            .iter_mut()
            .zip(v.as_slice())
            .zip(r.as_slice())
        {
            *pi = ri + beta * (*pi - omega * vi);
        }
        a.matvec_into(p.as_slice(), v.as_mut_slice())?;
        let r0_v = r0.dot(&v);
        if r0_v == T::zero() || !r0_v.is_finite() {
            return Ok(report(x, SolveStatus::Breakdown, it, r.norm()));
        }
        alpha = rho_new / r0_v;
        // s = r - alpha * v
        s.as_mut_slice().copy_from_slice(r.as_slice());
        axpy_slice(s.as_mut_slice(), -alpha, v.as_slice());
        // Half-step exit: if s is already small enough, x + alpha·p is the answer.
        if s.norm() <= target {
            axpy_slice(x.as_mut_slice(), alpha, p.as_slice());
            return Ok(report(x, SolveStatus::Converged, it + 1, s.norm()));
        }
        a.matvec_into(s.as_slice(), t.as_mut_slice())?;
        let t_sq = t.dot(&t);
        if t_sq == T::zero() {
            return Ok(report(x, SolveStatus::Breakdown, it, r.norm()));
        }
        omega = t.dot(&s) / t_sq;
        if omega == T::zero() || !omega.is_finite() {
            return Ok(report(x, SolveStatus::Breakdown, it, r.norm()));
        }
        // x += alpha * p + omega * s; r = s - omega * t.
        axpy_slice(x.as_mut_slice(), alpha, p.as_slice());
        axpy_slice(x.as_mut_slice(), omega, s.as_slice());
        r.as_mut_slice().copy_from_slice(s.as_slice());
        axpy_slice(r.as_mut_slice(), -omega, t.as_slice());
        if r.norm() <= target {
            return Ok(report(x, SolveStatus::Converged, it + 1, r.norm()));
        }
        rho = rho_new;
    }
    Ok(report(x, SolveStatus::NotConverged, max_iter, r.norm()))
}

/// Euclidean norm of a slice.
#[cfg(feature = "alloc")]
fn slice_norm<T: Scalar + Copy>(v: &[T]) -> T {
    v.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt()
}

/// Dot product of two equal-length slices.
#[cfg(feature = "alloc")]
fn slice_dot<T: Scalar + Copy>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// In-place `y ← y + α·x`.
#[cfg(feature = "alloc")]
fn axpy_slice<T: Scalar + Copy>(y: &mut [T], alpha: T, x: &[T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = *yi + alpha * xi;
    }
}

/// In-place `y ← x + β·y` (the search-direction update in CG).
#[cfg(feature = "alloc")]
fn xpay_slice<T: Scalar + Copy>(y: &mut [T], beta: T, x: &[T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = xi + beta * *yi;
    }
}

/// Givens rotation `(c, s)` with `c·a + s·b = r` and `−s·a + c·b = 0`.
#[cfg(feature = "alloc")]
fn givens<T: Scalar + Copy>(a: T, b: T) -> (T, T) {
    if b == T::zero() {
        return (T::one(), T::zero());
    }
    if b.abs() > a.abs() {
        let t = a / b;
        let s = T::one() / (T::one() + t * t).sqrt();
        (s * t, s)
    } else {
        let t = b / a;
        let c = T::one() / (T::one() + t * t).sqrt();
        (c, c * t)
    }
}

/// Solve `A x = b` with restarted GMRES(`restart`) for a general square `A`.
///
/// Each cycle builds an Arnoldi basis of the Krylov subspace by modified
/// Gram–Schmidt, and minimises the residual over that subspace with Givens
/// rotations. The basis is discarded every `restart` steps. The method works for
/// non-symmetric systems that [`conjugate_gradient`] cannot handle, at the cost
/// of storing the whole basis for one cycle.
///
/// * `tol` — relative residual tolerance: stop when `‖r‖ ≤ tol·‖b‖` (absolute if
///   `b = 0`).
/// * `restart` — Krylov subspace dimension per cycle (clamped to at least 1).
/// * `max_iter` — bound on the total number of Arnoldi steps across all cycles.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square, or `b` or
/// `x0` has the wrong length. Returns [`SparseError::NotConverged`] if the
/// tolerance is not met within `max_iter` steps, and [`SparseError::Breakdown`]
/// if the Hessenberg system becomes singular.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{gmres, CooMatrix};
/// use tpt_math_linalg_dense::DVector;
///
/// // Non-symmetric tridiagonal system with exact solution (1, 2, 3).
/// let mut coo = CooMatrix::<f64>::new(3, 3);
/// for &(r, c, v) in &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 5.0), (1, 2, 1.0), (2, 1, 1.0), (2, 2, 3.0)] {
///     coo.push(r, c, v);
/// }
/// let a = coo.to_csr();
/// let b = DVector::from_vec(vec![6.0, 15.0, 11.0]);
/// let x = gmres(&a, &b, None, 1e-12, 3, 100).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-9 && (x[2] - 3.0).abs() < 1e-9);
/// ```
#[cfg(feature = "alloc")]
pub fn gmres<T: Scalar + Copy + 'static>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    restart: usize,
    max_iter: usize,
) -> Result<DVector<T>, SparseError> {
    gmres_right_preconditioned(a, b, x0, tol, restart, max_iter, |v: &[T]| {
        Ok::<Vec<T>, SparseError>(v.to_vec())
    })
}

/// Right-preconditioned restarted GMRES.
///
/// Solves `A x = b` through the preconditioned operator `A M⁻¹`, where
/// `precond` applies `M⁻¹` to a vector. Right preconditioning leaves the
/// residual unchanged, so the tolerance and budget have the same meaning as in
/// [`gmres`]. A good `M` makes `A M⁻¹` close to the identity, which cuts the
/// number of Arnoldi steps.
///
/// # Errors
///
/// As [`gmres`], plus any error returned by `precond`.
#[cfg(feature = "alloc")]
pub fn gmres_right_preconditioned<T, P>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    restart: usize,
    max_iter: usize,
    precond: P,
) -> Result<DVector<T>, SparseError>
where
    T: Scalar + Copy + 'static,
    P: Fn(&[T]) -> Result<Vec<T>, SparseError>,
{
    if a.nrows() != a.ncols() || b.len() != a.nrows() {
        return Err(SparseError::DimensionMismatch);
    }
    let n = b.len();
    let m = restart.max(1);
    let mut x: Vec<T> = match x0 {
        Some(v) if v.len() == n => v.as_slice().to_vec(),
        Some(_) => return Err(SparseError::DimensionMismatch),
        None => vec![T::zero(); n],
    };
    let b_slice = b.as_slice();
    let b_norm = slice_norm(b_slice);
    let target = if b_norm > T::zero() {
        tol * b_norm
    } else {
        tol
    };

    let mut iterations = 0usize;
    let mut ax = vec![T::zero(); n];
    loop {
        a.matvec_into(&x, &mut ax)?;
        let r: Vec<T> = (0..n).map(|i| b_slice[i] - ax[i]).collect();
        let beta = slice_norm(&r);
        if beta <= target {
            return Ok(DVector::from_vec(x));
        }
        if iterations >= max_iter {
            return Err(SparseError::NotConverged { iterations });
        }

        // Arnoldi cycle on the preconditioned operator A M⁻¹. `basis[j]` is the
        // j-th orthonormal Krylov vector, `hess[j]` the j-th Hessenberg column
        // after the Givens rotations.
        let mut basis: Vec<Vec<T>> = vec![r.iter().map(|&v| v / beta).collect()];
        let mut hess: Vec<Vec<T>> = Vec::with_capacity(m);
        let mut rotations: Vec<(T, T)> = Vec::with_capacity(m);
        let mut g = vec![T::zero(); m + 1];
        g[0] = beta;
        let mut k = 0;
        while k < m && iterations < max_iter {
            iterations += 1;
            let z = precond(&basis[k])?;
            let mut w = vec![T::zero(); n];
            a.matvec_into(&z, &mut w)?;
            let mut col = vec![T::zero(); k + 2];
            for i in 0..=k {
                let h_ik = slice_dot(&w, &basis[i]);
                col[i] = h_ik;
                for (wj, &vj) in w.iter_mut().zip(&basis[i]) {
                    *wj = *wj - h_ik * vj;
                }
            }
            let h_next = slice_norm(&w);
            col[k + 1] = h_next;

            // Apply the earlier rotations, then a new one for the subdiagonal.
            for (i, &(c, s)) in rotations.iter().enumerate() {
                let temp = c * col[i] + s * col[i + 1];
                col[i + 1] = -s * col[i] + c * col[i + 1];
                col[i] = temp;
            }
            let (c, s) = givens(col[k], col[k + 1]);
            rotations.push((c, s));
            col[k] = c * col[k] + s * col[k + 1];
            col[k + 1] = T::zero();
            let g_k = g[k];
            g[k + 1] = -s * g_k;
            g[k] = c * g_k;
            hess.push(col);
            k += 1;

            // A zero subdiagonal means the subspace is invariant: the solution
            // is exact within it, so stop the cycle here.
            if g[k].abs() <= target || h_next == T::zero() {
                break;
            }
            basis.push(w.iter().map(|&v| v / h_next).collect());
        }

        // Solve the upper-triangular system R y = g for the cycle.
        let mut y = vec![T::zero(); k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for j in (i + 1)..k {
                s = s - hess[j][i] * y[j];
            }
            let diag = hess[i][i];
            if diag == T::zero() {
                return Err(SparseError::Breakdown { iterations });
            }
            y[i] = s / diag;
        }
        // The cycle's correction is M⁻¹ (Σ yⱼ vⱼ) because of right preconditioning.
        let mut combination = vec![T::zero(); n];
        for (j, &yj) in y.iter().enumerate() {
            for (ci, &vi) in combination.iter_mut().zip(&basis[j]) {
                *ci = *ci + yj * vi;
            }
        }
        let correction = precond(&combination)?;
        for (xi, &si) in x.iter_mut().zip(&correction) {
            *xi = *xi + si;
        }
    }
}

/// GMRES preconditioned by the ILU(0) factors of `a` itself.
///
/// Factorises `a` once, then runs [`gmres_right_preconditioned`]. This is the
/// usual pairing for sparse non-symmetric systems. It is cheap per iteration and
/// often cuts the iteration count a lot.
///
/// # Errors
///
/// As [`Ilu0::factor`] and [`gmres`].
#[cfg(feature = "alloc")]
pub fn gmres_ilu0(
    a: &CsrMatrix<f64>,
    b: &DVector<f64>,
    x0: Option<DVector<f64>>,
    tol: f64,
    restart: usize,
    max_iter: usize,
) -> Result<DVector<f64>, SparseError> {
    let ilu = Ilu0::factor(a)?;
    gmres_right_preconditioned(a, b, x0, tol, restart, max_iter, |v: &[f64]| {
        ilu.apply(&DVector::from_vec(v.to_vec()))
            .map(|d| d.as_slice().to_vec())
    })
}

/// Solve `A x = b` with conjugate gradients preconditioned by the Jacobi
/// (diagonal) preconditioner `M = diag(A)`.
///
/// Each iteration applies `M⁻¹ r` by dividing the residual by the diagonal
/// entries. This is cheap and helps most when the diagonal varies widely in
/// scale, as with badly scaled stiffness matrices. For SPD `A` with a constant
/// diagonal it does nothing useful, and plain [`conjugate_gradient`] is
/// equivalent.
///
/// The tolerance and error conditions are those of [`conjugate_gradient`].
///
/// # Errors
///
/// As [`conjugate_gradient`], plus [`SparseError::Breakdown`] at iteration zero
/// if a diagonal entry is zero, since the preconditioner is then undefined.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{jacobi_preconditioned_cg, CooMatrix};
/// use tpt_math_linalg_dense::DVector;
///
/// // Diagonal spans four orders of magnitude.
/// let mut coo = CooMatrix::<f64>::new(2, 2);
/// coo.push(0, 0, 1.0);
/// coo.push(1, 1, 1e4);
/// let a = coo.to_csr();
/// let b = DVector::from_vec(vec![2.0, 3e4]);
/// let x = jacobi_preconditioned_cg(&a, &b, None, 1e-12, 10).unwrap();
/// assert!((x[0] - 2.0).abs() < 1e-9 && (x[1] - 3.0).abs() < 1e-9);
/// ```
/// Solve `A x = b` with conjugate gradients under a caller-supplied SPD
/// preconditioner `M`.
///
/// `precond` must return `M⁻¹ r` for a residual `r`, and `M` must be symmetric
/// positive definite for the method to be sound. The tolerance and error
/// conditions are those of [`conjugate_gradient`]. [`jacobi_preconditioned_cg`]
/// and [`ic0_preconditioned_cg`] are this function with specific preconditioners.
///
/// # Errors
///
/// As [`conjugate_gradient`], plus any error returned by `precond`.
#[cfg(feature = "alloc")]
pub fn preconditioned_conjugate_gradient<T, P>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
    precond: P,
) -> Result<DVector<T>, SparseError>
where
    T: Scalar + Copy + 'static,
    P: Fn(&DVector<T>) -> Result<DVector<T>, SparseError>,
{
    if a.nrows() != a.ncols() || b.len() != a.nrows() {
        return Err(SparseError::DimensionMismatch);
    }
    let n = b.len();
    let mut x: Vec<T> = match x0 {
        Some(v) if v.len() == n => v.as_slice().to_vec(),
        Some(_) => return Err(SparseError::DimensionMismatch),
        None => vec![T::zero(); n],
    };
    let b_slice = b.as_slice();
    let b_norm = slice_norm(b_slice);
    let target = if b_norm > T::zero() {
        tol * b_norm
    } else {
        tol
    };

    let mut ax = vec![T::zero(); n];
    a.matvec_into(&x, &mut ax)?;
    let mut r: Vec<T> = (0..n).map(|i| b_slice[i] - ax[i]).collect();
    if slice_norm(&r) <= target {
        return Ok(DVector::from_vec(x));
    }
    let mut z = precond(&DVector::from_vec(r.clone()))?.as_slice().to_vec();
    let mut p = z.clone();
    let mut rz = slice_dot(&r, &z);
    let mut ap = vec![T::zero(); n];

    for it in 0..max_iter {
        a.matvec_into(&p, &mut ap)?;
        let p_ap = slice_dot(&p, &ap);
        if p_ap.is_nan() || p_ap <= T::zero() {
            return Err(SparseError::Breakdown { iterations: it });
        }
        let alpha = rz / p_ap;
        for (xi, &pi) in x.iter_mut().zip(&p) {
            *xi = *xi + alpha * pi;
        }
        for (ri, &api) in r.iter_mut().zip(&ap) {
            *ri = *ri - alpha * api;
        }
        if slice_norm(&r) <= target {
            return Ok(DVector::from_vec(x));
        }
        z = precond(&DVector::from_vec(r.clone()))?.as_slice().to_vec();
        let rz_new = slice_dot(&r, &z);
        if !rz_new.is_finite() {
            return Err(SparseError::Breakdown { iterations: it });
        }
        let beta = rz_new / rz;
        for (pi, &zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
        rz = rz_new;
    }
    Err(SparseError::NotConverged {
        iterations: max_iter,
    })
}

/// Solve `A x = b` with conjugate gradients preconditioned by the Jacobi
/// (diagonal) preconditioner `M = diag(A)`.
///
/// Each iteration applies `M⁻¹ r` by dividing the residual by the diagonal
/// entries. This is cheap and helps most when the diagonal varies widely in
/// scale, as with badly scaled stiffness matrices. For SPD `A` with a constant
/// diagonal it does nothing useful, and plain [`conjugate_gradient`] is
/// equivalent.
///
/// The tolerance and error conditions are those of [`conjugate_gradient`].
///
/// # Errors
///
/// As [`conjugate_gradient`], plus [`SparseError::Breakdown`] at iteration zero
/// if a diagonal entry is zero, since the preconditioner is then undefined.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{jacobi_preconditioned_cg, CooMatrix};
/// use tpt_math_linalg_dense::DVector;
///
/// // Diagonal spans four orders of magnitude.
/// let mut coo = CooMatrix::<f64>::new(2, 2);
/// coo.push(0, 0, 1.0);
/// coo.push(1, 1, 1e4);
/// let a = coo.to_csr();
/// let b = DVector::from_vec(vec![2.0, 3e4]);
/// let x = jacobi_preconditioned_cg(&a, &b, None, 1e-12, 10).unwrap();
/// assert!((x[0] - 2.0).abs() < 1e-9 && (x[1] - 3.0).abs() < 1e-9);
/// ```
#[cfg(feature = "alloc")]
pub fn jacobi_preconditioned_cg<T: Scalar + Copy + 'static>(
    a: &CsrMatrix<T>,
    b: &DVector<T>,
    x0: Option<DVector<T>>,
    tol: T,
    max_iter: usize,
) -> Result<DVector<T>, SparseError> {
    if a.nrows() != a.ncols() || b.len() != a.nrows() {
        return Err(SparseError::DimensionMismatch);
    }
    let n = b.len();

    // Extract diag(A); a missing or zero diagonal entry makes M singular.
    let mut diag = vec![T::zero(); n];
    for (i, j, &v) in a.iter() {
        if i == j {
            diag[i] = v;
        }
    }
    if diag.iter().any(|&d| d == T::zero()) {
        return Err(SparseError::Breakdown { iterations: 0 });
    }
    preconditioned_conjugate_gradient(a, b, x0, tol, max_iter, |r: &DVector<T>| {
        Ok(DVector::from_vec(
            r.iter().zip(&diag).map(|(&ri, &di)| ri / di).collect(),
        ))
    })
}

/// Sparse Cholesky factor `A = L Lᵀ` of a symmetric positive-definite matrix.
///
/// Stored by columns: column `j` holds the entries `L(i, j)` with `i ≥ j`,
/// sorted by row, with the diagonal first. No fill-reducing ordering is applied,
/// so fill-in can be larger than with a reordered factorisation. The method is
/// exact up to rounding, and the factor is reused for any number of right-hand
/// sides.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct SparseCholesky {
    n: usize,
    columns: Vec<Vec<(usize, f64)>>,
}

#[cfg(feature = "alloc")]
impl SparseCholesky {
    /// Factorise a symmetric positive-definite CSR matrix.
    ///
    /// Only the lower triangle of `a` is read; symmetry is assumed, not checked.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `a` is not square, and
    /// [`SparseError::Breakdown`] if a pivot is not positive, which means the
    /// matrix is not positive definite.
    pub fn factor(a: &CsrMatrix<f64>) -> Result<Self, SparseError> {
        if a.nrows() != a.ncols() {
            return Err(SparseError::DimensionMismatch);
        }
        let n = a.nrows();
        let mut columns: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut work = vec![0.0; n];

        for j in 0..n {
            // Load column j of A (the lower part of row j, by symmetry).
            for (row, col, &v) in a.iter() {
                if row == j && col >= j {
                    work[col] = v;
                }
            }
            // Subtract the contributions of the earlier columns.
            for column in &columns[..j] {
                let Ok(pos) = column.binary_search_by_key(&j, |&(i, _)| i) else {
                    continue;
                };
                let ljk = column[pos].1;
                for &(i, lik) in &column[pos..] {
                    work[i] -= lik * ljk;
                }
            }

            let pivot = work[j];
            if pivot.is_nan() || pivot <= 0.0 {
                return Err(SparseError::Breakdown { iterations: j });
            }
            let diag = tpt_math_numeric::prelude::Float::sqrt(pivot);
            let mut column = vec![(j, diag)];
            for (i, &w) in work.iter().enumerate().skip(j + 1) {
                if w != 0.0 {
                    column.push((i, w / diag));
                }
            }
            // Clear the workspace for the next column.
            for w in work.iter_mut().skip(j) {
                *w = 0.0;
            }
            columns.push(column);
        }
        Ok(SparseCholesky { n, columns })
    }

    /// Dimension of the factorised matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Number of stored entries in `L`, including the diagonal. Comparing this
    /// with the entries of `A` shows how much fill-in occurred.
    pub fn nnz(&self) -> usize {
        self.columns.iter().map(Vec::len).sum()
    }

    /// Solve `A x = b` with the stored factor.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `b` has the wrong length.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, SparseError> {
        if b.len() != self.n {
            return Err(SparseError::DimensionMismatch);
        }
        let mut x: Vec<f64> = b.as_slice().to_vec();
        // Forward substitution L y = b, column by column.
        for j in 0..self.n {
            let column = &self.columns[j];
            x[j] /= column[0].1;
            let xj = x[j];
            for &(i, v) in &column[1..] {
                x[i] -= v * xj;
            }
        }
        // Back substitution Lᵀ x = y, using the same columns as rows of Lᵀ.
        for j in (0..self.n).rev() {
            let column = &self.columns[j];
            let mut s = x[j];
            for &(i, v) in &column[1..] {
                s -= v * x[i];
            }
            x[j] = s / column[0].1;
        }
        Ok(DVector::from_vec(x))
    }
}

/// Incomplete LU factorisation with zero fill-in, ILU(0).
///
/// Computes `L U ≈ A` where `L` is unit lower triangular and `U` upper
/// triangular, and keeps only the sparsity pattern of `A`. The product matches
/// `A` on that pattern and differs elsewhere. Applying `(LU)⁻¹` to a vector is
/// a cheap approximate inverse, which makes ILU(0) a standard preconditioner.
/// For a matrix with no fill-in under LU, such as a tridiagonal one, it is exact.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct Ilu0 {
    n: usize,
    /// Each row sorted by column. Strictly lower entries hold `L`, the diagonal
    /// and upper entries hold `U`.
    rows: Vec<Vec<(usize, f64)>>,
    /// Position of the diagonal entry within each row.
    diagonal: Vec<usize>,
}

#[cfg(feature = "alloc")]
impl Ilu0 {
    /// Compute the ILU(0) factors of a square CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `a` is not square, and
    /// [`SparseError::Breakdown`] if a row has no diagonal entry or a pivot is
    /// zero.
    pub fn factor(a: &CsrMatrix<f64>) -> Result<Self, SparseError> {
        if a.nrows() != a.ncols() {
            return Err(SparseError::DimensionMismatch);
        }
        let n = a.nrows();
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (r, c, &v) in a.iter() {
            rows[r].push((c, v));
        }
        for row in &mut rows {
            row.sort_by_key(|&(c, _)| c);
        }
        let mut diagonal = vec![0usize; n];
        for (i, row) in rows.iter().enumerate() {
            match row.iter().position(|&(c, _)| c == i) {
                Some(p) => diagonal[i] = p,
                None => return Err(SparseError::Breakdown { iterations: i }),
            }
        }

        // IKJ-ordered elimination restricted to the existing pattern.
        const ABSENT: usize = usize::MAX;
        let mut pos = vec![ABSENT; n];
        for i in 0..n {
            for (p, &(c, _)) in rows[i].iter().enumerate() {
                pos[c] = p;
            }
            for kp in 0..rows[i].len() {
                let k = rows[i][kp].0;
                if k >= i {
                    break;
                }
                let pivot = rows[k][diagonal[k]].1;
                if pivot == 0.0 || pivot.is_nan() {
                    return Err(SparseError::Breakdown { iterations: k });
                }
                let multiplier = rows[i][kp].1 / pivot;
                rows[i][kp].1 = multiplier;
                // Row k is finished, so reading it while updating row i is safe.
                let row_k = rows[k].clone();
                for &(j, ukj) in row_k.iter().filter(|&&(j, _)| j > k) {
                    let p = pos[j];
                    if p != ABSENT {
                        rows[i][p].1 -= multiplier * ukj;
                    }
                }
            }
            for &(c, _) in &rows[i] {
                pos[c] = ABSENT;
            }
        }
        Ok(Ilu0 { n, rows, diagonal })
    }

    /// Apply the preconditioner: returns `(LU)⁻¹ r`, an approximation to `A⁻¹ r`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `r` has the wrong length.
    pub fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, SparseError> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch);
        }
        let mut y: Vec<f64> = r.as_slice().to_vec();
        // Forward substitution with unit lower L.
        for i in 0..self.n {
            let mut s = y[i];
            for &(j, v) in self.rows[i].iter().take_while(|&&(j, _)| j < i) {
                s -= v * y[j];
            }
            y[i] = s;
        }
        // Back substitution with upper U.
        for i in (0..self.n).rev() {
            let mut s = y[i];
            for &(j, v) in &self.rows[i][self.diagonal[i] + 1..] {
                s -= v * y[j];
            }
            y[i] = s / self.rows[i][self.diagonal[i]].1;
        }
        Ok(DVector::from_vec(y))
    }
}

/// Incomplete Cholesky factorisation with zero fill-in, IC(0).
///
/// For a symmetric positive-definite `A`, computes a lower-triangular `L` whose
/// sparsity pattern is exactly the lower triangle of `A`, with `L Lᵀ ≈ A`. It is
/// the symmetric counterpart of [`Ilu0`] and the usual preconditioner for CG.
/// The product matches `A` on the pattern and differs elsewhere. For a matrix
/// whose Cholesky factor has no fill-in, such as a tridiagonal one, it is exact.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct Ic0 {
    factor: SparseCholesky,
}

#[cfg(feature = "alloc")]
impl Ic0 {
    /// Compute the IC(0) factor of a symmetric positive-definite CSR matrix.
    ///
    /// Only the lower triangle of `a` is read; symmetry is assumed, not checked.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `a` is not square, and
    /// [`SparseError::Breakdown`] if a column has no diagonal entry or a pivot is
    /// not positive. A non-positive pivot means the incomplete factor does not
    /// exist for this pattern, which can happen even for an SPD matrix.
    pub fn factor(a: &CsrMatrix<f64>) -> Result<Self, SparseError> {
        if a.nrows() != a.ncols() {
            return Err(SparseError::DimensionMismatch);
        }
        let n = a.nrows();
        // Lower-triangle columns of A: lower[j] holds (i, A_ij) for i ≥ j.
        let mut lower: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (r, c, &v) in a.iter() {
            if r >= c {
                lower[c].push((r, v));
            }
        }
        for col in &mut lower {
            col.sort_by_key(|&(i, _)| i);
        }

        let mut columns: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        // contributors[i] lists the finished columns k < i with L(i, k) in their
        // pattern, so column j only visits the columns that can update it.
        let mut contributors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut work = vec![0.0; n];
        let mut in_pattern = vec![false; n];

        for j in 0..n {
            for &(i, v) in &lower[j] {
                work[i] = v;
                in_pattern[i] = true;
            }
            for &k in &contributors[j] {
                let col_k = &columns[k];
                let Ok(pos) = col_k.binary_search_by_key(&j, |&(i, _)| i) else {
                    continue;
                };
                let ljk = col_k[pos].1;
                for &(i, lik) in &col_k[pos..] {
                    if in_pattern[i] {
                        work[i] -= lik * ljk;
                    }
                }
            }
            let pivot = work[j];
            if pivot.is_nan() || pivot <= 0.0 {
                for &(i, _) in &lower[j] {
                    work[i] = 0.0;
                    in_pattern[i] = false;
                }
                return Err(SparseError::Breakdown { iterations: j });
            }
            let diag = tpt_math_numeric::prelude::Float::sqrt(pivot);
            let mut column = vec![(j, diag)];
            for &(i, _) in lower[j].iter().skip(1) {
                column.push((i, work[i] / diag));
                contributors[i].push(j);
            }
            for &(i, _) in &lower[j] {
                work[i] = 0.0;
                in_pattern[i] = false;
            }
            columns.push(column);
        }
        Ok(Ic0 {
            factor: SparseCholesky { n, columns },
        })
    }

    /// Dimension of the factorised matrix.
    pub fn dim(&self) -> usize {
        self.factor.dim()
    }

    /// Apply the preconditioner: returns `(L Lᵀ)⁻¹ r`, an approximation to
    /// `A⁻¹ r`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `r` has the wrong length.
    pub fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, SparseError> {
        self.factor.solve(r)
    }
}

/// Solve `A x = b` with conjugate gradients preconditioned by the IC(0) factor
/// of `a`.
///
/// Factorises `a` once and then runs [`preconditioned_conjugate_gradient`]. Use
/// it for sparse SPD systems where Jacobi scaling is not enough, such as the
/// 2D Laplacian. The tolerance and error conditions are those of
/// [`conjugate_gradient`].
///
/// # Errors
///
/// As [`Ic0::factor`] and [`conjugate_gradient`].
#[cfg(feature = "alloc")]
pub fn ic0_preconditioned_cg(
    a: &CsrMatrix<f64>,
    b: &DVector<f64>,
    x0: Option<DVector<f64>>,
    tol: f64,
    max_iter: usize,
) -> Result<DVector<f64>, SparseError> {
    let ic = Ic0::factor(a)?;
    preconditioned_conjugate_gradient(a, b, x0, tol, max_iter, |r: &DVector<f64>| ic.apply(r))
}

/// Sparse LU factorisation with partial pivoting, `P A = L U`.
///
/// Left-looking (Gilbert–Peierls) elimination: column `j` of `A` is solved
/// against the already-computed `L` columns, touching only the pivot steps whose
/// entries are non-zero, in increasing order. Pivoting picks the largest-magnitude
/// candidate in the column, which keeps the factorisation stable for
/// non-symmetric and indefinite matrices. Fill-in is not reduced by any ordering,
/// so `L` and `U` can be much denser than `A` on unstructured problems.
///
/// The factor is reusable for any number of right-hand sides via [`SparseLu::solve`].
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct SparseLu {
    n: usize,
    /// `pivot_row[k]` is the original row chosen as the pivot at step `k`.
    pivot_row: Vec<usize>,
    /// `L` column `k`: unit lower entries `(row, value)` for rows that were not
    /// pivots at steps `≤ k`. The unit diagonal is implicit.
    l_cols: Vec<Vec<(usize, f64)>>,
    /// `U` column `j`: strictly upper entries `(step, value)` with `step < j`.
    u_cols: Vec<Vec<(usize, f64)>>,
    /// `U` diagonal, one entry per step.
    diag: Vec<f64>,
}

#[cfg(feature = "alloc")]
impl SparseLu {
    /// Factorise a square CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `a` is not square, and
    /// [`SparseError::Breakdown`] at step `k` if column `k` has no usable pivot,
    /// which means the matrix is singular (structurally or numerically).
    pub fn factor(a: &CsrMatrix<f64>) -> Result<Self, SparseError> {
        if a.nrows() != a.ncols() {
            return Err(SparseError::DimensionMismatch);
        }
        let n = a.nrows();
        // Column-wise copy of A.
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (r, c, &v) in a.iter() {
            cols[c].push((r, v));
        }

        const UNPIVOTED: usize = usize::MAX;
        let mut pinv = vec![UNPIVOTED; n];
        let mut pivot_row = Vec::with_capacity(n);
        let mut l_cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut u_cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut diag = Vec::with_capacity(n);

        // Dense accumulator for the current column, plus the list of rows it
        // touches so that only those entries are visited and cleared.
        let mut x = vec![0.0; n];
        let mut touched = vec![false; n];
        let mut nz: Vec<usize> = Vec::new();

        for (j, col) in cols.iter().enumerate() {
            nz.clear();
            for &(r, v) in col {
                x[r] = v;
                if !touched[r] {
                    touched[r] = true;
                    nz.push(r);
                }
            }
            // Pivot steps to apply, in increasing order.
            let mut pending: alloc::collections::BTreeSet<usize> = nz
                .iter()
                .filter(|&&r| pinv[r] != UNPIVOTED)
                .map(|&r| pinv[r])
                .collect();
            let mut u_col = Vec::new();
            while let Some(k) = pending.pop_first() {
                let xk = x[pivot_row[k]];
                if xk == 0.0 {
                    continue;
                }
                u_col.push((k, xk));
                for &(i, l) in &l_cols[k] {
                    if !touched[i] {
                        touched[i] = true;
                        nz.push(i);
                    }
                    x[i] -= l * xk;
                    if pinv[i] != UNPIVOTED {
                        pending.insert(pinv[i]);
                    }
                }
            }

            // Partial pivoting: largest magnitude among the unpivoted rows.
            let mut best: Option<(usize, f64)> = None;
            for &r in &nz {
                if pinv[r] == UNPIVOTED && best.is_none_or(|(_, b)| x[r].abs() > b.abs()) {
                    best = Some((r, x[r]));
                }
            }
            let (p, pivot) = match best {
                Some((p, v)) if v != 0.0 && v.is_finite() => (p, v),
                _ => {
                    for &r in &nz {
                        x[r] = 0.0;
                        touched[r] = false;
                    }
                    return Err(SparseError::Breakdown { iterations: j });
                }
            };
            pinv[p] = j;
            pivot_row.push(p);
            diag.push(pivot);
            let l_col: Vec<(usize, f64)> = nz
                .iter()
                .filter(|&&r| pinv[r] == UNPIVOTED && x[r] != 0.0)
                .map(|&r| (r, x[r] / pivot))
                .collect();
            l_cols.push(l_col);
            u_cols.push(u_col);

            for &r in &nz {
                x[r] = 0.0;
                touched[r] = false;
            }
        }
        Ok(SparseLu {
            n,
            pivot_row,
            l_cols,
            u_cols,
            diag,
        })
    }

    /// Dimension of the factorised matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Stored entries in `L` and `U` together, counting the diagonal of `U`.
    /// Comparing this with the entries of `A` shows how much fill-in occurred.
    pub fn nnz(&self) -> usize {
        self.l_cols.iter().map(Vec::len).sum::<usize>()
            + self.u_cols.iter().map(Vec::len).sum::<usize>()
            + self.n
    }

    /// Solve `A x = b` with the stored factor.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `b` has the wrong length.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, SparseError> {
        if b.len() != self.n {
            return Err(SparseError::DimensionMismatch);
        }
        // Forward substitution L y = P b, in pivot-step order.
        let mut w: Vec<f64> = b.as_slice().to_vec();
        let mut z = vec![0.0; self.n];
        for k in 0..self.n {
            let yk = w[self.pivot_row[k]];
            z[k] = yk;
            for &(i, l) in &self.l_cols[k] {
                w[i] -= l * yk;
            }
        }
        // Back substitution U x = y. Column j of U holds the entries above its
        // diagonal, so x_j is final once j is reached in descending order.
        for j in (0..self.n).rev() {
            let xj = z[j] / self.diag[j];
            z[j] = xj;
            for &(k, u) in &self.u_cols[j] {
                z[k] -= u * xj;
            }
        }
        Ok(DVector::from_vec(z))
    }
}

#[cfg(test)]
mod lu_tests {
    use super::*;
    use tpt_math_linalg_dense::DMatrix;

    fn from_dense(rows: &[&[f64]]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(rows.len(), rows[0].len());
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v);
                }
            }
        }
        coo.to_csr()
    }

    /// Non-symmetric, diagonally dominant band with an off-band coupling, so
    /// the factor has real fill-in.
    fn convection_like(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.3);
                coo.push(i + 1, i, -0.7);
            }
            if i + 5 < n {
                coo.push(i, i + 5, 0.4);
            }
            if i >= 3 {
                coo.push(i, i - 3, -0.2);
            }
        }
        coo.to_csr()
    }

    fn residual_norm(a: &CsrMatrix<f64>, x: &DVector<f64>, b: &DVector<f64>) -> f64 {
        let ax = a.matvec(x).unwrap();
        b.iter()
            .zip(ax.iter())
            .map(|(&bi, &ai)| (bi - ai) * (bi - ai))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn solves_matrix_whose_leading_entry_is_zero() {
        // A(0,0) = 0, so the first step must pivot on a different row.
        let a = from_dense(&[&[0.0, 2.0, 1.0], &[1.0, 1.0, 0.0], &[2.0, 0.0, 3.0]]);
        let x_true = DVector::from_vec(vec![1.0, -2.0, 0.5]);
        let b = a.matvec(&x_true).unwrap();
        let lu = SparseLu::factor(&a).unwrap();
        let x = lu.solve(&b).unwrap();
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-12, "entry {i}: {}", x[i]);
        }
    }

    #[test]
    fn agrees_with_dense_lu_on_small_matrix() {
        let rows: [&[f64]; 4] = [
            &[1.0, 4.0, 0.0, 2.0],
            &[3.0, 0.5, 1.0, 0.0],
            &[0.0, 2.0, 6.0, 1.0],
            &[5.0, 0.0, 1.0, 2.0],
        ];
        let a = from_dense(&rows);
        let dense = DMatrix::from_fn(4, 4, |i, j| rows[i][j]);
        let b = DVector::from_vec(vec![1.0, -1.0, 2.0, 0.25]);
        let expected = dense.solve(&b).unwrap();
        let x = SparseLu::factor(&a).unwrap().solve(&b).unwrap();
        for i in 0..4 {
            assert!((x[i] - expected[i]).abs() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn small_residual_on_nonsymmetric_band_system() {
        let n = 120;
        let a = convection_like(n);
        let b = DVector::from_fn(n, |i| ((i as f64) * 0.21).cos());
        let lu = SparseLu::factor(&a).unwrap();
        let x = lu.solve(&b).unwrap();
        let r = residual_norm(&a, &x, &b);
        assert!(r < 1e-10, "residual {r}");
        // The factor really is sparse-but-filled: more than A's entries.
        assert!(lu.nnz() > a.nnz());
    }

    #[test]
    fn factor_is_reusable_for_several_right_hand_sides() {
        let n = 40;
        let a = convection_like(n);
        let lu = SparseLu::factor(&a).unwrap();
        for seed in 0..3 {
            let b = DVector::from_fn(n, |i| ((i + seed) as f64).sin());
            let x = lu.solve(&b).unwrap();
            assert!(residual_norm(&a, &x, &b) < 1e-10, "seed {seed}");
        }
    }

    #[test]
    fn singular_matrix_is_reported_as_breakdown() {
        // The second row is twice the first.
        let a = from_dense(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(matches!(
            SparseLu::factor(&a),
            Err(SparseError::Breakdown { .. })
        ));
    }

    #[test]
    fn rejects_non_square_and_wrong_rhs_length() {
        let rect = CooMatrix::<f64>::new(2, 3).to_csr();
        assert!(matches!(
            SparseLu::factor(&rect),
            Err(SparseError::DimensionMismatch)
        ));
        let lu = SparseLu::factor(&from_dense(&[&[2.0, 0.0], &[0.0, 2.0]])).unwrap();
        assert!(matches!(
            lu.solve(&DVector::from_vec(vec![1.0])),
            Err(SparseError::DimensionMismatch)
        ));
    }
}

#[cfg(test)]
mod ic0_tests {
    use super::*;

    fn tridiagonal_spd(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        coo.to_csr()
    }

    fn laplacian_2d(m: usize) -> CsrMatrix<f64> {
        let n = m * m;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for r in 0..m {
            for c in 0..m {
                let i = r * m + c;
                coo.push(i, i, 4.0);
                if c + 1 < m {
                    coo.push(i, i + 1, -1.0);
                    coo.push(i + 1, i, -1.0);
                }
                if r + 1 < m {
                    coo.push(i, i + m, -1.0);
                    coo.push(i + m, i, -1.0);
                }
            }
        }
        coo.to_csr()
    }

    #[test]
    fn exact_when_cholesky_has_no_fill_in() {
        // A tridiagonal SPD matrix has a bidiagonal Cholesky factor, so IC(0)
        // is the exact factor and applying it inverts A.
        let a = tridiagonal_spd(8);
        let x_true = DVector::from_fn(8, |i| (i as f64 + 1.0).ln());
        let b = a.matvec(&x_true).unwrap();
        let ic = Ic0::factor(&a).unwrap();
        let x = ic.apply(&b).unwrap();
        for i in 0..8 {
            assert!((x[i] - x_true[i]).abs() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn pcg_with_ic0_solves_2d_laplacian() {
        let a = laplacian_2d(12);
        let n = a.nrows();
        let x_true = DVector::from_fn(n, |i| (i as f64 * 0.37).sin());
        let b = a.matvec(&x_true).unwrap();
        let x = ic0_preconditioned_cg(&a, &b, None, 1e-12, 500).unwrap();
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() < 1e-8, "entry {i}");
        }
    }

    #[test]
    fn ic0_preconditioning_needs_fewer_iterations_than_plain_cg() {
        // Find the smallest iteration budget that solves the system: the
        // preconditioned solve must need strictly fewer iterations.
        let a = laplacian_2d(20);
        let n = a.nrows();
        let b = DVector::from_fn(n, |i| if i % 3 == 0 { 1.0 } else { -0.5 });
        let budget = |solve: &dyn Fn(usize) -> bool| (1..=400).find(|&k| solve(k)).unwrap();
        let plain = budget(&|k| conjugate_gradient(&a, &b, None, 1e-10, k).is_ok());
        let precond = budget(&|k| ic0_preconditioned_cg(&a, &b, None, 1e-10, k).is_ok());
        assert!(precond < plain, "preconditioned {precond} vs plain {plain}");
    }

    #[test]
    fn rejects_non_positive_pivot() {
        // Symmetric but indefinite: the second pivot is negative.
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(0, 1, 2.0);
        coo.push(1, 0, 2.0);
        coo.push(1, 1, 1.0);
        assert!(matches!(
            Ic0::factor(&coo.to_csr()),
            Err(SparseError::Breakdown { .. })
        ));
    }

    #[test]
    fn rejects_non_square_and_missing_diagonal() {
        let rect = CooMatrix::<f64>::new(2, 3).to_csr();
        assert!(matches!(
            Ic0::factor(&rect),
            Err(SparseError::DimensionMismatch)
        ));
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 0, 0.5);
        assert!(Ic0::factor(&coo.to_csr()).is_err());
    }
}

#[cfg(test)]
mod ilu_tests {
    use super::*;

    fn tridiagonal(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        coo.to_csr()
    }

    #[test]
    fn exact_when_lu_has_no_fill_in() {
        // A tridiagonal matrix produces no fill-in, so ILU(0) equals LU.
        let a = tridiagonal(8);
        let x_true = DVector::from_fn(8, |i| (i as f64 + 1.0).ln());
        let b = a.matvec(&x_true).unwrap();
        let ilu = Ilu0::factor(&a).unwrap();
        let x = ilu.apply(&b).unwrap();
        assert!((x - x_true).norm() < 1e-12);
    }

    #[test]
    fn approximate_inverse_reduces_residual_on_laplacian() {
        // 2D Laplacian: LU fills in, so ILU(0) is only approximate, but applying
        // it should still reduce the residual a lot compared with doing nothing.
        let m = 6;
        let n = m * m;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for r in 0..m {
            for c in 0..m {
                let k = r * m + c;
                coo.push(k, k, 4.0);
                if c > 0 {
                    coo.push(k, k - 1, -1.0);
                }
                if c + 1 < m {
                    coo.push(k, k + 1, -1.0);
                }
                if r > 0 {
                    coo.push(k, k - m, -1.0);
                }
                if r + 1 < m {
                    coo.push(k, k + m, -1.0);
                }
            }
        }
        let a = coo.to_csr();
        let b = DVector::from_fn(n, |i| if i % 3 == 0 { 1.0 } else { -0.5 });
        let ilu = Ilu0::factor(&a).unwrap();
        let x = ilu.apply(&b).unwrap();
        let residual = (b.clone() - a.matvec(&x).unwrap()).norm();
        assert!(
            residual < 0.5 * b.norm(),
            "residual {residual} vs |b| {}",
            b.norm()
        );
    }

    #[test]
    fn missing_diagonal_is_a_breakdown() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        assert_eq!(
            Ilu0::factor(&coo.to_csr()),
            Err(SparseError::Breakdown { iterations: 1 })
        );
    }
}

#[cfg(test)]
mod cholesky_tests {
    use super::*;

    /// 2D Laplacian (5-point stencil) on a `m × m` grid, which is SPD.
    fn laplacian_2d(m: usize) -> CsrMatrix<f64> {
        let n = m * m;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for r in 0..m {
            for c in 0..m {
                let k = r * m + c;
                coo.push(k, k, 4.0);
                if c > 0 {
                    coo.push(k, k - 1, -1.0);
                }
                if c + 1 < m {
                    coo.push(k, k + 1, -1.0);
                }
                if r > 0 {
                    coo.push(k, k - m, -1.0);
                }
                if r + 1 < m {
                    coo.push(k, k + m, -1.0);
                }
            }
        }
        coo.to_csr()
    }

    #[test]
    fn solves_2d_laplacian_and_matches_cg() {
        let a = laplacian_2d(6);
        let n = a.nrows();
        let x_true = DVector::from_fn(n, |i| (i as f64 * 0.37).sin());
        let b = a.matvec(&x_true).unwrap();
        let factor = SparseCholesky::factor(&a).unwrap();
        let x = factor.solve(&b).unwrap();
        assert!((x.clone() - x_true.clone()).norm() < 1e-10);
        let cg = conjugate_gradient(&a, &b, None, 1e-13, 1000).unwrap();
        assert!((x - cg).norm() < 1e-8);
    }

    #[test]
    fn factor_is_reusable_for_several_right_hand_sides() {
        let a = laplacian_2d(4);
        let factor = SparseCholesky::factor(&a).unwrap();
        for seed in 1..4 {
            let x_true = DVector::from_fn(a.nrows(), |i| ((i + seed) as f64).cos());
            let b = a.matvec(&x_true).unwrap();
            let x = factor.solve(&b).unwrap();
            assert!((x - x_true).norm() < 1e-10);
        }
    }

    #[test]
    fn rejects_indefinite_matrix() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 1, -1.0);
        assert_eq!(
            SparseCholesky::factor(&coo.to_csr()),
            Err(SparseError::Breakdown { iterations: 1 })
        );
    }

    #[test]
    fn rejects_wrong_rhs_length() {
        let factor = SparseCholesky::factor(&laplacian_2d(2)).unwrap();
        assert_eq!(
            factor.solve(&DVector::from_vec(vec![1.0])),
            Err(SparseError::DimensionMismatch)
        );
    }
}

#[cfg(test)]
mod preconditioner_tests {
    use super::*;

    /// SPD tridiagonal with a diagonal spanning 1 to 10⁴.
    fn scaled_tridiagonal(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            let d = 1.0 + 10_000.0 * (i as f64 / (n - 1) as f64);
            coo.push(i, i, d);
            if i + 1 < n {
                coo.push(i, i + 1, 0.5);
                coo.push(i + 1, i, 0.5);
            }
        }
        coo.to_csr()
    }

    #[test]
    fn solves_badly_scaled_spd_system() {
        let n = 50;
        let a = scaled_tridiagonal(n);
        let x_true = DVector::from_fn(n, |i| (i as f64 * 0.3).cos());
        let b = a.matvec(&x_true).unwrap();
        let x = jacobi_preconditioned_cg(&a, &b, None, 1e-12, 500).unwrap();
        let err = (x - x_true).norm();
        assert!(err < 1e-8, "error {err}");
    }

    #[test]
    fn zero_diagonal_is_a_breakdown() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 2.0);
        let a = coo.to_csr();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        assert_eq!(
            jacobi_preconditioned_cg(&a, &b, None, 1e-10, 10),
            Err(SparseError::Breakdown { iterations: 0 })
        );
    }
}

#[cfg(test)]
mod gmres_tests {
    use super::*;

    fn tridiagonal_3() -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        for &(r, c, v) in &[
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 2.0),
            (1, 1, 5.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 3.0),
        ] {
            coo.push(r, c, v);
        }
        coo.to_csr()
    }

    #[test]
    fn solves_non_symmetric_system_in_one_cycle() {
        let a = tridiagonal_3();
        let b = DVector::from_vec(vec![6.0, 15.0, 11.0]);
        let x = gmres(&a, &b, None, 1e-12, 3, 100).unwrap();
        for (got, want) in x.as_slice().iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-9, "{:?}", x.as_slice());
        }
    }

    #[test]
    fn restarted_cycles_converge_on_diagonally_dominant_system() {
        // n = 30 with a diagonal 4 and a non-symmetric off-diagonal pattern.
        let n = 30;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.5);
            }
            if i >= 1 {
                coo.push(i, i - 1, -0.5);
            }
        }
        let a = coo.to_csr();
        let x_true = DVector::from_fn(n, |i| (i as f64 + 1.0).sin());
        let b = a.matvec(&x_true).unwrap();
        // Restart 4 forces many cycles, so the restart logic is exercised.
        let x = gmres(&a, &b, None, 1e-12, 4, 500).unwrap();
        let err = (x - x_true).norm();
        assert!(err < 1e-8, "error {err}");
    }

    #[test]
    fn reports_non_convergence() {
        let a = tridiagonal_3();
        let b = DVector::from_vec(vec![6.0, 15.0, 11.0]);
        // One Arnoldi step cannot resolve a 3×3 system to 1e-14.
        assert_eq!(
            gmres(&a, &b, None, 1e-14, 1, 1),
            Err(SparseError::NotConverged { iterations: 1 })
        );
    }

    #[test]
    fn ilu_preconditioning_converges_where_plain_gmres_runs_out_of_budget() {
        // 2D Laplacian on a 10×10 grid (n = 100).
        let m = 10;
        let n = m * m;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for r in 0..m {
            for c in 0..m {
                let k = r * m + c;
                coo.push(k, k, 4.0);
                if c > 0 {
                    coo.push(k, k - 1, -1.0);
                }
                if c + 1 < m {
                    coo.push(k, k + 1, -1.0);
                }
                if r > 0 {
                    coo.push(k, k - m, -1.0);
                }
                if r + 1 < m {
                    coo.push(k, k + m, -1.0);
                }
            }
        }
        let a = coo.to_csr();
        let b = DVector::from_fn(n, |i| if i % 3 == 0 { 1.0 } else { -0.5 });
        // Measured: plain GMRES needs 26 steps for this tolerance and ILU(0)
        // needs 14. A budget of 20 separates the two.
        assert!(matches!(
            gmres(&a, &b, None, 1e-10, 40, 20),
            Err(SparseError::NotConverged { .. })
        ));
        let x = gmres_ilu0(&a, &b, None, 1e-10, 40, 20).unwrap();
        let residual = (b.clone() - a.matvec(&x).unwrap()).norm();
        assert!(residual < 1e-8 * b.norm(), "residual {residual}");
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let a = tridiagonal_3();
        let b = DVector::from_vec(vec![1.0, 2.0]);
        assert_eq!(
            gmres(&a, &b, None, 1e-10, 3, 10),
            Err(SparseError::DimensionMismatch)
        );
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let bad_guess = DVector::from_vec(vec![0.0]);
        assert_eq!(
            gmres(&a, &b, Some(bad_guess), 1e-10, 3, 10),
            Err(SparseError::DimensionMismatch)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_tolerance_is_relative_to_rhs() {
        // ‖b‖ ≈ 2e-12 is already below the absolute tol 1e-10, so the old
        // absolute test returned x0 = 0 without solving anything.
        let a = dense_2x2_csr();
        let b = DVector::from_vec(vec![1e-12, 2e-12]);
        let x = conjugate_gradient(&a, &b, None, 1e-10, 100).unwrap();
        let resid = b.clone() - a.matvec(&x).unwrap();
        assert!(resid.norm() <= 1e-10 * b.norm());
        assert!(x.norm() > 0.0);
    }

    #[test]
    fn cg_reports_breakdown_on_indefinite_matrix() {
        // diag(1, -1) with b = (1, 1): pᵀAp = 1 - 1 = 0 on the first step.
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 1, -1.0);
        let a = coo.to_csr();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        assert_eq!(
            conjugate_gradient(&a, &b, None, 1e-12, 10),
            Err(SparseError::Breakdown { iterations: 0 })
        );
    }

    #[test]
    fn matvec_into_matches_matvec_and_checks_lengths() {
        let a = dense_2x2_csr();
        let x = [1.0, 2.0];
        let mut y = [0.0; 2];
        a.matvec_into(&x, &mut y).unwrap();
        // [4 1; 1 3] * [1; 2] = [6; 7]
        assert_eq!(y, [6.0, 7.0]);
        let reference = a.matvec(&DVector::from_vec(x.to_vec())).unwrap();
        assert_eq!(reference.as_slice(), &y);
        let mut short = [0.0; 1];
        assert_eq!(
            a.matvec_into(&x, &mut short),
            Err(SparseError::DimensionMismatch)
        );
        assert_eq!(
            a.matvec_into(&[1.0], &mut y),
            Err(SparseError::DimensionMismatch)
        );
    }

    fn dense_2x2_csr() -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        coo.to_csr()
    }

    #[test]
    fn coo_to_csr_roundtrip_and_duplicate_sum() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        // Two contributions to (0,0) that must be summed, plus a duplicate that
        // is out of order.
        coo.push(0, 0, 1.0);
        coo.push(1, 1, 5.0);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, 7.0);
        coo.push(1, 0, 9.0);
        let csr = coo.to_csr();
        assert_eq!(csr.nrows(), 2);
        assert_eq!(csr.ncols(), 2);
        assert_eq!(csr.nnz(), 4);
        // Row 0: (0,0)=3, (0,1)=7 ; row 1: (1,0)=9, (1,1)=5.
        assert_eq!(csr.row_ptr, vec![0, 2, 4]);
        assert_eq!(csr.col_idx, vec![0, 1, 0, 1]);
        assert_eq!(csr.values, vec![3.0, 7.0, 9.0, 5.0]);

        let csc = coo.to_csc();
        assert_eq!(csc.col_ptr, vec![0, 2, 4]);
        assert_eq!(csc.row_idx, vec![0, 1, 0, 1]);
        assert_eq!(csc.values, vec![3.0, 9.0, 7.0, 5.0]);
    }

    #[test]
    fn counting_sort_conversion_matches_dense_accumulation() {
        // Unsorted triplets with duplicates in both row- and column-major order.
        let entries = [
            (2usize, 1usize, 1.5_f64),
            (0, 3, 2.0),
            (2, 1, -0.5),
            (1, 0, 4.0),
            (0, 0, 1.0),
            (2, 3, 3.0),
            (0, 3, 0.25),
        ];
        let (nrows, ncols) = (3, 4);
        let mut dense = [[0.0_f64; 4]; 3];
        for &(r, c, v) in &entries {
            dense[r][c] += v;
        }
        let mut coo = CooMatrix::new(nrows, ncols);
        for &(r, c, v) in &entries {
            coo.push(r, c, v);
        }
        let csr = coo.to_csr();
        let csc = coo.to_csc();
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csc.nnz(), 5);
        for (i, row) in dense.iter().enumerate() {
            for (j, &expected) in row.iter().enumerate() {
                let from_csr: f64 = csr
                    .iter()
                    .filter(|e| e.0 == i && e.1 == j)
                    .map(|e| *e.2)
                    .sum();
                let from_csc: f64 = csc
                    .iter()
                    .filter(|e| e.0 == i && e.1 == j)
                    .map(|e| *e.2)
                    .sum();
                assert_eq!(from_csr, expected, "CSR ({i},{j})");
                assert_eq!(from_csc, expected, "CSC ({i},{j})");
            }
        }
        // CSR stores columns ascending within each row.
        let mut prev: Option<(usize, usize)> = None;
        for (r, c, _) in csr.iter() {
            if let Some((pr, pc)) = prev {
                assert!(r > pr || (r == pr && c > pc), "order at ({r},{c})");
            }
            prev = Some((r, c));
        }
    }

    #[test]
    fn spmv_against_known_matrix() {
        let a = dense_2x2_csr();
        let x = DVector::from_vec(vec![2.0_f64, -1.0]);
        // [[4,1],[1,3]] * [2,-1] = [7, -1].
        let y = a.matvec(&x).unwrap();
        assert_eq!(y, DVector::from_vec(vec![7.0, -1.0]));
        // Operator form.
        let y2 = &a * &x;
        assert_eq!(y, y2);
    }

    #[test]
    fn spmv_dimension_mismatch() {
        let a = dense_2x2_csr();
        let x = DVector::from_vec(vec![1.0_f64]);
        assert_eq!(a.matvec(&x), Err(SparseError::DimensionMismatch));
    }

    #[test]
    fn transpose_csr_to_csc_and_back() {
        let a = dense_2x2_csr();
        let csc = a.transpose();
        // A^T = [[4,1],[1,3]] (symmetric) -> CSC col 0: (0,4),(1,1); col 1: (0,1),(1,3).
        assert_eq!(csc.col_ptr, vec![0, 2, 4]);
        assert_eq!(csc.row_idx, vec![0, 1, 0, 1]);
        assert_eq!(csc.values, vec![4.0, 1.0, 1.0, 3.0]);
        let back = csc.transpose();
        assert_eq!(back.row_ptr, vec![0, 2, 4]);
        assert_eq!(back.col_idx, vec![0, 1, 0, 1]);
        assert_eq!(back.values, vec![4.0, 1.0, 1.0, 3.0]);
    }

    #[test]
    fn cg_solves_spd_system() {
        let a = dense_2x2_csr();
        let b = DVector::from_vec(vec![1.0_f64, 2.0]);
        let x = conjugate_gradient(&a, &b, None, 1e-12, 100).unwrap();
        let expected = DVector::from_vec(vec![1.0 / 11.0, 7.0 / 11.0]);
        assert!((x[0] - expected[0]).abs() < 1e-10);
        assert!((x[1] - expected[1]).abs() < 1e-10);
        // Residual check.
        let r = b - a.matvec(&x).unwrap();
        assert!(r.norm() < 1e-10);
    }

    #[test]
    fn to_csr_preserves_declared_dimensions() {
        // Empty matrix and trailing empty columns keep their declared shape.
        let empty = CooMatrix::<f64>::new(3, 3).to_csr();
        assert_eq!((empty.nrows(), empty.ncols()), (3, 3));
        let mut coo = CooMatrix::<f64>::new(2, 5);
        coo.push(0, 1, 2.0);
        let a = coo.to_csr();
        assert_eq!(a.ncols(), 5);
        let x = DVector::from_vec(vec![1.0_f64; 5]);
        assert_eq!(a.matvec(&x).unwrap().len(), 2);
        assert_eq!(a.transpose().nrows(), 5);
    }

    #[test]
    fn transpose_non_square_dimensions() {
        // 2x3 matrix -> transpose is 3x2.
        let mut coo = CooMatrix::<f64>::new(2, 3);
        coo.push(0, 0, 1.0);
        coo.push(0, 2, 2.0);
        coo.push(1, 1, 3.0);
        let a = coo.to_csr();
        let csc = a.transpose();
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 2);
        // CSC column 0: (0,1),(2,2); col 1: (1,3).
        assert_eq!(csc.col_ptr, vec![0, 2, 3]);
        assert_eq!(csc.row_idx, vec![0, 2, 1]);
        assert_eq!(csc.values, vec![1.0, 2.0, 3.0]);
        let back = csc.transpose();
        assert_eq!(back.nrows(), 2);
        assert_eq!(back.ncols(), 3);
        assert_eq!(back.row_ptr, vec![0, 2, 3]);
        assert_eq!(back.col_idx, vec![0, 2, 1]);
        assert_eq!(back.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cg_non_convergence_reports_error() {
        let a = dense_2x2_csr();
        let b = DVector::from_vec(vec![1.0_f64, 2.0]);
        // One iteration is far from convergence -> NotConverged.
        let res = conjugate_gradient(&a, &b, None, 1e-14, 1);
        assert!(matches!(
            res,
            Err(SparseError::NotConverged { iterations: 1 })
        ));
    }

    #[test]
    fn bicgstab_non_convergence_reports_error() {
        let a = dense_2x2_csr();
        let b = DVector::from_vec(vec![1.0_f64, 2.0]);
        let res = bicgstab(&a, &b, None, 1e-14, 1);
        assert!(matches!(
            res,
            Err(SparseError::NotConverged { iterations: 1 })
        ));
    }

    #[test]
    fn csc_matvec_matches_csr() {
        // A = [[4,1],[1,3]] as CSC; x = [2,-1] -> [7,-1].
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        let csc = coo.to_csc();
        let x = DVector::from_vec(vec![2.0_f64, -1.0]);
        let y = csc.matvec(&x).unwrap();
        assert_eq!(y, DVector::from_vec(vec![7.0, -1.0]));
    }

    #[test]
    fn cg_2d_laplacian() {
        // 5-point 2D Laplacian on a 3x3 interior grid (9 unknowns), SPD.
        let n = 9usize;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for r in 0..3 {
            for c in 0..3 {
                let i = r * 3 + c;
                coo.push(i, i, 4.0);
                if r > 0 {
                    coo.push(i, i - 3, -1.0);
                }
                if r < 2 {
                    coo.push(i, i + 3, -1.0);
                }
                if c > 0 {
                    coo.push(i, i - 1, -1.0);
                }
                if c < 2 {
                    coo.push(i, i + 1, -1.0);
                }
            }
        }
        let a = coo.to_csr();
        let b = DVector::from_vec(vec![1.0; n]);
        let x = conjugate_gradient(&a, &b, None, 1e-10, 1000).unwrap();
        let r = b - a.matvec(&x).unwrap();
        assert!(r.norm() < 1e-8);
    }

    #[test]
    fn bicgstab_solves_nonsymmetric() {
        // A non-symmetric matrix with a known solution x = [1, 2].
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 3.0); // breaks symmetry
        coo.push(1, 1, 4.0);
        let a = coo.to_csr();
        // b = A * [1, 2] = [4, 11].
        let b = DVector::from_vec(vec![4.0_f64, 11.0]);
        let x = bicgstab(&a, &b, None, 1e-12, 200).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn solver_dimension_mismatch() {
        let a = dense_2x2_csr();
        let b = DVector::from_vec(vec![1.0_f64]);
        assert_eq!(
            conjugate_gradient(&a, &b, None, 1e-12, 10),
            Err(SparseError::DimensionMismatch)
        );
        assert_eq!(
            bicgstab(&a, &b, None, 1e-12, 10),
            Err(SparseError::DimensionMismatch)
        );
    }
}

#[cfg(test)]
mod report_tests {
    use super::*;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        coo.to_csr()
    }

    #[test]
    fn cg_report_on_convergence_counts_iterations() {
        let a = laplacian_1d(20);
        let b = DVector::from_fn(20, |i| (i as f64).sin());
        let (x, report) = conjugate_gradient_with_report(&a, &b, None, 1e-10, 500).unwrap();
        assert_eq!(report.status, SolveStatus::Converged);
        assert!(report.iterations > 0 && report.iterations <= 20);
        assert!(report.residual <= 1e-10 * b.norm());
        let ax = a.matvec(&x).unwrap();
        assert!((ax[3] - b[3]).abs() < 1e-8);
    }

    #[test]
    fn cg_report_on_budget_exhaustion_is_ok_not_err() {
        let a = laplacian_1d(50);
        let b = DVector::from_fn(50, |i| (i as f64 * 0.1).cos());
        let (_, report) = conjugate_gradient_with_report(&a, &b, None, 1e-14, 2).unwrap();
        assert_eq!(report.status, SolveStatus::NotConverged);
        assert_eq!(report.iterations, 2);
        // The plain solver reports the same situation as an error.
        assert!(matches!(
            conjugate_gradient(&a, &b, None, 1e-14, 2),
            Err(SparseError::NotConverged { iterations: 2 })
        ));
    }

    #[test]
    fn cg_report_on_indefinite_matrix_is_breakdown() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 1, -1.0);
        let a = coo.to_csr();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let (_, report) = conjugate_gradient_with_report(&a, &b, None, 1e-12, 10).unwrap();
        assert_eq!(report.status, SolveStatus::Breakdown);
    }

    #[test]
    fn bicgstab_report_matches_plain_solver() {
        let a = laplacian_1d(15);
        let b = DVector::from_fn(15, |i| 1.0 + i as f64);
        let (x, report) = bicgstab_with_report(&a, &b, None, 1e-10, 200).unwrap();
        assert!(report.status.is_converged());
        let plain = bicgstab(&a, &b, None, 1e-10, 200).unwrap();
        for i in 0..15 {
            assert!((x[i] - plain[i]).abs() < 1e-12);
        }
        let (_, starved) = bicgstab_with_report(&a, &b, None, 1e-14, 1).unwrap();
        assert_eq!(starved.status, SolveStatus::NotConverged);
    }

    #[test]
    fn report_still_rejects_dimension_mismatch() {
        let a = laplacian_1d(3);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            conjugate_gradient_with_report(&a, &b, None, 1e-8, 10),
            Err(SparseError::DimensionMismatch)
        ));
    }
}
