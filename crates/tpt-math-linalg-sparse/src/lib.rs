#![no_std]
#![forbid(unsafe_code)]
//! Sparse linear algebra implemented entirely in-house (no external backend).
//!
//! This crate provides general-purpose sparse matrix storage — coordinate
//! (triplet, [`CooMatrix`]), compressed-sparse-row ([`CsrMatrix`]) and
//! compressed-sparse-column ([`CscMatrix`]) — plus a pair of iterative solvers:
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
use tpt_math_numeric::Scalar;

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
        to_csr_impl(self.nrows, &self.rows, &self.cols, &self.vals)
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
        let nrows = self.nrows;
        let col_idx = &self.col_idx;
        let values = &self.values;
        let row_ptr = &self.row_ptr;
        let y = DVector::from_fn(nrows, |i| {
            let mut s = T::zero();
            for k in row_ptr[i]..row_ptr[i + 1] {
                s = s + values[k] * x[col_idx[k]];
            }
            s
        });
        Ok(y)
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
        to_csr_impl(self.ncols, &rows, &cols, &vals)
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

fn merge_triplets<T: Scalar + Copy>(
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> Vec<(usize, usize, T)> {
    let mut entries: Vec<(usize, usize, T)> = Vec::with_capacity(rows.len());
    for k in 0..rows.len() {
        entries.push((rows[k], cols[k], vals[k]));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut merged: Vec<(usize, usize, T)> = Vec::new();
    let mut k = 0;
    while k < entries.len() {
        let (r, c, v) = (entries[k].0, entries[k].1, entries[k].2);
        let mut summed = v;
        let mut kk = k + 1;
        while kk < entries.len() && entries[kk].0 == r && entries[kk].1 == c {
            summed = summed + entries[kk].2;
            kk += 1;
        }
        merged.push((r, c, summed));
        k = kk;
    }
    merged
}

fn to_csr_impl<T: Scalar + Copy>(
    nrows: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> CsrMatrix<T> {
    let merged = merge_triplets(rows, cols, vals);
    let mut counts = vec![0usize; nrows];
    for (r, _, _) in &merged {
        counts[*r] += 1;
    }
    let mut row_ptr = vec![0usize; nrows + 1];
    for i in 0..nrows {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }
    let mut col_idx = Vec::with_capacity(merged.len());
    let mut values = Vec::with_capacity(merged.len());
    for (_, c, v) in &merged {
        col_idx.push(*c);
        values.push(*v);
    }
    CsrMatrix {
        nrows,
        ncols: if cols.is_empty() {
            0
        } else {
            cols.iter().copied().max().map(|m| m + 1).unwrap_or(0)
        },
        row_ptr,
        col_idx,
        values,
    }
}

fn to_csc_impl<T: Scalar + Copy>(
    nrows: usize,
    ncols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> CscMatrix<T> {
    let n = rows.len();
    let mut entries: Vec<(usize, usize, T)> = Vec::with_capacity(n);
    for k in 0..n {
        entries.push((rows[k], cols[k], vals[k]));
    }
    // Sort by (col, row) for CSC layout.
    entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut merged: Vec<(usize, usize, T)> = Vec::new();
    let mut k = 0;
    while k < entries.len() {
        let (r, c, v) = (entries[k].0, entries[k].1, entries[k].2);
        let mut summed = v;
        let mut kk = k + 1;
        while kk < entries.len() && entries[kk].1 == c && entries[kk].0 == r {
            summed = summed + entries[kk].2;
            kk += 1;
        }
        merged.push((r, c, summed));
        k = kk;
    }
    let mut counts = vec![0usize; ncols];
    for (_, c, _) in &merged {
        counts[*c] += 1;
    }
    let mut col_ptr = vec![0usize; ncols + 1];
    for j in 0..ncols {
        col_ptr[j + 1] = col_ptr[j] + counts[j];
    }
    let mut row_idx = Vec::with_capacity(merged.len());
    let mut values = Vec::with_capacity(merged.len());
    for (r, _, v) in &merged {
        row_idx.push(*r);
        values.push(*v);
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
pub fn conjugate_gradient<T: Scalar + Copy>(
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
    let mut x = x0.unwrap_or_else(|| DVector::zeros(n));
    let mut r = b.clone() - a.matvec(&x)?;
    if r.norm() <= tol {
        return Ok(x);
    }
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    for it in 0..max_iter {
        let ap = a.matvec(&p)?;
        let p_ap = p.dot(&ap);
        if p_ap.abs() <= T::zero() {
            return Err(SparseError::NotConverged { iterations: it });
        }
        let alpha = rs_old / p_ap;
        x = x.clone() + p.clone() * alpha;
        r = r.clone() - ap * alpha;
        if r.norm() <= tol {
            return Ok(x);
        }
        let rs_new = r.dot(&r);
        let beta = rs_new / rs_old;
        p = r.clone() + p * beta;
        rs_old = rs_new;
    }
    Err(SparseError::NotConverged {
        iterations: max_iter,
    })
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
pub fn bicgstab<T: Scalar + Copy>(
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
    let mut x = x0.unwrap_or_else(|| DVector::zeros(n));
    let mut r = b.clone() - a.matvec(&x)?;
    if r.norm() <= tol {
        return Ok(x);
    }
    let r0 = r.clone();
    let mut p = DVector::zeros(n);
    let mut v = DVector::zeros(n);
    let mut rho = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    for it in 0..max_iter {
        let rho_new = r0.dot(&r);
        if rho_new.abs() <= T::zero() {
            return Err(SparseError::NotConverged { iterations: it });
        }
        let beta = (rho_new / rho) * (alpha / omega);
        p = r.clone() + (p - v.clone() * omega) * beta;
        let ap = a.matvec(&p)?;
        v = ap.clone();
        let r0_ap = r0.dot(&ap);
        if r0_ap.abs() <= T::zero() {
            return Err(SparseError::NotConverged { iterations: it });
        }
        alpha = rho_new / r0_ap;
        let s = r.clone() - ap * alpha;
        let t = a.matvec(&s)?;
        let t_sq = t.dot(&t);
        omega = if t_sq.abs() <= T::zero() {
            T::one()
        } else {
            t.dot(&s) / t_sq
        };
        x = x.clone() + p.clone() * alpha + s.clone() * omega;
        r = s.clone() - t.clone() * omega;
        if r.norm() <= tol {
            return Ok(x);
        }
        rho = rho_new;
    }
    Err(SparseError::NotConverged {
        iterations: max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
