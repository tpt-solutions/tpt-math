#![no_std]
#![forbid(unsafe_code)]
// Dense numeric kernels are clearest with explicit indexing; the indexed-loop
// lint does not fit this code.
#![allow(clippy::needless_range_loop)]
//! Dense linear algebra implemented entirely in-house (no external backend).
//!
//! This crate owns the dynamically-sized [`DVector`] and [`DMatrix`] types used
//! as the storage backend for `tpt-math-linalg` and `tpt-math-optimize`. The
//! matrices are stored column-major in a plain `Vec<T>`, so there is no
//! `faer`/`nalgebra` dependency and no license exposure. The arithmetic, norms,
//! and the partial-pivot-LU `solve`/`inverse` are all hand-rolled.
//!
//! Unlike the old `nalgebra`-backed path, this crate is the single storage
//! backend for `tpt-math-linalg` and `tpt-math-optimize`, so it owns its own
//! `DVector`/`DMatrix` types (which also resolves an orphan-rule problem for
//! any external solver trait impls).
//!
//! # Features
//!
//! * `std` (default) — enable the allocator and the `std` support of deps.
//! * `alloc` — signal allocator availability (dynamic vectors need it).
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg_dense::{DMatrix, DVector};
//!
//! let a = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
//! let b = DVector::from_vec(vec![4.0_f64, 5.0, 6.0]);
//! assert_eq!(a.dot(&b), 32.0);
//!
//! let m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
//! let v = DVector::from_vec(vec![1.0_f64, 1.0]);
//! let mv = m * v;
//! assert_eq!(mv[0], 3.0);
//! assert_eq!(mv[1], 7.0);
//! ```

extern crate alloc;

use core::fmt;
use core::ops::Index;
#[cfg(feature = "alloc")]
use core::ops::{Add, Div, Mul, Neg, Sub};

use tpt_math_numeric::Scalar;

// The storage types own a `Vec` whether or not the `alloc` feature is on, so the
// import is unconditional. The feature gates the constructors and algorithms.
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec};

#[cfg(feature = "alloc")]
#[cfg(feature = "alloc")]
mod diagnostics;
#[cfg(feature = "alloc")]
pub use diagnostics::{SolveDiagnostics, ILL_CONDITIONED_THRESHOLD};
#[cfg(feature = "alloc")]
mod eig;
#[cfg(feature = "alloc")]
pub use eig::Eigenvalue;
#[cfg(feature = "alloc")]
mod banded;
#[cfg(feature = "alloc")]
pub use banded::solve_banded;
#[cfg(feature = "alloc")]
mod factor;
#[cfg(feature = "alloc")]
pub use factor::{solve_tridiagonal, Cholesky};

/// A dynamically-sized column vector of `T`, stored as a contiguous `Vec`.
#[derive(Clone)]
pub struct DVector<T = f64> {
    data: Vec<T>,
}

/// A dynamically-sized matrix of `T`, stored column-major in a contiguous `Vec`
/// (`(i, j)` lives at `i + j * nrows`).
#[derive(Clone)]
pub struct DMatrix<T = f64> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

// ---------------------------------------------------------------------------
// Construction (allocator required)
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<T> DVector<T> {
    /// A zero vector of length `n`.
    pub fn zeros(n: usize) -> Self
    where
        T: Scalar,
    {
        DVector::from_fn(n, |_| T::zero())
    }

    /// Build from a `Vec` (elements in order).
    pub fn from_vec(data: Vec<T>) -> Self {
        DVector { data }
    }

    /// Build from a slice (elements in order).
    pub fn from_row_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        DVector {
            data: data.to_vec(),
        }
    }

    /// Build element-by-element with `f(i)`.
    pub fn from_fn(n: usize, f: impl FnMut(usize) -> T) -> Self {
        DVector {
            data: (0..n).map(f).collect(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<T> DMatrix<T> {
    /// A zero matrix of the given shape.
    pub fn zeros(nrows: usize, ncols: usize) -> Self
    where
        T: Scalar,
    {
        DMatrix::from_fn(nrows, ncols, |_, _| T::zero())
    }

    /// Build from a `Vec` laid out **column-major**: element `(i, j)` is
    /// `data[i + j * nrows]`.
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} does not match a {nrows}x{ncols} matrix",
            data.len()
        );
        DMatrix { nrows, ncols, data }
    }

    /// Build from a slice laid out **row-major**.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_row_slice(nrows: usize, ncols: usize, data: &[T]) -> Self
    where
        T: Clone,
    {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} does not match a {nrows}x{ncols} matrix",
            data.len()
        );
        DMatrix {
            nrows,
            ncols,
            data: (0..ncols)
                .flat_map(|j| (0..nrows).map(move |i| data[i * ncols + j].clone()))
                .collect(),
        }
    }

    /// Build element-by-element with `f(i, j)`.
    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(i, j));
            }
        }
        DMatrix { nrows, ncols, data }
    }

    /// A square matrix with `v`'s elements on the diagonal, zeros elsewhere.
    pub fn from_diagonal(v: &DVector<T>) -> Self
    where
        T: Scalar + Clone,
    {
        let n = v.len();
        DMatrix::from_fn(n, n, |i, j| if i == j { v[i] } else { T::zero() })
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl<T> DVector<T> {
    /// Number of components.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the vector has no components.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over the elements (in order).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// The elements as a contiguous slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The elements as a contiguous mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> DMatrix<T> {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// True if the matrix has no entries.
    pub fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    /// Iterate over all elements in column-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// The elements as a contiguous column-major slice (`(i, j)` is at
    /// `i + j * nrows`).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The elements as a contiguous mutable column-major slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Column-major linear index of `(i, j)`.
    fn offset(&self, i: usize, j: usize) -> usize {
        i + j * self.nrows
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

impl<T> Index<usize> for DVector<T> {
    type Output = T;
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T> Index<(usize, usize)> for DMatrix<T> {
    type Output = T;
    /// # Panics
    ///
    /// Panics if `(i, j)` is out of bounds.
    fn index(&self, (i, j): (usize, usize)) -> &T {
        // Checked per axis: a row index past `nrows` would otherwise alias
        // another element in column-major storage.
        assert!(
            i < self.nrows && j < self.ncols,
            "index ({i}, {j}) out of bounds for a {}x{} matrix",
            self.nrows,
            self.ncols
        );
        &self.data[self.offset(i, j)]
    }
}

// ---------------------------------------------------------------------------
// Norms / dot products (allocator-free)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "simd"))]
impl<T: Scalar> DVector<T> {
    /// The Euclidean (L2) norm.
    pub fn norm(&self) -> T {
        self.norm_generic()
    }

    /// Dot product with another vector of the same length.
    pub fn dot(&self, other: &DVector<T>) -> T {
        self.dot_generic(other)
    }
}

/// With the `simd` feature, `f32`/`f64` vectors use `tpt-simd-blas` (reassociated
/// sums; may differ from the scalar loop by a few ulps).
#[cfg(feature = "simd")]
impl<T: Scalar + 'static> DVector<T> {
    /// The Euclidean (L2) norm.
    pub fn norm(&self) -> T {
        if let Some(v) = simd::as_f64(&self.data) {
            return simd::from_f64(tpt_simd_blas::nrm2_f64(v));
        }
        if let Some(v) = simd::as_f32(&self.data) {
            return simd::from_f32(tpt_simd_blas::nrm2_f32(v));
        }
        self.norm_generic()
    }

    /// Dot product with another vector of the same length.
    pub fn dot(&self, other: &DVector<T>) -> T {
        if self.len() == other.len() {
            if let (Some(a), Some(b)) = (simd::as_f64(&self.data), simd::as_f64(&other.data)) {
                return simd::from_f64(tpt_simd_blas::dot_f64(a, b));
            }
            if let (Some(a), Some(b)) = (simd::as_f32(&self.data), simd::as_f32(&other.data)) {
                return simd::from_f32(tpt_simd_blas::dot_f32(a, b));
            }
        }
        self.dot_generic(other)
    }
}

impl<T: Scalar> DVector<T> {
    fn norm_generic(&self) -> T {
        // Scale by the largest magnitude so squaring cannot overflow or underflow
        // (a plain sum of squares overflows for entries near `sqrt(T::MAX)`).
        let scale = self.iter().fold(T::zero(), |acc, x| {
            let a = x.abs();
            if a > acc {
                a
            } else {
                acc
            }
        });
        if scale == T::zero() || !scale.is_finite() {
            // All-zero: norm is zero. Non-finite: let inf/NaN propagate as before.
            let s = self.iter().fold(T::zero(), |acc, x| acc + (*x) * (*x));
            return s.sqrt();
        }
        let s = self.iter().fold(T::zero(), |acc, x| {
            let r = *x / scale;
            acc + r * r
        });
        scale * s.sqrt()
    }

    fn dot_generic(&self, other: &DVector<T>) -> T {
        assert_eq!(
            self.len(),
            other.len(),
            "dot product requires equal lengths"
        );
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |acc, (a, b)| acc + (*a) * (*b))
    }
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<T: Scalar + Clone> DVector<T> {
    /// Transpose to a `1 x n` row matrix.
    pub fn transpose(&self) -> DMatrix<T> {
        let n = self.len();
        DMatrix::from_fn(1, n, |_, j| self[j])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Clone> DMatrix<T> {
    /// Transpose, swapping rows and columns.
    pub fn transpose(&self) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(n, m, |i, j| self[(j, i)])
    }
}

// ---------------------------------------------------------------------------
// Elementwise + scalar arithmetic
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<T: Scalar> Add for DVector<T> {
    type Output = DVector<T>;
    fn add(self, rhs: DVector<T>) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| self[i] + rhs[i])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Sub for DVector<T> {
    type Output = DVector<T>;
    fn sub(self, rhs: DVector<T>) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| self[i] - rhs[i])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Neg for DVector<T> {
    type Output = DVector<T>;
    fn neg(self) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| -self[i])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Mul<T> for DVector<T> {
    type Output = DVector<T>;
    fn mul(self, rhs: T) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| self[i] * rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Div<T> for DVector<T> {
    type Output = DVector<T>;
    fn div(self, rhs: T) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| self[i] / rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Add<T> for DVector<T> {
    type Output = DVector<T>;
    fn add(self, rhs: T) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| self[i] + rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Sub<T> for DVector<T> {
    type Output = DVector<T>;
    fn sub(self, rhs: T) -> DVector<T> {
        let n = self.len();
        DVector::from_fn(n, |i| self[i] - rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Add for DMatrix<T> {
    type Output = DMatrix<T>;
    fn add(self, rhs: DMatrix<T>) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| self[(i, j)] + rhs[(i, j)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Sub for DMatrix<T> {
    type Output = DMatrix<T>;
    fn sub(self, rhs: DMatrix<T>) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| self[(i, j)] - rhs[(i, j)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Neg for DMatrix<T> {
    type Output = DMatrix<T>;
    fn neg(self) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| -self[(i, j)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Mul<T> for DMatrix<T> {
    type Output = DMatrix<T>;
    fn mul(self, rhs: T) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| self[(i, j)] * rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Div<T> for DMatrix<T> {
    type Output = DMatrix<T>;
    fn div(self, rhs: T) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| self[(i, j)] / rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Add<T> for DMatrix<T> {
    type Output = DMatrix<T>;
    fn add(self, rhs: T) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| self[(i, j)] + rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Sub<T> for DMatrix<T> {
    type Output = DMatrix<T>;
    fn sub(self, rhs: T) -> DMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        DMatrix::from_fn(m, n, |i, j| self[(i, j)] - rhs)
    }
}

// ---------------------------------------------------------------------------
// In-place operators and `*_into` products (no allocation)
// ---------------------------------------------------------------------------

impl<T: Scalar> DVector<T> {
    /// `self ← self + α·x` in place.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    pub fn axpy(&mut self, alpha: T, x: &DVector<T>) {
        assert_eq!(self.len(), x.len(), "axpy: length mismatch");
        for (s, &xi) in self.data.iter_mut().zip(&x.data) {
            *s = *s + alpha * xi;
        }
    }
}

impl<T: Scalar> DMatrix<T> {
    /// `self ← self + α·x` in place.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    pub fn axpy(&mut self, alpha: T, x: &DMatrix<T>) {
        assert_eq!(
            (self.nrows, self.ncols),
            (x.nrows, x.ncols),
            "axpy: shape mismatch"
        );
        for (s, &xi) in self.data.iter_mut().zip(&x.data) {
            *s = *s + alpha * xi;
        }
    }

    /// `y ← self · x`, written into the caller's buffer `y`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.ncols()` or `y.len() != self.nrows()`.
    pub fn mul_vector_into(&self, x: &DVector<T>, y: &mut DVector<T>) {
        assert_eq!(
            self.ncols,
            x.len(),
            "mul_vector_into: column/vector mismatch"
        );
        assert_eq!(
            self.nrows,
            y.len(),
            "mul_vector_into: output length mismatch"
        );
        // Column-oriented accumulation: each column of `self` is contiguous.
        for o in y.data.iter_mut() {
            *o = T::zero();
        }
        for (k, &xk) in x.data.iter().enumerate() {
            let col = &self.data[k * self.nrows..(k + 1) * self.nrows];
            for (o, &aik) in y.data.iter_mut().zip(col) {
                *o = *o + aik * xk;
            }
        }
    }

    /// `out ← self · rhs`, written into the caller's buffer `out`.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match or `out` is not
    /// `self.nrows() × rhs.ncols()`.
    pub fn mul_matrix_into(&self, rhs: &DMatrix<T>, out: &mut DMatrix<T>) {
        assert_eq!(
            self.ncols, rhs.nrows,
            "mul_matrix_into: inner dimension mismatch"
        );
        assert_eq!(
            (out.nrows, out.ncols),
            (self.nrows, rhs.ncols),
            "mul_matrix_into: output shape mismatch"
        );
        let m = self.nrows;
        for j in 0..rhs.ncols {
            let out_col = &mut out.data[j * m..(j + 1) * m];
            for o in out_col.iter_mut() {
                *o = T::zero();
            }
            for kk in 0..self.ncols {
                let bkj = rhs.data[j * rhs.nrows + kk];
                let a_col = &self.data[kk * m..(kk + 1) * m];
                for (o, &aik) in out_col.iter_mut().zip(a_col) {
                    *o = *o + aik * bkj;
                }
            }
        }
    }
}

macro_rules! impl_assign_ops {
    ($type:ident) => {
        impl<T: Scalar> core::ops::AddAssign<&$type<T>> for $type<T> {
            /// # Panics
            ///
            /// Panics on a shape mismatch.
            fn add_assign(&mut self, rhs: &$type<T>) {
                self.axpy(T::one(), rhs);
            }
        }

        impl<T: Scalar> core::ops::SubAssign<&$type<T>> for $type<T> {
            /// # Panics
            ///
            /// Panics on a shape mismatch.
            fn sub_assign(&mut self, rhs: &$type<T>) {
                self.axpy(-T::one(), rhs);
            }
        }

        impl<T: Scalar> core::ops::MulAssign<T> for $type<T> {
            fn mul_assign(&mut self, rhs: T) {
                for s in self.data.iter_mut() {
                    *s = *s * rhs;
                }
            }
        }

        impl<T: Scalar> core::ops::DivAssign<T> for $type<T> {
            fn div_assign(&mut self, rhs: T) {
                for s in self.data.iter_mut() {
                    *s = *s / rhs;
                }
            }
        }
    };
}
impl_assign_ops!(DVector);
impl_assign_ops!(DMatrix);

#[cfg(feature = "rayon")]
impl DMatrix<f64> {
    /// Matrix product `self · rhs`, with output columns computed in parallel.
    ///
    /// Each column is accumulated in the same order as the serial product, so
    /// the result is bitwise equal to the serial scalar product. Without the `simd`
    /// feature that is `self * rhs`; with it, `*` uses a BLAS kernel that
    /// reassociates sums, so the two can differ by a few ulps. Only `f64` is
    /// parallelised: the generic `Mul` impls cannot require `Send + Sync`
    /// without breaking existing callers.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    pub fn par_mul(&self, rhs: &DMatrix<f64>) -> DMatrix<f64> {
        use rayon::prelude::*;

        assert_eq!(
            self.ncols, rhs.nrows,
            "inner dimensions do not match: {}x{} * {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols
        );
        let (m, k, n) = (self.nrows, self.ncols, rhs.ncols);
        let mut data = vec![0.0; m * n];
        if m > 0 {
            data.par_chunks_mut(m).enumerate().for_each(|(j, out_col)| {
                for kk in 0..k {
                    let bkj = rhs.data[j * rhs.nrows + kk];
                    let a_col = &self.data[kk * m..(kk + 1) * m];
                    for (o, &aik) in out_col.iter_mut().zip(a_col) {
                        *o += aik * bkj;
                    }
                }
            });
        }
        DMatrix {
            nrows: m,
            ncols: n,
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix * matrix and matrix * vector
// ---------------------------------------------------------------------------

#[cfg(all(feature = "alloc", not(feature = "simd")))]
impl<T: Scalar> Mul<DMatrix<T>> for DMatrix<T> {
    type Output = DMatrix<T>;
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    fn mul(self, rhs: DMatrix<T>) -> DMatrix<T> {
        mul_matrix_generic(&self, &rhs)
    }
}

#[cfg(all(feature = "alloc", not(feature = "simd")))]
impl<T: Scalar> Mul<DVector<T>> for DMatrix<T> {
    type Output = DVector<T>;
    /// # Panics
    ///
    /// Panics if the matrix column count does not match the vector length.
    fn mul(self, rhs: DVector<T>) -> DVector<T> {
        mul_vector_generic(&self, &rhs)
    }
}

/// With the `simd` feature, `f32`/`f64` products use `tpt-simd-blas` (blocked
/// FMA `gemm`; sums reassociate, so results can differ from the scalar loop by a
/// few ulps). Other scalar types use the generic loop.
#[cfg(feature = "simd")]
impl<T: Scalar + 'static> Mul<DMatrix<T>> for DMatrix<T> {
    type Output = DMatrix<T>;
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    fn mul(self, rhs: DMatrix<T>) -> DMatrix<T> {
        assert_eq!(
            self.ncols, rhs.nrows,
            "inner dimensions do not match: {}x{} * {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols
        );
        let (m, k, n) = (self.nrows, self.ncols, rhs.ncols);
        if let (Some(a), Some(b)) = (simd::as_f64(&self.data), simd::as_f64(&rhs.data)) {
            let mut c = vec![0.0f64; m * n];
            tpt_simd_blas::gemm_f64(
                m,
                n,
                k,
                1.0,
                a,
                m.max(1),
                b,
                k.max(1),
                0.0,
                &mut c,
                m.max(1),
            );
            return DMatrix {
                nrows: m,
                ncols: n,
                data: simd::vec_from_f64(c),
            };
        }
        if let (Some(a), Some(b)) = (simd::as_f32(&self.data), simd::as_f32(&rhs.data)) {
            let mut c = vec![0.0f32; m * n];
            tpt_simd_blas::gemm_f32(
                m,
                n,
                k,
                1.0,
                a,
                m.max(1),
                b,
                k.max(1),
                0.0,
                &mut c,
                m.max(1),
            );
            return DMatrix {
                nrows: m,
                ncols: n,
                data: simd::vec_from_f32(c),
            };
        }
        mul_matrix_generic(&self, &rhs)
    }
}

#[cfg(feature = "simd")]
impl<T: Scalar + 'static> Mul<DVector<T>> for DMatrix<T> {
    type Output = DVector<T>;
    /// # Panics
    ///
    /// Panics if the matrix column count does not match the vector length.
    fn mul(self, rhs: DVector<T>) -> DVector<T> {
        assert_eq!(
            self.ncols,
            rhs.len(),
            "matrix has {} columns but vector has length {}",
            self.ncols,
            rhs.len()
        );
        let (m, n) = (self.nrows, self.ncols);
        if let (Some(a), Some(x)) = (simd::as_f64(&self.data), simd::as_f64(&rhs.data)) {
            let mut y = vec![0.0f64; m];
            tpt_simd_blas::gemv_f64(m, n, 1.0, a, m.max(1), x, 0.0, &mut y);
            return DVector {
                data: simd::vec_from_f64(y),
            };
        }
        if let (Some(a), Some(x)) = (simd::as_f32(&self.data), simd::as_f32(&rhs.data)) {
            let mut y = vec![0.0f32; m];
            tpt_simd_blas::gemv_f32(m, n, 1.0, a, m.max(1), x, 0.0, &mut y);
            return DVector {
                data: simd::vec_from_f32(y),
            };
        }
        mul_vector_generic(&self, &rhs)
    }
}

#[cfg(feature = "alloc")]
fn mul_matrix_generic<T: Scalar>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T> {
    assert_eq!(
        a.ncols, b.nrows,
        "inner dimensions do not match: {}x{} * {}x{}",
        a.nrows, a.ncols, b.nrows, b.ncols
    );
    let m = a.nrows;
    let k = a.ncols;
    let n = b.ncols;
    // j-k-i order: column-major storage means the inner loop walks contiguous
    // columns of `a` and of the output.
    let mut data = vec![T::zero(); m * n];
    for j in 0..n {
        let out_col = &mut data[j * m..(j + 1) * m];
        for kk in 0..k {
            let bkj = b[(kk, j)];
            let a_col = &a.data[kk * m..(kk + 1) * m];
            for (o, &aik) in out_col.iter_mut().zip(a_col) {
                *o = *o + aik * bkj;
            }
        }
    }
    DMatrix {
        nrows: m,
        ncols: n,
        data,
    }
}

#[cfg(feature = "alloc")]
fn mul_vector_generic<T: Scalar>(a: &DMatrix<T>, x: &DVector<T>) -> DVector<T> {
    assert_eq!(
        a.ncols,
        x.len(),
        "matrix has {} columns but vector has length {}",
        a.ncols,
        x.len()
    );
    let m = a.nrows;
    let k = a.ncols;
    DVector::from_fn(m, |i| {
        let mut s = T::zero();
        for kk in 0..k {
            s = s + a[(i, kk)] * x[kk];
        }
        s
    })
}

/// `Any`-based `f32`/`f64` detection (safe; no `unsafe` needed).
#[cfg(feature = "simd")]
mod simd {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::any::Any;
    use tpt_math_numeric::Scalar;

    #[allow(clippy::ptr_arg)] // `Any` downcasting needs the concrete `Vec<T>`
    pub(crate) fn as_f64<T: 'static>(v: &Vec<T>) -> Option<&[f64]> {
        (v as &dyn Any)
            .downcast_ref::<Vec<f64>>()
            .map(|v| v.as_slice())
    }
    #[allow(clippy::ptr_arg)]
    pub(crate) fn as_f32<T: 'static>(v: &Vec<T>) -> Option<&[f32]> {
        (v as &dyn Any)
            .downcast_ref::<Vec<f32>>()
            .map(|v| v.as_slice())
    }
    pub(crate) fn vec_from_f64<T: 'static>(v: Vec<f64>) -> Vec<T> {
        *(Box::new(v) as Box<dyn Any>)
            .downcast::<Vec<T>>()
            .unwrap_or_else(|_| unreachable!("checked by as_f64"))
    }
    pub(crate) fn vec_from_f32<T: 'static>(v: Vec<f32>) -> Vec<T> {
        *(Box::new(v) as Box<dyn Any>)
            .downcast::<Vec<T>>()
            .unwrap_or_else(|_| unreachable!("checked by as_f32"))
    }
    pub(crate) fn from_f64<T: Scalar + 'static>(x: f64) -> T {
        *(Box::new(x) as Box<dyn Any>)
            .downcast::<T>()
            .unwrap_or_else(|_| unreachable!("checked by as_f64"))
    }
    pub(crate) fn from_f32<T: Scalar + 'static>(x: f32) -> T {
        *(Box::new(x) as Box<dyn Any>)
            .downcast::<T>()
            .unwrap_or_else(|_| unreachable!("checked by as_f32"))
    }
}

// ---------------------------------------------------------------------------
// Equality + Debug
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for DVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: PartialEq> PartialEq for DMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows && self.ncols == other.ncols && self.data == other.data
    }
}

impl<T: fmt::Debug> fmt::Debug for DVector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: fmt::Debug> fmt::Debug for DMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DMatrix")
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Fallible dense solve / inverse (partial-pivot LU over the in-house storage)
// ---------------------------------------------------------------------------

/// Errors returned by the fallible dense linear-algebra routines.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DenseError {
    /// The matrix is (numerically) singular and cannot be inverted/solved.
    Singular {
        /// Which routine detected the singular matrix.
        what: &'static str,
    },
    /// A dimension mismatch between operands.
    DimensionMismatch {
        /// Human-readable description of the conflict.
        what: String,
    },
    /// An iterative routine hit its sweep budget before converging, for example
    /// because the input contains NaN or infinity.
    NoConvergence {
        /// Which routine failed to converge.
        what: &'static str,
    },
}

#[cfg(feature = "alloc")]
impl fmt::Display for DenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenseError::Singular { what } => write!(f, "singular matrix in {what}"),
            DenseError::DimensionMismatch { what } => write!(f, "dimension mismatch: {what}"),
            DenseError::NoConvergence { what } => write!(f, "{what} did not converge"),
        }
    }
}

#[cfg(feature = "alloc")]
impl core::error::Error for DenseError {}

#[cfg(feature = "alloc")]
impl DMatrix<f64> {
    /// Solve `A x = b` for `x`, where `A` is `self`. Errors if `A` is singular
    /// or the dimensions do not match.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, DenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        if b.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("rhs has length {}, expected {n}", b.len()),
            });
        }

        let (lu, piv) = partial_pivot_lu(self)?;
        let x = solve_with_lu(&lu, &piv, b);
        Ok(DVector::from_vec(x))
    }

    /// Compute the inverse of `self`. Errors if `self` is singular.
    pub fn inverse(&self) -> Result<DMatrix<f64>, DenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        let factor = Lu::new(self)?;
        let identity = DMatrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        factor.solve_matrix(&identity)
    }
}

/// A reusable LU factorisation `P A = L U` of a square matrix.
///
/// Factor once with [`Lu::new`], then solve any number of right-hand sides with
/// [`Lu::solve`] at O(n²) per solve instead of refactoring at O(n³) each time.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Lu {
    /// Flat row-major `n × n` factors; see [`partial_pivot_lu_with`].
    lu: Vec<f64>,
    piv: Vec<usize>,
}

#[cfg(feature = "alloc")]
impl Lu {
    /// Factor the square matrix `m`.
    ///
    /// A pivot is treated as zero when it is below `1e-12` times the largest
    /// entry magnitude of `m`, so the test is scale-invariant. This is also the
    /// threshold used by [`DMatrix::solve`], [`DMatrix::inverse`] and
    /// [`DMatrix::det`].
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `m` is not square, and
    /// [`DenseError::Singular`] if it is numerically singular.
    pub fn new(m: &DMatrix<f64>) -> Result<Self, DenseError> {
        if m.nrows != m.ncols {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", m.nrows, m.ncols),
            });
        }
        let (lu, piv) = partial_pivot_lu(m)?;
        Ok(Lu { lu, piv })
    }

    /// Factor `m`, treating a pivot as singular when its magnitude is below
    /// `tol` (an absolute threshold).
    ///
    /// Use this to pin the threshold explicitly, for example to keep the absolute
    /// `1e-12` that [`Lu::new`] used before it became relative. `tol = 0.0`
    /// accepts any non-zero finite pivot.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `m` is not square, and
    /// [`DenseError::Singular`] if a pivot falls below `tol`.
    pub fn with_pivot_tolerance(m: &DMatrix<f64>, tol: f64) -> Result<Self, DenseError> {
        if m.nrows != m.ncols {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", m.nrows, m.ncols),
            });
        }
        let (lu, piv) = partial_pivot_lu_with(m, tol)?;
        Ok(Lu { lu, piv })
    }

    /// Factor `m`, treating a pivot as singular when it is below `rel` times the
    /// largest entry magnitude of `m`.
    ///
    /// This makes the singularity test scale-invariant: `A` and `1e-14 · A` are
    /// accepted or rejected together. A zero matrix is always singular.
    ///
    /// # Errors
    ///
    /// As [`Lu::with_pivot_tolerance`].
    pub fn with_relative_tolerance(m: &DMatrix<f64>, rel: f64) -> Result<Self, DenseError> {
        let scale = m.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Self::with_pivot_tolerance(m, rel * scale)
    }

    /// The dimension `n` of the factored `n × n` matrix.
    pub fn dim(&self) -> usize {
        self.piv.len()
    }

    /// Solve `A x = b` using the stored factorisation.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `b.len() != dim()`.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, DenseError> {
        let n = self.dim();
        if b.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("rhs has length {}, expected {n}", b.len()),
            });
        }
        Ok(DVector::from_vec(solve_with_lu(&self.lu, &self.piv, b)))
    }

    /// Solve `A X = B` for every column of `b` with the stored factorisation.
    ///
    /// The factors are reused across columns, so this costs O(n²) per column
    /// rather than a fresh factorisation per column.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `b` does not have `dim()` rows.
    pub fn solve_matrix(&self, b: &DMatrix<f64>) -> Result<DMatrix<f64>, DenseError> {
        let n = self.dim();
        if b.nrows() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("rhs has {} rows, expected {n}", b.nrows()),
            });
        }
        #[cfg(feature = "simd-runtime")]
        {
            if n >= SIMD_DENSE_CUTOFF {
                return simd_lu::solve_matrix(&self.lu, &self.piv, b);
            }
        }
        let mut out = Vec::with_capacity(n * b.ncols());
        for col in 0..b.ncols() {
            let column = DVector::from_fn(n, |i| b[(i, col)]);
            out.extend(solve_with_lu(&self.lu, &self.piv, &column));
        }
        // Column-major output, matching `DMatrix::from_vec`.
        Ok(DMatrix::from_vec(n, b.ncols(), out))
    }
}

/// A Householder QR factorisation `A = Q R` of a tall (`m ≥ n`) matrix, used
/// for least-squares solves. `Q` is kept implicitly as its reflectors.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Qr {
    m: usize,
    n: usize,
    /// Upper-triangular `R`, flat and row-major: entry `(i, j)` is `r[i*n + j]`.
    r: Vec<f64>,
    /// Reflectors `(k, v, vᵀv)` acting on rows `k..m`, in application order.
    reflectors: Vec<(usize, Vec<f64>, f64)>,
}

#[cfg(feature = "alloc")]
impl Qr {
    /// Factor `a` (with `nrows ≥ ncols`).
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `a` has fewer rows than
    /// columns, and [`DenseError::Singular`] if `a` is rank deficient.
    pub fn new(a: &DMatrix<f64>) -> Result<Self, DenseError> {
        let (m, n) = (a.nrows, a.ncols);
        if m < n {
            return Err(DenseError::DimensionMismatch {
                what: format!("QR needs nrows >= ncols, got {m}x{n}"),
            });
        }
        // Flat row-major working copy: entry (i, j) is `w[i*n + j]`.
        let mut w: Vec<f64> = (0..m * n).map(|idx| a[(idx / n, idx % n)]).collect();
        let mut reflectors = Vec::with_capacity(n);
        for k in 0..n {
            let norm = (k..m).map(|i| w[i * n + k] * w[i * n + k]).sum::<f64>();
            let norm = sqrt_f64(norm);
            if norm == 0.0 {
                continue;
            }
            // Choose the sign that avoids cancellation in v = x - alpha e1.
            let alpha = if w[k * n + k] > 0.0 { -norm } else { norm };
            let mut v: Vec<f64> = (k..m).map(|i| w[i * n + k]).collect();
            v[0] -= alpha;
            let vtv: f64 = v.iter().map(|x| x * x).sum();
            if vtv == 0.0 {
                continue;
            }
            for j in k..n {
                let dot: f64 = v
                    .iter()
                    .enumerate()
                    .map(|(t, vt)| vt * w[(k + t) * n + j])
                    .sum();
                let f = 2.0 * dot / vtv;
                for (t, vt) in v.iter().enumerate() {
                    w[(k + t) * n + j] -= f * vt;
                }
            }
            reflectors.push((k, v, vtv));
        }
        let r: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                if j >= i {
                    w[i * n + j]
                } else {
                    0.0
                }
            })
            .collect();
        let scale = r.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        let tol = (n as f64) * f64::EPSILON * scale;
        if (0..n).any(|i| {
            let d = r[i * n + i].abs();
            d.is_nan() || d <= tol
        }) {
            return Err(DenseError::Singular { what: "Qr::new" });
        }
        Ok(Qr {
            m,
            n,
            r,
            reflectors,
        })
    }

    /// Solve the least-squares problem `min ‖A x − b‖₂`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `b.len() != nrows`.
    pub fn solve_least_squares(&self, b: &DVector<f64>) -> Result<DVector<f64>, DenseError> {
        if b.len() != self.m {
            return Err(DenseError::DimensionMismatch {
                what: format!("rhs has length {}, expected {}", b.len(), self.m),
            });
        }
        // y = Qᵀ b by applying the reflectors in order.
        let mut y: Vec<f64> = b.iter().copied().collect();
        for (k, v, vtv) in &self.reflectors {
            let dot: f64 = v.iter().enumerate().map(|(t, vt)| vt * y[k + t]).sum();
            let f = 2.0 * dot / vtv;
            for (t, vt) in v.iter().enumerate() {
                y[k + t] -= f * vt;
            }
        }
        // Back-substitute R x = y[..n].
        let n = self.n;
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n {
                s -= self.r[i * n + j] * x[j];
            }
            x[i] = s / self.r[i * n + i];
        }
        Ok(DVector::from_vec(x))
    }
}

#[cfg(feature = "alloc")]
impl DMatrix<f64> {
    /// Determinant, from the LU factorisation.
    ///
    /// Singularity is judged relative to the largest entry (see
    /// [`Lu::new`]), so a well-conditioned matrix of small scale such as
    /// `1e-13 · I` has its true determinant. A numerically singular matrix
    /// has determinant `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the matrix is not square.
    pub fn det(&self) -> Result<f64, DenseError> {
        if self.nrows != self.ncols {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", self.nrows, self.ncols),
            });
        }
        match partial_pivot_lu(self) {
            Ok((lu, piv)) => {
                // Sign of the row permutation, from its cycle count.
                let n = piv.len();
                let mut seen = vec![false; n];
                let mut cycles = 0;
                for start in 0..n {
                    if !seen[start] {
                        cycles += 1;
                        let mut k = start;
                        while !seen[k] {
                            seen[k] = true;
                            k = piv[k];
                        }
                    }
                }
                let sign = if (n - cycles) % 2 == 0 { 1.0 } else { -1.0 };
                Ok(sign * (0..n).map(|i| lu[i * n + i]).product::<f64>())
            }
            Err(DenseError::Singular { .. }) => Ok(0.0),
            Err(e) => Err(e),
        }
    }

    /// Cholesky factorisation `A = L Lᵀ` of a symmetric positive-definite
    /// matrix, returning the lower-triangular `L`. Only the lower triangle of
    /// `self` is read.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the matrix is not square, and
    /// [`DenseError::Singular`] if it is not positive definite.
    pub fn cholesky(&self) -> Result<DMatrix<f64>, DenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        #[cfg(feature = "simd-runtime")]
        {
            if n >= SIMD_DENSE_CUTOFF {
                return simd_chol::factor(self);
            }
        }
        scalar_cholesky(self)
    }
}

/// The portable Cholesky kernel behind [`DMatrix::cholesky`].
#[cfg(feature = "alloc")]
fn scalar_cholesky(a: &DMatrix<f64>) -> Result<DMatrix<f64>, DenseError> {
    let n = a.nrows;
    // Flat row-major factor. Row i of `L` is `l[i*n..i*n + n]`, so each inner
    // sum is a dot product of two contiguous slices.
    let mut l = vec![0.0_f64; n * n];
    for j in 0..n {
        let row_j_head: f64 = l[j * n..j * n + j].iter().map(|x| x * x).sum();
        let d = a[(j, j)] - row_j_head;
        if d.is_nan() || d <= 0.0 {
            return Err(DenseError::Singular {
                what: "cholesky (not positive definite)",
            });
        }
        let ljj = sqrt_f64(d);
        l[j * n + j] = ljj;
        let (upto_i, from_i) = l.split_at_mut((j + 1) * n);
        let row_j = &upto_i[j * n..j * n + j];
        for (r, row_i) in from_i.chunks_exact_mut(n).enumerate() {
            let i = j + 1 + r;
            let dot: f64 = row_i[..j].iter().zip(row_j).map(|(a, b)| a * b).sum();
            row_i[j] = (a[(i, j)] - dot) / ljj;
        }
    }
    Ok(DMatrix::from_fn(n, n, |i, j| l[i * n + j]))
}

#[cfg(feature = "alloc")]
impl DMatrix<f64> {
    /// Eigendecomposition of a symmetric matrix by the cyclic Jacobi method.
    ///
    /// Returns `(values, vectors)` with the eigenvalues in ascending order and
    /// the orthonormal eigenvectors as the columns of `vectors`, so that
    /// `A v_k = λ_k v_k`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the matrix is not square or
    /// not symmetric (to a relative tolerance of `1e-12`).
    pub fn symmetric_eigen(&self) -> Result<(DVector<f64>, DMatrix<f64>), DenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        let scale = self.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let frob_sq: f64 = self.iter().map(|v| v * v).sum();
        for i in 0..n {
            for j in 0..i {
                if (self[(i, j)] - self[(j, i)]).abs() > 1e-12 * scale {
                    return Err(DenseError::DimensionMismatch {
                        what: "matrix is not symmetric".into(),
                    });
                }
            }
        }
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| self[(i, j)]).collect())
            .collect();
        let mut v: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();

        for _sweep in 0..100 {
            let off: f64 = (0..n)
                .flat_map(|p| ((p + 1)..n).map(move |q| (p, q)))
                .map(|(p, q)| a[p][q] * a[p][q])
                .sum();
            // Stop once the off-diagonal part is at rounding level relative to
            // ‖A‖_F, so tiny matrices such as [[0, 1e-16], [1e-16, 0]] still rotate.
            if off <= f64::EPSILON * f64::EPSILON * frob_sq {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    if a[p][q] == 0.0 {
                        continue;
                    }
                    // Rotation angle that annihilates a[p][q].
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = theta.signum() / (theta.abs() + sqrt_f64(theta * theta + 1.0));
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / sqrt_f64(t * t + 1.0);
                    let s = t * c;
                    // A ← Pᵀ A P, with P the Givens rotation in the (p, q) plane.
                    for k in 0..n {
                        let (akp, akq) = (a[k][p], a[k][q]);
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let (apk, aqk) = (a[p][k], a[q][k]);
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for row in v.iter_mut() {
                        let (vkp, vkq) = (row[p], row[q]);
                        row[p] = c * vkp - s * vkq;
                        row[q] = s * vkp + c * vkq;
                    }
                }
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
        let values = DVector::from_vec(order.iter().map(|&k| a[k][k]).collect());
        let vectors = DMatrix::from_fn(n, n, |i, j| v[i][order[j]]);
        Ok((values, vectors))
    }

    /// Least-squares solution of `min ‖self · x − b‖₂` via Householder QR.
    ///
    /// # Errors
    ///
    /// As [`Qr::new`] and [`Qr::solve_least_squares`].
    pub fn least_squares(&self, b: &DVector<f64>) -> Result<DVector<f64>, DenseError> {
        Qr::new(self)?.solve_least_squares(b)
    }
}

/// Square root for `f64` that also works without `std`. The inherent
/// `f64::sqrt` is unavailable there, so this goes through `num-traits`, which
/// uses `libm` in that configuration.
#[cfg(feature = "alloc")]
fn sqrt_f64(x: f64) -> f64 {
    tpt_math_numeric::prelude::Float::sqrt(x)
}

/// The factors `(U, σ, V)` of a thin singular value decomposition.
#[cfg(feature = "alloc")]
pub type SvdFactors = (DMatrix<f64>, DVector<f64>, DMatrix<f64>);

/// Sweep budget for the one-sided Jacobi SVD. Each sweep is a full pass over
/// all column pairs; convergence is quadratic, so this is far more than needed.
#[cfg(feature = "alloc")]
const SVD_MAX_SWEEPS: usize = 60;

/// Largest 1-norm for which the degree-13 Padé approximant meets unit-roundoff
/// accuracy without scaling (Higham 2005, θ₁₃). Above it the matrix is halved
/// until it is below.
#[cfg(feature = "alloc")]
const PADE13_THETA: f64 = 5.371_920_351_148_152;

/// Rotates columns `p` and `q` of `cols` by the plane rotation `(c, s)`.
#[cfg(feature = "alloc")]
fn rotate_columns(cols: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    let (lo, hi) = cols.split_at_mut(q);
    for (x, y) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
        let (up, uq) = (*x, *y);
        *x = c * up - s * uq;
        *y = s * up + c * uq;
    }
}

/// Dense matrix product for square operands of equal size.
#[cfg(feature = "alloc")]
fn matmul_square(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    // The flat j-k-i kernel, rather than an indexed triple loop.
    let mut out = DMatrix::zeros(n, n);
    a.mul_matrix_into(b, &mut out);
    out
}

#[cfg(feature = "alloc")]
impl DMatrix<f64> {
    /// Thin singular value decomposition `A = U · diag(σ) · Vᵀ`.
    ///
    /// For an `m × n` matrix with `k = min(m, n)`, returns `U` (`m × k`), the
    /// singular values `σ` (length `k`, sorted in descending order) and `V`
    /// (`n × k`), whose columns are orthonormal. Columns of `U` belonging to
    /// zero singular values are zero.
    ///
    /// Uses one-sided Jacobi rotations, which give high relative accuracy for
    /// small singular values. Works for any shape.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::NoConvergence`] if the rotations do not settle
    /// within the sweep budget, for example on NaN or infinite input.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_linalg_dense::DMatrix;
    ///
    /// let a = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, -4.0]);
    /// let (_, s, _) = a.svd().unwrap();
    /// assert!((s[0] - 4.0).abs() < 1e-12 && (s[1] - 3.0).abs() < 1e-12);
    /// ```
    pub fn svd(&self) -> Result<SvdFactors, DenseError> {
        let (m, n) = (self.nrows(), self.ncols());
        if m < n {
            // A = (Aᵀ)ᵀ: decompose the transpose and swap the factors.
            let (u, s, v) = self.transpose().svd()?;
            return Ok((v, s, u));
        }

        // Work column-wise: rotate pairs of columns of U = A until they are
        // mutually orthogonal, accumulating the rotations in V.
        let mut u: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..m).map(|i| self[(i, j)]).collect())
            .collect();
        let mut v: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut converged = false;
        // Squared column norms. They are recomputed exactly at the start of each
        // sweep and then carried through the rotations, which costs one dot
        // product per pair (the cross term) instead of three.
        let mut norms: Vec<f64> = vec![0.0; n];
        for _ in 0..SVD_MAX_SWEEPS {
            for (j, col) in u.iter().enumerate() {
                norms[j] = col.iter().map(|x| x * x).sum();
            }
            let mut rotated = false;
            for p in 0..n {
                for q in (p + 1)..n {
                    let (alpha, beta) = (norms[p], norms[q]);
                    let gamma: f64 = u[p].iter().zip(&u[q]).map(|(x, y)| x * y).sum();
                    if gamma == 0.0 || gamma.abs() <= f64::EPSILON * sqrt_f64(alpha * beta) {
                        continue;
                    }
                    rotated = true;
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = zeta.signum() / (zeta.abs() + sqrt_f64(1.0 + zeta * zeta));
                    let c = 1.0 / sqrt_f64(1.0 + t * t);
                    let s = c * t;
                    rotate_columns(&mut u, p, q, c, s);
                    rotate_columns(&mut v, p, q, c, s);
                    // With the rotation chosen to annihilate the cross term, the new
                    // squared norms are α − tγ and β + tγ.
                    norms[p] = alpha - t * gamma;
                    norms[q] = beta + t * gamma;
                }
            }
            if !rotated {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(DenseError::NoConvergence { what: "svd" });
        }

        // The column norms of the orthogonalised U are the singular values.
        let sigma: Vec<f64> = u
            .iter()
            .map(|col| {
                let sum: f64 = col.iter().map(|x| x * x).sum();
                sqrt_f64(sum)
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            sigma[b]
                .partial_cmp(&sigma[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let u_out = DMatrix::from_fn(m, n, |i, j| {
            let col = order[j];
            if sigma[col] > 0.0 {
                u[col][i] / sigma[col]
            } else {
                0.0
            }
        });
        let s_out = DVector::from_fn(n, |j| sigma[order[j]]);
        let v_out = DMatrix::from_fn(n, n, |i, j| v[order[j]][i]);
        Ok((u_out, s_out, v_out))
    }

    /// The singular values, sorted in descending order.
    ///
    /// # Errors
    ///
    /// As [`DMatrix::svd`].
    pub fn singular_values(&self) -> Result<DVector<f64>, DenseError> {
        Ok(self.svd()?.1)
    }

    /// Numerical rank: the number of singular values above
    /// `max(m, n) · σ_max · ε`, the usual tolerance for floating-point input.
    ///
    /// # Errors
    ///
    /// As [`DMatrix::svd`].
    pub fn rank(&self) -> Result<usize, DenseError> {
        let s = self.singular_values()?;
        let tol = self.rank_tolerance(&s);
        Ok(s.iter().filter(|&&x| x > tol).count())
    }

    /// The 2-norm condition number `σ_max / σ_min`.
    ///
    /// Returns `∞` when the matrix is rank-deficient (`σ_min` is within the
    /// [`DMatrix::rank`] tolerance) and `1` for an empty matrix.
    ///
    /// # Errors
    ///
    /// As [`DMatrix::svd`].
    pub fn cond(&self) -> Result<f64, DenseError> {
        let s = self.singular_values()?;
        if s.is_empty() {
            return Ok(1.0);
        }
        let s_min = s[s.len() - 1];
        if s_min <= self.rank_tolerance(&s) {
            return Ok(f64::INFINITY);
        }
        Ok(s[0] / s_min)
    }

    /// Moore–Penrose pseudo-inverse `A⁺ = V Σ⁺ Uᵀ`, from the thin SVD.
    ///
    /// Singular values at or below the [`DMatrix::rank`] tolerance are treated
    /// as zero, so rank-deficient and non-square matrices are handled. The
    /// result is `n × m` for an `m × n` input.
    ///
    /// # Errors
    ///
    /// As [`DMatrix::svd`].
    pub fn pseudo_inverse(&self) -> Result<DMatrix<f64>, DenseError> {
        let (m, n) = (self.nrows(), self.ncols());
        let (u, s, v) = self.svd()?;
        let tol = self.rank_tolerance(&s);
        let kept: Vec<usize> = (0..s.len()).filter(|&r| s[r] > tol).collect();
        Ok(DMatrix::from_fn(n, m, |i, j| {
            kept.iter().map(|&r| v[(i, r)] * u[(j, r)] / s[r]).sum()
        }))
    }

    /// Minimum-norm least-squares solution of `min ‖A x − b‖₂`, which is
    /// `A⁺ b`. Unlike [`Qr::solve_least_squares`] it accepts rank-deficient
    /// and wide matrices, and among the minimisers it returns the one of least
    /// norm.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `b` does not have one entry
    /// per row, and as [`DMatrix::svd`] otherwise.
    pub fn solve_min_norm(&self, b: &DVector<f64>) -> Result<DVector<f64>, DenseError> {
        if b.len() != self.nrows() {
            return Err(DenseError::DimensionMismatch {
                what: format!("rhs has length {}, expected {}", b.len(), self.nrows()),
            });
        }
        let pinv = self.pseudo_inverse()?;
        Ok(DVector::from_fn(self.ncols(), |i| {
            (0..self.nrows()).map(|j| pinv[(i, j)] * b[j]).sum()
        }))
    }

    fn rank_tolerance(&self, s: &DVector<f64>) -> f64 {
        let sigma_max = s.iter().copied().fold(0.0, f64::max);
        self.nrows().max(self.ncols()) as f64 * sigma_max * f64::EPSILON
    }

    /// The matrix exponential `e^A`, by scaling and squaring.
    ///
    /// `A` is scaled by `2^-s` until its infinity norm is at most `1/2`, the
    /// Taylor series is summed on the scaled matrix, and the result is squared
    /// `s` times.
    ///
    /// # Errors
    ///
    /// A matrix with a NaN or infinite entry has no finite exponential, so the
    /// result is a matrix of NaNs.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the matrix is not square.
    pub fn matrix_exp(&self) -> Result<DMatrix<f64>, DenseError> {
        let n = self.nrows();
        if self.ncols() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("matrix_exp needs a square matrix, got {n}x{}", self.ncols()),
            });
        }
        if self.iter().any(|v| !v.is_finite()) {
            // Without this guard, an infinite norm makes the squaring loop below
            // run forever.
            return Ok(DMatrix::from_fn(n, n, |_, _| f64::NAN));
        }
        // Higham (2005), Padé approximant of degree 13 with scaling and squaring.
        // The 1-norm (largest absolute column sum) decides how far to scale.
        let norm = (0..n)
            .map(|j| (0..n).map(|i| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0, f64::max);
        // Halve until the norm is at most θ₁₃: the count is ⌈log₂(norm/θ₁₃)⌉, found
        // without log2 or powi, which the no_std build does not have.
        let mut squarings = 0i32;
        let mut scaled_norm = norm;
        let mut scale = 1.0_f64;
        while scaled_norm > PADE13_THETA {
            scaled_norm *= 0.5;
            scale *= 0.5;
            squarings += 1;
        }
        let a = DMatrix::from_fn(n, n, |i, j| self[(i, j)] * scale);

        let identity = DMatrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        let a2 = matmul_square(&a, &a);
        let a4 = matmul_square(&a2, &a2);
        let a6 = matmul_square(&a4, &a2);
        let lin = |terms: &[(f64, &DMatrix<f64>)]| {
            DMatrix::from_fn(n, n, |i, j| terms.iter().map(|(c, m)| c * m[(i, j)]).sum())
        };
        const B: [f64; 14] = [
            64_764_752_532_480_000.0,
            32_382_376_266_240_000.0,
            7_771_770_303_897_600.0,
            1_187_353_796_428_800.0,
            129_060_195_264_000.0,
            10_559_470_521_600.0,
            670_442_572_800.0,
            33_522_128_640.0,
            1_323_241_920.0,
            40_840_800.0,
            960_960.0,
            16_380.0,
            182.0,
            1.0,
        ];
        // U = A [A⁶(b₁₃A⁶ + b₁₁A⁴ + b₉A²) + b₇A⁶ + b₅A⁴ + b₃A² + b₁I]
        let inner_u = lin(&[(B[13], &a6), (B[11], &a4), (B[9], &a2)]);
        // One scratch buffer serves every intermediate product, not a new matrix each time.
        let mut scratch = DMatrix::zeros(n, n);
        a6.mul_matrix_into(&inner_u, &mut scratch);
        let u_poly = lin(&[
            (1.0, &scratch),
            (B[7], &a6),
            (B[5], &a4),
            (B[3], &a2),
            (B[1], &identity),
        ]);
        let u = matmul_square(&a, &u_poly);
        // V = A⁶(b₁₂A⁶ + b₁₀A⁴ + b₈A²) + b₆A⁶ + b₄A⁴ + b₂A² + b₀I
        let inner_v = lin(&[(B[12], &a6), (B[10], &a4), (B[8], &a2)]);
        a6.mul_matrix_into(&inner_v, &mut scratch);
        let v = lin(&[
            (1.0, &scratch),
            (B[6], &a6),
            (B[4], &a4),
            (B[2], &a2),
            (B[0], &identity),
        ]);
        // exp(A) ≈ (V − U)⁻¹ (V + U)
        let p = lin(&[(1.0, &v), (1.0, &u)]);
        let q = lin(&[(1.0, &v), (-1.0, &u)]);
        let mut result = Lu::new(&q)?.solve_matrix(&p)?;
        for _ in 0..squarings {
            result.mul_matrix_into(&result, &mut scratch);
            core::mem::swap(&mut result, &mut scratch);
        }
        Ok(result)
    }
}

/// Default singularity threshold, relative to the largest entry magnitude of the
/// matrix being factored. A pivot below `DEFAULT_PIVOT_RELATIVE · max|a_ij|` is
/// treated as zero.
#[cfg(feature = "alloc")]
const DEFAULT_PIVOT_RELATIVE: f64 = 1e-12;

/// Returns the LU factorization (in place) and the row permutation, using the
/// default relative pivot tolerance.
#[cfg(feature = "alloc")]
fn partial_pivot_lu(m: &DMatrix<f64>) -> Result<(Vec<f64>, Vec<usize>), DenseError> {
    let scale = m.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    partial_pivot_lu_with(m, DEFAULT_PIVOT_RELATIVE * scale)
}

/// [`partial_pivot_lu`] with an explicit absolute pivot tolerance.
///
/// The factors are returned as one flat row-major buffer of `n × n` entries,
/// with `L` below the diagonal (unit diagonal implied) and `U` on and above it.
/// Elimination works on whole row slices, so the inner loop is contiguous and
/// vectorises.
#[cfg(feature = "alloc")]
fn partial_pivot_lu_with(m: &DMatrix<f64>, tol: f64) -> Result<(Vec<f64>, Vec<usize>), DenseError> {
    #[cfg(feature = "simd-runtime")]
    {
        if m.nrows >= SIMD_DENSE_CUTOFF {
            return simd_lu::factor(m, tol);
        }
    }
    scalar_lu(m, tol)
}

/// Dimension from which the `simd` feature factors with `tpt-simd-blas`. Below
/// this the scalar loop is as fast, and the conversion overhead is not repaid.
#[cfg(feature = "simd-runtime")]
const SIMD_DENSE_CUTOFF: usize = 64;

/// The portable LU kernel behind [`partial_pivot_lu_with`].
#[cfg(feature = "alloc")]
fn scalar_lu(m: &DMatrix<f64>, tol: f64) -> Result<(Vec<f64>, Vec<usize>), DenseError> {
    let n = m.nrows;
    let mut a: Vec<f64> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            a.push(m[(i, j)]);
        }
    }
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut p = k;
        let mut max = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max {
                max = v;
                p = i;
            }
        }
        if !max.is_finite() || max == 0.0 || max < tol {
            return Err(DenseError::Singular {
                what: "partial_pivot_lu",
            });
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            piv.swap(k, p);
        }
        let (top, bottom) = a.split_at_mut((k + 1) * n);
        let row_k = &top[k * n..];
        let pivot = row_k[k];
        for row_i in bottom.chunks_exact_mut(n) {
            let factor = row_i[k] / pivot;
            row_i[k] = factor;
            for (x, &u) in row_i[k + 1..].iter_mut().zip(&row_k[k + 1..]) {
                *x -= factor * u;
            }
        }
    }
    Ok((a, piv))
}

/// Solve `A x = b` given an already-computed partial-pivot LU factorization.
#[cfg(feature = "alloc")]
fn solve_with_lu(lu: &[f64], piv: &[usize], b: &DVector<f64>) -> Vec<f64> {
    let n = piv.len();
    // Forward solve L y = P b. Row i of the flat buffer holds L[i][..i] first.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let row = &lu[i * n..i * n + i];
        let dot: f64 = row.iter().zip(&y[..i]).map(|(l, yj)| l * yj).sum();
        y[i] = b[piv[i]] - dot;
    }
    // Back solve U x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let row = &lu[i * n + i + 1..i * n + n];
        let dot: f64 = row.iter().zip(&x[i + 1..]).map(|(u, xj)| u * xj).sum();
        x[i] = (y[i] - dot) / lu[i * n + i];
    }
    x
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn vector_construction_and_index() {
        let v = DVector::from_vec(vec![3.0_f64, 1.0, 4.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 3.0);
        assert_eq!(v[2], 4.0);
    }

    #[test]
    fn vector_add_sub_neg_scalar() {
        let a = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let s = a.clone() + b.clone();
        assert_eq!(s, DVector::from_vec(vec![5.0, 7.0, 9.0]));
        let d = b.clone() - a.clone();
        assert_eq!(d, DVector::from_vec(vec![3.0, 3.0, 3.0]));
        let n = -a.clone();
        assert_eq!(n, DVector::from_vec(vec![-1.0, -2.0, -3.0]));
        let sc = a * 2.0;
        assert_eq!(sc, DVector::from_vec(vec![2.0, 4.0, 6.0]));
    }

    #[test]
    fn vector_dot_and_norm() {
        let a = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0_f64, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-12);
        assert!((a.norm() - 14.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn matrix_construction_and_arithmetic() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
        let z = DMatrix::zeros(2, 2);
        assert_eq!(m.clone() - m.clone(), z);
    }

    #[test]
    fn matrix_vector_and_matrix_matrix() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        let v = DVector::from_vec(vec![1.0_f64, 1.0]);
        let mv = m.clone() * v;
        assert_eq!(mv, DVector::from_vec(vec![3.0, 7.0]));

        let n = DMatrix::from_row_slice(2, 2, &[0.0_f64, 1.0, 1.0, 0.0]);
        let mm = m.clone() * n;
        assert_eq!(mm, DMatrix::from_row_slice(2, 2, &[2.0_f64, 1.0, 4.0, 3.0]));
    }

    #[test]
    fn transpose_swaps_dimensions() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t[(0, 1)], 4.0);
    }

    #[test]
    fn solve_and_inverse() {
        // A = [[2, 0], [0, 2]] -> inverse [[0.5, 0], [0, 0.5]]
        let a = DMatrix::from_row_slice(2, 2, &[2.0_f64, 0.0, 0.0, 2.0]);
        let inv = a.inverse().unwrap();
        assert!((inv[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((inv[(1, 1)] - 0.5).abs() < 1e-12);

        let b = DVector::from_vec(vec![4.0_f64, 6.0]);
        let x = a.solve(&b).unwrap();
        assert_eq!(x, DVector::from_vec(vec![2.0, 3.0]));
    }

    #[test]
    fn determinant_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0_f64, 1.0, 2.0, 3.0]);
        assert!((a.det().unwrap() - 10.0).abs() < 1e-12);
        let b = DMatrix::from_row_slice(3, 3, &[0.0_f64, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0]);
        // Row swap (sign −1) times 2·1·3.
        assert!((b.det().unwrap() + 6.0).abs() < 1e-12);
        let singular = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 2.0, 4.0]);
        assert_eq!(singular.det().unwrap(), 0.0);
        assert!(DMatrix::<f64>::zeros(2, 3).det().is_err());
    }

    #[test]
    fn cholesky_reconstructs_and_rejects_indefinite() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
        let l = a.cholesky().unwrap();
        assert!((l[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((l[(1, 0)] - 1.0).abs() < 1e-12);
        assert!((l[(1, 1)] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(l[(0, 1)], 0.0);
        let indefinite = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 2.0, 1.0]);
        assert!(indefinite.cholesky().is_err());
    }

    #[test]
    fn symmetric_eigen_residuals_and_orthonormality() {
        let a = DMatrix::from_row_slice(3, 3, &[2.0_f64, 1.0, 0.5, 1.0, 3.0, -1.0, 0.5, -1.0, 1.0]);
        let (vals, vecs) = a.symmetric_eigen().unwrap();
        // Ascending order.
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);
        for k in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[(i, j)] * vecs[(j, k)]).sum();
                assert!(
                    (av - vals[k] * vecs[(i, k)]).abs() < 1e-10,
                    "pair {k}, row {i}"
                );
            }
        }
        for p in 0..3 {
            for q in 0..3 {
                let dot: f64 = (0..3).map(|i| vecs[(i, p)] * vecs[(i, q)]).sum();
                let want = if p == q { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10);
            }
        }
        // The trace is the sum of the eigenvalues.
        assert!((vals.iter().sum::<f64>() - 6.0).abs() < 1e-10);
        let nonsym = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 0.0, 1.0]);
        assert!(nonsym.symmetric_eigen().is_err());
    }

    #[test]
    fn least_squares_recovers_exact_line() {
        // y = 2x + 1 sampled exactly: the fit must be exact.
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for &x in &xs {
            data.push(1.0);
            data.push(x);
        }
        let a = DMatrix::from_row_slice(5, 2, &data);
        let b = DVector::from_vec(xs.iter().map(|x| 2.0 * x + 1.0).collect());
        let coef = a.least_squares(&b).unwrap();
        assert!((coef[0] - 1.0).abs() < 1e-12, "intercept {}", coef[0]);
        assert!((coef[1] - 2.0).abs() < 1e-12, "slope {}", coef[1]);
    }

    #[test]
    fn least_squares_satisfies_normal_equations() {
        // Noisy, overdetermined: the residual must be orthogonal to the columns,
        // i.e. Aᵀ(A x − b) = 0.
        let rows = 6;
        let a = DMatrix::from_fn(rows, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let b = DVector::from_vec(vec![1.1, 2.9, 5.2, 6.8, 9.1, 11.0]);
        let x = a.least_squares(&b).unwrap();
        for j in 0..2 {
            let mut g = 0.0;
            for i in 0..rows {
                let mut ri = -b[i];
                for k in 0..2 {
                    ri += a[(i, k)] * x[k];
                }
                g += a[(i, j)] * ri;
            }
            assert!(g.abs() < 1e-10, "column {j}: Aᵀr = {g}");
        }
    }

    #[test]
    fn least_squares_rejects_rank_deficient_and_wide_systems() {
        // Two identical columns: rank 1.
        let a = DMatrix::from_row_slice(3, 2, &[1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0]);
        assert!(a
            .least_squares(&DVector::from_vec(vec![1.0, 2.0, 3.0]))
            .is_err());
        // More unknowns than equations.
        let wide = DMatrix::<f64>::zeros(2, 3);
        assert!(wide
            .least_squares(&DVector::from_vec(vec![1.0, 2.0]))
            .is_err());
    }

    #[test]
    fn lu_factors_once_and_solves_many_right_hand_sides() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let lu = Lu::new(&a).unwrap();
        for k in 0..4 {
            let b = DVector::from_vec(vec![1.0 + k as f64, 2.0, -1.0]);
            let x = lu.solve(&b).unwrap();
            let want = a.solve(&b).unwrap();
            for i in 0..3 {
                assert!((x[i] - want[i]).abs() < 1e-12, "rhs {k}, component {i}");
            }
        }
    }

    #[test]
    fn lu_rejects_singular_and_mismatched_inputs() {
        let singular = DMatrix::from_row_slice(2, 2, &[1.0_f64, 1.0, 1.0, 1.0]);
        assert!(Lu::new(&singular).is_err());
        let a = DMatrix::from_row_slice(2, 2, &[2.0_f64, 0.0, 0.0, 2.0]);
        let lu = Lu::new(&a).unwrap();
        assert!(lu.solve(&DVector::from_vec(vec![1.0, 2.0, 3.0])).is_err());
        assert!(Lu::new(&DMatrix::<f64>::zeros(2, 3)).is_err());
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0_f64, 1.0, 1.0, 1.0]);
        assert!(a.inverse().is_err());
        assert!(a.solve(&DVector::from_vec(vec![1.0, 1.0])).is_err());
    }

    #[test]
    fn from_diagonal() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0_f64, 3.0, 4.0]));
        assert_eq!(d[(0, 0)], 2.0);
        assert_eq!(d[(1, 1)], 3.0);
        assert_eq!(d[(2, 2)], 4.0);
        assert_eq!(d[(0, 1)], 0.0);
    }
}

/// Checks the kernel-routed paths against the generic loops. These also run
/// (trivially) without the `simd` feature, where both sides are the same code.
#[cfg(all(test, feature = "alloc"))]
mod simd_tests {
    use super::*;

    fn a_f64(i: usize, j: usize) -> f64 {
        ((i * 7 + j * 3) % 17) as f64 * 0.25 - 2.0
    }
    fn b_f64(i: usize, j: usize) -> f64 {
        ((i * 5 + j * 11) % 13) as f64 * 0.25 - 1.5
    }

    #[test]
    fn matmul_f64_matches_generic() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (3, 5, 2),
            (17, 9, 33),
            (64, 64, 64),
            (70, 130, 65),
        ] {
            let a = DMatrix::from_fn(m, k, a_f64);
            let b = DMatrix::from_fn(k, n, b_f64);
            let want = mul_matrix_generic(&a, &b);
            let got = a * b;
            assert_eq!((got.nrows(), got.ncols()), (m, n));
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() <= 1e-9 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }
    }

    #[test]
    fn matmul_f32_matches_generic() {
        let a = DMatrix::from_fn(33, 21, |i, j| a_f64(i, j) as f32);
        let b = DMatrix::from_fn(21, 40, |i, j| b_f64(i, j) as f32);
        let want = mul_matrix_generic(&a, &b);
        let got = a * b;
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "{g} vs {w}");
        }
    }

    #[test]
    fn matvec_dot_norm_match_generic() {
        let a = DMatrix::from_fn(37, 29, a_f64);
        let x = DVector::from_fn(29, |i| b_f64(i, 1));
        let want = mul_vector_generic(&a, &x);
        let got = a * x.clone();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 1e-9 * (1.0 + w.abs()));
        }
        let y = DVector::from_fn(29, |i| a_f64(i, 2));
        assert!((x.dot(&y) - x.dot_generic(&y)).abs() <= 1e-9);
        assert!((x.norm() - x.norm_generic()).abs() <= 1e-9);
    }

    #[test]
    fn empty_and_degenerate_shapes() {
        let a = DMatrix::<f64>::zeros(0, 3);
        let b = DMatrix::<f64>::zeros(3, 4);
        let c = a * b;
        assert_eq!((c.nrows(), c.ncols()), (0, 4));
        let a = DMatrix::<f64>::zeros(2, 0);
        let b = DMatrix::<f64>::zeros(0, 3);
        let c = a * b;
        assert_eq!((c.nrows(), c.ncols()), (2, 3));
        assert!(c.iter().all(|v| *v == 0.0));
        assert_eq!(DVector::<f64>::from_vec(vec![]).norm(), 0.0);
    }

    #[test]
    fn slices_expose_column_major_data() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        assert_eq!(m.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
        let mut v = DVector::from_vec(vec![1.0_f64, 2.0]);
        v.as_mut_slice()[1] = 5.0;
        assert_eq!(v.as_slice(), &[1.0, 5.0]);
    }

    #[cfg(feature = "simd")]
    #[test]
    #[should_panic(expected = "inner dimensions")]
    fn matmul_dimension_mismatch_panics() {
        let _ = DMatrix::<f64>::zeros(2, 3) * DMatrix::<f64>::zeros(4, 2);
    }

    #[test]
    #[should_panic(expected = "inner dimensions")]
    fn generic_matmul_dimension_mismatch_panics() {
        let _ = mul_matrix_generic(&DMatrix::<f64>::zeros(2, 3), &DMatrix::<f64>::zeros(4, 2));
    }

    #[test]
    #[should_panic(expected = "columns")]
    fn generic_matvec_dimension_mismatch_panics() {
        let _ = mul_vector_generic(&DMatrix::<f64>::zeros(2, 3), &DVector::zeros(2));
    }

    #[test]
    #[should_panic(expected = "length")]
    fn dot_length_mismatch_panics() {
        let _ = DVector::from_vec(vec![1.0_f64, 2.0, 3.0]).dot(&DVector::from_vec(vec![1.0_f64]));
    }

    #[test]
    fn norm_does_not_overflow_for_large_entries() {
        // Naive sum of squares would be +inf here; the true norm is 1e200 * sqrt(2).
        let v = DVector::from_vec(vec![1e200_f64, 1e200]);
        let n = v.norm_generic();
        assert!(n.is_finite());
        assert!((n / (1e200 * 2.0_f64.sqrt()) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn norm_does_not_underflow_for_tiny_entries() {
        let v = DVector::from_vec(vec![1e-200_f64, 1e-200]);
        let n = v.norm_generic();
        assert!(n > 0.0);
        assert!((n / (1e-200 * 2.0_f64.sqrt()) - 1.0).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_row_out_of_bounds_panics() {
        // (3, 0) would alias (0, 1) in column-major storage without a check.
        let m = DMatrix::<f64>::zeros(3, 2);
        let _ = m[(3, 0)];
    }

    #[test]
    #[should_panic(expected = "length")]
    fn from_vec_length_mismatch_panics() {
        let _ = DMatrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "length")]
    fn from_row_slice_length_mismatch_panics() {
        let _ = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0, 5.0]);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod decomposition_tests {
    use super::*;

    fn assert_close(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64) {
        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() <= tol,
                    "({i},{j}): {} vs {}",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    /// `U · diag(σ) · Vᵀ`.
    fn reconstruct(u: &DMatrix<f64>, s: &DVector<f64>, v: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(u.nrows(), v.nrows(), |i, j| {
            (0..s.len()).map(|k| u[(i, k)] * s[k] * v[(j, k)]).sum()
        })
    }

    #[test]
    fn svd_reconstructs_tall_and_wide_matrices() {
        let tall = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, s, v) = tall.svd().unwrap();
        assert_close(&reconstruct(&u, &s, &v), &tall, 1e-12);
        let wide = tall.transpose();
        let (u, s, v) = wide.svd().unwrap();
        assert_close(&reconstruct(&u, &s, &v), &wide, 1e-12);
    }

    #[test]
    fn singular_values_are_sorted_and_exact_for_diagonal() {
        let d = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, -4.0]);
        let s = d.singular_values().unwrap();
        assert!((s[0] - 4.0).abs() < 1e-12);
        assert!((s[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rank_detects_dependent_rows() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(a.rank().unwrap(), 1);
        let eye = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(eye.rank().unwrap(), 3);
        assert_eq!(DMatrix::zeros(2, 2).rank().unwrap(), 0);
    }

    #[test]
    fn condition_number_of_diagonal_and_singular() {
        let a = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 1.0]);
        assert!((a.cond().unwrap() - 10.0).abs() < 1e-12);
        let singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(singular.cond().unwrap().is_infinite());
    }

    #[test]
    fn matrix_exponential_known_values() {
        let zero = DMatrix::zeros(2, 2);
        let eye = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_close(&zero.matrix_exp().unwrap(), &eye, 1e-14);

        let e = core::f64::consts::E;
        let d = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let want = DMatrix::from_row_slice(2, 2, &[e, 0.0, 0.0, e * e]);
        assert_close(&d.matrix_exp().unwrap(), &want, 1e-12);

        // Nilpotent: exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]].
        let nilpotent = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let want = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
        assert_close(&nilpotent.matrix_exp().unwrap(), &want, 1e-14);

        // Generator of rotations: exp([[0, -t], [t, 0]]) is a rotation by t.
        let t = 1.0_f64;
        let gen = DMatrix::from_row_slice(2, 2, &[0.0, -t, t, 0.0]);
        let want = DMatrix::from_row_slice(2, 2, &[t.cos(), -t.sin(), t.sin(), t.cos()]);
        assert_close(&gen.matrix_exp().unwrap(), &want, 1e-12);

        // Large entries exercise the scaling-and-squaring path.
        let big = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, -10.0]);
        let got = big.matrix_exp().unwrap();
        let want = 10f64.exp();
        assert!(((got[(0, 0)] - want) / want).abs() < 1e-12);
    }

    #[test]
    fn matrix_exp_requires_square() {
        assert!(matches!(
            DMatrix::zeros(2, 3).matrix_exp(),
            Err(DenseError::DimensionMismatch { .. })
        ));
    }
}

#[cfg(feature = "alloc")]
impl DMatrix<f64> {
    /// An eigenvector of a square matrix for a given real eigenvalue `lambda`,
    /// by inverse iteration, returned with its residual.
    ///
    /// The vector is found by repeatedly solving `(A − σI) y = x` with a shift
    /// `σ = λ + δ` just off the eigenvalue, so the solves are nearly singular and
    /// the iterates converge to the eigenvector. Returns `(v, r)` with `‖v‖ = 1`
    /// and `r = ‖A v − λ v‖`.
    ///
    /// This is for a *known* eigenvalue, for example one returned by another
    /// routine or read from a model. The matrix does not have to be symmetric.
    /// A wrong `lambda` gives a large residual, which is how to detect it. The
    /// crate does not yet compute the eigenvalues of a general real matrix.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the matrix is not square, and
    /// [`DenseError::NoConvergence`] if the residual is still above `1e-8·‖A‖` after
    /// the iteration budget. That usually means `lambda` is not an eigenvalue.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_linalg_dense::DMatrix;
    ///
    /// // A non-symmetric matrix with eigenvalues 1 and 3.
    /// let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    /// let (v, r) = a.eigenvector_for_eigenvalue(3.0).unwrap();
    /// assert!(r < 1e-9);
    /// assert!((v[0] - v[1]).abs() < 1e-9); // the eigenvector for 3 is (1, 1)/√2
    /// ```
    pub fn eigenvector_for_eigenvalue(
        &self,
        lambda: f64,
    ) -> Result<(DVector<f64>, f64), DenseError> {
        let n = self.nrows();
        if self.ncols() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!(
                    "eigenvector_for_eigenvalue needs a square matrix, got {n}x{}",
                    self.ncols()
                ),
            });
        }
        let scale = self.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
        // Shift just off the eigenvalue: singular enough to amplify the
        // eigenvector, but still factorable.
        let shift = lambda + 1e-9 * scale.max(lambda.abs());
        let shifted =
            DMatrix::from_fn(n, n, |i, j| self[(i, j)] - if i == j { shift } else { 0.0 });

        let mut x = DVector::from_fn(n, |i| 1.0 / sqrt_f64((i + 1) as f64));
        let mut residual = f64::INFINITY;
        for _ in 0..20 {
            let y = shifted.solve(&x)?;
            let norm = sqrt_f64(y.iter().map(|v| v * v).sum::<f64>());
            if !norm.is_finite() || norm == 0.0 {
                return Err(DenseError::NoConvergence {
                    what: "eigenvector_for_eigenvalue",
                });
            }
            x = DVector::from_fn(n, |i| y[i] / norm);
            // Residual of A x = λ x.
            let ax = DVector::from_fn(n, |i| (0..n).map(|j| self[(i, j)] * x[j]).sum::<f64>());
            residual = sqrt_f64(
                (0..n)
                    .map(|i| {
                        let d = ax[i] - lambda * x[i];
                        d * d
                    })
                    .sum::<f64>(),
            );
            if residual <= 1e-12 * scale {
                break;
            }
        }
        if residual <= 1e-8 * scale {
            Ok((x, residual))
        } else {
            Err(DenseError::NoConvergence {
                what: "eigenvector_for_eigenvalue",
            })
        }
    }
}

#[cfg(test)]
mod in_place_tests {
    use super::*;

    #[test]
    fn vector_compound_assignments() {
        let mut v = DVector::from_row_slice(&[1.0_f64, 2.0, 3.0]);
        let w = DVector::from_row_slice(&[10.0_f64, 20.0, 30.0]);
        v += &w;
        assert_eq!(v, DVector::from_row_slice(&[11.0, 22.0, 33.0]));
        v -= &w;
        assert_eq!(v, DVector::from_row_slice(&[1.0, 2.0, 3.0]));
        v *= 2.0;
        assert_eq!(v, DVector::from_row_slice(&[2.0, 4.0, 6.0]));
        v /= 4.0;
        assert_eq!(v, DVector::from_row_slice(&[0.5, 1.0, 1.5]));
    }

    #[test]
    fn axpy_updates_in_place() {
        let mut v = DVector::from_row_slice(&[1.0_f64, 1.0]);
        v.axpy(3.0, &DVector::from_row_slice(&[2.0, -1.0]));
        assert_eq!(v, DVector::from_row_slice(&[7.0, -2.0]));

        let mut m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 0.0, 0.0, 1.0]);
        m.axpy(-2.0, &DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(m, DMatrix::from_row_slice(2, 2, &[-1.0, -4.0, -6.0, -7.0]));
    }

    #[test]
    fn matrix_compound_assignments() {
        let mut m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        let n = DMatrix::from_row_slice(2, 2, &[1.0_f64, 1.0, 1.0, 1.0]);
        m += &n;
        m -= &n;
        assert_eq!(m, DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        m *= 0.5;
        assert_eq!(m, DMatrix::from_row_slice(2, 2, &[0.5, 1.0, 1.5, 2.0]));
    }

    #[test]
    fn mul_vector_into_matches_operator_and_overwrites_buffer() {
        let a = DMatrix::from_row_slice(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = DVector::from_row_slice(&[1.0, -1.0, 2.0]);
        let mut y = DVector::from_row_slice(&[99.0, 99.0]);
        a.mul_vector_into(&x, &mut y);
        assert_eq!(y, a.clone() * x);
    }

    #[test]
    fn mul_matrix_into_matches_operator_and_overwrites_buffer() {
        let a = DMatrix::from_row_slice(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DMatrix::from_row_slice(3, 2, &[7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = DMatrix::from_row_slice(2, 2, &[5.0_f64; 4]);
        a.mul_matrix_into(&b, &mut out);
        assert_eq!(out, a.clone() * b.clone());
        assert_eq!(
            out,
            DMatrix::from_row_slice(2, 2, &[58.0, 64.0, 139.0, 154.0])
        );
    }

    #[test]
    #[should_panic(expected = "output shape mismatch")]
    fn mul_matrix_into_rejects_wrong_output_shape() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0_f64, 0.0, 0.0, 1.0]);
        let mut out = DMatrix::zeros(3, 2);
        a.mul_matrix_into(&a.clone(), &mut out);
    }
}

#[cfg(test)]
mod lu_tolerance_tests {
    use super::*;

    #[test]
    fn default_accepts_small_scale_and_absolute_tolerance_still_rejects_it() {
        // Well-conditioned, but every pivot is below the old absolute 1e-12.
        let m = DMatrix::from_row_slice(2, 2, &[1e-13, 0.0, 0.0, 2e-13]);
        let lu = Lu::new(&m).unwrap();
        let x = lu.solve(&DVector::from_vec(vec![1e-13, 2e-13])).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
        assert!(matches!(
            Lu::with_pivot_tolerance(&m, 1e-12),
            Err(DenseError::Singular { .. })
        ));
        let lu = Lu::with_pivot_tolerance(&m, 0.0).unwrap();
        let x = lu.solve(&DVector::from_vec(vec![1e-13, 2e-13])).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn relative_tolerance_is_scale_invariant() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 2.0, 3.0]);
        let scaled = a.clone() * 1e-14;
        assert!(Lu::with_relative_tolerance(&a, 1e-10).is_ok());
        assert!(Lu::with_relative_tolerance(&scaled, 1e-10).is_ok());
        // The default test is relative, so the scaled copy is accepted too.
        assert!(Lu::new(&scaled).is_ok());
    }

    #[test]
    fn tolerance_constructors_still_reject_singular_and_zero_matrices() {
        let singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(Lu::with_relative_tolerance(&singular, 1e-10).is_err());
        assert!(Lu::with_pivot_tolerance(&singular, 0.0).is_err());
        let zero = DMatrix::zeros(3, 3);
        assert!(Lu::with_relative_tolerance(&zero, 1e-10).is_err());
        assert!(Lu::with_pivot_tolerance(&zero, 0.0).is_err());
    }
}

#[cfg(all(test, feature = "rayon"))]
mod par_mul_tests {
    use super::*;

    #[test]
    fn par_mul_is_bitwise_equal_to_serial_product() {
        // Non-square shapes, including a single column and an empty result.
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (3, 5, 2),
            (17, 9, 23),
            (64, 64, 64),
        ] {
            let a = DMatrix::from_fn(m, k, |i, j| ((i * 7 + j * 3) as f64).sin());
            let b = DMatrix::from_fn(k, n, |i, j| ((i * 5 + j) as f64).cos() * 1e-3);
            // Serial reference in the same accumulation order as `par_mul`. Not
            // `*`: with the `simd` feature that goes through a reassociating BLAS kernel.
            // Column-major reference, accumulated in the same order as `par_mul`.
            let mut reference = vec![0.0; m * n];
            for j in 0..n {
                for kk in 0..k {
                    for i in 0..m {
                        reference[j * m + i] += a[(i, kk)] * b[(kk, j)];
                    }
                }
            }
            assert_eq!(
                a.par_mul(&b).as_slice(),
                reference.as_slice(),
                "shape {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn par_mul_handles_empty_output() {
        let a = DMatrix::<f64>::zeros(0, 3);
        let b = DMatrix::<f64>::zeros(3, 4);
        assert_eq!(a.par_mul(&b).nrows(), 0);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod review_regression_tests {
    use super::*;

    #[test]
    fn matrix_exp_returns_nan_for_non_finite_input() {
        // Used to loop forever: the infinite norm never drops below 1/2.
        let m = DMatrix::from_row_slice(2, 2, &[f64::INFINITY, 0.0, 0.0, 1.0]);
        let e = m.matrix_exp().unwrap();
        assert!(e.iter().all(|v| v.is_nan()));
        let m = DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
        assert!(m.matrix_exp().unwrap().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn symmetric_eigen_resolves_values_at_rounding_scale() {
        // Used to return [0, 0]: the stop test had an absolute floor.
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1e-16, 1e-16, 0.0]);
        let (vals, _) = a.symmetric_eigen().unwrap();
        assert!((vals[0] + 1e-16).abs() < 1e-30, "got {}", vals[0]);
        assert!((vals[1] - 1e-16).abs() < 1e-30, "got {}", vals[1]);
    }

    #[test]
    fn symmetric_eigen_rejects_asymmetry_at_small_scale() {
        // Relative asymmetry of 100%, at a scale where the old absolute slack hid it.
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1e-16, 0.0, 0.0]);
        assert!(a.symmetric_eigen().is_err());
    }

    #[test]
    fn det_and_solve_work_for_small_scale_well_conditioned_matrices() {
        // Used to return Ok(0.0) and to fail with Singular.
        let a = DMatrix::from_row_slice(2, 2, &[1e-13, 0.0, 0.0, 1e-13]);
        assert!((a.det().unwrap() - 1e-26).abs() < 1e-38);
        let x = a.solve(&DVector::from_vec(vec![1e-13, 2e-13])).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        let inv = a.inverse().unwrap();
        assert!((inv[(0, 0)] - 1e13).abs() < 1e-3);
    }

    #[test]
    fn svd_reports_non_convergence_instead_of_unconverged_factors() {
        // NaN input never satisfies the rotation test, so the sweep budget runs out.
        let a = DMatrix::from_row_slice(2, 2, &[f64::NAN, 1.0, 1.0, 1.0]);
        assert!(matches!(
            a.svd(),
            Err(DenseError::NoConvergence { what: "svd" })
        ));
    }

    #[test]
    fn pseudo_inverse_satisfies_penrose_identities() {
        // Rank-deficient 3x2, and a wide 2x3 of rank 2.
        let tall = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        let wide = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 2.0, 0.0, 1.0, 1.0]);
        for a in [tall, wide] {
            let p = a.pseudo_inverse().unwrap();
            assert_eq!((p.nrows(), p.ncols()), (a.ncols(), a.nrows()));
            let apa = a.clone() * p.clone() * a.clone();
            let pap = p.clone() * a.clone() * p.clone();
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    assert!((apa[(i, j)] - a[(i, j)]).abs() < 1e-10);
                }
            }
            for i in 0..p.nrows() {
                for j in 0..p.ncols() {
                    assert!((pap[(i, j)] - p[(i, j)]).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn min_norm_solve_handles_rank_deficiency() {
        // Columns are dependent, so the minimisers form a line: x = (t, 1 - t) with
        // the minimum-norm point at t = 1/2.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let x = a
            .solve_min_norm(&DVector::from_vec(vec![1.0, 1.0]))
            .unwrap();
        assert!((x[0] - 0.5).abs() < 1e-12 && (x[1] - 0.5).abs() < 1e-12);
        assert!(a
            .solve_min_norm(&DVector::from_vec(vec![1.0, 2.0, 3.0]))
            .is_err());
    }
}

#[cfg(all(test, feature = "alloc"))]
mod solve_matrix_tests {
    use super::*;

    #[test]
    fn solve_matrix_matches_column_by_column_solves() {
        let a = DMatrix::from_fn(4, 4, |i, j| {
            if i == j {
                5.0
            } else {
                ((i + 2 * j) as f64).sin()
            }
        });
        let b = DMatrix::from_fn(4, 3, |i, j| (i * 3 + j) as f64 - 4.0);
        let lu = Lu::new(&a).unwrap();
        let x = lu.solve_matrix(&b).unwrap();
        assert_eq!((x.nrows(), x.ncols()), (4, 3));
        for col in 0..3 {
            let rhs = DVector::from_fn(4, |i| b[(i, col)]);
            let want = lu.solve(&rhs).unwrap();
            for i in 0..4 {
                assert!((x[(i, col)] - want[i]).abs() < 1e-12);
            }
        }
        assert!(lu.solve_matrix(&DMatrix::zeros(3, 1)).is_err());
    }
}

#[cfg(all(test, feature = "alloc"))]
mod pade_exp_tests {
    use super::*;

    #[test]
    fn rotation_generator_gives_a_rotation() {
        // exp([[0, -t], [t, 0]]) is the rotation by t.
        for t in [0.3_f64, 2.5, 7.0, 40.0] {
            let g = DMatrix::from_row_slice(2, 2, &[0.0, -t, t, 0.0]);
            let e = g.matrix_exp().unwrap();
            assert!((e[(0, 0)] - t.cos()).abs() < 1e-11, "t = {t}");
            assert!((e[(1, 0)] - t.sin()).abs() < 1e-11, "t = {t}");
            assert!((e[(0, 1)] + t.sin()).abs() < 1e-11, "t = {t}");
        }
    }

    #[test]
    fn large_norm_diagonal_matches_scalar_exp() {
        // The 1-norm is 20, far past θ₁₃, so several squarings run.
        let d = DMatrix::from_row_slice(2, 2, &[20.0, 0.0, 0.0, -3.0]);
        let e = d.matrix_exp().unwrap();
        assert!(((e[(0, 0)] - 20.0_f64.exp()) / 20.0_f64.exp()).abs() < 1e-12);
        assert!(((e[(1, 1)] - (-3.0_f64).exp()) / (-3.0_f64).exp()).abs() < 1e-12);
        assert!(e[(0, 1)].abs() < 1e-30 && e[(1, 0)].abs() < 1e-30);
    }

    #[test]
    fn exp_of_zero_is_identity_and_nilpotent_is_exact() {
        let z = DMatrix::<f64>::zeros(3, 3);
        let e = z.matrix_exp().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((e[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-14);
            }
        }
        // N = [[0, 1], [0, 0]] has exp(N) = I + N exactly.
        let nil = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let en = nil.matrix_exp().unwrap();
        assert!((en[(0, 1)] - 1.0).abs() < 1e-14 && (en[(0, 0)] - 1.0).abs() < 1e-14);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod svd_norm_carry_tests {
    use super::*;

    /// ‖A − U Σ Vᵀ‖_F for the thin factors returned by `svd`.
    fn residual(a: &DMatrix<f64>) -> f64 {
        let (u, s, v) = a.svd().unwrap();
        let k = s.len();
        let mut err = 0.0;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let rec: f64 = (0..k).map(|r| u[(i, r)] * s[r] * v[(j, r)]).sum();
                err += (a[(i, j)] - rec).powi(2);
            }
        }
        err.sqrt()
    }

    #[test]
    fn reconstructs_ill_conditioned_input() {
        // A Hilbert-like matrix has condition number about 1e10.
        let h = DMatrix::from_fn(8, 8, |i, j| 1.0 / (i + j + 1) as f64);
        let scale = h.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            residual(&h) < 1e-13 * scale * 8.0,
            "residual {}",
            residual(&h)
        );
        let tall = DMatrix::from_fn(10, 4, |i, j| ((i + 1) as f64).powi(j as i32));
        let scale = tall.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(residual(&tall) < 1e-12 * scale * 10.0);
    }

    #[test]
    fn singular_values_match_known_spectrum() {
        // diag(3, 1) rotated by an orthogonal matrix keeps singular values 3 and 1.
        let (c, s) = (0.6_f64, 0.8_f64);
        let rot = DMatrix::from_row_slice(2, 2, &[c, -s, s, c]);
        let d = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 1.0]);
        let a = rot.clone() * d * rot.transpose();
        let sv = a.singular_values().unwrap();
        assert!((sv[0] - 3.0).abs() < 1e-12 && (sv[1] - 1.0).abs() < 1e-12);
    }
}

/// Cholesky through `tpt-simd-blas`'s blocked `potrf`. Only the lower triangle
/// of the input is read, and the result is returned with a zero upper triangle,
/// as the scalar path does.
#[cfg(feature = "simd-runtime")]
mod simd_chol {
    use super::*;

    pub(super) fn factor(m: &DMatrix<f64>) -> Result<DMatrix<f64>, DenseError> {
        let n = m.nrows;
        let mut a = m.data.clone();
        if tpt_simd_blas::potrf_f64(n, &mut a, n).is_err() {
            return Err(DenseError::Singular {
                what: "cholesky (not positive definite)",
            });
        }
        Ok(DMatrix::from_fn(n, n, |i, j| {
            if i >= j {
                a[i + j * n]
            } else {
                0.0
            }
        }))
    }
}

/// LU through `tpt-simd-blas`'s blocked `getrf`. The kernel works on column-major
/// storage and returns LAPACK-style sequential swaps, so this module converts both
/// ways and produces the same row-major layout and permutation as the scalar path.
#[cfg(feature = "simd-runtime")]
mod simd_lu {
    use super::*;

    pub(super) fn factor(m: &DMatrix<f64>, tol: f64) -> Result<(Vec<f64>, Vec<usize>), DenseError> {
        let n = m.nrows;
        let singular = DenseError::Singular {
            what: "partial_pivot_lu",
        };
        // `DMatrix` is already column-major, which is the kernel's layout.
        let mut colmajor = m.data.clone();
        let mut ipiv = vec![0usize; n];
        if tpt_simd_blas::getrf_f64(n, n, &mut colmajor, n, &mut ipiv).is_err() {
            return Err(singular);
        }
        let mut a = vec![0.0_f64; n * n];
        for j in 0..n {
            for i in 0..n {
                a[i * n + j] = colmajor[i + j * n];
            }
        }
        // The same singularity test as the scalar path, on the pivots actually used.
        for k in 0..n {
            let u = a[k * n + k].abs();
            if !u.is_finite() || u == 0.0 || u < tol {
                return Err(singular);
            }
        }
        // Sequential swaps applied to the identity give the permutation `piv`.
        let mut piv: Vec<usize> = (0..n).collect();
        for (i, &p) in ipiv.iter().enumerate() {
            piv.swap(i, p);
        }
        Ok((a, piv))
    }

    /// Solve `A X = B` for all right-hand sides at once with `getrs`, from the
    /// row-major factors and permutation that [`factor`] produces.
    pub(super) fn solve_matrix(
        lu: &[f64],
        piv: &[usize],
        b: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, DenseError> {
        let n = piv.len();
        let nrhs = b.ncols();
        // Column-major factors, as the kernel expects.
        let mut colmajor = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                colmajor[i + j * n] = lu[i * n + j];
            }
        }
        // The kernel takes LAPACK-style sequential swaps. Rebuild them from the
        // permutation: position k must end up holding original row `piv[k]`.
        let mut current: Vec<usize> = (0..n).collect();
        let mut ipiv = vec![0usize; n];
        for k in 0..n {
            let p = (k..n)
                .find(|&p| current[p] == piv[k])
                .ok_or(DenseError::Singular {
                    what: "solve_matrix",
                })?;
            current.swap(k, p);
            ipiv[k] = p;
        }
        // `DMatrix` is column-major, so the right-hand sides can be passed as is.
        let mut x = b.data.clone();
        tpt_simd_blas::getrs_f64(n, nrhs, &colmajor, n, &ipiv, &mut x, n);
        Ok(DMatrix::from_vec(n, nrhs, x))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn dominant(n: usize) -> DMatrix<f64> {
            DMatrix::from_fn(n, n, |i, j| {
                if i == j {
                    n as f64 + 1.0
                } else {
                    ((i * 3 + j * 5) as f64).sin()
                }
            })
        }

        #[test]
        fn simd_factors_agree_with_scalar_factors() {
            for n in [5usize, 33, 200] {
                let a = dominant(n);
                let (s_lu, s_piv) = scalar_lu(&a, 0.0).unwrap();
                let (v_lu, v_piv) = factor(&a, 0.0).unwrap();
                assert_eq!(s_piv, v_piv, "permutation, n = {n}");
                for (x, y) in s_lu.iter().zip(&v_lu) {
                    assert!(
                        (x - y).abs() <= 1e-12 * (1.0 + x.abs()),
                        "n = {n}: {x} vs {y}"
                    );
                }
            }
        }

        #[test]
        fn simd_reports_singular_like_scalar() {
            // Two identical rows: exactly singular.
            let n = 80;
            let a = DMatrix::from_fn(n, n, |i, j| {
                if i <= 1 {
                    ((j + 1) as f64).sin()
                } else {
                    ((i * 7 + j) as f64).cos() + if i == j { 3.0 } else { 0.0 }
                }
            });
            assert!(matches!(factor(&a, 0.0), Err(DenseError::Singular { .. })));
            assert!(matches!(
                scalar_lu(&a, 0.0),
                Err(DenseError::Singular { .. })
            ));
        }
    }
}

#[cfg(all(test, feature = "simd-runtime"))]
mod simd_dense_solve_tests {
    use super::*;

    fn dominant_nonsymmetric(n: usize) -> DMatrix<f64> {
        DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 1.0
            } else {
                ((i * 3 + j * 5) as f64).sin()
            }
        })
    }

    fn spd(n: usize) -> DMatrix<f64> {
        // Symmetric, strictly diagonally dominant, so positive definite.
        DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 2.0
            } else {
                1.0 / (1.0 + i.abs_diff(j) as f64)
            }
        })
    }

    #[test]
    fn simd_solve_matrix_agrees_with_the_scalar_solve() {
        for n in [5usize, 33, 200] {
            let a = dominant_nonsymmetric(n);
            let (lu, piv) = scalar_lu(&a, 0.0).unwrap();
            let rhs = DMatrix::from_fn(n, 3, |i, j| (i as f64 * 0.3 + j as f64).cos());
            let simd = simd_lu::solve_matrix(&lu, &piv, &rhs).unwrap();
            for col in 0..3 {
                let column = DVector::from_fn(n, |i| rhs[(i, col)]);
                let want = solve_with_lu(&lu, &piv, &column);
                for i in 0..n {
                    let got = simd[(i, col)];
                    assert!(
                        (got - want[i]).abs() <= 1e-10 * want[i].abs().max(1.0),
                        "n = {n}, col {col}, row {i}: {got} vs {}",
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn simd_cholesky_agrees_with_the_scalar_cholesky() {
        for n in [5usize, 33, 200] {
            let a = spd(n);
            let want = scalar_cholesky(&a).unwrap();
            let got = simd_chol::factor(&a).unwrap();
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (got[(i, j)] - want[(i, j)]).abs() <= 1e-10 * want[(i, j)].abs().max(1.0),
                        "n = {n} at ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_cholesky_rejects_an_indefinite_matrix() {
        let n = 80;
        let a = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                if i == 40 {
                    -1.0
                } else {
                    2.0
                }
            } else {
                0.0
            }
        });
        assert!(simd_chol::factor(&a).is_err());
        assert!(a.cholesky().is_err());
    }

    #[test]
    fn determinant_sign_is_right_after_many_row_swaps() {
        // The cyclic shift on n rows is an n-cycle, with n − 1 transpositions, so
        // its determinant is (−1)^(n−1). Here n = 66 is above the SIMD cutoff.
        let n = 66;
        let shift = DMatrix::from_fn(n, n, |i, j| if j == (i + 1) % n { 1.0 } else { 0.0 });
        let det = shift.det().unwrap();
        assert!((det + 1.0).abs() < 1e-12, "det = {det}");
    }

    #[test]
    fn near_singular_inputs_are_classified_the_same_on_both_paths() {
        // Order 70 is above the SIMD cutoff. The last pivot sits on either side of
        // the relative threshold (1e-12 · max|aᵢⱼ| = 1e-12).
        let n = 70;
        for (last, singular) in [(1e-14, true), (1e-10, false)] {
            let a = DMatrix::from_fn(n, n, |i, j| match (i == j, i == n - 1) {
                (true, true) => last,
                (true, false) => 1.0,
                _ => 0.0,
            });
            let scale = 1.0;
            let simd = partial_pivot_lu_with(&a, DEFAULT_PIVOT_RELATIVE * scale);
            let scalar = scalar_lu(&a, DEFAULT_PIVOT_RELATIVE * scale);
            assert_eq!(simd.is_err(), singular, "simd path, last pivot {last}");
            assert_eq!(scalar.is_err(), singular, "scalar path, last pivot {last}");
            if !singular {
                let (lu_s, p_s) = scalar.unwrap();
                let (lu_v, p_v) = simd.unwrap();
                assert_eq!(p_s, p_v);
                for (x, y) in lu_s.iter().zip(&lu_v) {
                    assert!((x - y).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn lu_solve_matrix_dispatch_matches_small_path() {
        // Above the cutoff, `Lu::solve_matrix` takes the SIMD branch; check it
        // against a solve column by column.
        let n = 70;
        let a = dominant_nonsymmetric(n);
        let lu = Lu::new(&a).unwrap();
        let rhs = DMatrix::from_fn(n, 2, |i, j| (i + j) as f64);
        let x = lu.solve_matrix(&rhs).unwrap();
        let back = a.clone() * x;
        for i in 0..n {
            for j in 0..2 {
                assert!((back[(i, j)] - rhs[(i, j)]).abs() < 1e-9);
            }
        }
    }
}
