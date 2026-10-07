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
use core::ops::{Add, Div, Index, Mul, Neg, Sub};

use tpt_math_numeric::Scalar;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec};

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
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        DMatrix { nrows, ncols, data }
    }

    /// Build from a slice laid out **row-major**.
    pub fn from_row_slice(nrows: usize, ncols: usize, data: &[T]) -> Self
    where
        T: Clone,
    {
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
        let s = self.iter().fold(T::zero(), |acc, x| acc + (*x) * (*x));
        s.sqrt()
    }

    fn dot_generic(&self, other: &DVector<T>) -> T {
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
    let m = a.nrows;
    let k = a.ncols;
    let n = b.ncols;
    DMatrix::from_fn(m, n, |i, j| {
        let mut s = T::zero();
        for kk in 0..k {
            s = s + a[(i, kk)] * b[(kk, j)];
        }
        s
    })
}

#[cfg(feature = "alloc")]
fn mul_vector_generic<T: Scalar>(a: &DMatrix<T>, x: &DVector<T>) -> DVector<T> {
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
}

#[cfg(feature = "alloc")]
impl fmt::Display for DenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenseError::Singular { what } => write!(f, "singular matrix in {what}"),
            DenseError::DimensionMismatch { what } => write!(f, "dimension mismatch: {what}"),
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
        let (lu, piv) = partial_pivot_lu(self)?;
        let mut out = Vec::with_capacity(n * n);
        for col in 0..n {
            let mut e = vec![0.0_f64; n];
            e[col] = 1.0;
            let x = solve_with_lu(&lu, &piv, &DVector::from_vec(e));
            out.extend(x);
        }
        Ok(DMatrix::from_vec(n, n, out))
    }
}

/// Returns the LU factorization (in place) and the row permutation.
#[cfg(feature = "alloc")]
fn partial_pivot_lu(m: &DMatrix<f64>) -> Result<(Vec<Vec<f64>>, Vec<usize>), DenseError> {
    let n = m.nrows;
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| m[(i, j)]).collect())
        .collect();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut p = k;
        let mut max = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > max {
                max = v;
                p = i;
            }
        }
        if !max.is_finite() || max < 1e-12 {
            return Err(DenseError::Singular {
                what: "partial_pivot_lu",
            });
        }
        a.swap(k, p);
        piv.swap(k, p);
        for i in (k + 1)..n {
            a[i][k] /= a[k][k];
            for j in (k + 1)..n {
                a[i][j] -= a[i][k] * a[k][j];
            }
        }
    }
    Ok((a, piv))
}

/// Solve `A x = b` given an already-computed partial-pivot LU factorization.
#[cfg(feature = "alloc")]
fn solve_with_lu(lu: &[Vec<f64>], piv: &[usize], b: &DVector<f64>) -> Vec<f64> {
    let n = lu.len();
    // Forward solve L y = P b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[piv[i]];
        for j in 0..i {
            s -= lu[i][j] * y[j];
        }
        y[i] = s;
    }
    // Back solve U x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= lu[i][j] * x[j];
        }
        x[i] = s / lu[i][i];
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
}
