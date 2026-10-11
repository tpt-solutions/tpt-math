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

/// A reusable LU factorisation `P A = L U` of a square matrix.
///
/// Factor once with [`Lu::new`], then solve any number of right-hand sides with
/// [`Lu::solve`] at O(n²) per solve instead of refactoring at O(n³) each time.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Lu {
    lu: Vec<Vec<f64>>,
    piv: Vec<usize>,
}

#[cfg(feature = "alloc")]
impl Lu {
    /// Factor the square matrix `m`.
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

    /// The dimension `n` of the factored `n × n` matrix.
    pub fn dim(&self) -> usize {
        self.lu.len()
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
}

/// A Householder QR factorisation `A = Q R` of a tall (`m ≥ n`) matrix, used
/// for least-squares solves. `Q` is kept implicitly as its reflectors.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Qr {
    m: usize,
    n: usize,
    /// Upper-triangular `R`, row-major `n × n`.
    r: Vec<Vec<f64>>,
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
        let mut w: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..n).map(|j| a[(i, j)]).collect())
            .collect();
        let mut reflectors = Vec::with_capacity(n);
        for k in 0..n {
            let norm = w[k..m]
                .iter()
                .map(|row| row[k] * row[k])
                .sum::<f64>()
                .sqrt();
            if norm == 0.0 {
                continue;
            }
            // Choose the sign that avoids cancellation in v = x - alpha e1.
            let alpha = if w[k][k] > 0.0 { -norm } else { norm };
            let mut v: Vec<f64> = (k..m).map(|i| w[i][k]).collect();
            v[0] -= alpha;
            let vtv: f64 = v.iter().map(|x| x * x).sum();
            if vtv == 0.0 {
                continue;
            }
            for j in k..n {
                let dot: f64 = v.iter().enumerate().map(|(t, vt)| vt * w[k + t][j]).sum();
                let f = 2.0 * dot / vtv;
                for (t, vt) in v.iter().enumerate() {
                    w[k + t][j] -= f * vt;
                }
            }
            reflectors.push((k, v, vtv));
        }
        let r: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if j >= i { w[i][j] } else { 0.0 }).collect())
            .collect();
        let scale = r.iter().flatten().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        let tol = (n as f64) * f64::EPSILON * scale;
        if (0..n).any(|i| {
            let d = r[i][i].abs();
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
                s -= self.r[i][j] * x[j];
            }
            x[i] = s / self.r[i][i];
        }
        Ok(DVector::from_vec(x))
    }
}

#[cfg(feature = "alloc")]
impl DMatrix<f64> {
    /// Determinant, from the LU factorisation.
    ///
    /// A matrix the LU routine rejects as singular (pivot below `1e-12`) has
    /// determinant `0.0`.
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
                Ok(sign * (0..n).map(|i| lu[i][i]).product::<f64>())
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
        let mut l = vec![vec![0.0_f64; n]; n];
        for j in 0..n {
            let d = self[(j, j)] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
            if d.is_nan() || d <= 0.0 {
                return Err(DenseError::Singular {
                    what: "cholesky (not positive definite)",
                });
            }
            l[j][j] = d.sqrt();
            for i in (j + 1)..n {
                let s = self[(i, j)] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
                l[i][j] = s / l[j][j];
            }
        }
        Ok(DMatrix::from_fn(n, n, |i, j| l[i][j]))
    }

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
        for i in 0..n {
            for j in 0..i {
                if (self[(i, j)] - self[(j, i)]).abs() > 1e-12 * (1.0 + scale) {
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
            if off <= 1e-30 * (1.0 + scale * scale) {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    if a[p][q] == 0.0 {
                        continue;
                    }
                    // Rotation angle that annihilates a[p][q].
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
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
