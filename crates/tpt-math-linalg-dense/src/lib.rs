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
use alloc::{format, string::String};

#[cfg(all(feature = "alloc", any(test, not(feature = "simd"))))]
use alloc::vec;

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

impl<T: Scalar> DVector<T> {
    /// The Euclidean (L2) norm.
    pub fn norm(&self) -> T {
        let s = self.iter().fold(T::zero(), |acc, x| acc + (*x) * (*x));
        s.sqrt()
    }

    /// Dot product with another vector of the same length.
    pub fn dot(&self, other: &DVector<T>) -> T {
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

/// Note: with the `simd` feature, `T: 'static` is required (the operands are
/// moved into the backend without copying).
#[cfg(feature = "alloc")]
impl<T: Scalar + 'static> Mul<DMatrix<T>> for DMatrix<T> {
    type Output = DMatrix<T>;
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    fn mul(self, rhs: DMatrix<T>) -> DMatrix<T> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match simd::gemm(self, rhs) {
            Ok(out) => return out,
            Err(operands) => operands,
        };
        #[cfg(not(feature = "simd"))]
        let (lhs, rhs) = (self, rhs);

        let m = lhs.nrows;
        let k = lhs.ncols;
        let n = rhs.ncols;
        DMatrix::from_fn(m, n, |i, j| {
            let mut s = T::zero();
            for kk in 0..k {
                s = s + lhs[(i, kk)] * rhs[(kk, j)];
            }
            s
        })
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + 'static> Mul<DVector<T>> for DMatrix<T> {
    type Output = DVector<T>;
    /// # Panics
    ///
    /// Panics if the matrix column count does not match the vector length.
    fn mul(self, rhs: DVector<T>) -> DVector<T> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match simd::gemv(self, rhs) {
            Ok(out) => return out,
            Err(operands) => operands,
        };
        #[cfg(not(feature = "simd"))]
        let (lhs, rhs) = (self, rhs);

        let m = lhs.nrows;
        let k = lhs.ncols;
        DVector::from_fn(m, |i| {
            let mut s = T::zero();
            for kk in 0..k {
                s = s + lhs[(i, kk)] * rhs[kk];
            }
            s
        })
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

        #[cfg(feature = "simd")]
        let x = simd::lu_solve(self, b.iter().copied().collect())?;
        #[cfg(not(feature = "simd"))]
        let x = {
            let (lu, piv) = partial_pivot_lu(self)?;
            solve_with_lu(&lu, &piv, b)
        };
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
        #[cfg(feature = "simd")]
        let out = {
            // Identity right-hand side, column-major: 1 where row == column.
            let identity: Vec<f64> = (0..n * n)
                .map(|idx| if idx % n == idx / n { 1.0 } else { 0.0 })
                .collect();
            simd::lu_solve(self, identity)?
        };
        #[cfg(not(feature = "simd"))]
        let out = {
            let (lu, piv) = partial_pivot_lu(self)?;
            let mut out = Vec::with_capacity(n * n);
            for col in 0..n {
                let mut e = vec![0.0_f64; n];
                e[col] = 1.0;
                let x = solve_with_lu(&lu, &piv, &DVector::from_vec(e));
                out.extend(x);
            }
            out
        };
        Ok(DMatrix::from_vec(n, n, out))
    }
}

/// Returns the LU factorization (in place) and the row permutation.
#[cfg(all(feature = "alloc", not(feature = "simd")))]
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
#[cfg(all(feature = "alloc", not(feature = "simd")))]
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

// ---------------------------------------------------------------------------
// Optional SIMD backend (tpt-simd-blas): f32/f64 only, safe Rust throughout.
// ---------------------------------------------------------------------------

#[cfg(feature = "simd")]
mod simd {
    use super::{DMatrix, DVector, DenseError};
    use alloc::boxed::Box;
    use alloc::{vec, vec::Vec};
    use core::any::{Any, TypeId};
    use tpt_math_numeric::Scalar;

    /// Move a `Vec<U>` into a `Vec<V>`. Callers check the `TypeId`s first, so the
    /// downcast cannot fail; no element is copied.
    fn cast_vec<U: 'static, V: 'static>(v: Vec<U>) -> Vec<V> {
        let boxed: Box<dyn Any> = Box::new(v);
        *boxed
            .downcast::<Vec<V>>()
            .expect("cast_vec: TypeId checked by caller")
    }

    fn is_f64<T: 'static>() -> bool {
        TypeId::of::<T>() == TypeId::of::<f64>()
    }

    fn is_f32<T: 'static>() -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    /// `a * b` via `gemm`. Hands both operands back unchanged when the scalar
    /// type is not f32/f64, the shapes do not conform, or a dimension is zero, so
    /// the caller falls through to the scalar loop (which also owns the panics).
    pub(super) fn gemm<T: Scalar + 'static>(
        a: DMatrix<T>,
        b: DMatrix<T>,
    ) -> Result<DMatrix<T>, (DMatrix<T>, DMatrix<T>)> {
        let (m, k, n) = (a.nrows, a.ncols, b.ncols);
        if a.ncols != b.nrows || m == 0 || k == 0 || n == 0 || !(is_f64::<T>() || is_f32::<T>()) {
            return Err((a, b));
        }
        if is_f64::<T>() {
            let x: Vec<f64> = cast_vec(a.data);
            let y: Vec<f64> = cast_vec(b.data);
            let mut c = vec![0.0_f64; m * n];
            tpt_simd_blas::gemm_f64(m, n, k, 1.0, &x, m, &y, k, 0.0, &mut c, m);
            Ok(DMatrix {
                nrows: m,
                ncols: n,
                data: cast_vec(c),
            })
        } else {
            let x: Vec<f32> = cast_vec(a.data);
            let y: Vec<f32> = cast_vec(b.data);
            let mut c = vec![0.0_f32; m * n];
            tpt_simd_blas::gemm_f32(m, n, k, 1.0, &x, m, &y, k, 0.0, &mut c, m);
            Ok(DMatrix {
                nrows: m,
                ncols: n,
                data: cast_vec(c),
            })
        }
    }

    /// `a * x` via `gemv`, with the same hand-back rule as [`gemm`].
    pub(super) fn gemv<T: Scalar + 'static>(
        a: DMatrix<T>,
        x: DVector<T>,
    ) -> Result<DVector<T>, (DMatrix<T>, DVector<T>)> {
        let (m, k) = (a.nrows, a.ncols);
        if a.ncols != x.len() || m == 0 || k == 0 || !(is_f64::<T>() || is_f32::<T>()) {
            return Err((a, x));
        }
        if is_f64::<T>() {
            let a_data: Vec<f64> = cast_vec(a.data);
            let x_data: Vec<f64> = cast_vec(x.data);
            let mut y = vec![0.0_f64; m];
            tpt_simd_blas::gemv_f64(m, k, 1.0, &a_data, m, &x_data, 0.0, &mut y);
            Ok(DVector::from_vec(cast_vec(y)))
        } else {
            let a_data: Vec<f32> = cast_vec(a.data);
            let x_data: Vec<f32> = cast_vec(x.data);
            let mut y = vec![0.0_f32; m];
            tpt_simd_blas::gemv_f32(m, k, 1.0, &a_data, m, &x_data, 0.0, &mut y);
            Ok(DVector::from_vec(cast_vec(y)))
        }
    }

    /// Solve `A X = B` for every column of `B` (column-major; `nrhs = b.len() / n`)
    /// with blas `getrf`/`getrs`. Uses the same singularity rule as the scalar LU:
    /// a non-finite or `|u_ii| < 1e-12` pivot is `Err(Singular)`.
    pub(super) fn lu_solve(a: &DMatrix<f64>, mut b: Vec<f64>) -> Result<Vec<f64>, DenseError> {
        let n = a.nrows;
        if n == 0 {
            return Ok(b);
        }
        let nrhs = b.len() / n;
        let singular = DenseError::Singular {
            what: "partial_pivot_lu",
        };
        let mut lu = a.data.clone();
        let mut ipiv = vec![0_usize; n];
        tpt_simd_blas::getrf_f64(n, n, &mut lu, n, &mut ipiv).map_err(|_| singular.clone())?;
        let bad_pivot = (0..n).any(|i| {
            let p = lu[i + i * n].abs();
            !p.is_finite() || p < 1e-12
        });
        if bad_pivot {
            return Err(singular);
        }
        tpt_simd_blas::getrs_f64(n, nrhs, &lu, n, &ipiv, &mut b, n);
        Ok(b)
    }
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

#[cfg(all(test, feature = "simd"))]
mod simd_tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1) (no rand dependency).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0
    }

    fn random_f64(rows: usize, cols: usize, seed: &mut u64) -> DMatrix<f64> {
        DMatrix::from_fn(rows, cols, |_, _| lcg(seed))
    }

    #[test]
    fn gemm_f64_matches_naive_on_odd_sizes() {
        let mut seed = 1;
        let a = random_f64(37, 29, &mut seed);
        let b = random_f64(29, 23, &mut seed);
        let c = a.clone() * b.clone();
        for i in 0..37 {
            for j in 0..23 {
                let want: f64 = (0..29).map(|k| a[(i, k)] * b[(k, j)]).sum();
                assert!((c[(i, j)] - want).abs() < 1e-12, "({i},{j})");
            }
        }
    }

    #[test]
    fn gemm_f32_matches_naive_on_odd_sizes() {
        let a = DMatrix::from_fn(17, 9, |i, j| (i as f32) * 0.25 - (j as f32) * 0.5);
        let b = DMatrix::from_fn(9, 5, |i, j| (i as f32) * 0.125 + (j as f32));
        let c = a.clone() * b.clone();
        for i in 0..17 {
            for j in 0..5 {
                let want: f32 = (0..9).map(|k| a[(i, k)] * b[(k, j)]).sum();
                assert!((c[(i, j)] - want).abs() < 1e-3, "({i},{j})");
            }
        }
    }

    #[test]
    fn gemv_f64_and_f32_match_naive() {
        let mut seed = 7;
        let a = random_f64(31, 19, &mut seed);
        let x = DVector::from_fn(19, |_| lcg(&mut seed));
        let y = a.clone() * x.clone();
        for i in 0..31 {
            let want: f64 = (0..19).map(|k| a[(i, k)] * x[k]).sum();
            assert!((y[i] - want).abs() < 1e-12);
        }

        let a32 = DMatrix::from_fn(5, 3, |i, j| (i + 2 * j) as f32);
        let x32 = DVector::from_vec(vec![1.0_f32, -1.0, 0.5]);
        let y32 = a32 * x32;
        assert_eq!(y32[0], 0.0 - 2.0 + 0.5 * 4.0);
    }

    #[test]
    fn solve_and_inverse_residuals_are_small() {
        let mut seed = 42;
        let n = 23;
        // Diagonally dominant, so well conditioned and non-singular.
        let a = DMatrix::from_fn(n, n, |i, j| {
            lcg(&mut seed) + if i == j { n as f64 } else { 0.0 }
        });
        let b = DVector::from_fn(n, |_| lcg(&mut seed));
        let x = a.solve(&b).unwrap();
        let r = a.clone() * x - b.clone();
        assert!(r.norm() < 1e-10, "solve residual {}", r.norm());

        let inv = a.inverse().unwrap();
        let eye = a * inv;
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((eye[(i, j)] - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn near_singular_pivot_is_rejected_by_simd_path() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 2.0, 4.0 + 1e-15]);
        assert!(a.inverse().is_err());
    }
}
