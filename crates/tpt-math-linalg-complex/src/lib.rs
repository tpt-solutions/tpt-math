#![no_std]
#![forbid(unsafe_code)]
// Dense numeric kernels are clearest with explicit indexing; the indexed-loop
// lint does not fit this code.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]
//! Complex-valued linear algebra implemented entirely in-house (no external
//! backend), for EM / quantum and other complex-domain work.
//!
//! This crate builds on the storage pattern of [`tpt_math_linalg_dense`] and
//! extends it to a [`Complex`] scalar, providing:
//!
//! * [`ComplexDVector`] / [`ComplexDMatrix`] — dynamically-sized complex
//!   vectors and matrices stored column-major.
//! * [`ComplexDMatrix::lu`] — partial-pivot complex LU factorisation with
//!   [`ComplexDMatrix::solve`] / [`ComplexDMatrix::inverse`].
//! * [`ComplexDMatrix::cholesky`] — Cholesky factorisation of a Hermitian
//!   positive-definite matrix, with [`Cholesky::solve`].
//! * [`ComplexDMatrix::eigenvalues`] — a shifted-QR eigenvalue solver for
//!   general complex matrices.
//!
//! Like [`tpt_math_linalg_dense`], the storage is hand-rolled (a plain
//! `Vec<Complex<T>>`), so there is no `faer`/`nalgebra` dependency and no
//! license exposure.
//!
//! # Features
//!
//! * `std` (default) — enable the allocator and the `std` support of deps.
//! * `alloc` — signal allocator availability (dynamic vectors need it).
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg_complex::{Complex, ComplexDMatrix};
//!
//! let a = ComplexDMatrix::from_real_row_slice(
//!     2, 2,
//!     &[1.0_f64, 2.0, 3.0, 4.0],
//! );
//! let inv = a.inverse().unwrap();
//! let prod = a.clone() * inv;
//! assert!((prod[(0, 0)].re - 1.0).abs() < 1e-12);
//! assert!((prod[(1, 1)].re - 1.0).abs() < 1e-12);
//!
//! let z = Complex::new(0.0, 1.0);
//! assert_eq!(z * z, Complex::new(-1.0, 0.0));
//! ```

extern crate alloc;

use core::fmt;
use core::ops::{Add, Div, Index, Mul, Neg, Sub};

use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use tpt_math_linalg_dense::DMatrix;
#[cfg(feature = "alloc")]
use tpt_math_numeric::Float;
use tpt_math_numeric::Scalar;

#[cfg(feature = "alloc")]
use alloc::vec;

// ===========================================================================
// Complex scalar
// ===========================================================================

/// A complex number `z = re + i·im` with real and imaginary parts of type `T`.
///
/// `T` is normally a floating-point type (e.g. `f64`). All the usual complex
/// arithmetic is implemented, plus the modulus ([`Complex::norm`]), conjugate
/// ([`Complex::conj`]) and complex square root ([`Complex::sqrt`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex<T = f64> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Scalar> Complex<T> {
    /// Build a complex number from its real and imaginary parts.
    pub fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// A purely real complex number (`im == 0`).
    pub fn from_real(re: T) -> Self {
        Complex { re, im: T::zero() }
    }

    /// The complex conjugate `re - i·im`.
    pub fn conj(&self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// The squared modulus `re² + im²`.
    pub fn norm_sqr(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// The modulus `|z| = sqrt(re² + im²)`, computed with a scaled hypot so
    /// that large components do not overflow.
    pub fn norm(&self) -> T {
        let (a, b) = (self.re.abs(), self.im.abs());
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        if big == T::zero() {
            return T::zero();
        }
        let r = small / big;
        big * (T::one() + r * r).sqrt()
    }

    /// The argument (phase) `atan2(im, re)`.
    pub fn arg(&self) -> T {
        self.im.atan2(self.re)
    }

    /// The complex multiplicative inverse `1 / z`, or `None` if `z == 0`.
    pub fn inv(&self) -> Option<Self> {
        if self.re == T::zero() && self.im == T::zero() {
            return None;
        }
        // 1/z = (re - i·im)/(re² + im²), scaled by the larger component to
        // avoid forming re² + im² (which overflows for large |z|).
        if self.re.abs() >= self.im.abs() {
            let r = self.im / self.re;
            let den = self.re + self.im * r;
            Some(Complex::new(T::one() / den, -r / den))
        } else {
            let r = self.re / self.im;
            let den = self.re * r + self.im;
            Some(Complex::new(r / den, -T::one() / den))
        }
    }

    /// The complex square root.
    pub fn sqrt(&self) -> Self {
        // sqrt(z) = ±( sqrt((|z|+re)/2) + i·sign(im)·sqrt((|z|-re)/2) )
        let r = self.norm();
        let re = ((r + self.re) / (T::one() + T::one())).sqrt();
        let mut im = ((r - self.re) / (T::one() + T::one())).sqrt();
        if self.im < T::zero() {
            im = -im;
        }
        Complex::new(re, im)
    }
}

impl<T: Scalar + Copy> Complex<T> {
    /// `e^{i·theta}` — a point on the unit circle.
    pub fn from_polar(r: T, theta: T) -> Self {
        let (s, c) = theta.sin_cos();
        Complex::new(r * c, r * s)
    }
}

impl<T: Scalar + Copy> Add for Complex<T> {
    type Output = Complex<T>;
    fn add(self, rhs: Complex<T>) -> Complex<T> {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Scalar + Copy> Sub for Complex<T> {
    type Output = Complex<T>;
    fn sub(self, rhs: Complex<T>) -> Complex<T> {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: Scalar + Copy> Neg for Complex<T> {
    type Output = Complex<T>;
    fn neg(self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: Scalar + Copy> Mul for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, rhs: Complex<T>) -> Complex<T> {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl<T: Scalar + Copy> Div for Complex<T> {
    type Output = Complex<T>;
    /// Smith's algorithm: divides by the larger component of `rhs` first, so
    /// `|rhs|²` is never formed and cannot overflow.
    fn div(self, rhs: Complex<T>) -> Complex<T> {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let den = rhs.re + rhs.im * r;
            Complex::new((self.re + self.im * r) / den, (self.im - self.re * r) / den)
        } else {
            let r = rhs.re / rhs.im;
            let den = rhs.re * r + rhs.im;
            Complex::new((self.re * r + self.im) / den, (self.im * r - self.re) / den)
        }
    }
}

impl<T: Scalar + Copy> Mul<T> for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, rhs: T) -> Complex<T> {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

// ===========================================================================
// Storage
// ===========================================================================

/// A dynamically-sized complex column vector, stored as a contiguous `Vec`.
#[derive(Clone)]
pub struct ComplexDVector<T = f64> {
    data: Vec<Complex<T>>,
}

/// A dynamically-sized complex matrix, stored column-major in a contiguous
/// `Vec` (`(i, j)` lives at `i + j * nrows`).
#[derive(Clone)]
pub struct ComplexDMatrix<T = f64> {
    nrows: usize,
    ncols: usize,
    data: Vec<Complex<T>>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> ComplexDVector<T> {
    /// A zero vector of length `n`.
    pub fn zeros(n: usize) -> Self {
        ComplexDVector::from_fn(n, |_| Complex::from_real(T::zero()))
    }

    /// Build from a `Vec` of complex values (in order).
    pub fn from_vec(data: Vec<Complex<T>>) -> Self {
        ComplexDVector { data }
    }

    /// Build from a slice of complex values (in order).
    pub fn from_slice(data: &[Complex<T>]) -> Self
    where
        T: Clone,
    {
        ComplexDVector {
            data: data.to_vec(),
        }
    }

    /// Build element-by-element with `f(i)`.
    pub fn from_fn(n: usize, f: impl FnMut(usize) -> Complex<T>) -> Self {
        ComplexDVector {
            data: (0..n).map(f).collect(),
        }
    }

    /// Build from separate real and imaginary `Vec`s.
    pub fn from_real_imag(re: Vec<T>, im: Vec<T>) -> Self {
        let n = re.len().min(im.len());
        ComplexDVector {
            data: (0..n).map(|i| Complex::new(re[i], im[i])).collect(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> ComplexDMatrix<T> {
    /// A zero matrix of the given shape.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        ComplexDMatrix::from_fn(nrows, ncols, |_, _| Complex::from_real(T::zero()))
    }

    /// Build from a `Vec` laid out **column-major**: `(i, j)` is
    /// `data[i + j * nrows]`.
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<Complex<T>>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} does not match a {nrows}x{ncols} matrix",
            data.len()
        );
        ComplexDMatrix { nrows, ncols, data }
    }

    /// Build from a slice laid out **row-major**.
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_real_row_slice(nrows: usize, ncols: usize, data: &[T]) -> Self
    where
        T: Clone,
    {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} does not match a {nrows}x{ncols} matrix",
            data.len()
        );
        ComplexDMatrix {
            nrows,
            ncols,
            data: (0..ncols)
                .flat_map(|j| (0..nrows).map(move |i| Complex::from_real(data[i * ncols + j])))
                .collect(),
        }
    }

    /// Build from a slice of complex values laid out **row-major**.
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_row_slice(nrows: usize, ncols: usize, data: &[Complex<T>]) -> Self
    where
        T: Clone,
    {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length {} does not match a {nrows}x{ncols} matrix",
            data.len()
        );
        ComplexDMatrix {
            nrows,
            ncols,
            data: (0..ncols)
                .flat_map(|j| (0..nrows).map(move |i| data[i * ncols + j]))
                .collect(),
        }
    }

    /// Build element-by-element with `f(i, j)`.
    pub fn from_fn(
        nrows: usize,
        ncols: usize,
        mut f: impl FnMut(usize, usize) -> Complex<T>,
    ) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(i, j));
            }
        }
        ComplexDMatrix { nrows, ncols, data }
    }

    /// Convert a real [`DMatrix`] into a complex matrix.
    pub fn from_real_matrix(m: &DMatrix<f64>) -> ComplexDMatrix<f64> {
        let (r, c) = (m.nrows(), m.ncols());
        ComplexDMatrix::from_fn(r, c, |i, j| Complex::from_real(m[(i, j)]))
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl<T> ComplexDVector<T> {
    /// Number of components.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the vector has no components.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over the elements (in order).
    pub fn iter(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter()
    }
}

impl<T> ComplexDMatrix<T> {
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
    pub fn iter(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter()
    }

    /// Column-major linear index of `(i, j)`.
    fn offset(&self, i: usize, j: usize) -> usize {
        i + j * self.nrows
    }
}

impl<T> Index<usize> for ComplexDVector<T> {
    type Output = Complex<T>;
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    fn index(&self, i: usize) -> &Complex<T> {
        &self.data[i]
    }
}

impl<T> Index<(usize, usize)> for ComplexDMatrix<T> {
    type Output = Complex<T>;
    /// # Panics
    ///
    /// Panics if `(i, j)` is out of bounds.
    fn index(&self, (i, j): (usize, usize)) -> &Complex<T> {
        // Checked per axis: a row index past `nrows` would alias another entry.
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
// Norms / inner products
// ---------------------------------------------------------------------------

impl<T: Scalar + Copy> ComplexDVector<T> {
    /// The Euclidean (L2) norm of the vector.
    pub fn norm(&self) -> T {
        self.data
            .iter()
            .fold(T::zero(), |acc, z| acc + z.norm_sqr())
            .sqrt()
    }

    /// Conjugate dot product `Σ conj(self_i) · other_i`.
    pub fn dot(&self, other: &ComplexDVector<T>) -> Complex<T> {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "dot product requires equal lengths"
        );
        let mut s = Complex::from_real(T::zero());
        for i in 0..self.data.len() {
            s = s + self.data[i].conj() * other.data[i];
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Transpose / elementwise arithmetic
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> ComplexDVector<T> {
    /// Transpose to a `1 x n` row matrix.
    pub fn transpose(&self) -> ComplexDMatrix<T> {
        let n = self.len();
        ComplexDMatrix::from_fn(1, n, |_, j| self[j])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> ComplexDMatrix<T> {
    /// Transpose, swapping rows and columns.
    pub fn transpose(&self) -> ComplexDMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        ComplexDMatrix::from_fn(n, m, |i, j| self[(j, i)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Add for ComplexDVector<T> {
    type Output = ComplexDVector<T>;
    fn add(self, rhs: ComplexDVector<T>) -> ComplexDVector<T> {
        let n = self.len();
        ComplexDVector::from_fn(n, |i| self[i] + rhs[i])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Sub for ComplexDVector<T> {
    type Output = ComplexDVector<T>;
    fn sub(self, rhs: ComplexDVector<T>) -> ComplexDVector<T> {
        let n = self.len();
        ComplexDVector::from_fn(n, |i| self[i] - rhs[i])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Neg for ComplexDVector<T> {
    type Output = ComplexDVector<T>;
    fn neg(self) -> ComplexDVector<T> {
        let n = self.len();
        ComplexDVector::from_fn(n, |i| -self[i])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Mul<T> for ComplexDVector<T> {
    type Output = ComplexDVector<T>;
    fn mul(self, rhs: T) -> ComplexDVector<T> {
        let n = self.len();
        ComplexDVector::from_fn(n, |i| self[i] * rhs)
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Add for ComplexDMatrix<T> {
    type Output = ComplexDMatrix<T>;
    fn add(self, rhs: ComplexDMatrix<T>) -> ComplexDMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        ComplexDMatrix::from_fn(m, n, |i, j| self[(i, j)] + rhs[(i, j)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Sub for ComplexDMatrix<T> {
    type Output = ComplexDMatrix<T>;
    fn sub(self, rhs: ComplexDMatrix<T>) -> ComplexDMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        ComplexDMatrix::from_fn(m, n, |i, j| self[(i, j)] - rhs[(i, j)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Neg for ComplexDMatrix<T> {
    type Output = ComplexDMatrix<T>;
    fn neg(self) -> ComplexDMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        ComplexDMatrix::from_fn(m, n, |i, j| -self[(i, j)])
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Mul<T> for ComplexDMatrix<T> {
    type Output = ComplexDMatrix<T>;
    fn mul(self, rhs: T) -> ComplexDMatrix<T> {
        let (m, n) = (self.nrows, self.ncols);
        ComplexDMatrix::from_fn(m, n, |i, j| self[(i, j)] * rhs)
    }
}

// Matrix * matrix and matrix * vector.

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Mul<ComplexDMatrix<T>> for ComplexDMatrix<T> {
    type Output = ComplexDMatrix<T>;
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    fn mul(self, rhs: ComplexDMatrix<T>) -> ComplexDMatrix<T> {
        assert_eq!(
            self.ncols, rhs.nrows,
            "inner dimensions do not match: {}x{} * {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols
        );
        let m = self.nrows;
        let k = self.ncols;
        let n = rhs.ncols;
        ComplexDMatrix::from_fn(m, n, |i, j| {
            let mut s = Complex::from_real(T::zero());
            for kk in 0..k {
                s = s + self[(i, kk)] * rhs[(kk, j)];
            }
            s
        })
    }
}

#[cfg(feature = "alloc")]
impl<T: Scalar + Copy> Mul<ComplexDVector<T>> for ComplexDMatrix<T> {
    type Output = ComplexDVector<T>;
    /// # Panics
    ///
    /// Panics if the matrix column count does not match the vector length.
    fn mul(self, rhs: ComplexDVector<T>) -> ComplexDVector<T> {
        assert_eq!(
            self.ncols,
            rhs.len(),
            "matrix has {} columns but vector has length {}",
            self.ncols,
            rhs.len()
        );
        let m = self.nrows;
        let k = self.ncols;
        ComplexDVector::from_fn(m, |i| {
            let mut s = Complex::from_real(T::zero());
            for kk in 0..k {
                s = s + self[(i, kk)] * rhs[kk];
            }
            s
        })
    }
}

// ---------------------------------------------------------------------------
// Equality + Debug
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for ComplexDVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: PartialEq> PartialEq for ComplexDMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows && self.ncols == other.ncols && self.data == other.data
    }
}

#[cfg(feature = "alloc")]
impl<T: fmt::Debug> fmt::Debug for ComplexDVector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: fmt::Debug> fmt::Debug for ComplexDMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComplexDMatrix")
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .finish()
    }
}

// ===========================================================================
// Errors + decompositions (f64)
// ===========================================================================

/// Errors returned by the complex dense linear-algebra routines.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexDenseError {
    /// The matrix is (numerically) singular.
    Singular {
        /// Which routine detected the singular matrix.
        what: &'static str,
    },
    /// A dimension mismatch between operands.
    DimensionMismatch {
        /// Human-readable description of the conflict.
        what: alloc::string::String,
    },
    /// An iterative method exhausted its iteration budget.
    NotConverged {
        /// Number of iterations performed before giving up.
        iterations: usize,
    },
}

#[cfg(feature = "alloc")]
impl fmt::Display for ComplexDenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexDenseError::Singular { what } => write!(f, "singular matrix in {what}"),
            ComplexDenseError::DimensionMismatch { what } => {
                write!(f, "dimension mismatch: {what}")
            }
            ComplexDenseError::NotConverged { iterations } => {
                write!(f, "did not converge after {iterations} iterations")
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl core::error::Error for ComplexDenseError {}

#[cfg(feature = "alloc")]
impl ComplexDMatrix<f64> {
    /// [`ComplexDMatrix::lu`] with a scale-invariant threshold: a pivot below
    /// `rel` times the largest entry modulus counts as singular. Scaling the
    /// matrix by any non-zero constant does not change the outcome.
    ///
    /// # Errors
    ///
    /// As [`ComplexDMatrix::lu`].
    pub fn lu_with_relative_tolerance(
        &self,
        rel: f64,
    ) -> Result<(Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>), ComplexDenseError>
    {
        let scale = self.iter().fold(0.0_f64, |acc, z| acc.max(z.norm()));
        self.lu_with_tolerance(rel * scale)
    }

    /// Compute the partial-pivot LU factorisation of `self`.
    ///
    /// Returns the unit-lower and upper triangular factors `(L, U)` and the
    /// row permutation `p` such that `P A = L U`.
    pub fn lu(
        &self,
    ) -> Result<(Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>), ComplexDenseError>
    {
        self.lu_with_tolerance(1e-12)
    }

    /// [`ComplexDMatrix::lu`] with an absolute pivot threshold: a pivot whose
    /// modulus is below `tol` counts as singular. `tol = 0.0` accepts any
    /// non-zero finite pivot.
    ///
    /// # Errors
    ///
    /// As [`ComplexDMatrix::lu`].
    pub fn lu_with_tolerance(
        &self,
        tol: f64,
    ) -> Result<(Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>), ComplexDenseError>
    {
        let n = self.nrows;
        if self.ncols != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        #[cfg(feature = "simd-runtime")]
        {
            if n >= simd_lu::CUTOFF {
                let factors = simd_lu::factor(self, tol)?;
                return Ok(simd_lu::to_lu(&factors));
            }
        }
        let mut a: Vec<Vec<Complex<f64>>> = (0..n)
            .map(|i| (0..n).map(|j| self[(i, j)]).collect())
            .collect();
        let mut piv: Vec<usize> = (0..n).collect();

        for k in 0..n {
            let mut p = k;
            let mut max = a[k][k].norm();
            for i in (k + 1)..n {
                let v = a[i][k].norm();
                if v > max {
                    max = v;
                    p = i;
                }
            }
            if !max.is_finite() || max == 0.0 || max < tol {
                return Err(ComplexDenseError::Singular { what: "lu" });
            }
            a.swap(k, p);
            piv.swap(k, p);
            for i in (k + 1)..n {
                a[i][k] = a[i][k] / a[k][k];
                for j in (k + 1)..n {
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
                }
            }
        }
        let l: Vec<Vec<Complex<f64>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            Complex::from_real(1.0)
                        } else if i > j {
                            a[i][j]
                        } else {
                            Complex::from_real(0.0)
                        }
                    })
                    .collect()
            })
            .collect();
        let u: Vec<Vec<Complex<f64>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i <= j {
                            a[i][j]
                        } else {
                            Complex::from_real(0.0)
                        }
                    })
                    .collect()
            })
            .collect();
        Ok((l, u, piv))
    }

    /// Solve `A x = b` for `x`, where `A` is `self`.
    pub fn solve(&self, b: &ComplexDVector<f64>) -> Result<ComplexDVector<f64>, ComplexDenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        if b.len() != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("rhs has length {}, expected {n}", b.len()),
            });
        }
        #[cfg(feature = "simd-runtime")]
        {
            if n >= simd_lu::CUTOFF {
                let factors = simd_lu::factor(self, 1e-12)?;
                let rhs = ComplexDMatrix::from_fn(n, 1, |i, _| b[i]);
                let x = simd_lu::solve(&factors, &rhs);
                return Ok(ComplexDVector::from_fn(n, |i| x[(i, 0)]));
            }
        }
        let (l, u, piv) = self.lu()?;
        let x = solve_with_lu(&l, &u, &piv, b);
        Ok(ComplexDVector::from_vec(x))
    }

    /// Compute the inverse of `self`.
    pub fn inverse(&self) -> Result<ComplexDMatrix<f64>, ComplexDenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        #[cfg(feature = "simd-runtime")]
        {
            if n >= simd_lu::CUTOFF {
                let factors = simd_lu::factor(self, 1e-12)?;
                let identity = ComplexDMatrix::from_fn(n, n, |i, j| {
                    Complex::from_real(if i == j { 1.0 } else { 0.0 })
                });
                return Ok(simd_lu::solve(&factors, &identity));
            }
        }
        let (l, u, piv) = self.lu()?;
        let mut out = Vec::with_capacity(n * n);
        for col in 0..n {
            let mut e = vec![Complex::from_real(0.0_f64); n];
            e[col] = Complex::from_real(1.0);
            let x = solve_with_lu(&l, &u, &piv, &ComplexDVector::from_vec(e));
            out.extend(x);
        }
        Ok(ComplexDMatrix::from_vec(n, n, out))
    }

    /// Cholesky factorisation `A = L Lᴴ` for a Hermitian positive-definite
    /// matrix. Returns the lower-triangular factor `L`.
    pub fn cholesky(&self) -> Result<Cholesky<f64>, ComplexDenseError> {
        let n = self.nrows;
        if self.ncols != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("matrix is {}x{}, expected square", n, self.ncols),
            });
        }
        let mut l: Vec<Vec<Complex<f64>>> = vec![vec![Complex::from_real(0.0); n]; n];
        for j in 0..n {
            // Diagonal entry.
            let mut d = self[(j, j)];
            for k in 0..j {
                d = d - l[j][k] * l[j][k].conj();
            }
            if d.re <= 0.0 || !d.re.is_finite() {
                return Err(ComplexDenseError::Singular { what: "cholesky" });
            }
            let ljj = d.sqrt();
            l[j][j] = ljj;
            for i in (j + 1)..n {
                let mut s = self[(i, j)];
                for k in 0..j {
                    s = s - l[i][k] * l[j][k].conj();
                }
                l[i][j] = s / ljj;
            }
        }
        Ok(Cholesky { l, n })
    }

    /// Compute all eigenvalues of `self` via Hessenberg reduction and shifted QR.
    ///
    /// For a complex matrix the eigenvalues are themselves complex; the returned
    /// vector has one entry per row/column.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not square or the QR iteration does not converge.
    /// Use [`ComplexDMatrix::try_eigenvalues`] to handle these cases.
    pub fn eigenvalues(&self) -> Vec<Complex<f64>> {
        self.try_eigenvalues()
            .unwrap_or_else(|e| panic!("eigenvalues failed: {e}"))
    }

    /// Fallible [`ComplexDMatrix::eigenvalues`].
    ///
    /// # Errors
    ///
    /// Returns [`ComplexDenseError::DimensionMismatch`] if `self` is not square,
    /// and [`ComplexDenseError::NotConverged`] if the QR iteration exhausts its
    /// iteration budget.
    pub fn try_eigenvalues(&self) -> Result<Vec<Complex<f64>>, ComplexDenseError> {
        hessenberg_qr_eigenvalues(self)
    }
}

#[cfg(feature = "alloc")]
fn solve_with_lu(
    l: &[Vec<Complex<f64>>],
    u: &[Vec<Complex<f64>>],
    piv: &[usize],
    b: &ComplexDVector<f64>,
) -> Vec<Complex<f64>> {
    let n = l.len();
    // Forward solve L y = P b.
    let mut y = vec![Complex::from_real(0.0_f64); n];
    for i in 0..n {
        let mut s = b[piv[i]];
        for j in 0..i {
            s = s - l[i][j] * y[j];
        }
        y[i] = s;
    }
    // Back solve U x = y.
    let mut x = vec![Complex::from_real(0.0_f64); n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s = s - u[i][j] * x[j];
        }
        x[i] = s / u[i][i];
    }
    x
}

// ---------------------------------------------------------------------------
// Cholesky wrapper with a solver
// ---------------------------------------------------------------------------

/// A Cholesky factorisation `A = L Lᴴ` of a Hermitian positive-definite
/// complex matrix.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Cholesky<T = f64> {
    l: Vec<Vec<Complex<T>>>,
    n: usize,
}

#[cfg(feature = "alloc")]
impl Cholesky<f64> {
    /// Solve `A x = b` using the cached factorisation.
    pub fn solve(&self, b: &ComplexDVector<f64>) -> Result<ComplexDVector<f64>, ComplexDenseError> {
        let n = self.n;
        if b.len() != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("rhs has length {}, expected {n}", b.len()),
            });
        }
        // Forward solve L y = b.
        let mut y = vec![Complex::from_real(0.0_f64); n];
        for i in 0..n {
            let mut s = b[i];
            for j in 0..i {
                s = s - self.l[i][j] * y[j];
            }
            y[i] = s / self.l[i][i];
        }
        // Back solve Lᴴ x = y.
        let mut x = vec![Complex::from_real(0.0_f64); n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n {
                s = s - self.l[j][i].conj() * x[j];
            }
            x[i] = s / self.l[i][i].conj();
        }
        Ok(ComplexDVector::from_vec(x))
    }
}

// ---------------------------------------------------------------------------
// Eigenvalue solver (complex): Hessenberg reduction + single-shift QR
// ---------------------------------------------------------------------------

/// Eigenvalues of a square complex matrix.
///
/// Reduces `m` to upper Hessenberg form with Householder reflections, then runs
/// single-shift QR iteration with Wilkinson shifts (and an occasional
/// exceptional shift) using Givens rotations. Each rotation is a unitary
/// similarity, so the eigenvalues are preserved. Deflated diagonal entries are
/// returned in diagonal order.
#[cfg(feature = "alloc")]
fn hessenberg_qr_eigenvalues(
    m: &ComplexDMatrix<f64>,
) -> Result<Vec<Complex<f64>>, ComplexDenseError> {
    let n = m.nrows;
    if m.ncols != n {
        return Err(ComplexDenseError::DimensionMismatch {
            what: alloc::format!(
                "eigenvalues need a square matrix, got {}x{}",
                m.nrows,
                m.ncols
            ),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut a: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|i| (0..n).map(|j| m[(i, j)]).collect())
        .collect();
    hessenberg_reduce(&mut a);

    let zero = Complex::from_real(0.0_f64);
    let mut eig = vec![zero; n];
    // Iteration budget: generous for small matrices, still finite for pathological ones.
    let max_total = 50 * n.max(1);
    let mut total = 0_usize;
    let mut since_deflation = 0_usize;
    let mut hi = n - 1;

    while hi > 0 {
        // Find the start `lo` of the active unreduced Hessenberg block `[lo..=hi]`,
        // zeroing any subdiagonal entry that has become negligible.
        let mut lo = hi;
        while lo > 0 {
            let sub = a[lo][lo - 1].norm();
            let scale = a[lo - 1][lo - 1].norm() + a[lo][lo].norm();
            if sub <= f64::EPSILON * scale {
                a[lo][lo - 1] = zero;
                break;
            }
            lo -= 1;
        }
        if lo == hi {
            eig[hi] = a[hi][hi];
            hi -= 1;
            since_deflation = 0;
            continue;
        }

        if total >= max_total {
            return Err(ComplexDenseError::NotConverged { iterations: total });
        }
        total += 1;
        since_deflation += 1;
        let mu = if since_deflation % 11 == 0 {
            // Exceptional shift breaks cycles that the Wilkinson shift cannot.
            a[hi][hi] + Complex::from_real(0.75 * a[hi][hi - 1].norm())
        } else {
            wilkinson_shift(a[hi - 1][hi - 1], a[hi - 1][hi], a[hi][hi - 1], a[hi][hi])
        };
        qr_sweep(&mut a, lo, hi, mu);
    }
    eig[0] = a[0][0];
    Ok(eig)
}

/// Reduce `a` to upper Hessenberg form by unitary similarity (Householder).
#[cfg(feature = "alloc")]
fn hessenberg_reduce(a: &mut [Vec<Complex<f64>>]) {
    let n = a.len();
    let zero = Complex::from_real(0.0_f64);
    for k in 0..n.saturating_sub(2) {
        let len = n - k - 1;
        let x: Vec<Complex<f64>> = (0..len).map(|i| a[k + 1 + i][k]).collect();
        let nx = Float::sqrt(x.iter().fold(0.0_f64, |acc, z| acc + z.norm_sqr()));
        if nx == 0.0 {
            continue;
        }
        // alpha = -e^{i arg(x0)} |x| avoids cancellation in v = x - alpha e1.
        let x0_mod = x[0].norm();
        let phase = if x0_mod == 0.0 {
            Complex::from_real(1.0_f64)
        } else {
            x[0] * (1.0 / x0_mod)
        };
        let alpha = -(phase * Complex::from_real(nx));
        let mut v = x;
        v[0] = v[0] - alpha;
        let nv = Float::sqrt(v.iter().fold(0.0_f64, |acc, z| acc + z.norm_sqr()));
        if nv == 0.0 {
            continue;
        }
        let u: Vec<Complex<f64>> = v.iter().map(|z| *z * (1.0 / nv)).collect();

        // H = I - 2 u uᴴ on the trailing block: left-multiply rows k+1.., then
        // right-multiply columns k+1.. by H.
        for j in 0..n {
            let mut s = zero;
            for (i, ui) in u.iter().enumerate() {
                s = s + ui.conj() * a[k + 1 + i][j];
            }
            let s2 = s * 2.0;
            for (i, ui) in u.iter().enumerate() {
                a[k + 1 + i][j] = a[k + 1 + i][j] - *ui * s2;
            }
        }
        for row in a.iter_mut() {
            let mut s = zero;
            for (jj, uj) in u.iter().enumerate() {
                s = s + row[k + 1 + jj] * *uj;
            }
            let s2 = s * 2.0;
            for (jj, uj) in u.iter().enumerate() {
                row[k + 1 + jj] = row[k + 1 + jj] - s2 * uj.conj();
            }
        }
    }
}

/// Eigenvalue of the trailing 2×2 block `[[a, b], [c, d]]` closest to `d`.
#[cfg(feature = "alloc")]
fn wilkinson_shift(
    a: Complex<f64>,
    b: Complex<f64>,
    c: Complex<f64>,
    d: Complex<f64>,
) -> Complex<f64> {
    let half = Complex::from_real(0.5_f64);
    let mean = (a + d) * half;
    let half_diff = (a - d) * half;
    let disc = (half_diff * half_diff + b * c).sqrt();
    let l1 = mean + disc;
    let l2 = mean - disc;
    if (l1 - d).norm() < (l2 - d).norm() {
        l1
    } else {
        l2
    }
}

/// Givens rotation `G = [[c, s], [-s̄, c]]` (with real `c`) such that
/// `G [x; y] = [r; 0]`.
#[cfg(feature = "alloc")]
fn givens(x: Complex<f64>, y: Complex<f64>) -> (f64, Complex<f64>) {
    let nx = x.norm();
    let r = Float::sqrt(x.norm_sqr() + y.norm_sqr());
    if r == 0.0 {
        return (1.0, Complex::from_real(0.0));
    }
    if nx == 0.0 {
        return (0.0, Complex::from_real(1.0));
    }
    let c = nx / r;
    let s = (x * (1.0 / nx)) * y.conj() * (1.0 / r);
    (c, s)
}

/// One single-shift QR sweep (implicit bulge chase) on the block `[lo..=hi]`.
///
/// The rotations act on the whole matrix as `A ← G A Gᴴ`, so the spectrum is
/// preserved and the Hessenberg structure is restored after each step.
#[cfg(feature = "alloc")]
fn qr_sweep(a: &mut [Vec<Complex<f64>>], lo: usize, hi: usize, mu: Complex<f64>) {
    let n = a.len();
    let mut x = a[lo][lo] - mu;
    let mut y = a[lo + 1][lo];
    for k in lo..hi {
        let (c, s) = givens(x, y);
        // Rows k, k+1: A ← G A.
        for j in 0..n {
            let p = a[k][j];
            let q = a[k + 1][j];
            a[k][j] = p * c + s * q;
            a[k + 1][j] = -(s.conj() * p) + q * c;
        }
        // Columns k, k+1: A ← A Gᴴ.
        for row in a.iter_mut() {
            let p = row[k];
            let q = row[k + 1];
            row[k] = p * c + s.conj() * q;
            row[k + 1] = -(s * p) + q * c;
        }
        if k + 2 <= hi {
            // Chase the bulge that the rotation pushed to (k+2, k).
            x = a[k + 1][k];
            y = a[k + 2][k];
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    type C = Complex<f64>;

    fn c(re: f64, im: f64) -> C {
        Complex::new(re, im)
    }

    #[test]
    fn complex_arithmetic() {
        let z = c(1.0, 2.0);
        assert_eq!(z * z, c(-3.0, 4.0));
        assert_eq!(z + c(1.0, 0.0), c(2.0, 2.0));
        assert_eq!(-z, c(-1.0, -2.0));
        assert_eq!(z.conj(), c(1.0, -2.0));
        assert!((z.norm() - 5.0_f64.sqrt()).abs() < 1e-12);
        let inv = z.inv().unwrap();
        assert_relative_eq!((z * inv).re, 1.0, epsilon = 1e-12);
        assert_relative_eq!((z * inv).im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn division_and_norm_survive_large_magnitudes() {
        // |z|² overflows f64 at this scale, so the textbook formulas gave NaN/inf.
        let big = c(1e200, 1e200);
        let q = big / big;
        assert_relative_eq!(q.re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(q.im, 0.0, epsilon = 1e-12);
        assert_relative_eq!(big.norm(), 1e200 * 2.0_f64.sqrt(), max_relative = 1e-12);
        let inv = big.inv().unwrap();
        assert!(inv.re.is_finite() && inv.im.is_finite());
        assert_relative_eq!(inv.re, 0.5e-200, max_relative = 1e-12);
    }

    #[test]
    fn complex_sqrt() {
        // sqrt(-1) = i
        let r = c(-1.0, 0.0).sqrt();
        assert!((r.re).abs() < 1e-12);
        assert!((r.im - 1.0).abs() < 1e-12 || (r.im + 1.0).abs() < 1e-12);
        // sqrt(i) = (1+i)/sqrt(2)
        let r2 = c(0.0, 1.0).sqrt();
        assert_relative_eq!(r2.re, 2.0_f64.sqrt() / 2.0, epsilon = 1e-12);
        assert_relative_eq!(r2.im, 2.0_f64.sqrt() / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn vector_construction_index_norm_dot() {
        let v = ComplexDVector::from_fn(3, |i| c(i as f64, (i + 1) as f64));
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], c(0.0, 1.0));
        let n = v.norm();
        // |0+i|^2 + |1+2i|^2 + |2+3i|^2 = 1 + 5 + 13 = 19.
        assert!((n - 19.0_f64.sqrt()).abs() < 1e-12);
        let d = v.dot(&v);
        assert_relative_eq!(d.re, 19.0, epsilon = 1e-12);
        assert_relative_eq!(d.im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn matrix_arithmetic_and_mul() {
        let m = ComplexDMatrix::from_real_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let z = ComplexDMatrix::zeros(2, 2);
        assert_eq!(m.clone() - m.clone(), z);
        let v = ComplexDVector::from_vec(vec![c(1.0, 0.0), c(1.0, 0.0)]);
        let mv = m.clone() * v;
        assert_eq!(mv[0], c(3.0, 0.0));
        assert_eq!(mv[1], c(7.0, 0.0));
        let t = m.transpose();
        assert_eq!(t[(0, 1)], c(3.0, 0.0));
    }

    #[test]
    fn lu_solve_inverse_real() {
        let a = ComplexDMatrix::from_real_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let inv = a.inverse().unwrap();
        assert_relative_eq!(inv[(0, 0)].re, 0.5, epsilon = 1e-12);
        assert_relative_eq!(inv[(1, 1)].re, 0.5, epsilon = 1e-12);
        let b = ComplexDVector::from_vec(vec![c(4.0, 0.0), c(6.0, 0.0)]);
        let x = a.solve(&b).unwrap();
        assert_eq!(x[0], c(2.0, 0.0));
        assert_eq!(x[1], c(3.0, 0.0));
    }

    #[test]
    fn lu_solve_complex() {
        // A = [[1+i, 1], [0, 2-i]] (upper triangular), b = [2+i, 2]
        let a = ComplexDMatrix::from_row_slice(
            2,
            2,
            &[c(1.0, 1.0), c(1.0, 0.0), c(0.0, 0.0), c(2.0, -1.0)],
        );
        let b = ComplexDVector::from_vec(vec![c(2.0, 1.0), c(2.0, 0.0)]);
        let x = a.solve(&b).unwrap();
        // Back-substitution: x1 = b1/A11 = 2/(2-i) = 0.8 + 0.4 i.
        assert_relative_eq!(x[1].re, 0.8, epsilon = 1e-12);
        assert_relative_eq!(x[1].im, 0.4, epsilon = 1e-12);
        // Round-trip check A x == b.
        let ax = a.clone() * x;
        assert_relative_eq!(ax[0].re, b[0].re, epsilon = 1e-12);
        assert_relative_eq!(ax[0].im, b[0].im, epsilon = 1e-12);
        assert_relative_eq!(ax[1].re, b[1].re, epsilon = 1e-12);
        assert_relative_eq!(ax[1].im, b[1].im, epsilon = 1e-12);
    }

    #[test]
    fn singular_matrix_rejected() {
        let a = ComplexDMatrix::from_real_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        assert!(a.inverse().is_err());
        assert!(a
            .solve(&ComplexDVector::from_vec(vec![c(1.0, 0.0), c(1.0, 0.0)]))
            .is_err());
    }

    #[test]
    fn cholesky_hermitian_pd() {
        // A = [[4, 1-i], [1+i, 3]] is Hermitian PD.
        let a = ComplexDMatrix::from_row_slice(
            2,
            2,
            &[c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)],
        );
        let chol = a.cholesky().unwrap();
        // Reconstruct A = L Lᴴ.
        let l = ComplexDMatrix::from_vec(2, 2, {
            let mut v = Vec::new();
            for j in 0..2 {
                for i in 0..2 {
                    v.push(chol.l[i][j]);
                }
            }
            v
        });
        let recon = mul_mm_public(&l, &ComplexDMatrix::from_fn(2, 2, |i, j| l[(j, i)].conj()));
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(recon[(i, j)].re, a[(i, j)].re, epsilon = 1e-11);
                assert_relative_eq!(recon[(i, j)].im, a[(i, j)].im, epsilon = 1e-11);
            }
        }
        // Solve A x = [1, 0]^T.
        let b = ComplexDVector::from_vec(vec![c(1.0, 0.0), c(0.0, 0.0)]);
        let x = chol.solve(&b).unwrap();
        let ax = a.clone() * x;
        assert_relative_eq!(ax[0].re, 1.0, epsilon = 1e-11);
        assert_relative_eq!(ax[1].re, 0.0, epsilon = 1e-11);
    }

    #[test]
    fn eigenvalues_diagonal() {
        // Diagonal matrix diag(2, 3i) has eigenvalues 2 and 3i.
        let a = ComplexDMatrix::from_row_slice(
            2,
            2,
            &[c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 3.0)],
        );
        let ev = a.eigenvalues();
        assert_eq!(ev.len(), 2);
        assert!(find(&ev, 2.0, 0.0));
        assert!(find(&ev, 0.0, 3.0));
    }

    #[test]
    fn eigenvalues_rotation_like() {
        // Matrix [[0, -1], [1, 0]] has eigenvalues ±i.
        let a = ComplexDMatrix::from_real_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let ev = a.eigenvalues();
        assert_eq!(ev.len(), 2);
        assert!(find(&ev, 0.0, -1.0));
        assert!(find(&ev, 0.0, 1.0));
    }

    #[test]
    fn eigenvalues_hermitian_real_spectrum() {
        // Hermitian matrix -> real eigenvalues 2 ± sqrt(2).
        let a = ComplexDMatrix::from_row_slice(
            2,
            2,
            &[c(2.0, 0.0), c(1.0, 1.0), c(1.0, -1.0), c(2.0, 0.0)],
        );
        let ev = a.eigenvalues();
        assert_eq!(ev.len(), 2);
        let s = 2.0_f64.sqrt();
        assert!(find(&ev, 2.0 - s, 0.0));
        assert!(find(&ev, 2.0 + s, 0.0));
    }

    #[test]
    fn eigenvalues_3x3_companion_real_roots() {
        // Companion matrix of (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6.
        let a = ComplexDMatrix::from_real_row_slice(
            3,
            3,
            &[0.0, 0.0, 6.0, 1.0, 0.0, -11.0, 0.0, 1.0, 6.0],
        );
        let ev = a.eigenvalues();
        assert_eq!(ev.len(), 3);
        assert!(find(&ev, 1.0, 0.0));
        assert!(find(&ev, 2.0, 0.0));
        assert!(find(&ev, 3.0, 0.0));
    }

    #[test]
    fn eigenvalues_3x3_complex_roots() {
        // Companion matrix of x³ - 1 has roots 1, ω, ω² (cube roots of unity).
        let a = ComplexDMatrix::from_real_row_slice(
            3,
            3,
            &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let ev = a.eigenvalues();
        let h = 0.5_f64;
        let s = 3.0_f64.sqrt() / 2.0;
        assert!(find(&ev, 1.0, 0.0));
        assert!(find(&ev, -h, s));
        assert!(find(&ev, -h, -s));
    }

    #[test]
    fn eigenvalues_hermitian_3x3_with_repeated_value() {
        // Block [[1, i], [-i, 1]] has eigenvalues 0 and 2; the 1×1 block adds 2.
        let a = ComplexDMatrix::from_row_slice(
            3,
            3,
            &[
                c(1.0, 0.0),
                c(0.0, 1.0),
                c(0.0, 0.0),
                c(0.0, -1.0),
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(2.0, 0.0),
            ],
        );
        let ev = a.eigenvalues();
        assert_eq!(ev.len(), 3);
        assert!(find(&ev, 0.0, 0.0));
        assert_eq!(ev.iter().filter(|z| find(&[**z], 2.0, 0.0)).count(), 2);
    }

    #[test]
    fn eigenvalues_4x4_trace_and_spectrum_of_triangular() {
        // Upper-triangular: eigenvalues are the diagonal, and the trace is their sum.
        let a = ComplexDMatrix::from_row_slice(
            4,
            4,
            &[
                c(1.0, 0.0),
                c(5.0, 2.0),
                c(-3.0, 1.0),
                c(0.5, 0.0),
                c(0.0, 0.0),
                c(0.0, 2.0),
                c(4.0, 0.0),
                c(1.0, -1.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(-2.0, 0.0),
                c(2.0, 2.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, -3.0),
            ],
        );
        let ev = a.eigenvalues();
        assert_eq!(ev.len(), 4);
        for (want_re, want_im) in [(1.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -3.0)] {
            assert!(find(&ev, want_re, want_im), "missing {want_re}+{want_im}i");
        }
        let trace = ev.iter().fold(C::from_real(0.0), |acc, z| acc + *z);
        assert_relative_eq!(trace.re, -1.0, epsilon = 1e-9);
        assert_relative_eq!(trace.im, -1.0, epsilon = 1e-9);
    }

    #[test]
    fn try_eigenvalues_rejects_non_square() {
        let a = ComplexDMatrix::<f64>::zeros(2, 3);
        assert!(matches!(
            a.try_eigenvalues(),
            Err(ComplexDenseError::DimensionMismatch { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "length")]
    fn from_vec_length_mismatch_panics() {
        let _ = ComplexDMatrix::from_vec(2, 2, vec![c(1.0, 0.0); 3]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_row_out_of_bounds_panics() {
        // (2, 0) would alias (0, 1) in column-major storage without a check.
        let m = ComplexDMatrix::<f64>::zeros(2, 3);
        let _ = m[(2, 0)];
    }

    fn mul_mm_public(a: &ComplexDMatrix<f64>, b: &ComplexDMatrix<f64>) -> ComplexDMatrix<f64> {
        a.clone() * b.clone()
    }

    fn find(ev: &[Complex<f64>], target_re: f64, target_im: f64) -> bool {
        ev.iter()
            .any(|z| (z.re - target_re).abs() < 1e-9 && (z.im - target_im).abs() < 1e-9)
    }
}

#[cfg(test)]
mod lu_threshold_tests {
    use super::*;

    #[test]
    fn relative_threshold_is_scale_invariant() {
        let a = ComplexDMatrix::from_real_row_slice(2, 2, &[4.0_f64, 1.0, 2.0, 3.0]);
        let tiny = ComplexDMatrix::from_real_row_slice(2, 2, &[4e-14, 1e-14, 2e-14, 3e-14]);
        assert!(a.lu_with_relative_tolerance(1e-10).is_ok());
        assert!(tiny.lu_with_relative_tolerance(1e-10).is_ok());
        // The default absolute test rejects the scaled copy.
        assert!(tiny.lu().is_err());
    }

    #[test]
    fn zero_matrix_is_singular_for_any_threshold() {
        let zero = ComplexDMatrix::<f64>::zeros(2, 2);
        assert!(zero.lu_with_tolerance(0.0).is_err());
        assert!(zero.lu_with_relative_tolerance(1e-10).is_err());
    }
}

#[cfg(feature = "alloc")]
impl ComplexDMatrix<f64> {
    /// The determinant, from the LU factorisation: the sign of the row
    /// permutation times the product of the diagonal of `U`.
    ///
    /// A numerically singular matrix has determinant zero, as with the real
    /// [`DMatrix::det`](tpt_math_linalg_dense::DMatrix::det).
    ///
    /// # Errors
    ///
    /// Returns [`ComplexDenseError::DimensionMismatch`] if the matrix is not square.
    pub fn det(&self) -> Result<Complex<f64>, ComplexDenseError> {
        if self.nrows != self.ncols {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("matrix is {}x{}, expected square", self.nrows, self.ncols),
            });
        }
        let (_, u, piv) = match self.lu() {
            Ok(factors) => factors,
            Err(ComplexDenseError::Singular { .. }) => {
                return Ok(Complex::from_real(0.0));
            }
            Err(e) => return Err(e),
        };
        // Sign of the permutation from its cycle count.
        let n = piv.len();
        let mut seen = alloc::vec![false; n];
        let mut cycles = 0usize;
        for start in 0..n {
            if !seen[start] {
                cycles += 1;
                let mut j = start;
                while !seen[j] {
                    seen[j] = true;
                    j = piv[j];
                }
            }
        }
        let sign = if (n - cycles) % 2 == 0 { 1.0 } else { -1.0 };
        let mut det = Complex::from_real(sign);
        for (i, row) in u.iter().enumerate() {
            det = det * row[i];
        }
        Ok(det)
    }

    /// The matrix product `self · rhs`.
    ///
    /// With `simd-runtime` and matrices of order 64 or more, the product goes
    /// through `tpt-simd-blas`'s `gemm_c64`, on split real and imaginary planes.
    /// Otherwise it is the scalar product, the same as `*`. The `*` operator is
    /// generic over the scalar type and always uses the scalar product.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    pub fn product(&self, rhs: &ComplexDMatrix<f64>) -> ComplexDMatrix<f64> {
        assert_eq!(
            self.ncols, rhs.nrows,
            "inner dimensions do not match: {}x{} * {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols
        );
        #[cfg(feature = "simd-runtime")]
        {
            if self.nrows >= simd_lu::CUTOFF && self.ncols >= simd_lu::CUTOFF {
                return simd_lu::gemm(self, rhs);
            }
        }
        self.clone() * rhs.clone()
    }
}

/// SIMD LU for complex matrices through `tpt-simd-blas`'s `getrf_c64`/`getrs_c64`.
///
/// The kernels work on split real and imaginary planes in column-major order, so
/// this module converts to and from that layout. Pivots come back as LAPACK-style
/// sequential swaps, which become the permutation `piv` used by the public API.
#[cfg(feature = "simd-runtime")]
mod simd_lu {
    use super::*;

    /// Matrices from this size up use the kernels. Below it the scalar loops win.
    pub(super) const CUTOFF: usize = 64;

    /// An LU factorisation in the kernels' split-plane, column-major layout.
    pub(super) struct Factors {
        n: usize,
        ar: Vec<f64>,
        ai: Vec<f64>,
        ipiv: Vec<usize>,
    }

    /// Factor `m` with `getrf_c64`, rejecting pivots below `tol` in modulus.
    pub(super) fn factor(m: &ComplexDMatrix<f64>, tol: f64) -> Result<Factors, ComplexDenseError> {
        let n = m.nrows;
        let mut ar = vec![0.0_f64; n * n];
        let mut ai = vec![0.0_f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let z = m[(i, j)];
                ar[i + j * n] = z.re;
                ai[i + j * n] = z.im;
            }
        }
        let mut ipiv = vec![0usize; n];
        let singular = ComplexDenseError::Singular { what: "lu" };
        if tpt_simd_blas::getrf_c64(n, &mut ar, &mut ai, n, &mut ipiv).is_err() {
            return Err(singular);
        }
        // The same singularity test as the scalar path, on the pivots actually used.
        for k in 0..n {
            let (re, im) = (ar[k + k * n], ai[k + k * n]);
            let modulus = re.hypot(im);
            if !modulus.is_finite() || modulus == 0.0 || modulus < tol {
                return Err(singular);
            }
        }
        Ok(Factors { n, ar, ai, ipiv })
    }

    /// The `(L, U, piv)` triple that the scalar [`ComplexDMatrix::lu`] returns.
    pub(super) fn to_lu(
        f: &Factors,
    ) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>) {
        let n = f.n;
        let at = |i: usize, j: usize| Complex::new(f.ar[i + j * n], f.ai[i + j * n]);
        let mut piv: Vec<usize> = (0..n).collect();
        for (k, &p) in f.ipiv.iter().enumerate() {
            piv.swap(k, p);
        }
        let zero = Complex::from_real(0.0);
        let one = Complex::from_real(1.0);
        let l = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| match i.cmp(&j) {
                        core::cmp::Ordering::Equal => one,
                        core::cmp::Ordering::Greater => at(i, j),
                        core::cmp::Ordering::Less => zero,
                    })
                    .collect()
            })
            .collect();
        let u = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i <= j { at(i, j) } else { zero })
                    .collect()
            })
            .collect();
        (l, u, piv)
    }

    /// `A · B` through `gemm_c64`, on split planes in column-major order.
    pub(super) fn gemm(a: &ComplexDMatrix<f64>, b: &ComplexDMatrix<f64>) -> ComplexDMatrix<f64> {
        let (m, k, n) = (a.nrows, a.ncols, b.ncols);
        let mut ar = vec![0.0_f64; m * k];
        let mut ai = vec![0.0_f64; m * k];
        for j in 0..k {
            for i in 0..m {
                ar[i + j * m] = a[(i, j)].re;
                ai[i + j * m] = a[(i, j)].im;
            }
        }
        let mut br = vec![0.0_f64; k * n];
        let mut bi = vec![0.0_f64; k * n];
        for j in 0..n {
            for i in 0..k {
                br[i + j * k] = b[(i, j)].re;
                bi[i + j * k] = b[(i, j)].im;
            }
        }
        let mut cr = vec![0.0_f64; m * n];
        let mut ci = vec![0.0_f64; m * n];
        tpt_simd_blas::gemm_c64(
            m,
            n,
            k,
            [1.0, 0.0],
            &ar,
            &ai,
            m,
            &br,
            &bi,
            k,
            [0.0, 0.0],
            &mut cr,
            &mut ci,
            m,
        );
        ComplexDMatrix::from_fn(m, n, |i, j| Complex::new(cr[i + j * m], ci[i + j * m]))
    }

    /// Solve `A X = B` for the `ncols` right-hand sides of `rhs` with `getrs_c64`.
    pub(super) fn solve(f: &Factors, rhs: &ComplexDMatrix<f64>) -> ComplexDMatrix<f64> {
        let n = f.n;
        let k = rhs.ncols;
        let mut br = vec![0.0_f64; n * k];
        let mut bi = vec![0.0_f64; n * k];
        for c in 0..k {
            for i in 0..n {
                let z = rhs[(i, c)];
                br[i + c * n] = z.re;
                bi[i + c * n] = z.im;
            }
        }
        tpt_simd_blas::getrs_c64(n, k, &f.ar, &f.ai, n, &f.ipiv, &mut br, &mut bi, n);
        ComplexDMatrix::from_fn(n, k, |i, c| Complex::new(br[i + c * n], bi[i + c * n]))
    }
}

#[cfg(all(test, feature = "simd-runtime"))]
mod simd_complex_tests {
    use super::*;

    fn dominant(n: usize) -> ComplexDMatrix<f64> {
        ComplexDMatrix::from_fn(n, n, |i, j| {
            if i == j {
                Complex::new(n as f64 + 1.0, 0.5)
            } else {
                Complex::new(((i * 3 + j) as f64).sin(), ((i + 2 * j) as f64).cos() * 0.5)
            }
        })
    }

    #[test]
    fn simd_lu_factors_agree_with_the_scalar_factors() {
        for n in [5usize, 33] {
            let a = dominant(n);
            let (l_s, u_s, p_s) = a.lu_with_tolerance(0.0).unwrap();
            let factors = simd_lu::factor(&a, 0.0).unwrap();
            let (l_v, u_v, p_v) = simd_lu::to_lu(&factors);
            assert_eq!(p_s, p_v, "permutation, n = {n}");
            for i in 0..n {
                for j in 0..n {
                    let d = |x: Complex<f64>, y: Complex<f64>| (x.re - y.re).hypot(x.im - y.im);
                    assert!(d(l_s[i][j], l_v[i][j]) < 1e-10, "L at ({i},{j}), n = {n}");
                    assert!(d(u_s[i][j], u_v[i][j]) < 1e-10, "U at ({i},{j}), n = {n}");
                }
            }
        }
    }

    #[test]
    fn simd_solve_and_inverse_have_small_residuals_above_the_cutoff() {
        let n = 80;
        let a = dominant(n);
        let b = ComplexDVector::from_fn(n, |i| Complex::new(i as f64, 1.0));
        let x = a.solve(&b).unwrap();
        for i in 0..n {
            let mut acc = Complex::new(0.0, 0.0);
            for j in 0..n {
                let (p, q) = (a[(i, j)], x[j]);
                acc = Complex::new(
                    acc.re + p.re * q.re - p.im * q.im,
                    acc.im + p.re * q.im + p.im * q.re,
                );
            }
            assert!((acc.re - b[i].re).hypot(acc.im - b[i].im) < 1e-9, "row {i}");
        }
        let inv = a.inverse().unwrap();
        for i in 0..n {
            for j in 0..n {
                // (A · A⁻¹)_{ij} should be δ_ij.
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..n {
                    let (p, q) = (a[(i, k)], inv[(k, j)]);
                    acc = Complex::new(
                        acc.re + p.re * q.re - p.im * q.im,
                        acc.im + p.re * q.im + p.im * q.re,
                    );
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((acc.re - want).hypot(acc.im) < 1e-9, "({i},{j})");
            }
        }
    }

    #[test]
    fn determinant_of_a_small_complex_matrix() {
        // det [[1+i, 2], [3, 4−i]] = (1+i)(4−i) − 6 = −1 + 3i.
        let a = ComplexDMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => Complex::new(1.0, 1.0),
            (0, 1) => Complex::new(2.0, 0.0),
            (1, 0) => Complex::new(3.0, 0.0),
            _ => Complex::new(4.0, -1.0),
        });
        let d = a.det().unwrap();
        assert!(
            (d.re + 1.0).abs() < 1e-12 && (d.im - 3.0).abs() < 1e-12,
            "{d:?}"
        );
    }

    #[test]
    fn determinant_sign_after_many_swaps_on_the_simd_path() {
        // A cyclic shift on 66 rows is an n-cycle, with 65 transpositions: det = −1.
        let n = 66;
        let shift = ComplexDMatrix::from_fn(n, n, |i, j| {
            Complex::new(if j == (i + 1) % n { 1.0 } else { 0.0 }, 0.0)
        });
        let d = shift.det().unwrap();
        assert!((d.re + 1.0).abs() < 1e-10 && d.im.abs() < 1e-10, "{d:?}");
    }

    #[test]
    fn singular_determinant_is_zero_and_nonsquare_is_rejected() {
        let s = ComplexDMatrix::from_fn(2, 2, |_, _| Complex::new(1.0, 0.0));
        let d = s.det().unwrap();
        assert!(d.re == 0.0 && d.im == 0.0);
        assert!(ComplexDMatrix::from_fn(2, 3, |_, _| Complex::new(1.0, 0.0))
            .det()
            .is_err());
    }

    #[test]
    fn product_matches_the_scalar_product_above_the_cutoff() {
        let n = 70;
        let a = dominant(n);
        let b = ComplexDMatrix::from_fn(n, n, |i, j| {
            Complex::new((i + 2 * j) as f64 * 0.01, (i as f64 - j as f64) * 0.02)
        });
        let fast = a.product(&b);
        let slow = a.clone() * b.clone();
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (fast[(i, j)], slow[(i, j)]);
                assert!((x.re - y.re).hypot(x.im - y.im) < 1e-9, "({i},{j})");
            }
        }
    }
}
