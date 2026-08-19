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

use tpt_math_linalg_dense::DMatrix;
use tpt_math_numeric::Scalar;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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

    /// The modulus `|z| = sqrt(re² + im²)`.
    pub fn norm(&self) -> T {
        self.norm_sqr().sqrt()
    }

    /// The argument (phase) `atan2(im, re)`.
    pub fn arg(&self) -> T {
        self.im.atan2(self.re)
    }

    /// The complex multiplicative inverse `1 / z`, or `None` if `z == 0`.
    pub fn inv(&self) -> Option<Self> {
        let n = self.norm_sqr();
        if n == T::zero() {
            return None;
        }
        Some(Complex::new(self.re / n, -self.im / n))
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
    fn div(self, rhs: Complex<T>) -> Complex<T> {
        let d = rhs.norm_sqr();
        Complex::new(
            (self.re * rhs.re + self.im * rhs.im) / d,
            (self.im * rhs.re - self.re * rhs.im) / d,
        )
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
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<Complex<T>>) -> Self {
        ComplexDMatrix { nrows, ncols, data }
    }

    /// Build from a slice laid out **row-major**.
    pub fn from_real_row_slice(nrows: usize, ncols: usize, data: &[T]) -> Self
    where
        T: Clone,
    {
        ComplexDMatrix {
            nrows,
            ncols,
            data: (0..ncols)
                .flat_map(|j| (0..nrows).map(move |i| Complex::from_real(data[i * ncols + j])))
                .collect(),
        }
    }

    /// Build from a slice of complex values laid out **row-major**.
    pub fn from_row_slice(nrows: usize, ncols: usize, data: &[Complex<T>]) -> Self
    where
        T: Clone,
    {
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
}

#[cfg(feature = "alloc")]
impl fmt::Display for ComplexDenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexDenseError::Singular { what } => write!(f, "singular matrix in {what}"),
            ComplexDenseError::DimensionMismatch { what } => {
                write!(f, "dimension mismatch: {what}")
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl core::error::Error for ComplexDenseError {}

#[cfg(feature = "alloc")]
impl ComplexDMatrix<f64> {
    /// Compute the partial-pivot LU factorisation of `self`.
    ///
    /// Returns the unit-lower and upper triangular factors `(L, U)` and the
    /// row permutation `p` such that `P A = L U`.
    pub fn lu(
        &self,
    ) -> Result<(Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>), ComplexDenseError>
    {
        let n = self.nrows;
        if self.ncols != n {
            return Err(ComplexDenseError::DimensionMismatch {
                what: alloc::format!("matrix is {}x{}, expected square", n, self.ncols),
            });
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
            if !max.is_finite() || max < 1e-12 {
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

    /// Compute all eigenvalues of `self` via the shifted QR algorithm.
    ///
    /// For a complex matrix the eigenvalues are themselves complex; the returned
    /// vector has one entry per row/column.
    pub fn eigenvalues(&self) -> Vec<Complex<f64>> {
        shifted_qr_eigenvalues(self)
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
// Shifted-QR eigenvalue solver (complex)
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
fn shifted_qr_eigenvalues(m: &ComplexDMatrix<f64>) -> Vec<Complex<f64>> {
    let mut n = m.nrows;
    if m.ncols != n || n == 0 {
        return Vec::new();
    }
    // Working copy as a Vec<Vec<Complex<f64>>>.
    let mut a: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|i| (0..n).map(|j| m[(i, j)]).collect())
        .collect();
    let mut out: Vec<Complex<f64>> = Vec::with_capacity(n);
    let tol = 1e-10;
    let max_iter = 1000;

    while n > 2 {
        let mut iter = 0;
        let mut converged = false;
        while iter < max_iter && !converged {
            iter += 1;
            let mu = a[n - 1][n - 1];
            // (A - mu I) = R Q, then A' = R Q + mu I (similarity transform).
            let ashifted: Vec<Vec<Complex<f64>>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| a[i][j] - if i == j { mu } else { Complex::from_real(0.0) })
                        .collect()
                })
                .collect();
            let (q, r) = householder_qr(&ashifted);
            let rq = mul_mm(&r, &q);
            for i in 0..n {
                for j in 0..n {
                    a[i][j] = rq[i][j] + if i == j { mu } else { Complex::from_real(0.0) };
                }
            }
            if a[n - 1][n - 2].norm() < tol {
                converged = true;
            }
        }
        // Deflate: the bottom-right entry is an eigenvalue.
        out.push(a[n - 1][n - 1]);
        n -= 1;
        if n > 0 {
            a.truncate(n);
            for row in a.iter_mut() {
                row.truncate(n);
            }
        }
    }

    if n == 2 {
        // Solve the 2x2 characteristic polynomial directly (the single-shift QR
        // does not reliably converge on a 2x2 complex block).
        let a00 = a[0][0];
        let a01 = a[0][1];
        let a10 = a[1][0];
        let a11 = a[1][1];
        let tr = a00 + a11;
        let det = a00 * a11 - a01 * a10;
        let disc = (tr * tr - Complex::from_real(4.0) * det).sqrt();
        let half = Complex::from_real(0.5);
        out.push((tr + disc) * half);
        out.push((tr - disc) * half);
    } else if n == 1 {
        out.push(a[0][0]);
    }
    out
}

#[cfg(feature = "alloc")]
fn householder_qr(a: &[Vec<Complex<f64>>]) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>) {
    let n = a.len();
    let m = a[0].len();
    let mut r: Vec<Vec<Complex<f64>>> = a.to_vec();
    let mut q: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        Complex::from_real(1.0)
                    } else {
                        Complex::from_real(0.0)
                    }
                })
                .collect()
        })
        .collect();

    for k in 0..m.min(n) {
        // Column vector x = r[k..][k].
        let mut x: Vec<Complex<f64>> = (k..n).map(|i| r[i][k]).collect();
        let norm_x = x.iter().fold(0.0_f64, |acc, z| acc + z.norm_sqr()).sqrt();
        if norm_x < 1e-14 {
            continue;
        }
        let sign = if x[0].re < 0.0 { -1.0 } else { 1.0 };
        x[0] = x[0] + Complex::from_real(sign * norm_x);
        let norm_u = x.iter().fold(0.0_f64, |acc, z| acc + z.norm_sqr()).sqrt();
        if norm_u < 1e-14 {
            continue;
        }
        // Apply H = I - 2 u uᴴ to the trailing submatrix of r (rows k..n, cols k..m).
        for j in k..m {
            let mut dot = Complex::from_real(0.0);
            for (i, xi) in x.iter().enumerate() {
                dot = dot + xi.conj() * r[k + i][j];
            }
            let coeff = dot / Complex::from_real(norm_u * norm_u / 2.0);
            for (i, xi) in x.iter().enumerate() {
                r[k + i][j] = r[k + i][j] - *xi * coeff;
            }
        }
        // Accumulate Q = Q H (apply H to columns k..n, rows k..n of q).
        for j in 0..n {
            let mut dot = Complex::from_real(0.0);
            for (i, xi) in x.iter().enumerate() {
                dot = dot + xi.conj() * q[k + i][j];
            }
            let coeff = dot / Complex::from_real(norm_u * norm_u / 2.0);
            for (i, xi) in x.iter().enumerate() {
                q[k + i][j] = q[k + i][j] - *xi * coeff;
            }
        }
    }
    (q, r)
}

#[cfg(feature = "alloc")]
fn mul_mm(a: &[Vec<Complex<f64>>], b: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let mut out = vec![vec![Complex::from_real(0.0); m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut s = Complex::from_real(0.0);
            for p in 0..k {
                s = s + a[i][p] * b[p][j];
            }
            out[i][j] = s;
        }
    }
    out
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

    fn mul_mm_public(a: &ComplexDMatrix<f64>, b: &ComplexDMatrix<f64>) -> ComplexDMatrix<f64> {
        a.clone() * b.clone()
    }

    fn find(ev: &[Complex<f64>], target_re: f64, target_im: f64) -> bool {
        ev.iter()
            .any(|z| (z.re - target_re).abs() < 1e-9 && (z.im - target_im).abs() < 1e-9)
    }
}
