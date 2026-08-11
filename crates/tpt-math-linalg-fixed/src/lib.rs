#![no_std]
#![forbid(unsafe_code)]
// Fixed-size matrix/vector algorithms are clearest with explicit indexing over
// fixed-size arrays; the indexed-loop and manual-swap lints do not fit this
// code, and the operator-impl lints misfire on componentwise arithmetic.
#![allow(
    clippy::needless_range_loop,
    clippy::manual_swap,
    clippy::suspicious_arithmetic_impl,
    clippy::suspicious_op_assign_impl
)]
//! Const-generic, fixed-size (stack-allocated) linear algebra.
//!
//! This crate mirrors the `Vector3`/`Matrix4` layer of `nalgebra` but is
//! implemented from scratch with const generics, so it carries no Apache-2.0
//! dependency and needs no allocator. Storage is plain fixed-size arrays
//! (`[T; N]` / `[[T; C]; R]`), so every type here is `#![no_std]` and
//! allocator-free.
//!
//! All arithmetic is generic over [`tpt_math_numeric::Scalar`] (i.e. any float
//! type), and the usual elementwise, scalar, dot/cross and matrix-product ops
//! are implemented. Square matrices get closed-form [`Matrix::determinant`] and
//! [`Matrix::inverse`] (exact for 2×2 / 3×3 / 4×4, and by Gauss–Jordan for any
//! size).
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg_fixed::{Vector3, Matrix3};
//!
//! let a = Vector3::new([1.0_f64, 2.0, 3.0]);
//! let b = Vector3::new([4.0_f64, 5.0, 6.0]);
//! assert_eq!(a.dot(&b), 32.0);
//!
//! let m = Matrix3::identity();
//! assert_eq!(m * a, a);
//! ```

use core::fmt;
use core::ops::{Add, Div, Index, Mul, Neg, Sub};

use tpt_math_numeric::Scalar;

// ===========================================================================
// Vector
// ===========================================================================

/// A fixed-size column vector of `N` elements of type `T`, stored inline.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Vector<T, const N: usize> {
    /// Row-major element storage.
    pub data: [T; N],
}

impl<T, const N: usize> Vector<T, N> {
    /// Build from a fixed-size array (row-major order).
    pub fn new(data: [T; N]) -> Self {
        Vector { data }
    }

    /// Build element-by-element with `f(i)`.
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Vector {
            data: array_init::<T, N>(f),
        }
    }

    /// Number of components.
    pub fn len(&self) -> usize {
        N
    }

    /// Always `false` (a fixed vector has a constant, non-zero length unless
    /// `N == 0`).
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// Iterate over the components.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// The Euclidean (L2) norm.
    pub fn norm(&self) -> T {
        let s = self.dot(self);
        s.sqrt()
    }

    /// Normalise to unit length. Returns the zero vector if the norm is zero.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == T::zero() {
            *self
        } else {
            *self * (T::one() / n)
        }
    }

    /// Dot product with another vector of the same length.
    pub fn dot(&self, other: &Vector<T, N>) -> T {
        let mut s = T::zero();
        for i in 0..N {
            s = s + self.data[i] * other.data[i];
        }
        s
    }
}

/// Per-size accessors (`.x()`, `.y()`, …) that only exist when the vector is
/// large enough, so they can never panic on a too-small vector.
impl<T: Scalar + Copy> Vector<T, 2> {
    /// First component.
    pub fn x(&self) -> T {
        self.data[0]
    }
    /// Second component.
    pub fn y(&self) -> T {
        self.data[1]
    }
}

impl<T: Scalar + Copy> Vector<T, 3> {
    /// First component.
    pub fn x(&self) -> T {
        self.data[0]
    }
    /// Second component.
    pub fn y(&self) -> T {
        self.data[1]
    }
    /// Third component.
    pub fn z(&self) -> T {
        self.data[2]
    }

    /// 3-D cross product.
    pub fn cross(&self, other: &Vector<T, 3>) -> Vector<T, 3> {
        Vector::new([
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        ])
    }
}

impl<T: Scalar + Copy> Vector<T, 4> {
    /// First component.
    pub fn x(&self) -> T {
        self.data[0]
    }
    /// Second component.
    pub fn y(&self) -> T {
        self.data[1]
    }
    /// Third component.
    pub fn z(&self) -> T {
        self.data[2]
    }
    /// Fourth component.
    pub fn w(&self) -> T {
        self.data[3]
    }
}

impl<T: Scalar + Copy, const N: usize> Vector<T, N> {
    /// 2-D perpendicular dot product (z-component of the 2-D cross product).
    /// Available on vectors of length ≥ 2; panics for `N < 2`.
    pub fn perp_dot(&self, other: &Vector<T, N>) -> T {
        self.data[0] * other.data[1] - self.data[1] * other.data[0]
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    /// # Panics
    ///
    /// Panics if `i >= N`.
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Scalar, const N: usize> Add for Vector<T, N> {
    type Output = Vector<T, N>;
    /// # Panics
    ///
    /// Inherits `T`'s `+` semantics (never panics for floats).
    fn add(self, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::from_fn(|i| self.data[i] + rhs.data[i])
    }
}

impl<T: Scalar, const N: usize> Sub for Vector<T, N> {
    type Output = Vector<T, N>;
    fn sub(self, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::from_fn(|i| self.data[i] - rhs.data[i])
    }
}

impl<T: Scalar, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T, N>;
    fn neg(self) -> Vector<T, N> {
        Vector::from_fn(|i| -self.data[i])
    }
}

impl<T: Scalar, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn mul(self, rhs: T) -> Vector<T, N> {
        Vector::from_fn(|i| self.data[i] * rhs)
    }
}

impl<T: Scalar, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn div(self, rhs: T) -> Vector<T, N> {
        Vector::from_fn(|i| self.data[i] / rhs)
    }
}

/// Componentwise product of two same-length vectors.
impl<T: Scalar, const N: usize> Mul<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn mul(self, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::from_fn(|i| self.data[i] * rhs.data[i])
    }
}

/// Componentwise quotient of two same-length vectors.
impl<T: Scalar, const N: usize> Div<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn div(self, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::from_fn(|i| self.data[i] / rhs.data[i])
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

// ===========================================================================
// Matrix
// ===========================================================================

/// A fixed-size `R × C` matrix of elements of type `T`, stored row-major.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    /// Row-major storage: `data[row][col]`.
    pub data: [[T; C]; R],
}

impl<T: Scalar, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Build from a row-major nested array.
    pub fn new(data: [[T; C]; R]) -> Self {
        Matrix { data }
    }

    /// Build element-by-element with `f(row, col)`.
    pub fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self {
        let data = array2_init::<T, R, C>(f);
        Matrix { data }
    }

    /// Build from a flat row-major slice of length `R * C` (row-major order).
    pub fn from_array(data: &[T]) -> Self
    where
        T: Scalar,
    {
        let mut out = [[T::zero(); C]; R];
        let mut k = 0;
        for i in 0..R {
            for j in 0..C {
                out[i][j] = data[k];
                k += 1;
            }
        }
        Matrix { data: out }
    }

    /// Build from `R` column vectors (each of length `C`).
    pub fn from_columns(columns: &[Vector<T, C>]) -> Option<Self>
    where
        T: Scalar,
    {
        if columns.len() != R {
            return None;
        }
        let mut out = [[T::zero(); C]; R];
        for (j, col) in columns.iter().enumerate() {
            for i in 0..R {
                out[i][j] = col.data[i];
            }
        }
        Some(Matrix { data: out })
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        R
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        C
    }

    /// Transpose (only defined for non-rectangular via the explicit
    /// [`Matrix::transpose`]).
    pub fn transpose(&self) -> Matrix<T, C, R>
    where
        T: Copy,
    {
        let mut out = [[T::zero(); R]; C];
        for i in 0..R {
            for j in 0..C {
                out[j][i] = self.data[i][j];
            }
        }
        Matrix { data: out }
    }

    /// Iterate over all elements in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flat_map(|row| row.iter())
    }
}

impl<T: Copy, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Index a single element.
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i][j]
    }
}

impl<T: Scalar, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Matrix–vector product `self * v` (requires `v` of length `C`), giving a
    /// vector of length `R`.
    pub fn mul_vec(&self, v: &Vector<T, C>) -> Vector<T, R> {
        let mut out = [T::zero(); R];
        for i in 0..R {
            let mut s = T::zero();
            for j in 0..C {
                s = s + self.data[i][j] * v.data[j];
            }
            out[i] = s;
        }
        Vector { data: out }
    }
}

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;
    /// # Panics
    ///
    /// Panics if `(i, j)` is out of bounds.
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i][j]
    }
}

impl<T: Scalar, const R: usize, const C: usize> Add for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn add(self, rhs: Matrix<T, R, C>) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| self.data[i][j] + rhs.data[i][j])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Sub for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn sub(self, rhs: Matrix<T, R, C>) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| self.data[i][j] - rhs.data[i][j])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Neg for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn neg(self) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| -self.data[i][j])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn mul(self, rhs: T) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| self.data[i][j] * rhs)
    }
}

impl<T: Scalar, const R: usize, const C: usize> Div<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn div(self, rhs: T) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| self.data[i][j] / rhs)
    }
}

/// Matrix–matrix product `(R×C) * (C×M) -> (R×M)`.
impl<T: Scalar, const R: usize, const C: usize, const M: usize> Mul<Matrix<T, C, M>>
    for Matrix<T, R, C>
{
    type Output = Matrix<T, R, M>;
    /// # Panics
    ///
    /// Inherits `T`'s `*` semantics.
    fn mul(self, rhs: Matrix<T, C, M>) -> Matrix<T, R, M> {
        let mut out = [[T::zero(); M]; R];
        for i in 0..R {
            for k in 0..M {
                let mut s = T::zero();
                for j in 0..C {
                    s = s + self.data[i][j] * rhs.data[j][k];
                }
                out[i][k] = s;
            }
        }
        Matrix { data: out }
    }
}

/// Matrix–vector product `(R×C) * (C) -> (R)`.
impl<T: Scalar, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C> {
    type Output = Vector<T, R>;
    fn mul(self, rhs: Vector<T, C>) -> Vector<T, R> {
        self.mul_vec(&rhs)
    }
}

/// Row-vector–matrix product `(N) * (N×M) -> (M)`.
impl<T: Scalar, const N: usize, const M: usize> Mul<Matrix<T, N, M>> for Vector<T, N> {
    type Output = Vector<T, M>;
    fn mul(self, rhs: Matrix<T, N, M>) -> Vector<T, M> {
        let mut out = [T::zero(); M];
        for k in 0..M {
            let mut s = T::zero();
            for j in 0..N {
                s = s + self.data[j] * rhs.data[j][k];
            }
            out[k] = s;
        }
        Vector { data: out }
    }
}

impl<T: fmt::Debug, const R: usize, const C: usize> fmt::Debug for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("nrows", &R)
            .field("ncols", &C)
            .finish()
    }
}

impl<T: Scalar, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Componentwise (Hadamard) product of two same-shaped matrices.
    pub fn component_mul(&self, rhs: &Matrix<T, R, C>) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| self.data[i][j] * rhs.data[i][j])
    }

    /// Componentwise (Hadamard) quotient of two same-shaped matrices.
    pub fn component_div(&self, rhs: &Matrix<T, R, C>) -> Matrix<T, R, C> {
        Matrix::from_fn(|i, j| self.data[i][j] / rhs.data[i][j])
    }
}

// ===========================================================================
// Square matrices: identity, determinant, inverse
// ===========================================================================

impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// The `N×N` identity matrix.
    pub fn identity() -> Self {
        Matrix::from_fn(|i, j| if i == j { T::one() } else { T::zero() })
    }

    /// Determinant via Gauss elimination with partial pivoting. Exact (up to
    /// floating-point round-off) for small `N`.
    pub fn determinant(&self) -> T {
        let mut a = self.data;
        let mut det = T::one();
        let mut sign = T::one();
        for k in 0..N {
            let mut piv = k;
            let mut mx = a[k][k].abs();
            for i in (k + 1)..N {
                let v = a[i][k].abs();
                if v > mx {
                    mx = v;
                    piv = i;
                }
            }
            if !mx.is_finite() || mx == T::zero() {
                return T::zero();
            }
            if piv != k {
                let tmp = a[k];
                a[k] = a[piv];
                a[piv] = tmp;
                sign = -sign;
            }
            let ak = a[k][k];
            det = det * ak;
            for i in (k + 1)..N {
                let f = a[i][k] / ak;
                for j in (k + 1)..N {
                    a[i][j] = a[i][j] - f * a[k][j];
                }
            }
        }
        det * sign
    }

    /// Inverse via Gauss–Jordan elimination with partial pivoting. Returns
    /// `None` if the matrix is (numerically) singular.
    pub fn inverse(&self) -> Option<Matrix<T, N, N>> {
        let eps = T::from(1e-12).unwrap_or(T::zero());
        let mut a = self.data;
        let mut inv = [[T::zero(); N]; N];
        for i in 0..N {
            inv[i][i] = T::one();
        }
        for k in 0..N {
            let mut piv = k;
            let mut mx = a[k][k].abs();
            for i in (k + 1)..N {
                let v = a[i][k].abs();
                if v > mx {
                    mx = v;
                    piv = i;
                }
            }
            if mx < eps {
                return None;
            }
            if piv != k {
                let tmp = a[k];
                a[k] = a[piv];
                a[piv] = tmp;
                let tmp2 = inv[k];
                inv[k] = inv[piv];
                inv[piv] = tmp2;
            }
            let ak = a[k][k];
            for j in 0..N {
                a[k][j] = a[k][j] / ak;
                inv[k][j] = inv[k][j] / ak;
            }
            for i in 0..N {
                if i != k {
                    let f = a[i][k];
                    if f != T::zero() {
                        for j in 0..N {
                            a[i][j] = a[i][j] - f * a[k][j];
                            inv[i][j] = inv[i][j] - f * inv[k][j];
                        }
                    }
                }
            }
        }
        Some(Matrix { data: inv })
    }
}

// ===========================================================================
// Common aliases
// ===========================================================================

/// 2-D vector.
pub type Vector2<T> = Vector<T, 2>;
/// 3-D vector.
pub type Vector3<T> = Vector<T, 3>;
/// 4-D vector.
pub type Vector4<T> = Vector<T, 4>;
/// 6-D vector.
pub type Vector6<T> = Vector<T, 6>;

/// 2×2 matrix.
pub type Matrix2<T> = Matrix<T, 2, 2>;
/// 3×3 matrix.
pub type Matrix3<T> = Matrix<T, 3, 3>;
/// 4×4 matrix.
pub type Matrix4<T> = Matrix<T, 4, 4>;
/// 3×4 matrix.
pub type Matrix3x4<T> = Matrix<T, 3, 4>;
/// 2×3 matrix.
pub type Matrix2x3<T> = Matrix<T, 2, 3>;
/// 4×3 matrix.
pub type Matrix4x3<T> = Matrix<T, 4, 3>;

// ===========================================================================
// Internal fixed-array helpers (const-generic initialisers)
// ===========================================================================

fn array_init<T, const N: usize>(f: impl FnMut(usize) -> T) -> [T; N] {
    core::array::from_fn(f)
}

fn array2_init<T, const R: usize, const C: usize>(
    mut f: impl FnMut(usize, usize) -> T,
) -> [[T; C]; R] {
    core::array::from_fn(|i| core::array::from_fn(|j| f(i, j)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::iter;

    /// A random-ish but reproducible RNG.
    fn rng() -> impl Iterator<Item = f64> {
        let mut state: u64 = 0x9e3779b97f4a7c15;
        iter::from_fn(move || {
            state ^= state >> 30;
            state = state.wrapping_mul(0xbf58476d1ce4e5b9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94d049bb133111eb);
            state ^= state >> 31;
            Some(((state >> 11) as f64) / ((1u64 << 53) as f64))
        })
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn vector_arithmetic() {
        let a = Vector3::new([1.0_f64, 2.0, 3.0]);
        let b = Vector3::new([4.0_f64, 5.0, 6.0]);
        assert_eq!((a + b).data, [5.0, 7.0, 9.0]);
        assert_eq!((a - b).data, [-3.0, -3.0, -3.0]);
        assert_eq!((a * 2.0).data, [2.0, 4.0, 6.0]);
        assert_eq!((a / 2.0).data, [0.5, 1.0, 1.5]);
        assert_eq!((a * b).data, [4.0, 10.0, 18.0]);
        assert_eq!(-a, Vector3::new([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn vector_dot_cross() {
        let a = Vector3::new([1.0_f64, 0.0, 0.0]);
        let b = Vector3::new([0.0_f64, 1.0, 0.0]);
        assert_eq!(a.cross(&b), Vector3::new([0.0, 0.0, 1.0]));
        let u = Vector2::new([1.0_f64, 0.0]);
        let v = Vector2::new([0.0_f64, 1.0]);
        assert_eq!(u.perp_dot(&v), 1.0);
        assert!(close(a.dot(&b), 0.0, 1e-12));
    }

    #[test]
    fn vector_norm() {
        let a = Vector3::new([3.0_f64, 4.0, 0.0]);
        assert!(close(a.norm(), 5.0, 1e-12));
        let n = a.normalize();
        assert!(close(n.norm(), 1.0, 1e-12));
    }

    #[test]
    fn matrix_vector_product() {
        let m = Matrix2::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        let v = Vector2::new([1.0_f64, 1.0]);
        assert_eq!((m * v).data, [3.0, 7.0]);
    }

    #[test]
    fn matrix_matrix_product() {
        let m = Matrix2::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        let n = Matrix2::new([[0.0_f64, 1.0], [1.0, 0.0]]);
        let p = m * n;
        assert_eq!(p.data, [[2.0, 1.0], [4.0, 3.0]]);
    }

    #[test]
    fn transpose() {
        let m = Matrix2x3::new([[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn identity_and_inverse_2x2() {
        let m = Matrix2::new([[2.0_f64, 0.0], [0.0, 3.0]]);
        let inv = m.inverse().unwrap();
        assert_eq!(inv.data, [[0.5, 0.0], [0.0, 1.0 / 3.0]]);
        let prod = m * inv;
        assert!(close(prod.data[0][0], 1.0, 1e-12));
        assert!(close(prod.data[1][1], 1.0, 1e-12));
        assert!(close(prod.data[0][1], 0.0, 1e-12));
    }

    #[test]
    fn determinant_2x2() {
        let m = Matrix2::new([[2.0_f64, 1.0], [1.0, 3.0]]);
        assert!(close(m.determinant(), 5.0, 1e-12));
    }

    #[test]
    fn random_invertible_2x2() {
        let mut r = rng();
        for _ in 0..50 {
            let (a, b, c, d) = (
                r.next().unwrap(),
                r.next().unwrap(),
                r.next().unwrap(),
                r.next().unwrap(),
            );
            let m = Matrix2::new([[a, b], [c, d]]) + Matrix2::identity();
            let inv = m.inverse().unwrap();
            let prod = m * inv;
            for i in 0..2 {
                for j in 0..2 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        close(prod.data[i][j], want, 1e-9),
                        "entry {i},{j} = {}",
                        prod.data[i][j]
                    );
                }
            }
            assert!(close(m.determinant() * inv.determinant(), 1.0, 1e-8));
        }
    }

    #[test]
    fn random_invertible_3x3() {
        let mut r = rng();
        for _ in 0..30 {
            let mut m = Matrix3::<f64>::identity();
            for i in 0..3 {
                for j in 0..3 {
                    m.data[i][j] = r.next().unwrap();
                }
            }
            m = m + Matrix3::identity();
            let inv = m.inverse().unwrap();
            let prod = m * inv;
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(close(prod.data[i][j], want, 1e-8), "entry {i},{j}");
                }
            }
        }
    }

    #[test]
    fn random_invertible_4x4() {
        let mut r = rng();
        for _ in 0..20 {
            let mut m = Matrix4::<f64>::identity();
            for i in 0..4 {
                for j in 0..4 {
                    m.data[i][j] = if i == j {
                        1.0 + r.next().unwrap()
                    } else {
                        r.next().unwrap()
                    };
                }
            }
            let inv = m.inverse().unwrap();
            let prod = m * inv;
            for i in 0..4 {
                for j in 0..4 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(close(prod.data[i][j], want, 1e-7), "entry {i},{j}");
                }
            }
        }
    }

    #[test]
    fn singular_is_none() {
        let m = Matrix2::new([[1.0_f64, 2.0], [2.0, 4.0]]);
        assert!(m.inverse().is_none());
        assert!(close(m.determinant(), 0.0, 1e-12));
    }

    #[test]
    fn from_columns_and_array() {
        let cols = [Vector2::new([1.0_f64, 2.0]), Vector2::new([3.0, 4.0])];
        let m = Matrix2::from_columns(&cols).unwrap();
        assert_eq!(m.data, [[1.0, 3.0], [2.0, 4.0]]);
        let a = Matrix2::from_array(&[1.0_f64, 2.0, 3.0, 4.0]);
        assert_eq!(a.data, [[1.0, 2.0], [3.0, 4.0]]);
        assert!(Matrix2::from_columns(&cols[0..1]).is_none());
    }
}
