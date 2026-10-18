#![no_std]
#![allow(clippy::wrong_self_convention)]
//! Linear algebra with compile-time dimensional safety.
//!
//! [`tpt_math_linalg_dense`] (a hand-rolled, in-house dense linear-algebra
//! crate with no external backend) provides the storage, but it is
//! *dimensionless*: a `DVector<f64>` is just numbers. [`uom`] (via
//! [`tpt_math_units`]) gives compile-time unit checking, but has no vector or
//! matrix types. This crate is the bridge: it wraps the dense storage and tags
//! it with a phantom unit type `U` (typically a `uom` quantity), so that adding
//! a length-vector to a mass-vector is a compile error, and matrix
//! multiplication propagates units correctly:
//!
//! ```text
//! Mat<U, V> * Mat<V, W> = Mat<U, W>
//! ```
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg::Vec;
//! use tpt_math_units::prelude::{Length, Time, Velocity};
//! use tpt_math_linalg_dense::DVector;
//!
//! // A position vector in metres. `Length`/`Time`/`Velocity` are the concrete
//! // `f64` SI quantity types re-exported by `tpt_math_units::prelude`.
//! let pos = Vec::<Length>::from_raw(DVector::from_row_slice(&[3.0_f64, 4.0]));
//! let dur = Vec::<Time>::from_raw(DVector::from_row_slice(&[2.0_f64, 1.0]));
//! // Cannot add `pos + dur`: different units (a compile error).
//! let _ = pos.raw().clone() + dur.raw().clone(); // raw dense ops remain available
//! ```
//!
//! ## Backend decision
//!
//! This crate wraps **`tpt-math-linalg-dense`** only — never `nalgebra`, which
//! is Apache-2.0-only and disqualified as a wrap target by the workspace
//! license policy (ADR-0007). `tpt-math-linalg-dense` is implemented in-house
//! (no `faer`/`nalgebra` dependency), so there is no license exposure. The
//! dense storage lives in `tpt-math-linalg-dense` so the `ArgminMath`
//! orphan-rule problem is solved there once, and this crate only adds the
//! unit-tagging layer on top. A dual facade was never considered, since
//! `tpt-math-linalg-dense` is the single storage backend.
//!
//! [`uom`]: tpt_math_units
//! [`tpt_math_linalg_dense`]: tpt_math_linalg_dense
//! [`tpt_math_units`]: tpt_math_units

pub use tpt_math_linalg_dense;
pub use tpt_math_units;

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_numeric::Scalar;

/// A vector of elements with unit `U` (typically a `uom` quantity type).
///
/// The unit is a phantom type: it costs nothing at runtime, but the type
/// system forbids mixing vectors of different units.
#[derive(Clone, Debug, PartialEq)]
pub struct Vec<U, T = f64> {
    raw: DVector<T>,
    _unit: PhantomData<U>,
}

/// A matrix whose rows have unit `U` and columns have unit `V`.
///
/// Matrix multiplication enforces dimensional consistency:
/// `Mat<U, V> * Mat<V, W> -> Mat<U, W>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<U, V, T = f64> {
    raw: DMatrix<T>,
    _unit: PhantomData<(U, V)>,
}

impl<U, T> Vec<U, T> {
    /// Wrap a raw dense dynamic vector, tagging it with unit `U`.
    pub fn from_raw(raw: DVector<T>) -> Self {
        Vec {
            raw,
            _unit: PhantomData,
        }
    }

    /// Borrow the underlying dimensionless dense vector.
    pub fn raw(&self) -> &DVector<T> {
        &self.raw
    }

    /// The number of components.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// True if the vector has no components.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}

impl<U, V, T> Mat<U, V, T> {
    /// Wrap a raw dense dynamic matrix, tagging rows with `U` and columns with
    /// `V`.
    pub fn from_raw(raw: DMatrix<T>) -> Self {
        Mat {
            raw,
            _unit: PhantomData,
        }
    }

    /// Borrow the underlying dimensionless dense matrix.
    pub fn raw(&self) -> &DMatrix<T> {
        &self.raw
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.raw.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.raw.ncols()
    }
}

#[cfg(feature = "alloc")]
impl<U, T: Clone> Vec<U, T> {
    /// A zero vector of length `n`.
    pub fn zeros(n: usize) -> Self
    where
        T: Scalar,
    {
        Vec::from_raw(DVector::zeros(n))
    }
}

#[cfg(feature = "alloc")]
impl<U, V, T: Clone> Mat<U, V, T> {
    /// A zero matrix of the given shape.
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Scalar,
    {
        Mat::from_raw(DMatrix::zeros(rows, cols))
    }
}

impl<U, T> core::ops::Index<usize> for Vec<U, T> {
    type Output = T;
    /// # Panics
    ///
    /// Panics if `i` is out of bounds for the underlying vector.
    fn index(&self, i: usize) -> &T {
        &self.raw[i]
    }
}

macro_rules! impl_vec_scalar {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl<U, T> $trait<T> for Vec<U, T>
        where
            T: Scalar + Clone,
            DVector<T>: $trait<T, Output = DVector<T>>,
        {
            type Output = Vec<U, T>;
            /// # Panics
            ///
            /// Inherits any panic of the equivalent dense scalar op (e.g.
            /// division by zero yields a non-finite value rather than panicking
            /// for `f64`; check the element type's contract).
            fn $fn(self, rhs: T) -> Vec<U, T> {
                Vec::from_raw(self.raw $op rhs)
            }
        }
    };
}
impl_vec_scalar!(Mul, mul, *);
impl_vec_scalar!(Div, div, /);

macro_rules! impl_same_unit_binop {
    ($type:ident, $trait:ident, $fn:ident, $op:tt) => {
        impl<U, T> $trait for $type<U, T>
        where
            T: Scalar + Clone,
            DVector<T>: $trait<Output = DVector<T>>,
        {
            type Output = $type<U, T>;
            /// # Panics
            ///
            /// Panics if the two operands have mismatched dimensions (the
            /// underlying dense op panics on shape mismatch).
            fn $fn(self, rhs: $type<U, T>) -> $type<U, T> {
                $type::from_raw(self.raw $op rhs.raw)
            }
        }
    };
}

impl_same_unit_binop!(Vec, Add, add, +);
impl_same_unit_binop!(Vec, Sub, sub, -);

impl<U, T> Neg for Vec<U, T>
where
    T: Scalar + Clone,
    DVector<T>: Neg<Output = DVector<T>>,
{
    type Output = Vec<U, T>;
    fn neg(self) -> Vec<U, T> {
        Vec::from_raw(-self.raw)
    }
}

impl<U, V, T> Add for Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Add<Output = DMatrix<T>>,
{
    type Output = Mat<U, V, T>;
    /// # Panics
    ///
    /// Panics if the two matrices have mismatched dimensions.
    fn add(self, rhs: Mat<U, V, T>) -> Mat<U, V, T> {
        Mat::from_raw(self.raw + rhs.raw)
    }
}

impl<U, V, T> Sub for Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Sub<Output = DMatrix<T>>,
{
    type Output = Mat<U, V, T>;
    /// # Panics
    ///
    /// Panics if the two matrices have mismatched dimensions.
    fn sub(self, rhs: Mat<U, V, T>) -> Mat<U, V, T> {
        Mat::from_raw(self.raw - rhs.raw)
    }
}

impl<U, V, T> Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Neg<Output = DMatrix<T>>,
{
    /// Negate every component.
    pub fn negate(self) -> Mat<U, V, T> {
        Mat::from_raw(-self.raw)
    }
}

impl<U, V, W, T> Mul<Mat<V, W, T>> for Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Mul<DMatrix<T>, Output = DMatrix<T>>,
{
    type Output = Mat<U, W, T>;
    /// # Panics
    ///
    /// Panics if the operand matrices have incompatible inner dimensions for
    /// multiplication (column count of `self` must equal row count of `rhs`).
    fn mul(self, rhs: Mat<V, W, T>) -> Mat<U, W, T> {
        Mat::from_raw(self.raw * rhs.raw)
    }
}

impl<U, V, T> Mul<Vec<V, T>> for Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Mul<DVector<T>, Output = DVector<T>>,
{
    type Output = Vec<U, T>;
    /// # Panics
    ///
    /// Panics if the matrix column count does not match the vector length.
    fn mul(self, rhs: Vec<V, T>) -> Vec<U, T> {
        Vec::from_raw(self.raw * rhs.raw)
    }
}

#[cfg(feature = "alloc")]
impl<U, V, T> Mat<U, V, T>
where
    T: Scalar + Clone,
{
    /// Transpose, swapping the row/column unit tags.
    pub fn transpose(self) -> Mat<V, U, T> {
        Mat::from_raw(self.raw.transpose())
    }
}

macro_rules! impl_mat_scalar {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl<U, V, T> $trait<T> for Mat<U, V, T>
        where
            T: Scalar + Clone,
            DMatrix<T>: $trait<T, Output = DMatrix<T>>,
        {
            type Output = Mat<U, V, T>;
            fn $fn(self, rhs: T) -> Mat<U, V, T> {
                Mat::from_raw(self.raw $op rhs)
            }
        }
    };
}
impl_mat_scalar!(Mul, mul, *);
impl_mat_scalar!(Div, div, /);

// By-reference operator forms. They clone the operands and delegate to the
// owned impls, so they cost one copy per operand; the owned forms remain the
// zero-copy path.
macro_rules! impl_ref_binop {
    ($type:ident, $trait:ident, $fn:ident, $op:tt) => {
        impl<'a, U, T> $trait<&'a $type<U, T>> for &'a $type<U, T>
        where
            T: Scalar + Clone,
            DVector<T>: $trait<Output = DVector<T>>,
        {
            type Output = $type<U, T>;
            fn $fn(self, rhs: &'a $type<U, T>) -> $type<U, T> {
                $type::from_raw(self.raw.clone() $op rhs.raw.clone())
            }
        }
    };
}
impl_ref_binop!(Vec, Add, add, +);
impl_ref_binop!(Vec, Sub, sub, -);

impl<'a, U, V, T> Add<&'a Mat<U, V, T>> for &'a Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Add<Output = DMatrix<T>>,
{
    type Output = Mat<U, V, T>;
    fn add(self, rhs: &'a Mat<U, V, T>) -> Mat<U, V, T> {
        Mat::from_raw(self.raw.clone() + rhs.raw.clone())
    }
}

impl<'a, U, V, T> Sub<&'a Mat<U, V, T>> for &'a Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Sub<Output = DMatrix<T>>,
{
    type Output = Mat<U, V, T>;
    fn sub(self, rhs: &'a Mat<U, V, T>) -> Mat<U, V, T> {
        Mat::from_raw(self.raw.clone() - rhs.raw.clone())
    }
}

impl<'a, U, V, W, T> Mul<&'a Mat<V, W, T>> for &'a Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Mul<DMatrix<T>, Output = DMatrix<T>>,
{
    type Output = Mat<U, W, T>;
    fn mul(self, rhs: &'a Mat<V, W, T>) -> Mat<U, W, T> {
        Mat::from_raw(self.raw.clone() * rhs.raw.clone())
    }
}

impl<'a, U, V, T> Mul<&'a Vec<V, T>> for &'a Mat<U, V, T>
where
    T: Scalar + Clone,
    DMatrix<T>: Mul<DVector<T>, Output = DVector<T>>,
{
    type Output = Vec<U, T>;
    fn mul(self, rhs: &'a Vec<V, T>) -> Vec<U, T> {
        Vec::from_raw(self.raw.clone() * rhs.raw.clone())
    }
}

impl<U, T> Vec<U, T>
where
    T: Scalar + Clone + 'static,
{
    /// Euclidean (L2) norm.
    ///
    /// The result carries the unit `U` in its meaning, but is returned as a bare
    /// scalar `T`: this crate does not model the unit algebra of `U * U`, so the
    /// caller tags the result.
    pub fn norm(&self) -> T {
        self.raw.norm()
    }

    /// Dot product with a vector of the same unit.
    ///
    /// Returns a bare scalar `T` for the same reason as [`Vec::norm`]: the
    /// product of two `U` quantities has unit `U²`, which is not modelled here.
    ///
    /// # Panics
    ///
    /// Panics if the two vectors have different lengths.
    pub fn dot(&self, other: &Vec<U, T>) -> T {
        self.raw.dot(&other.raw)
    }
}

#[cfg(feature = "alloc")]
impl<U, V> Mat<U, V, f64> {
    /// Solve `A x = b` for `x` by LU factorisation, where `A` is `self`.
    ///
    /// `b` has the row unit `U` and the solution `x` has the column unit `V`,
    /// so the solve is dimensionally consistent: `A: Mat<U, V>`, `x: Vec<V>`,
    /// `b: Vec<U>`.
    ///
    /// # Errors
    ///
    /// Returns the dense backend's [`DenseError`](tpt_math_linalg_dense::DenseError)
    /// for a non-square or singular matrix, or a length mismatch with `b`.
    pub fn solve(&self, b: &Vec<U, f64>) -> Result<Vec<V, f64>, tpt_math_linalg_dense::DenseError> {
        self.raw.solve(&b.raw).map(Vec::from_raw)
    }

    /// Inverse of a square matrix. Its rows carry the unit `V` and its columns
    /// the unit `U`, mirroring [`Mat::transpose`].
    ///
    /// # Errors
    ///
    /// Returns the dense backend's error for a non-square or singular matrix.
    pub fn inverse(&self) -> Result<Mat<V, U, f64>, tpt_math_linalg_dense::DenseError> {
        self.raw.inverse().map(Mat::from_raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_linalg_dense::{DMatrix, DVector};
    use tpt_math_units::prelude::{Length, Time, Velocity};

    #[test]
    fn vector_addition_keeps_unit() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0, 2.0]));
        let b = Vec::<Length>::from_raw(DVector::from_row_slice(&[3.0, 4.0]));
        let c = a + b;
        assert_eq!(c.raw(), &DVector::from_row_slice(&[4.0, 6.0]));
    }

    #[test]
    fn matrix_mul_propagates_units() {
        let m =
            Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]));
        let n = Mat::<Time, Velocity>::from_raw(DMatrix::from_row_slice(
            2,
            2,
            &[1.0_f64, 0.0, 0.0, 1.0],
        ));
        let p: Mat<Length, Velocity> = m * n;
        assert_eq!(
            p.raw(),
            &DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0])
        );
    }

    #[test]
    fn matrix_times_vector() {
        let m =
            Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[2.0_f64, 0.0, 0.0, 2.0]));
        let v = Vec::<Time>::from_raw(DVector::from_row_slice(&[1.0, 2.0]));
        let r: Vec<Length> = m * v;
        assert_eq!(r.raw(), &DVector::from_row_slice(&[2.0, 4.0]));
    }

    #[test]
    fn transpose_swaps_unit_tags() {
        let m = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(1, 2, &[1.0_f64, 2.0]));
        let t: Mat<Time, Length> = m.transpose();
        assert_eq!(t.nrows(), 2);
        assert_eq!(t.ncols(), 1);
    }

    #[test]
    fn mat_add_and_sub_keep_units() {
        let a = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        let b = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[4.0, 3.0, 2.0, 1.0]));
        let sum: Mat<Length, Time> = a.clone() + b.clone();
        assert_eq!(
            sum.raw(),
            &DMatrix::from_row_slice(2, 2, &[5.0, 5.0, 5.0, 5.0])
        );
        let diff: Mat<Length, Time> = a - b;
        assert_eq!(
            diff.raw(),
            &DMatrix::from_row_slice(2, 2, &[-3.0, -1.0, 1.0, 3.0])
        );
    }

    #[test]
    fn vec_sub_keeps_unit() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[5.0, 6.0]));
        let b = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0, 2.0]));
        let c = a - b;
        assert_eq!(c.raw(), &DVector::from_row_slice(&[4.0, 4.0]));
    }

    #[test]
    fn vec_scalar_mul_div_keep_unit() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[2.0, 4.0]));
        let doubled: Vec<Length> = a.clone() * 3.0;
        assert_eq!(doubled.raw(), &DVector::from_row_slice(&[6.0, 12.0]));
        let halved: Vec<Length> = a / 2.0;
        assert_eq!(halved.raw(), &DVector::from_row_slice(&[1.0, 2.0]));
    }

    #[test]
    fn vec_neg_keeps_unit() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0, -2.0]));
        let b: Vec<Length> = -a;
        assert_eq!(b.raw(), &DVector::from_row_slice(&[-1.0, 2.0]));
    }

    #[test]
    fn vec_index_and_len() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[3.0, 1.0, 4.0]));
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
        assert_eq!(a[0], 3.0);
        assert_eq!(a[2], 4.0);
        let empty = Vec::<Length>::from_raw(DVector::from_row_slice(&[]));
        assert!(empty.is_empty());
    }

    #[test]
    fn mat_scalar_mul_div_keep_units() {
        let m = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(1, 2, &[2.0_f64, 4.0]));
        let doubled: Mat<Length, Time> = m.clone() * 2.0;
        assert_eq!(doubled.raw(), &DMatrix::from_row_slice(1, 2, &[4.0, 8.0]));
        let halved: Mat<Length, Time> = m / 2.0;
        assert_eq!(halved.raw(), &DMatrix::from_row_slice(1, 2, &[1.0, 2.0]));
    }

    #[test]
    fn by_reference_ops_match_owned_ops() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0, 2.0]));
        let b = Vec::<Length>::from_raw(DVector::from_row_slice(&[3.0, 5.0]));
        assert_eq!(&a + &b, a.clone() + b.clone());
        assert_eq!(&a - &b, a.clone() - b.clone());

        let m = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        let n = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]));
        assert_eq!(&m + &n, m.clone() + n.clone());
        assert_eq!(&m - &n, m.clone() - n.clone());

        let p = Mat::<Time, Length>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
        let prod: Mat<Length, Length> = &m * &p;
        assert_eq!(prod, m.clone() * p.clone());

        let v = Vec::<Time>::from_raw(DVector::from_row_slice(&[1.0, 1.0]));
        let mv: Vec<Length> = &m * &v;
        assert_eq!(mv, m * v);
    }

    #[test]
    fn norm_and_dot() {
        let a = Vec::<Length>::from_raw(DVector::from_row_slice(&[3.0_f64, 4.0]));
        assert_eq!(a.norm(), 5.0);
        let b = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0_f64, 2.0]));
        assert_eq!(a.dot(&b), 11.0);
    }

    #[test]
    fn solve_carries_row_and_column_units() {
        // A: Length rows, Time columns; b: Length; x: Time.
        let a =
            Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[2.0_f64, 0.0, 0.0, 4.0]));
        let b = Vec::<Length>::from_raw(DVector::from_row_slice(&[2.0, 8.0]));
        let x: Vec<Time> = a.solve(&b).unwrap();
        assert_eq!(x.raw(), &DVector::from_row_slice(&[1.0, 2.0]));
    }

    #[test]
    fn solve_rejects_singular_matrix() {
        let a =
            Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 2.0, 4.0]));
        let b = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0, 1.0]));
        assert!(a.solve(&b).is_err());
    }

    #[test]
    fn inverse_swaps_unit_tags() {
        let a =
            Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[2.0_f64, 0.0, 0.0, 4.0]));
        let inv: Mat<Time, Length> = a.inverse().unwrap();
        assert_eq!(
            inv.raw(),
            &DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.25])
        );
    }

    #[test]
    fn zeros_construct_zero_filled() {
        let v = Vec::<Length>::zeros(3);
        assert_eq!(v.raw(), &DVector::from_row_slice(&[0.0, 0.0, 0.0]));
        let m = Mat::<Length, Time>::zeros(2, 3);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.raw(), &DMatrix::from_row_slice(2, 3, &[0.0; 6]));
    }
}
