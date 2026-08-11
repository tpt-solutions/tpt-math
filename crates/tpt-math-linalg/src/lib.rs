#![no_std]
#![allow(clippy::wrong_self_convention)]
//! Linear algebra with compile-time dimensional safety.
//!
//! [`nalgebra`] is a powerful, `no_std`-capable dense linear-algebra library,
//! but it is *dimensionless*: a `DVector<f64>` is just numbers. [`uom`] (via
//! [`tpt_math_units`]) gives compile-time unit checking, but has no vector or
//! matrix types. This crate is the bridge: it wraps nalgebra storage and tags
//! it with a phantom unit type `U` (typically a `uom` quantity), so that
//! adding a length-vector to a mass-vector is a compile error, and matrix
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
//!
//! // A position vector in metres. `Length`/`Time`/`Velocity` are the concrete
//! // `f64` SI quantity types re-exported by `tpt_math_units::prelude`.
//! let pos = Vec::<Length>::from_raw(nalgebra::dvector![3.0_f64, 4.0]);
//! let dur = Vec::<Time>::from_raw(nalgebra::dvector![2.0_f64, 1.0]);
//! // Cannot add `pos + dur`: different units (a compile error).
//! let _ = pos.raw() + dur.raw(); // raw nalgebra ops remain available
//! ```
//!
//! ## Backend decision
//!
//! This crate wraps **`nalgebra` only**. A dual `nalgebra`+`faer` facade was
//! explicitly rejected (see `spec.txt`): the two do not share a trait surface,
//! and rebuilding dense-linalg kernels is wasted effort. Platforms with proven
//! extreme dense-matrix performance needs should depend on `faer` directly.
//!
//! [`nalgebra`]: nalgebra
//! [`uom`]: tpt_math_units
//! [`tpt_math_units`]: tpt_math_units

pub use nalgebra;
pub use tpt_math_units;

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A vector of elements with unit `U` (typically a `uom` quantity type).
///
/// The unit is a phantom type: it costs nothing at runtime, but the type
/// system forbids mixing vectors of different units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vec<U, T = f64> {
    raw: nalgebra::DVector<T>,
    _unit: PhantomData<U>,
}

/// A matrix whose rows have unit `U` and columns have unit `V`.
///
/// Matrix multiplication enforces dimensional consistency:
/// `Mat<U, V> * Mat<V, W> -> Mat<U, W>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mat<U, V, T = f64> {
    raw: nalgebra::DMatrix<T>,
    _unit: PhantomData<(U, V)>,
}

impl<U, T> Vec<U, T> {
    /// Wrap a raw nalgebra dynamic vector, tagging it with unit `U`.
    pub fn from_raw(raw: nalgebra::DVector<T>) -> Self {
        Vec {
            raw,
            _unit: PhantomData,
        }
    }

    /// Borrow the underlying dimensionless nalgebra vector.
    pub fn raw(&self) -> &nalgebra::DVector<T> {
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
    /// Wrap a raw nalgebra dynamic matrix, tagging rows with `U` and columns
    /// with `V`.
    pub fn from_raw(raw: nalgebra::DMatrix<T>) -> Self {
        Mat {
            raw,
            _unit: PhantomData,
        }
    }

    /// Borrow the underlying dimensionless nalgebra matrix.
    pub fn raw(&self) -> &nalgebra::DMatrix<T> {
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

impl<U, T: Clone> Vec<U, T> {
    /// A zero vector of length `n`.
    pub fn zeros(n: usize) -> Self
    where
        T: nalgebra::Scalar + num_traits::Zero,
    {
        Vec::from_raw(nalgebra::DVector::zeros(n))
    }
}

impl<U, V, T: Clone> Mat<U, V, T> {
    /// A zero matrix of the given shape.
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: nalgebra::Scalar + num_traits::Zero,
    {
        Mat::from_raw(nalgebra::DMatrix::zeros(rows, cols))
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
            T: nalgebra::Scalar + Clone,
            nalgebra::DVector<T>: $trait<T, Output = nalgebra::DVector<T>>,
        {
            type Output = Vec<U, T>;
            /// # Panics
            ///
            /// Inherits any panic of the equivalent `nalgebra` scalar op (e.g.
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
            T: nalgebra::Scalar + Clone,
            nalgebra::DVector<T>: $trait<Output = nalgebra::DVector<T>>,
        {
            type Output = $type<U, T>;
            /// # Panics
            ///
            /// Panics if the two operands have mismatched dimensions (the
            /// underlying `nalgebra` op panics on shape mismatch).
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
    T: nalgebra::Scalar + Clone,
    nalgebra::DVector<T>: Neg<Output = nalgebra::DVector<T>>,
{
    type Output = Vec<U, T>;
    fn neg(self) -> Vec<U, T> {
        Vec::from_raw(-self.raw)
    }
}

impl<U, V, T> Add for Mat<U, V, T>
where
    T: nalgebra::Scalar + Clone,
    nalgebra::DMatrix<T>: Add<Output = nalgebra::DMatrix<T>>,
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
    T: nalgebra::Scalar + Clone,
    nalgebra::DMatrix<T>: Sub<Output = nalgebra::DMatrix<T>>,
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
    T: nalgebra::Scalar + Clone,
    nalgebra::DMatrix<T>: Neg<Output = nalgebra::DMatrix<T>>,
{
    /// Negate every component.
    pub fn negate(self) -> Mat<U, V, T> {
        Mat::from_raw(-self.raw)
    }
}

impl<U, V, W, T> Mul<Mat<V, W, T>> for Mat<U, V, T>
where
    T: nalgebra::Scalar + Clone,
    nalgebra::DMatrix<T>: Mul<nalgebra::DMatrix<T>, Output = nalgebra::DMatrix<T>>,
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
    T: nalgebra::Scalar + Clone,
    nalgebra::DMatrix<T>: Mul<nalgebra::DVector<T>, Output = nalgebra::DVector<T>>,
{
    type Output = Vec<U, T>;
    /// # Panics
    ///
    /// Panics if the matrix column count does not match the vector length.
    fn mul(self, rhs: Vec<V, T>) -> Vec<U, T> {
        Vec::from_raw(self.raw * rhs.raw)
    }
}

impl<U, V, T> Mat<U, V, T>
where
    T: nalgebra::Scalar + Clone,
{
    /// Transpose, swapping the row/column unit tags.
    pub fn transpose(self) -> Mat<V, U, T> {
        Mat::from_raw(self.raw.transpose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_units::prelude::{Length, Time, Velocity};

    #[test]
    fn vector_addition_keeps_unit() {
        let a = Vec::<Length>::from_raw(nalgebra::DVector::from_row_slice(&[1.0, 2.0]));
        let b = Vec::<Length>::from_raw(nalgebra::DVector::from_row_slice(&[3.0, 4.0]));
        let c = a + b;
        assert_eq!(c.raw(), &nalgebra::DVector::from_row_slice(&[4.0, 6.0]));
    }

    #[test]
    fn matrix_mul_propagates_units() {
        let m = Mat::<Length, Time>::from_raw(nalgebra::DMatrix::from_row_slice(
            2,
            2,
            &[1.0_f64, 2.0, 3.0, 4.0],
        ));
        let n = Mat::<Time, Velocity>::from_raw(nalgebra::DMatrix::from_row_slice(
            2,
            2,
            &[1.0_f64, 0.0, 0.0, 1.0],
        ));
        let p: Mat<Length, Velocity> = m * n;
        assert_eq!(
            p.raw(),
            &nalgebra::DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0])
        );
    }

    #[test]
    fn matrix_times_vector() {
        let m = Mat::<Length, Time>::from_raw(nalgebra::DMatrix::from_row_slice(
            2,
            2,
            &[2.0_f64, 0.0, 0.0, 2.0],
        ));
        let v = Vec::<Time>::from_raw(nalgebra::DVector::from_row_slice(&[1.0, 2.0]));
        let r: Vec<Length> = m * v;
        assert_eq!(r.raw(), &nalgebra::DVector::from_row_slice(&[2.0, 4.0]));
    }

    #[test]
    fn transpose_swaps_unit_tags() {
        let m =
            Mat::<Length, Time>::from_raw(nalgebra::DMatrix::from_row_slice(1, 2, &[1.0_f64, 2.0]));
        let t: Mat<Time, Length> = m.transpose();
        assert_eq!(t.nrows(), 2);
        assert_eq!(t.ncols(), 1);
    }
}
