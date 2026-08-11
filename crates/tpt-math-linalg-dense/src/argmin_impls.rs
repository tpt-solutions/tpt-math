//! `ArgminMath`-family trait impls for [`DVector<f64>`] / [`DMatrix<f64>`].
//!
//! These let the dense types drive `argmin` solvers without pulling in the
//! `nalgebra` backend of `argmin-math`. The implementations mirror the
//! semantics of `argmin-math`'s own backends but operate on this crate's
//! elementwise API.

use argmin_math::{
    ArgminAdd, ArgminConj, ArgminDiv, ArgminDot, ArgminEye, ArgminInv, ArgminL1Norm, ArgminL2Norm,
    ArgminMinMax, ArgminMul, ArgminSignum, ArgminSub, ArgminTranspose, ArgminZero, ArgminZeroLike,
    Error,
};

use crate::{DMatrix, DVector, DenseError};

// ===========================================================================
// DVector<f64>
// ===========================================================================

impl ArgminAdd<f64, DVector<f64>> for DVector<f64> {
    fn add(&self, other: &f64) -> DVector<f64> {
        self.clone() + *other
    }
}

impl ArgminAdd<DVector<f64>, DVector<f64>> for f64 {
    fn add(&self, other: &DVector<f64>) -> DVector<f64> {
        other.clone() + *self
    }
}

impl ArgminAdd<DVector<f64>, DVector<f64>> for DVector<f64> {
    fn add(&self, other: &DVector<f64>) -> DVector<f64> {
        self.clone() + other.clone()
    }
}

impl ArgminSub<f64, DVector<f64>> for DVector<f64> {
    fn sub(&self, other: &f64) -> DVector<f64> {
        self.clone() - *other
    }
}

impl ArgminSub<DVector<f64>, DVector<f64>> for f64 {
    fn sub(&self, other: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(other.iter().map(|x| *self - *x).collect())
    }
}

impl ArgminSub<DVector<f64>, DVector<f64>> for DVector<f64> {
    fn sub(&self, other: &DVector<f64>) -> DVector<f64> {
        self.clone() - other.clone()
    }
}

impl ArgminMul<f64, DVector<f64>> for DVector<f64> {
    fn mul(&self, other: &f64) -> DVector<f64> {
        self.clone() * *other
    }
}

impl ArgminMul<DVector<f64>, DVector<f64>> for DVector<f64> {
    fn mul(&self, other: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(
            self.iter()
                .zip(other.iter())
                .map(|(a, b)| *a * *b)
                .collect(),
        )
    }
}

impl ArgminMul<DVector<f64>, DVector<f64>> for f64 {
    fn mul(&self, other: &DVector<f64>) -> DVector<f64> {
        other.clone() * *self
    }
}

impl ArgminDiv<f64, DVector<f64>> for DVector<f64> {
    fn div(&self, other: &f64) -> DVector<f64> {
        self.clone() / *other
    }
}

impl ArgminDiv<DVector<f64>, DVector<f64>> for DVector<f64> {
    fn div(&self, other: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(
            self.iter()
                .zip(other.iter())
                .map(|(a, b)| *a / *b)
                .collect(),
        )
    }
}

impl ArgminDiv<DVector<f64>, DVector<f64>> for f64 {
    fn div(&self, other: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(other.iter().map(|x| *self / *x).collect())
    }
}

impl ArgminDot<DVector<f64>, f64> for DVector<f64> {
    fn dot(&self, other: &DVector<f64>) -> f64 {
        self.dot(other)
    }
}

impl ArgminDot<f64, DVector<f64>> for DVector<f64> {
    fn dot(&self, other: &f64) -> DVector<f64> {
        self.clone() * *other
    }
}

impl ArgminDot<DVector<f64>, DVector<f64>> for f64 {
    fn dot(&self, other: &DVector<f64>) -> DVector<f64> {
        other.clone() * *self
    }
}

impl ArgminL1Norm<f64> for DVector<f64> {
    fn l1_norm(&self) -> f64 {
        self.iter().map(|x| x.abs()).fold(0.0, |a, b| a + b)
    }
}

impl ArgminL2Norm<f64> for DVector<f64> {
    fn l2_norm(&self) -> f64 {
        self.norm()
    }
}

impl ArgminConj for DVector<f64> {
    fn conj(&self) -> Self {
        self.clone()
    }
}

impl ArgminZero for DVector<f64> {
    fn zero() -> Self {
        DVector::zeros(0)
    }
}

impl ArgminZeroLike for DVector<f64> {
    fn zero_like(&self) -> Self {
        DVector::zeros(self.len())
    }
}

// `ArgminScaledAdd` / `ArgminScaledSub` for `DVector<f64>` are provided by
// argmin-math's blanket impls (they require `f64: ArgminMul<DVector, DVector>`
// and `DVector: ArgminAdd`/`ArgminSub`, all of which this file implements).

impl ArgminMinMax for DVector<f64> {
    fn min(x: &Self, y: &Self) -> Self {
        DVector::from_vec(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| if *a < *b { *a } else { *b })
                .collect(),
        )
    }
    fn max(x: &Self, y: &Self) -> Self {
        DVector::from_vec(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| if *a > *b { *a } else { *b })
                .collect(),
        )
    }
}

impl ArgminSignum for DVector<f64> {
    fn signum(self) -> Self {
        DVector::from_vec(self.iter().map(|x| x.signum()).collect())
    }
}

// ===========================================================================
// DMatrix<f64>
// ===========================================================================

impl ArgminAdd<f64, DMatrix<f64>> for DMatrix<f64> {
    fn add(&self, other: &f64) -> DMatrix<f64> {
        self.clone() + *other
    }
}

impl ArgminAdd<DMatrix<f64>, DMatrix<f64>> for f64 {
    fn add(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        other.clone() + *self
    }
}

impl ArgminAdd<DMatrix<f64>, DMatrix<f64>> for DMatrix<f64> {
    fn add(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        self.clone() + other.clone()
    }
}

impl ArgminSub<f64, DMatrix<f64>> for DMatrix<f64> {
    fn sub(&self, other: &f64) -> DMatrix<f64> {
        self.clone() - *other
    }
}

impl ArgminSub<DMatrix<f64>, DMatrix<f64>> for f64 {
    fn sub(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(other.nrows(), other.ncols(), |i, j| *self - other[(i, j)])
    }
}

impl ArgminSub<DMatrix<f64>, DMatrix<f64>> for DMatrix<f64> {
    fn sub(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        self.clone() - other.clone()
    }
}

impl ArgminMul<f64, DMatrix<f64>> for DMatrix<f64> {
    fn mul(&self, other: &f64) -> DMatrix<f64> {
        self.clone() * *other
    }
}

impl ArgminMul<DMatrix<f64>, DMatrix<f64>> for DMatrix<f64> {
    fn mul(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(self.nrows(), self.ncols(), |i, j| {
            self[(i, j)] * other[(i, j)]
        })
    }
}

impl ArgminMul<DVector<f64>, DVector<f64>> for DMatrix<f64> {
    fn mul(&self, other: &DVector<f64>) -> DVector<f64> {
        self.clone() * other.clone()
    }
}

// NOTE: `ArgminMul<DVector<f64>, DVector<f64>> for f64` (scalar * vector) is
// provided once, in the `DVector` block above.

impl ArgminDiv<f64, DMatrix<f64>> for DMatrix<f64> {
    fn div(&self, other: &f64) -> DMatrix<f64> {
        self.clone() / *other
    }
}

impl ArgminDiv<DMatrix<f64>, DMatrix<f64>> for DMatrix<f64> {
    fn div(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(self.nrows(), self.ncols(), |i, j| {
            self[(i, j)] / other[(i, j)]
        })
    }
}

// NOTE: `ArgminDiv<DVector<f64>, DVector<f64>> for f64` (scalar / vector) is
// provided once, in the `DVector` block above.

impl ArgminDot<DMatrix<f64>, f64> for DMatrix<f64> {
    fn dot(&self, other: &DMatrix<f64>) -> f64 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a * *b)
            .fold(0.0, |a, b| a + b)
    }
}

impl ArgminDot<f64, DMatrix<f64>> for DMatrix<f64> {
    fn dot(&self, other: &f64) -> DMatrix<f64> {
        self.clone() * *other
    }
}

impl ArgminDot<DMatrix<f64>, DMatrix<f64>> for f64 {
    fn dot(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        other.clone() * *self
    }
}

impl ArgminDot<DVector<f64>, DVector<f64>> for DMatrix<f64> {
    fn dot(&self, other: &DVector<f64>) -> DVector<f64> {
        self.clone() * other.clone()
    }
}

impl ArgminDot<DMatrix<f64>, DMatrix<f64>> for DMatrix<f64> {
    fn dot(&self, other: &DMatrix<f64>) -> DMatrix<f64> {
        self.clone() * other.clone()
    }
}

impl ArgminEye for DMatrix<f64> {
    fn eye(n: usize) -> Self {
        DMatrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 })
    }
    fn eye_like(&self) -> Self {
        let n = self.nrows().min(self.ncols());
        DMatrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 })
    }
}

impl ArgminInv<DMatrix<f64>> for DMatrix<f64> {
    fn inv(&self) -> Result<DMatrix<f64>, Error> {
        self.inverse().map_err(map_dense_err)
    }
}

impl ArgminTranspose<DMatrix<f64>> for DMatrix<f64> {
    fn t(self) -> DMatrix<f64> {
        self.transpose()
    }
}

/// Map this crate's [`DenseError`] into `argmin`'s error type.
fn map_dense_err(e: DenseError) -> Error {
    Error::msg(e)
}
