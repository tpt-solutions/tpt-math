#![no_std]
#![forbid(unsafe_code)]
// Fixed-size matrix/vector algorithms are clearest with explicit indexing over
// fixed-size arrays; the indexed-loop and manual-swap lints do not fit this
// code, and the operator-impl lints misfire on componentwise arithmetic.
#![allow(
    clippy::needless_range_loop,
    clippy::manual_swap,
    clippy::suspicious_arithmetic_impl,
    clippy::suspicious_op_assign_impl,
    clippy::items_after_test_module
)]
//! Geometry module built on [`tpt_math_linalg_fixed`].
//!
//! A from-scratch, const-generic geometry layer (matching the breadth of
//! `nalgebra`'s geometry module) with **no `nalgebra` dependency** and **no
//! allocator** — everything here is stack-allocated fixed-size storage. It
//! provides points, translations, rotations (2-D and 3-D), unit quaternions,
//! isometries, similarities, uniform scaling and the perspective / orthographic
//! projection matrices.
//!
//! # Conventions (stated explicitly)
//!
//! * **Active (alibi) rotations.** A rotation acts on a vector `v` as `R * v`
//!   (matrix–vector multiplication with the vector on the right), which rotates
//!   the *point/vector* rather than the *coordinate frame*.
//! * **Column vectors.** Points and vectors are column vectors; transformations
//!   are applied as `M * v`.
//! * **Right-handed coordinates.** The 2-D rotation by a positive angle is
//!   counter-clockwise. The 3-D right-handed basis is `x × y = z`.
//! * **Euler angles** (3-D) use the intrinsic Tait–Bryan order
//!   `Rz(yaw) · Ry(pitch) · Rx(roll)` — that is, the vector is first rolled
//!   about `x`, then pitched about `y`, then yawed about `z`.
//! * **Quaternions are Hamilton quaternions** `q = w + x i + y j + z k` (scalar
//!   part `w` last), matching `nalgebra`'s memory layout.
//! * **Projection matrices** are right-handed, look down `-z`, with Normalised
//!   Device Coordinates `z` in `[-1, 1]` (OpenGL-style depth range).
//! * **Isometry composition** is written as `B * A` and means "apply `A`,
//!   then `B`": `(B * A)(p) = B.rotation * (A.rotation * p + A.translation)
//!   + B.translation`.

use core::ops::{Add, Mul, Neg, Sub};

use tpt_math_linalg_fixed::{Matrix, Matrix2, Matrix3, Matrix4, Vector, Vector3, Vector4};
use tpt_math_numeric::Scalar;

// ===========================================================================
// Point
// ===========================================================================

/// A point in `D`-dimensional Euclidean space.
///
/// A point is an affine location; unlike a [`Vector`], subtracting two points
/// yields a displacement [`Vector`], and a point can be offset by a [`Vector`]
/// but not directly added to another point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point<T, const D: usize> {
    /// The coordinates of the point.
    pub coords: Vector<T, D>,
}

impl<T: Scalar, const D: usize> Point<T, D> {
    /// Build a point from its coordinates.
    pub fn new(coords: Vector<T, D>) -> Self {
        Point { coords }
    }

    /// The origin (all coordinates zero).
    pub fn origin() -> Self {
        Point {
            coords: Vector::from_fn(|_| T::zero()),
        }
    }

    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Point<T, D>) -> T {
        (*self - *other).norm()
    }

    /// Translate the point by a vector, returning a new point.
    pub fn translate(&self, by: &Vector<T, D>) -> Point<T, D> {
        Point {
            coords: self.coords + *by,
        }
    }
}

impl<T: Copy, const D: usize> Point<T, D> {
    /// Build a point from a fixed-size coordinate array.
    pub fn from_array(data: [T; D]) -> Self {
        Point {
            coords: Vector::new(data),
        }
    }
}

impl<T: Copy> Point<T, 2> {
    /// First coordinate.
    pub fn x(&self) -> T {
        self.coords.data[0]
    }
    /// Second coordinate.
    pub fn y(&self) -> T {
        self.coords.data[1]
    }
}

impl<T: Copy> Point<T, 3> {
    /// First coordinate.
    pub fn x(&self) -> T {
        self.coords.data[0]
    }
    /// Second coordinate.
    pub fn y(&self) -> T {
        self.coords.data[1]
    }
    /// Third coordinate.
    pub fn z(&self) -> T {
        self.coords.data[2]
    }
}

impl<T: Scalar, const D: usize> Add<Vector<T, D>> for Point<T, D> {
    type Output = Point<T, D>;
    fn add(self, rhs: Vector<T, D>) -> Point<T, D> {
        Point {
            coords: self.coords + rhs,
        }
    }
}

impl<T: Scalar, const D: usize> Sub<Vector<T, D>> for Point<T, D> {
    type Output = Point<T, D>;
    fn sub(self, rhs: Vector<T, D>) -> Point<T, D> {
        Point {
            coords: self.coords - rhs,
        }
    }
}

impl<T: Scalar, const D: usize> Sub<Point<T, D>> for Point<T, D> {
    type Output = Vector<T, D>;
    fn sub(self, rhs: Point<T, D>) -> Vector<T, D> {
        self.coords - rhs.coords
    }
}

/// 2-D point.
pub type Point2<T> = Point<T, 2>;
/// 3-D point.
pub type Point3<T> = Point<T, 3>;

// ===========================================================================
// Translation
// ===========================================================================

/// A translation in `D`-dimensional space.
///
/// Translations form an abelian group under composition; `T2 * T1` shifts by
/// `T1.vector` and then by `T2.vector`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Translation<T, const D: usize> {
    /// The translation vector.
    pub vector: Vector<T, D>,
}

impl<T: Scalar, const D: usize> Translation<T, D> {
    /// Build a translation from a vector.
    pub fn new(vector: Vector<T, D>) -> Self {
        Translation { vector }
    }

    /// The identity translation (zero vector).
    pub fn identity() -> Self {
        Translation {
            vector: Vector::from_fn(|_| T::zero()),
        }
    }

    /// The inverse translation: negates the vector.
    pub fn inverse(&self) -> Self {
        Translation {
            vector: -self.vector,
        }
    }

    /// Apply the translation to a point.
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        *pt + self.vector
    }

    /// Apply the translation to a vector (a no-op: vectors are translation
    /// invariant).
    pub fn transform_vector(&self, vec: &Vector<T, D>) -> Vector<T, D> {
        *vec
    }
}

impl<T: Scalar, const D: usize> Mul for Translation<T, D> {
    type Output = Translation<T, D>;
    fn mul(self, rhs: Translation<T, D>) -> Translation<T, D> {
        Translation {
            vector: self.vector + rhs.vector,
        }
    }
}

// ===========================================================================
// Rotation
// ===========================================================================

/// An orthogonal, orientation-preserving (determinant `+1`) linear map in
/// `D` dimensions, stored as its rotation matrix.
///
/// Orthogonality is guaranteed by construction: every constructor produces a
/// proper rotation matrix, and [`Rotation::inverse`] is the exact transpose
/// (no renormalisation needed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rotation<T, const D: usize> {
    matrix: Matrix<T, D, D>,
}

impl<T: Scalar, const D: usize> Rotation<T, D> {
    /// The identity rotation (the `D×D` identity matrix).
    pub fn identity() -> Self {
        Rotation {
            matrix: Matrix::identity(),
        }
    }

    /// Build a rotation from a matrix *without* checking that it is a proper
    /// rotation. The caller must ensure orthogonality and unit determinant.
    pub fn from_matrix_unchecked(matrix: Matrix<T, D, D>) -> Self {
        Rotation { matrix }
    }

    /// Build a rotation from an (approximately) orthogonal matrix, by
    /// orthonormalising it via Gram–Schmidt. Returns `None` if the matrix is
    /// numerically singular.
    pub fn from_matrix(matrix: Matrix<T, D, D>) -> Option<Self> {
        // Orthonormalise columns via Gram–Schmidt, then re-orthogonalise the
        // last column against all the previous ones.
        let mut cols = [[T::zero(); D]; D];
        for j in 0..D {
            for i in 0..D {
                cols[j][i] = matrix.data[i][j]; // copy the j-th column of the input
            }
            for i in 0..j {
                // subtract projection onto column i
                let mut dot = T::zero();
                for k in 0..D {
                    dot = dot + cols[i][k] * cols[j][k];
                }
                for k in 0..D {
                    cols[j][k] = cols[j][k] - cols[i][k] * dot;
                }
            }
            // normalise column j
            let mut n = T::zero();
            for k in 0..D {
                n = n + cols[j][k] * cols[j][k];
            }
            n = n.sqrt();
            if n == T::zero() {
                return None;
            }
            for k in 0..D {
                cols[j][k] = cols[j][k] / n;
            }
        }
        // Re-orthogonalise the last column against the first D-1 (Gram–Schmidt
        // drift correction) and renormalise.
        let last = D - 1;
        if D > 1 {
            let mut dot = T::zero();
            for i in 0..last {
                let mut d = T::zero();
                for k in 0..D {
                    d = d + cols[i][k] * cols[last][k];
                }
                dot = dot + d;
                for k in 0..D {
                    cols[last][k] = cols[last][k] - cols[i][k] * d;
                }
            }
            let mut n = T::zero();
            for k in 0..D {
                n = n + cols[last][k] * cols[last][k];
            }
            n = n.sqrt();
            if n == T::zero() {
                return None;
            }
            for k in 0..D {
                cols[last][k] = cols[last][k] / n;
            }
            // silence unused warning when D == 1
            let _ = dot;
        }
        // Build a row-major matrix from the column storage.
        let mut data = [[T::zero(); D]; D];
        for j in 0..D {
            for i in 0..D {
                data[i][j] = cols[j][i];
            }
        }
        Some(Rotation {
            matrix: Matrix::new(data),
        })
    }

    /// Borrow the underlying rotation matrix.
    pub fn matrix(&self) -> &Matrix<T, D, D> {
        &self.matrix
    }

    /// The inverse rotation (the matrix transpose).
    pub fn inverse(&self) -> Self
    where
        T: Copy,
    {
        Rotation {
            matrix: self.matrix.transpose(),
        }
    }

    /// Rotate a vector: `R * v`.
    pub fn transform_vector(&self, vec: &Vector<T, D>) -> Vector<T, D> {
        self.matrix * *vec
    }

    /// Rotate a point about the origin: `R * p`.
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        Point {
            coords: self.matrix * pt.coords,
        }
    }
}

impl<T: Scalar, const D: usize> Mul for Rotation<T, D> {
    type Output = Rotation<T, D>;
    fn mul(self, rhs: Rotation<T, D>) -> Rotation<T, D> {
        Rotation {
            matrix: self.matrix * rhs.matrix,
        }
    }
}

impl<T: Scalar, const D: usize> Mul<Vector<T, D>> for Rotation<T, D> {
    type Output = Vector<T, D>;
    fn mul(self, rhs: Vector<T, D>) -> Vector<T, D> {
        self.matrix * rhs
    }
}

/// 2-D rotation.
pub type Rotation2<T> = Rotation<T, 2>;
/// 3-D rotation.
pub type Rotation3<T> = Rotation<T, 3>;

impl<T: Scalar> Rotation<T, 2> {
    /// Build a 2-D rotation by `angle` radians (counter-clockwise for a positive
    /// angle in a right-handed plane).
    pub fn from_angle(angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Rotation {
            matrix: Matrix2::new([[c, -s], [s, c]]),
        }
    }
}

impl<T: Scalar> Rotation<T, 3> {
    /// Build a 3-D rotation of `angle` radians about a (normalised) `axis`.
    ///
    /// Uses Rodrigues' rotation formula; the axis is normalised internally, so a
    /// non-unit axis is accepted.
    pub fn from_axis_angle(axis: &Vector<T, 3>, angle: T) -> Self {
        let n = axis.norm();
        let (kx, ky, kz) = if n == T::zero() {
            (T::zero(), T::zero(), T::zero())
        } else {
            (axis.data[0] / n, axis.data[1] / n, axis.data[2] / n)
        };
        let (s, c) = angle.sin_cos();
        let t = T::one() - c;
        let m = [
            [c + t * kx * kx, t * kx * ky - s * kz, t * kx * kz + s * ky],
            [t * ky * kx + s * kz, c + t * ky * ky, t * ky * kz - s * kx],
            [t * kz * kx - s * ky, t * kz * ky + s * kx, c + t * kz * kz],
        ];
        Rotation {
            matrix: Matrix3::new(m),
        }
    }

    /// Build a 3-D rotation from intrinsic Tait–Bryan Euler angles
    /// `(roll, pitch, yaw)` using the order `Rz(yaw) · Ry(pitch) · Rx(roll)`.
    pub fn from_euler(roll: T, pitch: T, yaw: T) -> Self {
        let rx = Rotation3::from_axis_angle(&Vector3::new([T::one(), T::zero(), T::zero()]), roll);
        let ry = Rotation3::from_axis_angle(&Vector3::new([T::zero(), T::one(), T::zero()]), pitch);
        let rz = Rotation3::from_axis_angle(&Vector3::new([T::zero(), T::zero(), T::one()]), yaw);
        rz * ry * rx
    }

    /// Override the generic [`Rotation::powf`] with a proper 3-D
    /// axis-angle interpolation.
    pub fn powf(&self, t: T) -> Rotation<T, 3>
    where
        T: Copy,
    {
        // Convert to an axis-angle via the trace identity, then scale the
        // angle. `cos(theta) = (trace - 1) / 2`.
        let tr = self.matrix.data[0][0] + self.matrix.data[1][1] + self.matrix.data[2][2];
        let cos_t = (tr - T::one()) / (T::one() + T::one());
        let angle = cos_t.acos();
        if angle < T::from(1e-12).unwrap_or(T::zero()) {
            return *self;
        }
        // Axis from the skew-symmetric part.
        let x = self.matrix.data[2][1] - self.matrix.data[1][2];
        let y = self.matrix.data[0][2] - self.matrix.data[2][0];
        let z = self.matrix.data[1][0] - self.matrix.data[0][1];
        let axis = Vector3::new([x, y, z]);
        Rotation3::from_axis_angle(&axis, angle * t)
    }
}

// ===========================================================================
// Quaternion
// ===========================================================================

/// A quaternion `q = w + x i + y j + z k` (Hamilton convention, scalar `w`
/// last).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quaternion<T> {
    /// Components `[x, y, z, w]` (scalar `w` last, matching `nalgebra`).
    pub coords: Vector4<T>,
}

impl<T: Scalar> Quaternion<T> {
    /// Build a quaternion from its `[x, y, z, w]` components.
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Quaternion {
            coords: Vector4::new([x, y, z, w]),
        }
    }

    /// The quaternion conjugate `w - x i - y j - z k`.
    pub fn conjugate(&self) -> Self {
        Quaternion::new(
            -self.coords.data[0],
            -self.coords.data[1],
            -self.coords.data[2],
            self.coords.data[3],
        )
    }

    /// The squared norm `w² + x² + y² + z²`.
    pub fn norm_squared(&self) -> T {
        self.coords.dot(&self.coords)
    }

    /// The norm `sqrt(w² + x² + y² + z²)`.
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Normalise to a unit quaternion. Returns `None` if the norm is zero.
    pub fn normalize(&self) -> Option<Self> {
        let n = self.norm();
        if n == T::zero() {
            return None;
        }
        let inv = T::one() / n;
        Some(Quaternion::new(
            self.coords.data[0] * inv,
            self.coords.data[1] * inv,
            self.coords.data[2] * inv,
            self.coords.data[3] * inv,
        ))
    }

    /// Hamilton product `self * rhs` (composition of rotations:
    /// `rhs` is applied first, then `self`).
    pub fn multiply(&self, rhs: &Quaternion<T>) -> Quaternion<T> {
        let (x1, y1, z1, w1) = (
            self.coords.data[0],
            self.coords.data[1],
            self.coords.data[2],
            self.coords.data[3],
        );
        let (x2, y2, z2, w2) = (
            rhs.coords.data[0],
            rhs.coords.data[1],
            rhs.coords.data[2],
            rhs.coords.data[3],
        );
        Quaternion::new(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    }

    /// Rotate a 3-D vector by this (unit) quaternion: `q * (0, v) * q⁻¹`.
    pub fn rotate_vector(&self, v: &Vector<T, 3>) -> Vector<T, 3> {
        let (vx, vy, vz) = (v.data[0], v.data[1], v.data[2]);
        let q = self;
        // p = q * (0, v)
        let pw = -(q.coords.data[0] * vx + q.coords.data[1] * vy + q.coords.data[2] * vz);
        let px = q.coords.data[3] * vx + (q.coords.data[1] * vz - q.coords.data[2] * vy);
        let py = q.coords.data[3] * vy + (q.coords.data[2] * vx - q.coords.data[0] * vz);
        let pz = q.coords.data[3] * vz + (q.coords.data[0] * vy - q.coords.data[1] * vx);
        // result = p * conj(q)
        let c = q.conjugate();
        let rx = -pw * c.coords.data[0] + c.coords.data[3] * px - c.coords.data[1] * pz
            + c.coords.data[2] * py;
        let ry = -pw * c.coords.data[1] + c.coords.data[3] * py - c.coords.data[2] * px
            + c.coords.data[0] * pz;
        let rz = -pw * c.coords.data[2] + c.coords.data[3] * pz - c.coords.data[0] * py
            + c.coords.data[1] * px;
        Vector3::new([rx, ry, rz])
    }

    /// Build a unit quaternion from a 3-D rotation matrix (the inverse of
    /// [`Rotation3::from_quaternion`]).
    pub fn from_rotation_matrix(rot: &Rotation3<T>) -> UnitQuaternion<T> {
        let m = rot.matrix().data;
        let m00 = m[0][0];
        let m01 = m[0][1];
        let m02 = m[0][2];
        let m10 = m[1][0];
        let m11 = m[1][1];
        let m12 = m[1][2];
        let m20 = m[2][0];
        let m21 = m[2][1];
        let m22 = m[2][2];
        let one = T::one();
        let two = one + one;
        let four = two + two;
        let (x, y, z, w);
        let trace = m00 + m11 + m22;
        if trace > T::zero() {
            let s = (trace + one).sqrt() / two;
            w = s;
            let inv = one / (two * two * s);
            x = (m21 - m12) * inv;
            y = (m02 - m20) * inv;
            z = (m10 - m01) * inv;
        } else if m00 > m11 && m00 > m22 {
            let s = (one + m00 - m11 - m22).sqrt() / two;
            x = s;
            let inv = one / (two * two * s);
            y = (m01 + m10) * inv;
            z = (m02 + m20) * inv;
            w = (m21 - m12) * inv;
        } else if m11 > m22 {
            let s = (one + m11 - m00 - m22).sqrt() / two;
            y = s;
            let inv = one / (two * two * s);
            x = (m01 + m10) * inv;
            z = (m12 + m21) * inv;
            w = (m02 - m20) * inv;
        } else {
            let s = (one + m22 - m00 - m11).sqrt() / two;
            z = s;
            let inv = one / (two * two * s);
            x = (m02 + m20) * inv;
            y = (m12 + m21) * inv;
            w = (m10 - m01) * inv;
        }
        let _ = four;
        UnitQuaternion::from_quaternion(Quaternion::new(x, y, z, w))
    }

    /// Spherical linear interpolation between two unit quaternions.
    pub fn slerp(a: &UnitQuaternion<T>, b: &UnitQuaternion<T>, t: T) -> UnitQuaternion<T> {
        let qa = a.quaternion();
        let qb = b.quaternion();
        let dot = qa.coords.dot(&qb.coords);
        // Take the shorter arc.
        let (bq, dot) = if dot < T::zero() {
            (
                Quaternion::new(
                    -qb.coords.data[0],
                    -qb.coords.data[1],
                    -qb.coords.data[2],
                    -qb.coords.data[3],
                ),
                -dot,
            )
        } else {
            (*qb, dot)
        };
        let one = T::one();
        let eps = T::from(1e-12).unwrap_or(T::zero());
        if dot > one - eps {
            // Nearly parallel: linear interp + renormalise.
            let r = Quaternion::new(
                qa.coords.data[0] + (bq.coords.data[0] - qa.coords.data[0]) * t,
                qa.coords.data[1] + (bq.coords.data[1] - qa.coords.data[1]) * t,
                qa.coords.data[2] + (bq.coords.data[2] - qa.coords.data[2]) * t,
                qa.coords.data[3] + (bq.coords.data[3] - qa.coords.data[3]) * t,
            );
            return UnitQuaternion::from_quaternion(r);
        }
        let theta0 = dot.acos();
        let theta = theta0 * t;
        let sin0 = theta0.sin();
        let (w1, w2) = if sin0 == T::zero() {
            (one - t, t)
        } else {
            ((theta0 - theta).sin() / sin0, theta.sin() / sin0)
        };
        let r = Quaternion::new(
            qa.coords.data[0] * w1 + bq.coords.data[0] * w2,
            qa.coords.data[1] * w1 + bq.coords.data[1] * w2,
            qa.coords.data[2] * w1 + bq.coords.data[2] * w2,
            qa.coords.data[3] * w1 + bq.coords.data[3] * w2,
        );
        UnitQuaternion::from_quaternion(r)
    }
}

impl<T: Scalar> Mul<Quaternion<T>> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: Quaternion<T>) -> Quaternion<T> {
        self.multiply(&rhs)
    }
}

impl<T: Scalar> Mul<T> for Quaternion<T> {
    type Output = Quaternion<T>;
    /// Component-wise scaling by a scalar.
    fn mul(self, rhs: T) -> Quaternion<T> {
        Quaternion::new(
            self.coords.data[0] * rhs,
            self.coords.data[1] * rhs,
            self.coords.data[2] * rhs,
            self.coords.data[3] * rhs,
        )
    }
}

impl<T: Scalar> Add for Quaternion<T> {
    type Output = Quaternion<T>;
    fn add(self, rhs: Quaternion<T>) -> Quaternion<T> {
        Quaternion::new(
            self.coords.data[0] + rhs.coords.data[0],
            self.coords.data[1] + rhs.coords.data[1],
            self.coords.data[2] + rhs.coords.data[2],
            self.coords.data[3] + rhs.coords.data[3],
        )
    }
}

impl<T: Scalar> Sub for Quaternion<T> {
    type Output = Quaternion<T>;
    fn sub(self, rhs: Quaternion<T>) -> Quaternion<T> {
        Quaternion::new(
            self.coords.data[0] - rhs.coords.data[0],
            self.coords.data[1] - rhs.coords.data[1],
            self.coords.data[2] - rhs.coords.data[2],
            self.coords.data[3] - rhs.coords.data[3],
        )
    }
}

impl<T: Scalar> Neg for Quaternion<T> {
    type Output = Quaternion<T>;
    fn neg(self) -> Quaternion<T> {
        Quaternion::new(
            -self.coords.data[0],
            -self.coords.data[1],
            -self.coords.data[2],
            -self.coords.data[3],
        )
    }
}

/// A unit (normalised) quaternion, representing a 3-D rotation.
///
/// The unit-norm invariant is enforced on construction, so composition and
/// inversion are numerically safe without re-normalisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitQuaternion<T> {
    quaternion: Quaternion<T>,
}

impl<T: Scalar> UnitQuaternion<T> {
    /// Build from an arbitrary quaternion, normalising it. Panics if the
    /// quaternion is zero.
    pub fn from_quaternion(q: Quaternion<T>) -> Self {
        UnitQuaternion {
            quaternion: q.normalize().expect("cannot normalise a zero quaternion"),
        }
    }

    /// Build from a rotation about an axis by an angle.
    pub fn from_axis_angle(axis: &Vector<T, 3>, angle: T) -> Self {
        let n = axis.norm();
        let (kx, ky, kz) = if n == T::zero() {
            (T::zero(), T::zero(), T::zero())
        } else {
            (axis.data[0] / n, axis.data[1] / n, axis.data[2] / n)
        };
        let half = angle / (T::one() + T::one());
        let (s, c) = half.sin_cos();
        UnitQuaternion::from_quaternion(Quaternion::new(kx * s, ky * s, kz * s, c))
    }

    /// Build from intrinsic Tait–Bryan Euler angles `(roll, pitch, yaw)` using
    /// the order `Rz(yaw) · Ry(pitch) · Rx(roll)`.
    pub fn from_euler(roll: T, pitch: T, yaw: T) -> Self {
        let qx =
            UnitQuaternion::from_axis_angle(&Vector3::new([T::one(), T::zero(), T::zero()]), roll);
        let qy =
            UnitQuaternion::from_axis_angle(&Vector3::new([T::zero(), T::one(), T::zero()]), pitch);
        let qz =
            UnitQuaternion::from_axis_angle(&Vector3::new([T::zero(), T::zero(), T::one()]), yaw);
        qz * qy * qx
    }

    /// Build from a rotation matrix (see
    /// [`Quaternion::from_rotation_matrix`]).
    pub fn from_rotation_matrix(rot: &Rotation3<T>) -> Self {
        Quaternion::from_rotation_matrix(rot)
    }

    /// Borrow the underlying unit quaternion.
    pub fn quaternion(&self) -> &Quaternion<T> {
        &self.quaternion
    }

    /// The inverse rotation (the conjugate, which for a unit quaternion equals
    /// the reciprocal).
    pub fn inverse(&self) -> Self {
        UnitQuaternion {
            quaternion: self.quaternion.conjugate(),
        }
    }

    /// Rotate a 3-D vector.
    pub fn rotate_vector(&self, v: &Vector<T, 3>) -> Vector<T, 3> {
        self.quaternion.rotate_vector(v)
    }

    /// Convert to the equivalent 3-D rotation matrix.
    pub fn to_rotation_matrix(&self) -> Rotation3<T> {
        Rotation3::from_quaternion(self)
    }
}

impl<T: Scalar> Mul<UnitQuaternion<T>> for UnitQuaternion<T> {
    type Output = UnitQuaternion<T>;
    fn mul(self, rhs: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: self.quaternion.multiply(&rhs.quaternion),
        }
    }
}

impl<T: Scalar> Rotation<T, 3> {
    /// Build a 3-D rotation from a unit quaternion.
    pub fn from_quaternion(q: &UnitQuaternion<T>) -> Self {
        let (x, y, z, w) = (
            q.quaternion().coords.data[0],
            q.quaternion().coords.data[1],
            q.quaternion().coords.data[2],
            q.quaternion().coords.data[3],
        );
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;
        let two = T::one() + T::one();
        let m = [
            [T::one() - two * (yy + zz), two * (xy - wz), two * (xz + wy)],
            [two * (xy + wz), T::one() - two * (xx + zz), two * (yz - wx)],
            [two * (xz - wy), two * (yz + wx), T::one() - two * (xx + yy)],
        ];
        Rotation {
            matrix: Matrix3::new(m),
        }
    }
}

// ===========================================================================
// Isometry
// ===========================================================================

/// A rigid motion: a rotation followed by a translation.
///
/// Composition is written `B * A` and means "apply `A`, then `B`":
/// `(B * A)(p) = B.rotation * (A.rotation * p + A.translation)
/// + B.translation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Isometry<T, const D: usize> {
    /// The (linear) rotation part.
    pub rotation: Rotation<T, D>,
    /// The (affine) translation part.
    pub translation: Translation<T, D>,
}

impl<T: Scalar, const D: usize> Isometry<T, D> {
    /// Build an isometry from a rotation and a translation.
    pub fn new(translation: Translation<T, D>, rotation: Rotation<T, D>) -> Self {
        Isometry {
            rotation,
            translation,
        }
    }

    /// The identity isometry (no rotation, no translation).
    pub fn identity() -> Self {
        Isometry {
            rotation: Rotation::identity(),
            translation: Translation::identity(),
        }
    }

    /// An isometry with only a rotation (zero translation).
    pub fn from_rotation(rotation: Rotation<T, D>) -> Self {
        Isometry {
            rotation,
            translation: Translation::identity(),
        }
    }

    /// An isometry with only a translation (identity rotation).
    pub fn from_translation(translation: Translation<T, D>) -> Self {
        Isometry {
            rotation: Rotation::identity(),
            translation,
        }
    }

    /// The inverse isometry.
    ///
    /// If `self(p) = R p + t`, then `self.inverse()(p) = Rᵀ p - Rᵀ t`.
    pub fn inverse(&self) -> Self
    where
        T: Copy,
    {
        let r_inv = self.rotation.inverse();
        let t_inv = Translation::new(-(r_inv.transform_vector(&self.translation.vector)));
        Isometry {
            rotation: r_inv,
            translation: t_inv,
        }
    }

    /// Apply to a point: `R * p + t`.
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D>
    where
        T: Copy,
    {
        Point {
            coords: self.rotation.transform_vector(&pt.coords) + self.translation.vector,
        }
    }

    /// Apply to a vector: `R * v` (translation invariant).
    pub fn transform_vector(&self, vec: &Vector<T, D>) -> Vector<T, D>
    where
        T: Copy,
    {
        self.rotation.transform_vector(vec)
    }
}

impl<T: Scalar, const D: usize> Mul for Isometry<T, D>
where
    T: Copy,
{
    type Output = Isometry<T, D>;
    fn mul(self, rhs: Isometry<T, D>) -> Isometry<T, D> {
        // (B * A): rotation = B.rotation * A.rotation,
        // translation = B.rotation * A.translation + B.translation.
        let rotation = self.rotation * rhs.rotation;
        let translated = self.rotation.transform_vector(&rhs.translation.vector);
        let translation = Translation::new(translated + self.translation.vector);
        Isometry {
            rotation,
            translation,
        }
    }
}

/// 2-D isometry.
pub type Isometry2<T> = Isometry<T, 2>;
/// 3-D isometry.
pub type Isometry3<T> = Isometry<T, 3>;

// ===========================================================================
// Similarity
// ===========================================================================

/// A similarity transform: a uniform scaling, then a rotation, then a
/// translation.
///
/// `self(p) = s * (R * p) + t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Similarity<T, const D: usize> {
    /// The (linear) rotation part.
    pub rotation: Rotation<T, D>,
    /// The (affine) translation part.
    pub translation: Translation<T, D>,
    /// The uniform scaling factor `s`.
    pub scaling: T,
}

impl<T: Scalar, const D: usize> Similarity<T, D> {
    /// Build a similarity from a scaling factor, a rotation and a translation.
    pub fn new(scaling: T, translation: Translation<T, D>, rotation: Rotation<T, D>) -> Self {
        Similarity {
            rotation,
            translation,
            scaling,
        }
    }

    /// The identity similarity (`s = 1`, no rotation, no translation).
    pub fn identity() -> Self {
        Similarity {
            rotation: Rotation::identity(),
            translation: Translation::identity(),
            scaling: T::one(),
        }
    }

    /// The inverse similarity.
    ///
    /// If `self(p) = s (R p) + t`, then
    /// `self.inverse()(p) = (1/s) Rᵀ (p - t)`.
    pub fn inverse(&self) -> Self
    where
        T: Copy,
    {
        let r_inv = self.rotation.inverse();
        let inv_s = T::one() / self.scaling;
        // translation of the inverse: -(1/s) Rᵀ t
        let t = r_inv.transform_vector(&self.translation.vector) * inv_s;
        Similarity {
            rotation: r_inv,
            translation: Translation::new(-t),
            scaling: inv_s,
        }
    }

    /// Apply to a point: `s * (R * p) + t`.
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D>
    where
        T: Copy,
    {
        let rotated = self.rotation.transform_vector(&pt.coords);
        Point {
            coords: rotated * self.scaling + self.translation.vector,
        }
    }

    /// Apply to a vector: `s * (R * v)` (translation invariant).
    pub fn transform_vector(&self, vec: &Vector<T, D>) -> Vector<T, D>
    where
        T: Copy,
    {
        self.rotation.transform_vector(vec) * self.scaling
    }
}

/// 3-D similarity.
pub type Similarity3<T> = Similarity<T, 3>;

// ===========================================================================
// Scale
// ===========================================================================

/// A non-uniform (axis-aligned) scaling.
///
/// `self(p) = diag(components) * p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale<T, const D: usize> {
    /// Per-axis scaling factors.
    pub vector: Vector<T, D>,
}

impl<T: Scalar, const D: usize> Scale<T, D> {
    /// Build a scaling from per-axis factors.
    pub fn new(vector: Vector<T, D>) -> Self {
        Scale { vector }
    }

    /// Uniform scaling by `s`.
    pub fn uniform(s: T) -> Self {
        Scale {
            vector: Vector::from_fn(|_| s),
        }
    }

    /// The identity scaling (all factors `1`).
    pub fn identity() -> Self {
        Scale {
            vector: Vector::from_fn(|_| T::one()),
        }
    }

    /// The inverse scaling (componentwise reciprocals).
    pub fn inverse(&self) -> Self {
        Scale {
            vector: Vector::from_fn(|i| T::one() / self.vector.data[i]),
        }
    }

    /// Apply to a point: componentwise multiply.
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        Point {
            coords: Vector::from_fn(|i| self.vector.data[i] * pt.coords.data[i]),
        }
    }

    /// Apply to a vector: componentwise multiply.
    pub fn transform_vector(&self, vec: &Vector<T, D>) -> Vector<T, D> {
        Vector::from_fn(|i| self.vector.data[i] * vec.data[i])
    }
}

// ===========================================================================
// Projections
// ===========================================================================

/// A right-handed perspective projection (view space → clip space), looking
/// down `-z`, with NDC depth in `[-1, 1]` (OpenGL-style).
///
/// Use [`Perspective3::from_fov`] to build from a vertical field-of-view and
/// aspect ratio, or [`Perspective3::from_frustum`] for explicit frustum bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Perspective3<T> {
    matrix: Matrix4<T>,
}

impl<T: Scalar> Perspective3<T> {
    /// Build from vertical field-of-view `fovy`, `aspect` ratio and near/far
    /// clip planes (`znear`, `zfar`).
    pub fn from_fov(aspect: T, fovy: T, znear: T, zfar: T) -> Self {
        let f = T::one() / (fovy / (T::one() + T::one())).tan();
        let m00 = f / aspect;
        let m11 = f;
        let m22 = (zfar + znear) / (znear - zfar);
        let m23 = (T::one() + T::one()) * zfar * znear / (znear - zfar);
        let m32 = -T::one();
        let mut data = [[T::zero(); 4]; 4];
        data[0][0] = m00;
        data[1][1] = m11;
        data[2][2] = m22;
        data[2][3] = m23;
        data[3][2] = m32;
        Perspective3 {
            matrix: Matrix4::new(data),
        }
    }

    /// Build from explicit frustum bounds (right-handed, `-z` forward).
    pub fn from_frustum(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> Self {
        let two = T::one() + T::one();
        let m00 = two * znear / (right - left);
        let m11 = two * znear / (top - bottom);
        let m02 = (right + left) / (right - left);
        let m12 = (top + bottom) / (top - bottom);
        let m22 = (zfar + znear) / (znear - zfar);
        let m23 = two * zfar * znear / (znear - zfar);
        let m32 = -T::one();
        let mut data = [[T::zero(); 4]; 4];
        data[0][0] = m00;
        data[0][2] = m02;
        data[1][1] = m11;
        data[1][2] = m12;
        data[2][2] = m22;
        data[2][3] = m23;
        data[3][2] = m32;
        Perspective3 {
            matrix: Matrix4::new(data),
        }
    }

    /// The 4×4 projection matrix.
    pub fn matrix(&self) -> &Matrix4<T> {
        &self.matrix
    }
}

/// A right-handed orthographic projection (view space → clip space), looking
/// down `-z`, with NDC depth in `[-1, 1]` (OpenGL-style).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orthographic3<T> {
    matrix: Matrix4<T>,
}

impl<T: Scalar> Orthographic3<T> {
    /// Build from the axis-aligned frustum bounds (right-handed, `-z` forward),
    /// mapping `z ∈ [-znear, -zfar]` to NDC `z ∈ [-1, 1]`.
    pub fn from_frustum(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> Self {
        let two = T::one() + T::one();
        let m00 = two / (right - left);
        let m11 = two / (top - bottom);
        let m22 = -two / (zfar - znear);
        let m03 = -(right + left) / (right - left);
        let m13 = -(top + bottom) / (top - bottom);
        let m23 = -(zfar + znear) / (zfar - znear);
        let mut data = [[T::zero(); 4]; 4];
        data[0][0] = m00;
        data[0][3] = m03;
        data[1][1] = m11;
        data[1][3] = m13;
        data[2][2] = m22;
        data[2][3] = m23;
        data[3][3] = T::one();
        Orthographic3 {
            matrix: Matrix4::new(data),
        }
    }

    /// The 4×4 projection matrix.
    pub fn matrix(&self) -> &Matrix4<T> {
        &self.matrix
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_linalg_fixed::{Vector2, Vector3, Vector4};

    // Convenience aliases used by the tests below.
    type Similarity2<T> = Similarity<T, 2>;
    type Scale3<T> = Scale<T, 3>;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn point_vector_arithmetic() {
        let p = Point3::new(Vector3::new([1.0, 2.0, 3.0]));
        let v = Vector3::new([1.0, 0.0, -1.0]);
        let q = p + v;
        assert_eq!(q.coords.data, [2.0, 2.0, 2.0]);
        let w = q - p;
        assert_eq!(w.data, [1.0, 0.0, -1.0]);
        assert!(close(
            p.distance(&Point3::new(Vector3::new([1.0, 2.0, 6.0]))),
            3.0,
            1e-12
        ));
    }

    #[test]
    fn translation_composition_and_inverse() {
        let t1 = Translation::new(Vector2::new([1.0, 0.0]));
        let t2 = Translation::new(Vector2::new([0.0, 2.0]));
        let p = Point2::new(Vector2::new([0.0, 0.0]));
        let q = (t2 * t1).transform_point(&p);
        assert_eq!(q.coords.data, [1.0, 2.0]);
        let back = t1.inverse().transform_point(&q);
        assert_eq!(back.coords.data, [0.0, 2.0]);
    }

    #[test]
    fn rotation2_known_values() {
        let r90 = Rotation2::<f64>::from_angle(core::f64::consts::FRAC_PI_2);
        let v = Vector2::new([1.0, 0.0]);
        let rv = r90.transform_vector(&v);
        assert!(close(rv.data[0], 0.0, 1e-12));
        assert!(close(rv.data[1], 1.0, 1e-12));
        // R * Rᵀ ≈ I
        let id = (*r90.matrix()) * (*r90.inverse().matrix());
        for i in 0..2 {
            for j in 0..2 {
                assert!(close(id.data[i][j], if i == j { 1.0 } else { 0.0 }, 1e-12));
            }
        }
    }

    #[test]
    fn rotation3_axis_known_values() {
        let rz90 = Rotation3::<f64>::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let v = Vector3::new([1.0, 0.0, 0.0]);
        let rv = rz90.transform_vector(&v);
        assert!(close(rv.data[0], 0.0, 1e-12));
        assert!(close(rv.data[1], 1.0, 1e-12));
        assert!(close(rv.data[2], 0.0, 1e-12));

        let rz180 = Rotation3::<f64>::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::PI,
        );
        let rv2 = rz180.transform_vector(&v);
        assert!(close(rv2.data[0], -1.0, 1e-12));
        assert!(close(rv2.data[1], 0.0, 1e-12));
    }

    #[test]
    fn rotation3_from_matrix_roundtrip() {
        let r = Rotation3::<f64>::from_axis_angle(
            &Vector3::new([1.0, 1.0, 1.0]),
            core::f64::consts::FRAC_PI_3,
        );
        let rebuilt = Rotation::from_matrix(*r.matrix()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(rebuilt.matrix().data[i][j], r.matrix().data[i][j], 1e-10),
                    "entry {i},{j}"
                );
            }
        }
    }

    #[test]
    fn quaternion_vector_rotation() {
        let q = UnitQuaternion::<f64>::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let v = Vector3::new([1.0, 0.0, 0.0]);
        let rv = q.rotate_vector(&v);
        assert!(close(rv.data[0], 0.0, 1e-12));
        assert!(close(rv.data[1], 1.0, 1e-12));
        assert!(close(rv.data[2], 0.0, 1e-12));
    }

    #[test]
    fn rotation3_quaternion_roundtrip() {
        // Start from a rotation built via axis-angle ...
        let r = Rotation3::<f64>::from_axis_angle(
            &Vector3::new([1.0, 2.0, 3.0]),
            core::f64::consts::FRAC_PI_4,
        );
        // ... convert to a unit quaternion and back.
        let q = UnitQuaternion::from_rotation_matrix(&r);
        let r2 = q.to_rotation_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(r2.matrix().data[i][j], r.matrix().data[i][j], 1e-10),
                    "entry {i},{j}"
                );
            }
        }
        // And back to a quaternion from the rebuilt rotation.
        let q2 = UnitQuaternion::from_rotation_matrix(&r2);
        let dot = q.quaternion().coords.dot(&q2.quaternion().coords).abs();
        assert!(close(dot, 1.0, 1e-10), "quaternion double-roundtrip");
    }

    #[test]
    fn isometry_composition_inverse_identity() {
        let rot = Rotation3::<f64>::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let iso = Isometry3::new(Translation::new(Vector3::new([1.0, 2.0, 3.0])), rot);
        let inv = iso.inverse();
        // inv * iso should be the identity (within tolerance).
        let composed = inv * iso;
        let id = Isometry3::<f64>::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(
                    composed.rotation.matrix().data[i][j],
                    id.rotation.matrix().data[i][j],
                    1e-10
                ));
            }
        }
        for i in 0..3 {
            assert!(close(composed.translation.vector.data[i], 0.0, 1e-10));
        }

        // transform_point then inverse should recover the original point.
        let p = Point3::new(Vector3::new([3.0, 1.0, 0.5]));
        let q = iso.transform_point(&p);
        let back = inv.transform_point(&q);
        for i in 0..3 {
            assert!(
                close(back.coords.data[i], p.coords.data[i], 1e-9),
                "coord {i}"
            );
        }
    }

    #[test]
    fn similarity_inverse_recovers_point() {
        let rot = Rotation2::<f64>::from_angle(core::f64::consts::FRAC_PI_2);
        let sim = Similarity2::new(2.0, Translation::new(Vector2::new([1.0, 1.0])), rot);
        let p = Point2::new(Vector2::new([1.0, 0.0]));
        let q = sim.transform_point(&p);
        let back = sim.inverse().transform_point(&q);
        assert!(close(back.coords.data[0], 1.0, 1e-10));
        assert!(close(back.coords.data[1], 0.0, 1e-10));
    }

    #[test]
    fn scale_inverse() {
        let s = Scale3::new(Vector3::new([2.0, 3.0, 4.0]));
        let p = Point3::new(Vector3::new([1.0, 1.0, 1.0]));
        let q = s.transform_point(&p);
        assert_eq!(q.coords.data, [2.0, 3.0, 4.0]);
        let back = s.inverse().transform_point(&q);
        assert!(close(back.coords.data[0], 1.0, 1e-12));
    }

    #[test]
    fn perspective_maps_known_depth() {
        let proj = Perspective3::<f64>::from_fov(1.0, core::f64::consts::FRAC_PI_2, 1.0, 10.0);
        // A point at z = -znear in view space maps to NDC z = -1 (w = znear).
        let view = Vector4::new([0.0, 0.0, -1.0, 1.0]);
        let clip = *proj.matrix() * view;
        assert!(close(clip.data[3], 1.0, 1e-12));
        assert!(close(clip.data[2] / clip.data[3], -1.0, 1e-12));
        // A point at z = -zfar maps to NDC z = +1.
        let view2 = Vector4::new([0.0, 0.0, -10.0, 1.0]);
        let clip2 = *proj.matrix() * view2;
        assert!(close(clip2.data[2] / clip2.data[3], 1.0, 1e-12));
    }

    #[test]
    fn orthographic_maps_known_depth() {
        let proj = Orthographic3::<f64>::from_frustum(-1.0, 1.0, -1.0, 1.0, 1.0, 10.0);
        let view = Vector4::new([1.0, 1.0, -1.0, 1.0]);
        let clip = *proj.matrix() * view;
        assert!(close(clip.data[0], 1.0, 1e-12));
        assert!(close(clip.data[1], 1.0, 1e-12));
        assert!(close(clip.data[2], -1.0, 1e-12));
        let view2 = Vector4::new([0.0, 0.0, -10.0, 1.0]);
        let clip2 = *proj.matrix() * view2;
        assert!(close(clip2.data[2], 1.0, 1e-12));
    }

    #[test]
    fn slerp_half_way() {
        let id = UnitQuaternion::<f64>::from_axis_angle(&Vector3::new([0.0, 0.0, 1.0]), 0.0);
        let q90 = UnitQuaternion::<f64>::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let mid = Quaternion::slerp(&id, &q90, 0.5);
        // Halfway should equal the 45° rotation.
        let expected = UnitQuaternion::<f64>::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_4,
        );
        let dot = mid
            .quaternion()
            .coords
            .dot(&expected.quaternion().coords)
            .abs();
        assert!(close(dot, 1.0, 1e-10), "slerp midpoint");
    }
}
