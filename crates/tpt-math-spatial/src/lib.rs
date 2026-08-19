#![no_std]
#![forbid(unsafe_code)]
// Fixed-size 6-D spatial algebra is clearest with explicit indexing over
// fixed-size arrays; the indexed-loop and arithmetic-impl lints do not fit
// this code.
#![allow(
    clippy::needless_range_loop,
    clippy::manual_swap,
    clippy::suspicious_arithmetic_impl,
    clippy::suspicious_op_assign_impl
)]
//! In-house (no external-crate) spatial kinematics for the `tpt-math`
//! substrate: Featherstone 6-D spatial vectors, Plücker / adjoint transforms,
//! dual quaternions and screw theory (exponential / logarithmic maps).
//!
//! Everything is built on [`tpt_math_geometry`] ([`Isometry3`],
//! `Quaternion`, `UnitQuaternion`, …) and [`tpt_math_linalg_fixed`], is
//! `#![no_std]` and allocator-free (stack-allocated fixed-size storage), and
//! carries **no `nalgebra` dependency** (matching the workspace license policy,
//! ADR-0007). It exists to fill the spatial-algebra gap in-house.
//!
//! # Conventions (stated explicitly)
//!
//! * **Active (alibi) rotations** — a transform acts on a vector `v` as
//!   `R * v`; the *point/vector* is rotated, not the coordinate frame.
//! * **Column vectors** — transformations apply as `M * v`.
//! * **Right-handed coordinates** — positive angle about `+z` rotates `x → y`;
//!   `x × y = z`.
//! * **Spatial (6-D) vectors** are stored `[angular (top 3); linear (bottom
//!   3)]`: a motion vector `v = (ω, v)` and a force/wrench
//!   `f = (n, f)` (moment, force).
//! * **Spatial cross product** follows Featherstone's `crm` / `crf`
//!   operators (see [`SpatialVector::cross`] and friends); the result kind is
//!   determined by the operand kinds (motion⊗motion→motion, motion⊗force→force,
//!   force⊗motion→force, force⊗force→motion).
//! * **Screw exponential** `exp` maps a twist `(ω, v) ∈ se(3)` to an
//!   [`Isometry3`]; `log` is its inverse. The convention is the standard
//!   Lynch–Park `se(3)` exponential with an **active**, right-handed rotation.
//! * **Dual quaternions** are Hamilton quaternions `q + ε·qₑ` representing a
//!   rigid transform (rotation quaternion + half the translation quaternion
//!   pre-multiplied by the rotation).
//!
//! # Examples
//!
//! ```
//! use tpt_math_spatial::{MotionVector, Screw};
//! use tpt_math_geometry::{Isometry3, Rotation3, Translation};
//! use tpt_math_linalg_fixed::Vector3;
//!
//! // A 90° active rotation about +Z.
//! let rot = Rotation3::from_axis_angle(&Vector3::new([0.0_f64, 0.0, 1.0]), std::f64::consts::FRAC_PI_2);
//! let iso = Isometry3::new(Translation::new(Vector3::new([0.0, 0.0, 0.0])), rot);
//!
//! // A pure-linear motion (1,0,0): after the rotation its linear part is (0,1,0).
//! let m = MotionVector::new(Vector3::new([0.0, 0.0, 0.0]), Vector3::new([1.0, 0.0, 0.0]));
//! let m2 = m.transform_by(&iso);
//! assert!((m2.linear().x()).abs() < 1e-12);
//! assert!((m2.linear().y() - 1.0).abs() < 1e-12);
//!
//! // A screw twist exponentiates to the same isometry.
//! let screw = Screw::new(Vector3::new([0.0, 0.0, std::f64::consts::FRAC_PI_2]), Vector3::new([0.0, 0.0, 0.0]));
//! let iso2 = screw.exp();
//! assert!((iso2.rotation.matrix().data[0][1] - (-1.0)).abs() < 1e-12);
//! ```

extern crate alloc;

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

use tpt_math_geometry::{Isometry3, Quaternion, Rotation3, Translation, UnitQuaternion};
use tpt_math_linalg_fixed::{Matrix, Matrix3, Vector3, Vector6};
use tpt_math_numeric::Scalar;

/// A `6 × 6` matrix (fixed-size, row-major).
pub type Matrix6<T> = Matrix<T, 6, 6>;

#[inline]
fn c<T: Scalar>(x: f64) -> T {
    T::from(x).unwrap()
}

#[inline]
fn two<T: Scalar>() -> T {
    T::one() + T::one()
}

/// Skew-symmetric (`×`) matrix of a 3-vector.
#[inline]
fn skew<T: Scalar + Copy>(v: &Vector3<T>) -> Matrix3<T> {
    Matrix3::new([
        [T::zero(), -v.z(), v.y()],
        [v.z(), T::zero(), -v.x()],
        [-v.y(), v.x(), T::zero()],
    ])
}

// ===========================================================================
// Spatial (6-D) vectors
// ===========================================================================

/// Marker type for **motion**-type spatial vectors (velocities, accelerations).
pub struct Motion;
/// Marker type for **force**-type spatial vectors (wrenches: moment + force).
pub struct Force;

/// A 6-D spatial vector `v = [angular (top 3); linear (bottom 3)]`, tagged by
/// `K ∈ {`[`Motion`], [`Force`]`}` to control cross-product and transform
/// semantics (Featherstone's `crm` / `crf` operators).
///
/// * A [`MotionVector`] is `(ω, v)` (angular velocity, linear velocity).
/// * A [`ForceVector`] is `(n, f)` (moment, force).
///
/// See the [crate-level documentation](crate) for the full convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialVector<T, K> {
    /// The raw 6-components `[angular₀, angular₁, angular₂, linear₀, linear₁,
    /// linear₂]`.
    pub data: Vector6<T>,
    _marker: PhantomData<K>,
}

/// A **motion** spatial vector `(ω, v)`.
pub type MotionVector<T> = SpatialVector<T, Motion>;
/// A **force** / wrench spatial vector `(n, f)`.
pub type ForceVector<T> = SpatialVector<T, Force>;

impl<T: Scalar + Copy> SpatialVector<T, Motion> {
    /// Build a motion vector from its angular (top) and linear (bottom) halves.
    pub fn new(angular: Vector3<T>, linear: Vector3<T>) -> Self {
        SpatialVector {
            data: Vector6::new([
                angular.x(),
                angular.y(),
                angular.z(),
                linear.x(),
                linear.y(),
                linear.z(),
            ]),
            _marker: PhantomData,
        }
    }

    /// `motion ⊗ motion → motion` (Featherstone `crm`).
    pub fn cross(&self, rhs: &SpatialVector<T, Motion>) -> SpatialVector<T, Motion> {
        // (mω × nω, mω × nv + mv × nω)
        let mw = self.angular();
        let mv = self.linear();
        let nw = rhs.angular();
        let nv = rhs.linear();
        let top = mw.cross(&nw);
        let bot = mw.cross(&nv) + mv.cross(&nw);
        Self::new(top, bot)
    }

    /// `motion ⊗ force → force` (Featherstone `crm` applied to a force).
    pub fn cross_force(&self, rhs: &SpatialVector<T, Force>) -> SpatialVector<T, Force> {
        // (mω × fω + mv × fv, mω × fv)
        let mw = self.angular();
        let mv = self.linear();
        let fw = rhs.angular();
        let fv = rhs.linear();
        let top = mw.cross(&fw) + mv.cross(&fv);
        let bot = mw.cross(&fv);
        ForceVector::new(top, bot)
    }

    /// Transform this motion vector by an [`Isometry3`] using the motion
    /// Plücker adjoint `[R, 0; t×R, R]`.
    ///
    /// For a pure rotation this reduces to `R` acting on both halves, matching
    /// [`Isometry3::transform_vector`] componentwise.
    pub fn transform_by(&self, iso: &Isometry3<T>) -> Self {
        SpatialVector {
            data: adjoint_motion(iso) * self.data,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Copy> SpatialVector<T, Force> {
    /// Build a force vector from its angular (moment, top) and linear (force,
    /// bottom) halves.
    pub fn new(angular: Vector3<T>, linear: Vector3<T>) -> Self {
        SpatialVector {
            data: Vector6::new([
                angular.x(),
                angular.y(),
                angular.z(),
                linear.x(),
                linear.y(),
                linear.z(),
            ]),
            _marker: PhantomData,
        }
    }

    /// `force ⊗ motion → force` (Featherstone `crf`).
    pub fn cross_motion(&self, rhs: &SpatialVector<T, Motion>) -> SpatialVector<T, Force> {
        // (fω × mω, fω × mv + fv × mω)
        let fw = self.angular();
        let fv = self.linear();
        let mw = rhs.angular();
        let mv = rhs.linear();
        let top = fw.cross(&mw);
        let bot = fw.cross(&mv) + fv.cross(&mw);
        Self::new(top, bot)
    }

    /// `force ⊗ force → motion` (Featherstone `crf`).
    pub fn cross(&self, rhs: &SpatialVector<T, Force>) -> SpatialVector<T, Motion> {
        // (fω × gv + fv × gω, fv × gω)
        let fw = self.angular();
        let fv = self.linear();
        let gw = rhs.angular();
        let gv = rhs.linear();
        let top = fw.cross(&gv) + fv.cross(&gw);
        let bot = fv.cross(&gw);
        MotionVector::new(top, bot)
    }

    /// Transform this force vector by an [`Isometry3`] using the force
    /// Plücker adjoint `[R, t×R; 0, R]`.
    ///
    /// For a pure rotation this reduces to `R` acting on both halves, matching
    /// [`Isometry3::transform_vector`] componentwise.
    pub fn transform_by(&self, iso: &Isometry3<T>) -> Self {
        SpatialVector {
            data: adjoint_force(iso) * self.data,
            _marker: PhantomData,
        }
    }
}

// Shared accessors / transforms, generic over the kind marker.
impl<T: Scalar + Copy, K> SpatialVector<T, K> {
    /// The angular half (top 3 components).
    pub fn angular(&self) -> Vector3<T> {
        Vector3::new([self.data.data[0], self.data.data[1], self.data.data[2]])
    }

    /// The linear half (bottom 3 components).
    pub fn linear(&self) -> Vector3<T> {
        Vector3::new([self.data.data[3], self.data.data[4], self.data.data[5]])
    }

    /// Borrow the raw 6-vector.
    pub fn as_vector(&self) -> &Vector6<T> {
        &self.data
    }
}

impl<T: Scalar + Copy, K> Add for SpatialVector<T, K> {
    type Output = Self;
    /// # Panics
    ///
    /// Inherits `T`'s `+` semantics (never panics for floats).
    fn add(self, rhs: Self) -> Self {
        SpatialVector {
            data: self.data + rhs.data,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Copy, K> Sub for SpatialVector<T, K> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        SpatialVector {
            data: self.data - rhs.data,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Copy, K> Neg for SpatialVector<T, K> {
    type Output = Self;
    fn neg(self) -> Self {
        SpatialVector {
            data: -self.data,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Copy, K> Mul<T> for SpatialVector<T, K> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        SpatialVector {
            data: self.data * rhs,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Copy, K> fmt::Display for SpatialVector<T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[ω: {:?}; v: {:?}]",
            self.angular().data,
            self.linear().data
        )
    }
}

// ===========================================================================
// Adjoint (Plücker) transforms
// ===========================================================================

/// The **motion** adjoint `Ad_X` of an isometry `(R, t)`:
///
/// ```text
/// Ad_X = [ R      0   ]
///        [ t×R    R   ]
/// ```
///
/// Acting on a [`MotionVector`] `(ω, v)` it yields
/// `(R ω, R v + t × (R ω))`.
pub fn adjoint_motion<T: Scalar + Copy>(iso: &Isometry3<T>) -> Matrix6<T> {
    let r = *iso.rotation.matrix();
    let t = iso.translation.vector;
    let tl = skew(&t) * r;
    Matrix6::from_fn(|i, j| {
        if i < 3 && j < 3 {
            r.data[i][j]
        } else if i >= 3 && j < 3 {
            tl.data[i - 3][j]
        } else if i >= 3 && j >= 3 {
            r.data[i - 3][j - 3]
        } else {
            T::zero()
        }
    })
}

/// The **force** adjoint `Ad_X*` of an isometry `(R, t)`:
///
/// ```text
/// Ad_X* = [ R      t×R ]
///         [ 0      R   ]
/// ```
///
/// Acting on a [`ForceVector`] `(n, f)` it yields
/// `(R n + t × (R f), R f)`.
pub fn adjoint_force<T: Scalar + Copy>(iso: &Isometry3<T>) -> Matrix6<T> {
    let r = *iso.rotation.matrix();
    let t = iso.translation.vector;
    let tr = skew(&t) * r;
    Matrix6::from_fn(|i, j| {
        if i < 3 && j < 3 {
            r.data[i][j]
        } else if i < 3 && j >= 3 {
            tr.data[i][j - 3]
        } else if i >= 3 && j >= 3 {
            r.data[i - 3][j - 3]
        } else {
            T::zero()
        }
    })
}

// ===========================================================================
// Dual quaternions
// ===========================================================================

/// A dual quaternion `q = real + ε·dual`, each part a Hamilton
/// [`Quaternion`] (`w` last). A *unit* dual quaternion represents a rigid
/// transform (rotation + translation) with no scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualQuaternion<T> {
    /// The real (rotation) part.
    pub real: Quaternion<T>,
    /// The dual (translation) part.
    pub dual: Quaternion<T>,
}

impl<T: Scalar> DualQuaternion<T> {
    /// Build a dual quaternion from its real and dual quaternion parts.
    pub fn new(real: Quaternion<T>, dual: Quaternion<T>) -> Self {
        DualQuaternion { real, dual }
    }

    /// A dual quaternion with a zero dual part (a pure rotation).
    pub fn from_real(real: Quaternion<T>) -> Self {
        DualQuaternion {
            real,
            dual: Quaternion::new(T::zero(), T::zero(), T::zero(), T::zero()),
        }
    }

    /// The (Hamilton) conjugate `(real*, dual*)`.
    pub fn conjugate(&self) -> Self {
        DualQuaternion {
            real: self.real.conjugate(),
            dual: self.dual.conjugate(),
        }
    }

    /// Normalise to a unit dual quaternion by dividing both parts by the norm
    /// of the real part. Returns `None` if the real part has zero norm.
    pub fn normalize(&self) -> Option<Self> {
        let n = self.real.norm();
        if n == T::zero() {
            return None;
        }
        let inv = T::one() / n;
        Some(DualQuaternion {
            real: self.real * inv,
            dual: self.dual * inv,
        })
    }

    /// Hamilton product of two dual quaternions
    /// `(r₁, d₁) * (r₂, d₂) = (r₁ r₂, r₁ d₂ + d₁ r₂)`.
    pub fn multiply(&self, rhs: &DualQuaternion<T>) -> DualQuaternion<T> {
        DualQuaternion {
            real: self.real.multiply(&rhs.real),
            dual: self.real.multiply(&rhs.dual) + self.dual.multiply(&rhs.real),
        }
    }

    /// Build a unit dual quaternion from a rigid transform.
    ///
    /// Given an isometry with rotation quaternion `q` and translation `t`, the
    /// dual quaternion is `q + ε·(½·t_q·q)` where `t_q = (t_x, t_y, t_z, 0)`.
    pub fn from_isometry(iso: &Isometry3<T>) -> Self {
        let q = UnitQuaternion::<T>::from_rotation_matrix(&iso.rotation).quaternion().clone();
        let t = iso.translation.vector;
        let tq = Quaternion::new(t.x(), t.y(), t.z(), T::zero());
        let dual = tq.multiply(&q) * (T::one() / (T::one() + T::one()));
        DualQuaternion { real: q, dual }
    }

    /// Recover the rigid transform represented by this (unit) dual quaternion.
    ///
    /// The rotation is the real part; the translation is recovered from
    /// `2·dual·real*` (taking the vector part).
    pub fn to_isometry(&self) -> Isometry3<T> {
        let real = self.real.normalize().unwrap_or(self.real);
        let rotation = Rotation3::from_quaternion(&UnitQuaternion::from_quaternion(real));
        let tq = self.dual.multiply(&real.conjugate()) * two();
        let translation = Translation::new(Vector3::new([
            tq.coords.data[0],
            tq.coords.data[1],
            tq.coords.data[2],
        ]));
        Isometry3::new(translation, rotation)
    }
}

// ===========================================================================
// Screw theory
// ===========================================================================

/// A screw (twist) in `se(3)`, stored as the 6-D spatial motion vector
/// `(ω, v)` where `ω` is the angular part (rotation axis × angle) and `v` is
/// the linear part.
///
/// The [`Screw::exp`] map sends this twist to an [`Isometry3`] via the standard
/// `se(3)` matrix exponential (Lynch–Park convention, active right-handed
/// rotation), and [`Screw::log`] is its inverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Screw<T> {
    /// The angular part (axis × angle).
    pub angular: Vector3<T>,
    /// The linear part.
    pub linear: Vector3<T>,
}

impl<T: Scalar + Copy> Screw<T> {
    /// Build a screw from its angular and linear 3-vectors.
    pub fn new(angular: Vector3<T>, linear: Vector3<T>) -> Self {
        Screw { angular, linear }
    }

    /// View the screw as a [`MotionVector`].
    pub fn as_motion(&self) -> MotionVector<T> {
        MotionVector::new(self.angular, self.linear)
    }

    /// The exponential map `exp: se(3) → SE(3)`.
    ///
    /// For `ω = 0` this is a pure translation by `v`; otherwise it is a rotation
    /// by `|ω|` about `ω/|ω|` with the Lynch–Park transported translation.
    pub fn exp(&self) -> Isometry3<T> {
        let w = self.angular;
        let v = self.linear;
        let theta = w.norm();
        let eps = c::<T>(1e-12);
        let rotation = if theta < eps {
            Rotation3::identity()
        } else {
            let axis = w / theta;
            Rotation3::from_axis_angle(&axis, theta)
        };
        let p = if theta < eps {
            v
        } else {
            let axis = w / theta;
            let vl = v / theta;
            let av = axis.cross(&vl); // A·vl
            let aav = axis.cross(&av); // A²·vl
            let (s, cth) = theta.sin_cos();
            let term1 = vl * theta;
            let term2 = av * (T::one() - cth);
            let term3 = aav * (theta - s);
            term1 + term2 + term3
        };
        Isometry3::new(Translation::new(p), rotation)
    }

    /// The logarithmic map `log: SE(3) → se(3)` — the inverse of [`Screw::exp`].
    ///
    /// Recovers the screw `(ω, v)` whose exponential equals `iso`. For a pure
    /// translation this is `(0, translation)`.
    pub fn log(iso: &Isometry3<T>) -> Screw<T> {
        let m = iso.rotation.matrix().data;
        let tr = m[0][0] + m[1][1] + m[2][2];
        let cos_t = (tr - T::one()) / two();
        let theta = cos_t.acos();
        let eps = c::<T>(1e-12);
        if theta < eps {
            return Screw::new(
                Vector3::new([T::zero(), T::zero(), T::zero()]),
                iso.translation.vector,
            );
        }
        let two_ = two::<T>();
        let (s, cth) = theta.sin_cos();
        let axis = Vector3::new([
            (m[2][1] - m[1][2]) / (two_ * s),
            (m[0][2] - m[2][0]) / (two_ * s),
            (m[1][0] - m[0][1]) / (two_ * s),
        ]);
        let p = iso.translation.vector;
        let ap = axis.cross(&p); // A·p
        let aap = axis.cross(&ap); // A²·p
        let coef = (T::one() / theta) - (T::one() + cth) / (two_ * s);
        let vl = (p / theta) - (ap * (T::one() / (T::one() + T::one()))) + (aap * coef);
        let angular = axis * theta;
        let linear = vl * theta;
        Screw::new(angular, linear)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_geometry::Point3;

    type MVec = MotionVector<f64>;
    type FVec = ForceVector<f64>;

    #[inline]
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn iso_z90() -> Isometry3<f64> {
        let rot = Rotation3::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        Isometry3::new(Translation::new(Vector3::new([0.0, 0.0, 0.0])), rot)
    }

    #[test]
    fn spatial_vector_accessors_and_arithmetic() {
        let a = MVec::new(
            Vector3::new([1.0, 2.0, 3.0]),
            Vector3::new([4.0, 5.0, 6.0]),
        );
        assert!(close(a.angular().x(), 1.0));
        assert!(close(a.linear().z(), 6.0));
        let b = MVec::new(
            Vector3::new([0.0, 0.0, 0.0]),
            Vector3::new([1.0, 1.0, 1.0]),
        );
        let s = a + b;
        assert!(close(s.linear().x(), 5.0));
        let n = -a;
        assert!(close(n.angular().x(), -1.0));
    }

    #[test]
    fn cross_motion_motion_known() {
        // m = (ω=(0,0,1), v=0), n = (ω=0, v=(1,0,0)).
        // m × n = (0, 0×0 + (0,0,1)×(1,0,0)) = (0, (0,1,0)).
        let m = MVec::new(Vector3::new([0.0, 0.0, 1.0]), Vector3::new([0.0, 0.0, 0.0]));
        let n = MVec::new(Vector3::new([0.0, 0.0, 0.0]), Vector3::new([1.0, 0.0, 0.0]));
        let r = m.cross(&n);
        assert!(close(r.angular().norm(), 0.0));
        assert!(close(r.linear().x(), 0.0));
        assert!(close(r.linear().y(), 1.0));
        assert!(close(r.linear().z(), 0.0));
    }

    #[test]
    fn cross_motion_force_known() {
        // m = (ω=(0,0,1), v=0), f = (n=0, f=(1,0,0)).
        // m × f = (0×0 + 0×(1,0,0), 0×(1,0,0)) = (0, 0).
        let m = MVec::new(Vector3::new([0.0, 0.0, 1.0]), Vector3::new([0.0, 0.0, 0.0]));
        let f = FVec::new(Vector3::new([0.0, 0.0, 0.0]), Vector3::new([1.0, 0.0, 0.0]));
        let r = m.cross_force(&f);
        assert!(close(r.angular().norm(), 0.0));
        assert!(close(r.linear().norm(), 0.0));
    }

    #[test]
    fn cross_force_force_known() {
        // f = (n=(0,0,1), f=0), g = (n=0, f=(1,0,0)).
        // f × g = ((0,0,1)×(1,0,0) + 0, 0) = ((0,1,0), 0) → motion.
        let f = FVec::new(Vector3::new([0.0, 0.0, 1.0]), Vector3::new([0.0, 0.0, 0.0]));
        let g = FVec::new(Vector3::new([0.0, 0.0, 0.0]), Vector3::new([1.0, 0.0, 0.0]));
        let r = f.cross(&g);
        assert!(close(r.angular().x(), 0.0));
        assert!(close(r.angular().y(), 1.0));
        assert!(close(r.angular().z(), 0.0));
        assert!(close(r.linear().norm(), 0.0));
    }

    #[test]
    fn transform_by_rotation_matches_isometry_vector() {
        let iso = iso_z90();
        // Pure-linear motion (1,0,0) → linear part rotates to (0,1,0), angular to 0.
        let m = MVec::new(
            Vector3::new([0.0, 0.0, 0.0]),
            Vector3::new([1.0, 0.0, 0.0]),
        );
        let mt = m.transform_by(&iso);
        assert!(close(mt.angular().norm(), 0.0));
        assert!(close(mt.linear().x(), 0.0));
        assert!(close(mt.linear().y(), 1.0));

        // Angular part should transform like a vector under R.
        let m2 = MVec::new(
            Vector3::new([1.0, 0.0, 0.0]),
            Vector3::new([0.0, 0.0, 0.0]),
        );
        let m2t = m2.transform_by(&iso);
        // R applied to (1,0,0) about z by 90° → (0,1,0).
        let expected = iso.rotation.transform_vector(&Vector3::new([1.0, 0.0, 0.0]));
        assert!(close(m2t.angular().x(), expected.data[0]));
        assert!(close(m2t.angular().y(), expected.data[1]));
        assert!(close(m2t.angular().z(), expected.data[2]));
    }

    #[test]
    fn adjoint_matches_transform() {
        let iso = iso_z90();
        let ad = adjoint_motion(&iso);
        let m = MVec::new(
            Vector3::new([1.0, 2.0, 3.0]),
            Vector3::new([4.0, 5.0, 6.0]),
        );
        let via_matrix = MVec {
            data: ad * m.data,
            _marker: PhantomData,
        };
        let via_fn = m.transform_by(&iso);
        for i in 0..6 {
            assert!(close(via_matrix.data.data[i], via_fn.data.data[i]), "idx {i}");
        }
    }

    #[test]
    fn transform_by_with_translation() {
        // Rotation 90° about Z plus translation (10, 0, 0).
        let rot = Rotation3::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let iso = Isometry3::new(Translation::new(Vector3::new([10.0, 0.0, 0.0])), rot);
        // A force at the origin with linear force (1,0,0) and zero moment.
        // Force adjoint: fω' = R fω + t × (R fv), fv' = R fv.
        let f = FVec::new(
            Vector3::new([0.0, 0.0, 0.0]),
            Vector3::new([1.0, 0.0, 0.0]),
        );
        let ft = f.transform_by(&iso);
        let rfv = rot.transform_vector(&Vector3::new([1.0, 0.0, 0.0])); // (0,1,0)
        let expected_top = Vector3::new([10.0, 0.0, 0.0]).cross(&rfv); // t × (R fv)
        assert!(close(ft.angular().x(), expected_top.data[0]));
        assert!(close(ft.angular().y(), expected_top.data[1]));
        assert!(close(ft.angular().z(), expected_top.data[2]));
        assert!(close(ft.linear().x(), 0.0));
        assert!(close(ft.linear().y(), 1.0));
    }

    #[test]
    fn dual_quaternion_roundtrip() {
        let rot = Rotation3::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let iso = Isometry3::new(Translation::new(Vector3::new([1.0, 2.0, 3.0])), rot);
        let dq = DualQuaternion::from_isometry(&iso);
        let back = dq.to_isometry();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(back.rotation.matrix().data[i][j], iso.rotation.matrix().data[i][j]),
                    "rot {i},{j}"
                );
            }
            assert!(close(
                back.translation.vector.data[i],
                iso.translation.vector.data[i]
            ), "trans {i}");
        }
    }

    #[test]
    fn dual_quaternion_composition() {
        let r1 = Rotation3::from_axis_angle(
            &Vector3::new([0.0, 0.0, 1.0]),
            core::f64::consts::FRAC_PI_2,
        );
        let iso1 = Isometry3::new(Translation::new(Vector3::new([1.0, 0.0, 0.0])), r1);
        let r2 = Rotation3::from_axis_angle(
            &Vector3::new([0.0, 1.0, 0.0]),
            core::f64::consts::FRAC_PI_3,
        );
        let iso2 = Isometry3::new(Translation::new(Vector3::new([0.0, 2.0, 0.0])), r2);

        let dq1 = DualQuaternion::from_isometry(&iso1);
        let dq2 = DualQuaternion::from_isometry(&iso2);
        let composed_dq = dq1.multiply(&dq2);
        let composed_iso = iso1 * iso2;
        let from_dq = composed_dq.to_isometry();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(
                        from_dq.rotation.matrix().data[i][j],
                        composed_iso.rotation.matrix().data[i][j]
                    ),
                    "rot {i},{j}"
                );
            }
            assert!(close(
                from_dq.translation.vector.data[i],
                composed_iso.translation.vector.data[i]
            ), "trans {i}");
        }
    }

    #[test]
    fn screw_exp_log_roundtrip() {
        // Random-ish isometry.
        let rot = Rotation3::from_axis_angle(
            &Vector3::new([1.0, 2.0, 3.0]).normalize(),
            core::f64::consts::FRAC_PI_4,
        );
        let iso = Isometry3::new(Translation::new(Vector3::new([0.5, -1.0, 2.0])), rot);
        let screw = Screw::log(&iso);
        let iso2 = screw.exp();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(iso2.rotation.matrix().data[i][j], iso.rotation.matrix().data[i][j]),
                    "rot {i},{j}"
                );
            }
            assert!(close(
                iso2.translation.vector.data[i],
                iso.translation.vector.data[i]
            ), "trans {i}");
        }
    }

    #[test]
    fn screw_pure_translation() {
        let iso = Isometry3::new(
            Translation::new(Vector3::new([3.0, -2.0, 1.0])),
            Rotation3::identity(),
        );
        let screw = Screw::log(&iso);
        assert!(close(screw.angular.norm(), 0.0));
        for i in 0..3 {
            assert!(close(screw.linear.data[i], iso.translation.vector.data[i]), "lin {i}");
        }
        let iso2 = screw.exp();
        for i in 0..3 {
            assert!(close(
                iso2.translation.vector.data[i],
                iso.translation.vector.data[i]
            ), "trans {i}");
        }
    }

    #[test]
    fn screw_exp_log_point_recovery() {
        // exp(log(iso)) must act identically to iso on an arbitrary point.
        let rot = Rotation3::from_axis_angle(
            &Vector3::new([0.3, 1.0, -0.7]).normalize(),
            core::f64::consts::FRAC_PI_3,
        );
        let iso = Isometry3::new(Translation::new(Vector3::new([2.0, 1.0, -3.0])), rot);
        let screw = Screw::log(&iso);
        let iso2 = screw.exp();
        let p = Point3::new(Vector3::new([1.0, -2.0, 0.5]));
        let q1 = iso.transform_point(&p);
        let q2 = iso2.transform_point(&p);
        for i in 0..3 {
            assert!(close(q1.coords.data[i], q2.coords.data[i]), "pt {i}");
        }
    }

    #[test]
    fn screw_composition_identity() {
        // exp(a) * exp(b) exponentiates (as an isometry product) to a transform
        // whose log exponentiates back to itself.
        let a = Screw::new(
            Vector3::new([0.0, 0.0, 0.1]),
            Vector3::new([0.2, 0.0, 0.0]),
        );
        let b = Screw::new(
            Vector3::new([0.1, 0.0, 0.0]),
            Vector3::new([0.0, 0.3, 0.0]),
        );
        let iso_a = a.exp();
        let iso_b = b.exp();
        let iso_ab = iso_a * iso_b;
        let screw_ab = Screw::log(&iso_ab);
        let iso_ab2 = screw_ab.exp();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(
                        iso_ab2.rotation.matrix().data[i][j],
                        iso_ab.rotation.matrix().data[i][j]
                    ),
                    "rot {i},{j}"
                );
            }
            assert!(close(
                iso_ab2.translation.vector.data[i],
                iso_ab.translation.vector.data[i]
            ), "trans {i}");
        }
    }

    #[test]
    fn screw_small_twist_bch_approx() {
        // For small twists, exp(a) * exp(b) ≈ exp(a + b).
        let a = Screw::new(
            Vector3::new([0.01, 0.0, 0.0]),
            Vector3::new([0.0, 0.02, 0.0]),
        );
        let b = Screw::new(
            Vector3::new([0.0, 0.01, 0.0]),
            Vector3::new([0.0, 0.0, 0.01]),
        );
        let sum = Screw::new(a.angular + b.angular, a.linear + b.linear);
        let iso_ab = a.exp() * b.exp();
        let iso_sum = sum.exp();
        for i in 0..3 {
            assert!(close(
                iso_ab.translation.vector.data[i],
                iso_sum.translation.vector.data[i]
            ), "trans {i}");
            for j in 0..3 {
                assert!(
                    close(
                        iso_ab.rotation.matrix().data[i][j],
                        iso_sum.rotation.matrix().data[i][j]
                    ),
                    "rot {i},{j}"
                );
            }
        }
    }
}
