#![no_std]
//! Scalar numeric trait glue for the `tpt-math` substrate.
//!
//! This crate is a thin wrapper over [`num_traits`] (and, when the `libm`
//! feature is enabled, [`libm`]) that re-exports the numeric traits the rest
//! of `tpt-math` builds on, and provides a single [`Scalar`] supertrait that
//! captures "a floating-point scalar type we can do math on".
//!
//! # Features
//!
//! * `std` (default) — enable [`num_traits`]' `std`-dependent items.
//! * `alloc` — signal that an allocator is available (no extra deps).
//! * `libm` — re-export [`libm`] so callers can do transcendental math in
//!   `no_std` environments.
//!
//! [`num_traits`]: num_traits
//! [`libm`]: libm

pub use num_traits::{
    cast,
    float::FloatConst,
    real::Real,
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedRem, CheckedShl, CheckedShr,
    CheckedSub, Float, FromPrimitive, Inv, Num, NumAssign, NumAssignOps, NumCast, NumOps,
    One, Pow, PrimInt, Signed, ToPrimitive, Unsigned, Zero,
};

pub use num_traits;

#[cfg(feature = "libm")]
pub use libm;

/// A floating-point scalar type usable throughout `tpt-math`.
///
/// This is a convenience supertrait: anything that is a [`Float`], can be
/// constructed from/cast to primitive floats via [`NumCast`], and carries the
/// mathematical constants from [`FloatConst`] satisfies it.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::Scalar;
///
/// fn midpoint<T: Scalar>(a: T, b: T) -> T {
///     (a + b) / (T::one() + T::one())
/// }
///
/// assert_eq!(midpoint(2.0_f64, 4.0_f64), 3.0_f64);
/// ```
pub trait Scalar: Float + NumCast + FloatConst {}
impl<T: Float + NumCast + FloatConst> Scalar for T {}

/// Re-export of the [`Num`]/`Float` numeric hierarchy root for convenience.
pub mod prelude {
    pub use crate::{Float, FloatConst, Num, NumCast, One, Real, Scalar, Signed, Unsigned, Zero};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_is_implemented_for_f64() {
        fn takes_scalar<T: Scalar>(_: T) {}
        takes_scalar(1.0_f64);
        takes_scalar(1.0_f32);
    }

    #[test]
    fn constants_are_available() {
        assert!((f64::E() - core::f64::consts::E).abs() < 1e-12);
        assert!((f64::PI() - core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn cast_roundtrip() {
        let x: f64 = 3.5;
        let y: i64 = NumCast::from(x).unwrap();
        assert_eq!(y, 3);
    }
}
