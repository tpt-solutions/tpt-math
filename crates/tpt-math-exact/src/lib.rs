#![no_std]
//! Arbitrary-precision rational arithmetic and interval arithmetic.
//!
//! This crate thin-wraps [`num_bigint`] and [`num_rational`] to expose exact
//! rational numbers ([`BigRational`]), and layers an [`Interval`] type on top
//! for rigorous (rounding-error-free) bounds. Because the bounds are exact
//! rationals, interval arithmetic here never accumulates floating-point error.
//!
//! # Features
//!
//! * `std` (default) — enable the `std` feature of the wrapped crates.
//! * `alloc` — signal an allocator is available (`no_std + alloc`).
//!
//! [`num_bigint`]: num_bigint
//! [`num_rational`]: num_rational

extern crate alloc;

pub use num_bigint::{BigInt, BigUint};
pub use num_rational::BigRational;
pub use tpt_math_numeric as numeric;
use tpt_math_numeric::FromPrimitive;

/// Exact rational number type alias.
pub type Rational = BigRational;

/// A closed interval `[lo, hi]` over an ordered, arithmetic type.
///
/// The endpoints are inclusive. Intervals support the usual arithmetic, where
/// each operation computes the exact enclosure of the result set:
///
/// ```
/// use tpt_math_exact::{Interval, Rational};
/// use num_rational::BigRational;
/// use num_traits::One;
///
/// let a = Interval::new(Rational::new(1.into(), 1.into()), Rational::new(2.into(), 1.into()));
/// let b = Interval::new(Rational::new(3.into(), 1.into()), Rational::new(4.into(), 1.into()));
/// let c = &a + &b;
/// assert_eq!(c.lo(), &Rational::new(4.into(), 1.into()));
/// assert_eq!(c.hi(), &Rational::new(6.into(), 1.into()));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval<T> {
    lo: T,
    hi: T,
}

impl<T> Interval<T>
where
    T: Clone + PartialOrd,
{
    /// Construct an interval. Panics in debug builds if `lo > hi`.
    pub fn new(lo: T, hi: T) -> Self {
        debug_assert!(
            !(lo.clone().partial_cmp(&hi) == Some(core::cmp::Ordering::Greater)),
            "interval lower bound must not exceed upper bound"
        );
        Interval { lo, hi }
    }

    /// The lower bound.
    pub fn lo(&self) -> &T {
        &self.lo
    }

    /// The upper bound.
    pub fn hi(&self) -> &T {
        &self.hi
    }

    /// True if `x` lies within `[lo, hi]`.
    pub fn contains(&self, x: &T) -> bool {
        self.lo
            .partial_cmp(x)
            .is_some_and(|o| o != core::cmp::Ordering::Greater)
            && self
                .hi
                .partial_cmp(x)
                .is_some_and(|o| o != core::cmp::Ordering::Less)
    }

    /// The smallest interval containing both `self` and `other`.
    pub fn hull(&self, other: &Interval<T>) -> Interval<T>
    where
        T: Ord,
    {
        let lo = core::cmp::min(self.lo.clone(), other.lo.clone());
        let hi = core::cmp::max(self.hi.clone(), other.hi.clone());
        Interval::new(lo, hi)
    }

    /// The overlap of `self` and `other`, or `None` if they are disjoint.
    pub fn intersect(&self, other: &Interval<T>) -> Option<Interval<T>>
    where
        T: Ord,
    {
        let lo = core::cmp::max(self.lo.clone(), other.lo.clone());
        let hi = core::cmp::min(self.hi.clone(), other.hi.clone());
        if lo <= hi {
            Some(Interval::new(lo, hi))
        } else {
            None
        }
    }

    /// True if the interval contains exactly one point.
    pub fn is_point(&self) -> bool
    where
        T: PartialEq,
    {
        self.lo == self.hi
    }
}

impl<T> Interval<T>
where
    T: Clone + Ord + core::ops::Add<Output = T> + core::ops::Sub<Output = T>,
{
    /// The width `hi - lo`.
    pub fn width(&self) -> T {
        self.hi.clone() - self.lo.clone()
    }

    /// The midpoint `(lo + hi) / 2`.
    pub fn midpoint(&self) -> T
    where
        T: core::ops::Div<Output = T> + FromPrimitive,
    {
        let two = T::from_u8(2).expect("two");
        (self.lo.clone() + self.hi.clone()) / two
    }
}

macro_rules! impl_binop {
    ($trait:ident, $fn:ident, $op:ident) => {
        impl<T> core::ops::$trait<&Interval<T>> for &Interval<T>
        where
            T: Clone + Ord + core::ops::$trait<Output = T>,
        {
            type Output = Interval<T>;
            fn $fn(self, rhs: &Interval<T>) -> Interval<T> {
                let a = self.lo.clone().$op(rhs.lo.clone());
                let b = self.lo.clone().$op(rhs.hi.clone());
                let c = self.hi.clone().$op(rhs.lo.clone());
                let d = self.hi.clone().$op(rhs.hi.clone());
                let lo = core::cmp::min(
                    core::cmp::min(a.clone(), b.clone()),
                    core::cmp::min(c.clone(), d.clone()),
                );
                let hi = core::cmp::max(
                    core::cmp::max(a.clone(), b.clone()),
                    core::cmp::max(c.clone(), d.clone()),
                );
                Interval::new(lo, hi)
            }
        }
        impl<T> core::ops::$trait<Interval<T>> for Interval<T>
        where
            T: Clone + Ord + core::ops::$trait<Output = T>,
        {
            type Output = Interval<T>;
            fn $fn(self, rhs: Interval<T>) -> Interval<T> {
                (&self).$fn(&rhs)
            }
        }
    };
}

impl_binop!(Add, add, add);
impl_binop!(Sub, sub, sub);
impl_binop!(Mul, mul, mul);

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;

    fn rat(n: i64) -> BigRational {
        BigRational::new(n.into(), 1.into())
    }
    fn iv(a: i64, b: i64) -> Interval<BigRational> {
        Interval::new(rat(a), rat(b))
    }

    #[test]
    fn add_encloses() {
        let c = iv(1, 2) + iv(3, 4);
        assert_eq!(c.lo(), &rat(4));
        assert_eq!(c.hi(), &rat(6));
    }

    #[test]
    fn mul_encloses_all_corners() {
        let c = iv(-1, 2) * iv(-3, 4);
        assert_eq!(c.lo(), &rat(-6));
        assert_eq!(c.hi(), &rat(8));
    }

    #[test]
    fn contains_and_hull() {
        let a = iv(0, 5);
        assert!(a.contains(&rat(3)));
        assert!(!a.contains(&rat(6)));
        let b = iv(3, 10);
        let h = a.hull(&b);
        assert_eq!(h.lo(), &rat(0));
        assert_eq!(h.hi(), &rat(10));
    }

    #[test]
    fn intersect() {
        let a = iv(0, 5);
        let b = iv(3, 10);
        assert_eq!(a.intersect(&b), Some(iv(3, 5)));
        assert_eq!(iv(0, 1).intersect(&iv(2, 3)), None);
    }

    #[test]
    fn midpoint_and_width() {
        let a = iv(2, 4);
        assert_eq!(a.width(), rat(2));
        let m = a.midpoint();
        assert_eq!(m, rat(3));
    }
}
