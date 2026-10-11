#![no_std]
#![allow(clippy::wrong_self_convention)]
//! Forward-mode automatic differentiation via dual numbers.
//!
//! A [`Dual<T, N>`] pairs a value with a vector of `N` first-derivatives, one
//! per independent variable. Arithmetic and the common transcendental
//! functions propagate derivatives exactly, so evaluating a function on a
//! [`Dual`] yields both its value and its gradient with respect to each
//! variable in a single forward pass.
//!
//! The default `N = 1` gives the classic scalar dual number; `N > 1` computes
//! all first partials at once.
//!
//! # Examples
//!
//! ```
//! use tpt_math_autodiff_fwd::Dual;
//!
//! // f(x) = x^2, derivative at x = 3 should be 6.
//! let x = Dual::<f64>::variable(3.0, 0);
//! let y = x * x;
//! assert_eq!(y.re(), 9.0);
//! assert_eq!(y.du(0), 6.0);
//! ```
//!
//! [`Dual`]: Dual

use core::ops::{Add, Div, Mul, Neg, Sub};
use tpt_math_numeric::{Float, One, Zero};

/// A dual number: a value `re` together with `N` derivatives `du`.
///
/// `du[i]` is the partial derivative of `self` with respect to the `i`-th
/// independent variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dual<T, const N: usize = 1> {
    re: T,
    du: [T; N],
}

impl<T: Zero + Copy, const N: usize> Dual<T, N> {
    /// A constant (zero derivative in every direction).
    pub fn constant(re: T) -> Self {
        Dual {
            re,
            du: [T::zero(); N],
        }
    }
}

impl<T: Zero + One + Copy, const N: usize> Dual<T, N> {
    /// An independent variable: `re` with a unit derivative in direction `idx`.
    pub fn variable(re: T, idx: usize) -> Self {
        let mut du = [T::zero(); N];
        if idx < N {
            du[idx] = T::one();
        }
        Dual { re, du }
    }
}

impl<T, const N: usize> Dual<T, N> {
    /// The primal value.
    pub fn re(&self) -> T
    where
        T: Copy,
    {
        self.re
    }

    /// The derivative in direction `idx`.
    pub fn du(&self, idx: usize) -> T
    where
        T: Copy,
    {
        self.du[idx]
    }

    /// The full derivative vector.
    pub fn deriv(&self) -> &[T; N] {
        &self.du
    }
}

impl<T, const N: usize> Dual<T, N>
where
    T: Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Zero,
{
    /// Construct from a value and explicit derivative vector.
    pub fn new(re: T, du: [T; N]) -> Self {
        Dual { re, du }
    }
}

macro_rules! impl_binop {
    ($trait:ident, $fn:ident, $re:expr, $du:expr) => {
        impl<T, const N: usize> core::ops::$trait for Dual<T, N>
        where
            T: Copy
                + Add<T, Output = T>
                + Sub<T, Output = T>
                + Mul<T, Output = T>
                + Div<T, Output = T>
                + Zero,
        {
            type Output = Dual<T, N>;
            fn $fn(self, rhs: Dual<T, N>) -> Dual<T, N> {
                let re = $re(self.re, rhs.re);
                let du = core::array::from_fn(|i| $du(self.re, self.du[i], rhs.re, rhs.du[i]));
                Dual { re, du }
            }
        }
    };
}

impl_binop!(Add, add, |a, b| a + b, |_a, da, _b, db| da + db);
impl_binop!(Sub, sub, |a, b| a - b, |_a, da, _b, db| da - db);
impl_binop!(Mul, mul, |a, b| a * b, |a, da, b, db| a * db + b * da);
impl_binop!(Div, div, |a, b| a / b, |a, da, b, db| (da * b - a * db)
    / (b * b));

impl<T, const N: usize> Dual<T, N>
where
    T: Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Zero
        + Neg<Output = T>,
{
    /// Negate.
    pub fn negate(self) -> Dual<T, N> {
        Dual {
            re: -self.re,
            du: core::array::from_fn(|i| -self.du[i]),
        }
    }
}

impl<T, const N: usize> Dual<T, N>
where
    T: Float,
{
    /// `sin(self)` with propagated derivative `cos(re) * du`.
    pub fn sin(self) -> Dual<T, N> {
        let c = self.re.cos();
        Dual {
            re: self.re.sin(),
            du: core::array::from_fn(|i| self.du[i] * c),
        }
    }

    /// `cos(self)` with propagated derivative `-sin(re) * du`.
    pub fn cos(self) -> Dual<T, N> {
        let s = self.re.sin();
        Dual {
            re: self.re.cos(),
            du: core::array::from_fn(|i| -self.du[i] * s),
        }
    }

    /// `exp(self)` with propagated derivative `exp(re) * du`.
    pub fn exp(self) -> Dual<T, N> {
        let e = self.re.exp();
        Dual {
            re: e,
            du: core::array::from_fn(|i| self.du[i] * e),
        }
    }

    /// `ln(self)` with propagated derivative `du / re`.
    pub fn ln(self) -> Dual<T, N> {
        Dual {
            re: self.re.ln(),
            du: core::array::from_fn(|i| self.du[i] / self.re),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_powi_powf_derivatives() {
        let x = Dual::<f64>::variable(4.0, 0);
        // d/dx sqrt(x) = 1/(2 sqrt x) = 0.25 at x = 4.
        assert!((x.sqrt().du(0) - 0.25).abs() < 1e-12);
        // d/dx x^3 = 3x^2 = 48 at x = 4.
        assert!((x.powi(3).du(0) - 48.0).abs() < 1e-12);
        // d/dx x^2.5 = 2.5 x^1.5 = 20 at x = 4.
        assert!((x.powf(2.5).du(0) - 20.0).abs() < 1e-9);
        // The zero exponent is constant, so its derivative is 0 (not 0·0⁻¹).
        let z = Dual::<f64>::variable(0.0, 0);
        assert_eq!(z.powi(0).du(0), 0.0);
    }

    #[test]
    fn tan_tanh_atan_abs_derivatives() {
        let x = Dual::<f64>::variable(0.5, 0);
        let t = 0.5_f64.tan();
        assert!((x.tan().du(0) - (1.0 + t * t)).abs() < 1e-12);
        let y = Dual::<f64>::variable(0.3, 0);
        let th = 0.3_f64.tanh();
        assert!((y.tanh().du(0) - (1.0 - th * th)).abs() < 1e-12);
        let a = Dual::<f64>::variable(1.0, 0);
        // d/dx atan(x) = 1/(1 + x²) = 0.5 at x = 1.
        assert!((a.atan().du(0) - 0.5).abs() < 1e-12);
        let n = Dual::<f64>::variable(-2.0, 0);
        assert_eq!(n.abs().re(), 2.0);
        assert_eq!(n.abs().du(0), -1.0);
    }

    #[test]
    fn dual_with_scalar_operators() {
        let x = Dual::<f64>::variable(2.0, 0);
        assert_eq!((x * 3.0).du(0), 3.0);
        assert_eq!((x + 1.0).re(), 3.0);
        assert_eq!((x + 1.0).du(0), 1.0);
        assert_eq!((x - 1.0).re(), 1.0);
        assert_eq!((x / 4.0).du(0), 0.25);
        assert_eq!((x / 4.0).re(), 0.5);
    }

    #[test]
    fn polynomial_derivative() {
        let x = Dual::<f64>::variable(3.0, 0);
        let y = x * x * x + x * Dual::constant(2.0);
        assert!((y.re() - 33.0).abs() < 1e-12);
        // d/dx (x^3 + 2x) = 3x^2 + 2 = 29 at x=3
        assert!((y.du(0) - 29.0).abs() < 1e-12);
    }

    #[test]
    fn sin_derivative() {
        let x = Dual::<f64>::variable(0.0, 0);
        let y = x.sin();
        assert!((y.re() - 0.0).abs() < 1e-12);
        assert!((y.du(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn multi_variable_gradient() {
        let x = Dual::<f64, 2>::variable(2.0, 0);
        let y = Dual::<f64, 2>::variable(3.0, 1);
        let f = x * y + x;
        // f = x*y + x ; df/dx = y+1 = 4, df/dy = x = 2
        assert!((f.re() - 8.0).abs() < 1e-12);
        assert!((f.du(0) - 4.0).abs() < 1e-12);
        assert!((f.du(1) - 2.0).abs() < 1e-12);
    }
}
