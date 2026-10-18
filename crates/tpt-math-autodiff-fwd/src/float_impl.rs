//! `num_traits::Float` for [`Dual<f64, N>`], so generic numerical code written for
//! `Float` (the solvers in `tpt-math-numeric`, for instance) runs on dual numbers
//! and returns exact derivatives with respect to the `N` independent variables.
//!
//! Every function applies the chain rule to the derivative array. Functions that
//! are piecewise constant (`floor`, `round`, `signum`, …) have zero derivative
//! almost everywhere. At their jumps the derivative is zero by convention.

use core::cmp::Ordering;
use core::num::FpCategory;

use tpt_math_numeric::{Float, FloatConst, Num, NumCast, One, ToPrimitive, Zero};

use crate::Dual;

type D<const N: usize> = Dual<f64, N>;

impl<const N: usize> D<N> {
    /// `self` with the value `value` and every derivative multiplied by `slope`:
    /// the chain rule for a function whose derivative at `self.re` is `slope`.
    fn chain(self, value: f64, slope: f64) -> Self {
        Dual {
            re: value,
            du: self.du.map(|d| d * slope),
        }
    }

    /// A value with no derivative, for piecewise-constant functions.
    fn flat(value: f64) -> Self {
        Dual {
            re: value,
            du: [0.0; N],
        }
    }
}

impl<const N: usize> core::ops::Neg for D<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Dual {
            re: -self.re,
            du: self.du.map(|d| -d),
        }
    }
}

impl<const N: usize> Zero for D<N> {
    fn zero() -> Self {
        Dual::constant(0.0)
    }
    fn is_zero(&self) -> bool {
        self.re == 0.0
    }
}

impl<const N: usize> One for D<N> {
    fn one() -> Self {
        Dual::constant(1.0)
    }
}

impl<const N: usize> core::ops::Rem for D<N> {
    type Output = Self;
    /// `x % y` with `d(x % y) = dx − dy · trunc(x / y)`.
    fn rem(self, rhs: Self) -> Self {
        let q = (self.re / rhs.re).trunc();
        Dual {
            re: self.re % rhs.re,
            du: core::array::from_fn(|i| self.du[i] - rhs.du[i] * q),
        }
    }
}

impl<const N: usize> Num for D<N> {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, <f64 as Num>::FromStrRadixErr> {
        <f64 as Num>::from_str_radix(s, radix).map(Dual::constant)
    }
}

impl<const N: usize> ToPrimitive for D<N> {
    fn to_i64(&self) -> Option<i64> {
        self.re.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.re.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.re)
    }
}

impl<const N: usize> NumCast for D<N> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Dual::constant)
    }
}

impl<const N: usize> PartialOrd for D<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl<const N: usize> FloatConst for D<N> {
    fn E() -> Self {
        Dual::constant(f64::E())
    }
    fn FRAC_1_PI() -> Self {
        Dual::constant(f64::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        Dual::constant(f64::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        Dual::constant(f64::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Dual::constant(f64::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        Dual::constant(f64::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        Dual::constant(f64::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        Dual::constant(f64::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        Dual::constant(f64::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        Dual::constant(f64::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        Dual::constant(f64::LN_10())
    }
    fn LN_2() -> Self {
        Dual::constant(f64::LN_2())
    }
    fn LOG10_E() -> Self {
        Dual::constant(f64::LOG10_E())
    }
    fn LOG2_E() -> Self {
        Dual::constant(f64::LOG2_E())
    }
    fn PI() -> Self {
        Dual::constant(f64::PI())
    }
    fn SQRT_2() -> Self {
        Dual::constant(f64::SQRT_2())
    }
    fn TAU() -> Self {
        Dual::constant(f64::TAU())
    }
    fn LOG2_10() -> Self {
        Dual::constant(f64::LOG2_10())
    }
    fn LOG10_2() -> Self {
        Dual::constant(f64::LOG10_2())
    }
}

impl<const N: usize> Float for D<N> {
    fn nan() -> Self {
        Dual::constant(f64::NAN)
    }
    fn infinity() -> Self {
        Dual::constant(f64::INFINITY)
    }
    fn neg_infinity() -> Self {
        Dual::constant(f64::NEG_INFINITY)
    }
    fn neg_zero() -> Self {
        Dual::constant(-0.0)
    }
    fn min_value() -> Self {
        Dual::constant(f64::MIN)
    }
    fn min_positive_value() -> Self {
        Dual::constant(f64::MIN_POSITIVE)
    }
    fn epsilon() -> Self {
        Dual::constant(f64::EPSILON)
    }
    fn max_value() -> Self {
        Dual::constant(f64::MAX)
    }
    fn is_nan(self) -> bool {
        self.re.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.re.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.re.is_finite()
    }
    fn is_normal(self) -> bool {
        self.re.is_normal()
    }
    fn classify(self) -> FpCategory {
        self.re.classify()
    }
    fn floor(self) -> Self {
        Self::flat(self.re.floor())
    }
    fn ceil(self) -> Self {
        Self::flat(self.re.ceil())
    }
    fn round(self) -> Self {
        Self::flat(self.re.round())
    }
    fn trunc(self) -> Self {
        Self::flat(self.re.trunc())
    }
    fn fract(self) -> Self {
        Dual {
            re: self.re.fract(),
            du: self.du,
        }
    }
    fn abs(self) -> Self {
        let s = if self.re < 0.0 { -1.0 } else { 1.0 };
        self.chain(self.re.abs(), s)
    }
    fn signum(self) -> Self {
        Self::flat(self.re.signum())
    }
    fn is_sign_positive(self) -> bool {
        self.re.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.re.is_sign_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    fn recip(self) -> Self {
        let r = 1.0 / self.re;
        self.chain(r, -r * r)
    }
    fn powi(self, n: i32) -> Self {
        Dual::powi(self, n)
    }
    fn powf(self, n: Self) -> Self {
        // d(xⁿ) = xⁿ · (n′ ln x + n x′ / x), with both arguments differentiated.
        let value = self.re.powf(n.re);
        let ln = self.re.ln();
        Dual {
            re: value,
            du: core::array::from_fn(|i| value * (n.du[i] * ln + n.re * self.du[i] / self.re)),
        }
    }
    fn sqrt(self) -> Self {
        Dual::sqrt(self)
    }
    fn exp(self) -> Self {
        Dual::exp(self)
    }
    fn exp2(self) -> Self {
        let v = self.re.exp2();
        self.chain(v, v * core::f64::consts::LN_2)
    }
    fn ln(self) -> Self {
        Dual::ln(self)
    }
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    fn log2(self) -> Self {
        self.ln() / Self::constant_f(core::f64::consts::LN_2)
    }
    fn log10(self) -> Self {
        self.ln() / Self::constant_f(core::f64::consts::LN_10)
    }
    fn to_degrees(self) -> Self {
        self * Self::constant_f(180.0 / core::f64::consts::PI)
    }
    fn to_radians(self) -> Self {
        self * Self::constant_f(core::f64::consts::PI / 180.0)
    }
    fn max(self, other: Self) -> Self {
        if self.re >= other.re || other.re.is_nan() {
            self
        } else {
            other
        }
    }
    fn min(self, other: Self) -> Self {
        if self.re <= other.re || other.re.is_nan() {
            self
        } else {
            other
        }
    }
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        self.chain(c, 1.0 / (3.0 * c * c))
    }
    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }
    fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin())
    }
    fn tan(self) -> Self {
        Dual::tan(self)
    }
    fn asin(self) -> Self {
        self.chain(self.re.asin(), 1.0 / (1.0 - self.re * self.re).sqrt())
    }
    fn acos(self) -> Self {
        self.chain(self.re.acos(), -1.0 / (1.0 - self.re * self.re).sqrt())
    }
    fn atan(self) -> Self {
        Dual::atan(self)
    }
    fn atan2(self, other: Self) -> Self {
        // d atan2(y, x) = (x dy − y dx) / (x² + y²).
        let denom = other.re * other.re + self.re * self.re;
        Dual {
            re: self.re.atan2(other.re),
            du: core::array::from_fn(|i| (other.re * self.du[i] - self.re * other.du[i]) / denom),
        }
    }
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }
    fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), 1.0 / (1.0 + self.re))
    }
    fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }
    fn tanh(self) -> Self {
        Dual::tanh(self)
    }
    fn asinh(self) -> Self {
        self.chain(self.re.asinh(), 1.0 / (self.re * self.re + 1.0).sqrt())
    }
    fn acosh(self) -> Self {
        self.chain(self.re.acosh(), 1.0 / (self.re * self.re - 1.0).sqrt())
    }
    fn atanh(self) -> Self {
        self.chain(self.re.atanh(), 1.0 / (1.0 - self.re * self.re))
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.re.integer_decode()
    }
    fn abs_sub(self, other: Self) -> Self {
        if self.re > other.re {
            self - other
        } else {
            Self::flat(0.0)
        }
    }
    fn copysign(self, sign: Self) -> Self {
        let s = if (sign.re < 0.0) == (self.re < 0.0) {
            1.0
        } else {
            -1.0
        };
        self.chain(self.re.copysign(sign.re), s)
    }
}

impl<const N: usize> D<N> {
    /// A constant with value `v` and no derivative.
    fn constant_f(v: f64) -> Self {
        Dual::constant(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_numeric::ode::dopri5_integrate;

    #[test]
    fn chain_rule_through_common_functions() {
        // f(x) = sin(x²) · exp(x), so f′(x) = (2x cos(x²) + sin(x²)) · eˣ.
        let x = Dual::<f64, 1>::variable(0.7, 0);
        let f = Float::sin(x * x) * Float::exp(x);
        let want = (2.0 * 0.7 * (0.49f64).cos() + (0.49f64).sin()) * 0.7f64.exp();
        assert!((f.du(0) - want).abs() < 1e-12, "{} vs {want}", f.du(0));
    }

    #[test]
    fn powf_differentiates_both_base_and_exponent() {
        // d/dx x^y at (2, 3) is 3·2² = 12; d/dy x^y is 2³·ln 2.
        let x = Dual::<f64, 2>::variable(2.0, 0);
        let y = Dual::<f64, 2>::variable(3.0, 1);
        let f = Float::powf(x, y);
        assert!((f.du(0) - 12.0).abs() < 1e-12);
        assert!((f.du(1) - 8.0 * 2f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn atan2_and_hypot_have_the_right_partials() {
        let y = Dual::<f64, 2>::variable(3.0, 0);
        let x = Dual::<f64, 2>::variable(4.0, 1);
        let r = Float::hypot(x, y);
        assert!((r.re() - 5.0).abs() < 1e-12);
        assert!((r.du(0) - 0.6).abs() < 1e-12 && (r.du(1) - 0.8).abs() < 1e-12);
        let a = Float::atan2(y, x);
        assert!((a.du(0) - 4.0 / 25.0).abs() < 1e-12);
        assert!((a.du(1) + 3.0 / 25.0).abs() < 1e-12);
    }

    #[test]
    fn generic_ode_solver_returns_the_exact_sensitivity_of_a_parameter() {
        // y' = −k y, with k a dual variable. Then ∂y(t)/∂k = −t y₀ e^{−kt}.
        let k = Dual::<f64, 1>::variable(0.5, 0);
        let y0 = Dual::<f64, 1>::constant(2.0);
        let y = dopri5_integrate(
            |_t, y: &[Dual<f64, 1>; 1]| [-k * y[0]],
            Dual::constant(0.0),
            Dual::constant(2.0),
            [y0],
            Dual::constant(1e-11),
            Dual::constant(1e-13),
            100_000,
        )
        .unwrap();
        let want = -2.0 * 2.0 * (-0.5f64 * 2.0).exp();
        assert!((y[0].du(0) - want).abs() < 1e-7, "{} vs {want}", y[0].du(0));
        assert!((y[0].re() - 2.0 * (-1.0f64).exp()).abs() < 1e-8);
    }
}
