//! The tape-bound scalar handle and its arithmetic.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ptr;

use crate::op::Op;
use crate::tape::GradientTape;

/// A scalar bound to a [`GradientTape`].
///
/// A `Variable` is a lightweight handle: the index of its node on the tape,
/// its primal value, and a back-pointer to the tape. It is [`Copy`], so it can
/// be used several times in one expression (`x * x`) — the backward pass
/// accumulates each use.
///
/// Every operation on a `Variable` appends a node to the tape and returns a
/// handle to that node, which is what lets [`GradientTape::backward`] replay
/// the computation in reverse.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff_rev::GradientTape;
///
/// let tape = GradientTape::new();
/// let x = tape.var(0.5);
/// let y = (2.0 * x).exp() - x.ln();
/// // dy/dx = 2 exp(2x) - 1/x
/// let expected = 2.0 * 1.0_f64.exp() - 2.0;
/// assert!((tape.gradient(y, &[x])[0] - expected).abs() < 1e-12);
/// ```
#[derive(Clone, Copy)]
pub struct Variable<'t> {
    tape: &'t GradientTape,
    index: usize,
    value: f64,
}

impl<'t> Variable<'t> {
    /// Bind an already-recorded node to a handle.
    pub(crate) fn new(tape: &'t GradientTape, index: usize, value: f64) -> Self {
        Variable { tape, index, value }
    }

    /// The primal (forward) value of this node.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The index of this node on its tape.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// The tape this variable was recorded on.
    #[must_use]
    pub fn tape(&self) -> &'t GradientTape {
        self.tape
    }

    /// Record a unary operation whose primal result is `value`.
    fn unary(self, make: impl FnOnce(usize) -> Op, value: f64) -> Variable<'t> {
        self.tape.push(make(self.index), value)
    }

    /// Record a binary operation whose primal result is `value`.
    ///
    /// # Panics
    ///
    /// Panics if the two operands live on different tapes.
    fn binary(
        self,
        rhs: Variable<'t>,
        make: impl FnOnce(usize, usize) -> Op,
        value: f64,
    ) -> Variable<'t> {
        assert!(
            ptr::eq(self.tape, rhs.tape),
            "cannot combine variables from different GradientTapes"
        );
        self.tape.push(make(self.index, rhs.index), value)
    }

    /// `sin(self)`, with local derivative `cos(self)`.
    #[must_use]
    pub fn sin(self) -> Variable<'t> {
        self.unary(Op::Sin, self.value.sin())
    }

    /// `cos(self)`, with local derivative `-sin(self)`.
    #[must_use]
    pub fn cos(self) -> Variable<'t> {
        self.unary(Op::Cos, self.value.cos())
    }

    /// `tan(self)`, with local derivative `1 + tan²(self)`.
    #[must_use]
    pub fn tan(self) -> Variable<'t> {
        self.unary(Op::Tan, self.value.tan())
    }

    /// `exp(self)`, with local derivative `exp(self)`.
    #[must_use]
    pub fn exp(self) -> Variable<'t> {
        self.unary(Op::Exp, self.value.exp())
    }

    /// `ln(self)`, with local derivative `1 / self`.
    #[must_use]
    pub fn ln(self) -> Variable<'t> {
        self.unary(Op::Ln, self.value.ln())
    }

    /// `sqrt(self)`, with local derivative `1 / (2 sqrt(self))`.
    #[must_use]
    pub fn sqrt(self) -> Variable<'t> {
        self.unary(Op::Sqrt, self.value.sqrt())
    }

    /// `self` raised to the integer power `n`, with local derivative
    /// `n * self^(n - 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_autodiff_rev::GradientTape;
    ///
    /// let tape = GradientTape::new();
    /// let x = tape.var(2.0);
    /// let y = x.powi(3);
    /// assert!((y.value() - 8.0).abs() < 1e-12);
    /// assert!((tape.gradient(y, &[x])[0] - 12.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn powi(self, n: i32) -> Variable<'t> {
        self.unary(|a| Op::Powi(a, n), self.value.powi(n))
    }

    /// `self` raised to the real power `p`, with local derivative
    /// `p * self^(p - 1)`.
    #[must_use]
    pub fn powf(self, p: f64) -> Variable<'t> {
        self.unary(|a| Op::Powf(a, p), self.value.powf(p))
    }

    /// `1 / self`, with local derivative `-1 / self²`.
    #[must_use]
    pub fn recip(self) -> Variable<'t> {
        self.tape.constant(1.0) / self
    }
}

impl fmt::Debug for Variable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("index", &self.index)
            .field("value", &self.value)
            .finish()
    }
}

impl fmt::Display for Variable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

/// Implement `Variable op Variable`, `Variable op f64` and `f64 op Variable`.
macro_rules! impl_binop {
    ($Trait:ident, $method:ident, $Variant:ident, $eval:expr) => {
        impl<'t> $Trait<Variable<'t>> for Variable<'t> {
            type Output = Variable<'t>;

            #[inline]
            fn $method(self, rhs: Variable<'t>) -> Variable<'t> {
                let eval: fn(f64, f64) -> f64 = $eval;
                self.binary(rhs, Op::$Variant, eval(self.value(), rhs.value()))
            }
        }

        impl<'t> $Trait<f64> for Variable<'t> {
            type Output = Variable<'t>;

            #[inline]
            fn $method(self, rhs: f64) -> Variable<'t> {
                let rhs = self.tape().constant(rhs);
                $Trait::$method(self, rhs)
            }
        }

        impl<'t> $Trait<Variable<'t>> for f64 {
            type Output = Variable<'t>;

            #[inline]
            fn $method(self, rhs: Variable<'t>) -> Variable<'t> {
                let lhs = rhs.tape().constant(self);
                $Trait::$method(lhs, rhs)
            }
        }
    };
}

impl_binop!(Add, add, Add, |a, b| a + b);
impl_binop!(Sub, sub, Sub, |a, b| a - b);
impl_binop!(Mul, mul, Mul, |a, b| a * b);
impl_binop!(Div, div, Div, |a, b| a / b);

/// Implement the `*Assign` counterpart of an already-implemented operator.
macro_rules! impl_assign_op {
    ($Trait:ident, $method:ident, $op:tt) => {
        impl<'t> $Trait<Variable<'t>> for Variable<'t> {
            #[inline]
            fn $method(&mut self, rhs: Variable<'t>) {
                *self = *self $op rhs;
            }
        }

        impl $Trait<f64> for Variable<'_> {
            #[inline]
            fn $method(&mut self, rhs: f64) {
                *self = *self $op rhs;
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, +);
impl_assign_op!(SubAssign, sub_assign, -);
impl_assign_op!(MulAssign, mul_assign, *);
impl_assign_op!(DivAssign, div_assign, /);

impl<'t> Neg for Variable<'t> {
    type Output = Variable<'t>;

    #[inline]
    fn neg(self) -> Variable<'t> {
        self.unary(Op::Neg, -self.value)
    }
}

#[cfg(test)]
mod tests {
    use crate::GradientTape;

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < TOL,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn scalar_operands_on_either_side() {
        let tape = GradientTape::new();
        let x = tape.var(4.0);
        let a = 2.0 * x + 1.0; // 9
        let b = 10.0 - x; // 6
        let c = 12.0 / x; // 3
        let d = x / 2.0; // 2
        assert_close(a.value(), 9.0);
        assert_close(b.value(), 6.0);
        assert_close(c.value(), 3.0);
        assert_close(d.value(), 2.0);

        assert_close(tape.gradient(a, &[x])[0], 2.0);
        assert_close(tape.gradient(b, &[x])[0], -1.0);
        assert_close(tape.gradient(c, &[x])[0], -12.0 / 16.0);
        assert_close(tape.gradient(d, &[x])[0], 0.5);
    }

    #[test]
    fn negation_and_subtraction() {
        let tape = GradientTape::new();
        let x = tape.var(3.0);
        let y = tape.var(7.0);
        let z = -x - y;
        assert_close(z.value(), -10.0);
        let g = tape.gradient(z, &[x, y]);
        assert_close(g[0], -1.0);
        assert_close(g[1], -1.0);
    }

    #[test]
    fn assign_operators_accumulate_on_the_tape() {
        let tape = GradientTape::new();
        let xs = tape.vars(&[1.0, 2.0, 3.0]);
        let mut acc = tape.constant(0.0);
        for &x in &xs {
            acc += x * x;
        }
        acc *= 2.0;
        // f = 2 * sum(x_i^2) => df/dx_i = 4 x_i
        assert_close(acc.value(), 28.0);
        let g = tape.gradient(acc, &xs);
        assert_close(g[0], 4.0);
        assert_close(g[1], 8.0);
        assert_close(g[2], 12.0);
    }

    #[test]
    fn recip_and_powf() {
        let tape = GradientTape::new();
        let x = tape.var(2.0);
        let r = x.recip();
        assert_close(r.value(), 0.5);
        assert_close(tape.gradient(r, &[x])[0], -0.25);

        let p = x.powf(2.5);
        assert_close(p.value(), 2.0_f64.powf(2.5));
        assert_close(tape.gradient(p, &[x])[0], 2.5 * 2.0_f64.powf(1.5));
    }

    #[test]
    fn debug_and_display() {
        let tape = GradientTape::new();
        let x = tape.var(1.25);
        assert_eq!(x.to_string(), "1.25");
        assert!(format!("{x:?}").contains("index: 0"));
        assert_eq!(x.index(), 0);
        assert!(tape.owns(x));
    }

    #[test]
    #[should_panic(expected = "different GradientTapes")]
    fn mixing_tapes_panics() {
        let a = GradientTape::new();
        let b = GradientTape::new();
        let x = a.var(1.0);
        let y = b.var(2.0);
        let _ = x + y;
    }
}
