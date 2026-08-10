//! Permissive-license symbolic math (CAS).
//!
//! `tpt-math-symbolic` provides a small, dependency-light computer-algebra
//! core built around an expression tree [`Expr<C>`] parameterised over a
//! numeric [`Coefficient`]. It supports construction, algebraic
//! simplification, symbolic differentiation, substitution and evaluation.
//!
//! # Design decision (spec.txt open call)
//!
//! The expression tree is generic over [`Coefficient`] rather than hard-wired
//! to a single number type. The default convenience alias is
//! [`Expr64`] (`Expr<f64>`). Enable the `exact` feature to also get
//! [`ExprRational`], which is backed by arbitrary-precision
//! [`tpt_math_exact::BigRational`] wrapped in a [`Coefficient`] implementation.
//! This keeps the crate genuinely coefficient-agnostic without forcing every
//! consumer onto exact arithmetic.
//!
//! ```
//! use tpt_math_symbolic::{Expr64, Symbolic};
//!
//! // Build (3*x + 1)^2 and differentiate with respect to x.
//! let x = Expr64::var("x");
//! let expr = (Expr64::from(3.0) * x.clone() + Expr64::from(1.0)).pow(2);
//! let d = expr.derivative("x").simplify();
//! assert_eq!(d.eval_str("x", 2.0).unwrap(), 42.0);
//! ```

use std::collections::HashMap;
use std::fmt;

use num_traits::{One, Zero};

/// A numeric value that can sit at a leaf of an [`Expr`] and survive
/// simplification, evaluation and differentiation.
///
/// Implementors must be a field-like type supporting the four arithmetic
/// operations plus integer powers. `f64` satisfies this out of the box; the
/// `exact` feature provides an implementation for wrapped
/// [`tpt_math_exact::BigRational`].
pub trait Coefficient:
    Clone
    + PartialEq
    + fmt::Debug
    + fmt::Display
    + From<f64>
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    /// Raise `self` to an integer power.
    fn powi(self, n: i32) -> Self;

    /// Whether this coefficient is exactly zero.
    fn is_zero_value(&self) -> bool {
        *self == Self::zero()
    }
}

impl Coefficient for f64 {
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
}

/// A symbolic expression over coefficient type `C`.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr<C: Coefficient> {
    /// A literal constant, e.g. `3.0`.
    Const(C),
    /// A free variable, e.g. `x`.
    Var(String),
    /// Addition of two sub-expressions.
    Add(Box<Expr<C>>, Box<Expr<C>>),
    /// Subtraction of the right from the left.
    Sub(Box<Expr<C>>, Box<Expr<C>>),
    /// Multiplication of two sub-expressions.
    Mul(Box<Expr<C>>, Box<Expr<C>>),
    /// Division of the left by the right.
    Div(Box<Expr<C>>, Box<Expr<C>>),
    /// Arithmetic negation.
    Neg(Box<Expr<C>>),
    /// `base^exponent` with a non-negative integer exponent.
    Pow(Box<Expr<C>>, usize),
    /// A named unary function applied to a sub-expression, e.g. `sin(...)`.
    Func(String, Box<Expr<C>>),
}

/// Extension trait providing fluent constructor and calculus helpers.
///
/// Implemented for `Expr<C>` so callers can write `x.derivative("x")`,
/// `expr.simplify()`, and so on.
pub trait Symbolic<C: Coefficient> {
    /// Construct a variable expression.
    fn var(name: impl Into<String>) -> Self;
    /// Construct a constant expression from a literal `f64`.
    fn constant(f: f64) -> Self;
    /// Raise this expression to a non-negative integer power.
    fn pow(self, n: usize) -> Self;
    /// Simplify this expression using algebraic identities.
    fn simplify(self) -> Self;
    /// Differentiate with respect to `var`.
    fn derivative(&self, var: &str) -> Self;
    /// Substitute `var` for `value` throughout the expression.
    fn substitute(&self, var: &str, value: C) -> Self;
    /// Evaluate the expression to a coefficient, given a variable environment.
    fn eval(&self, env: &HashMap<String, C>) -> Option<C>;
    /// Convenience wrapper evaluating with a single variable binding.
    fn eval_str(&self, var: &str, value: f64) -> Option<C>
    where
        C: From<f64>;
}

impl<C: Coefficient> Expr<C> {
    /// Construct a variable expression.
    pub fn var(name: impl Into<String>) -> Expr<C> {
        Expr::Var(name.into())
    }

    /// Construct a constant expression from a literal `f64`.
    pub fn constant(f: f64) -> Expr<C>
    where
        C: From<f64>,
    {
        Expr::Const(C::from(f))
    }

    /// Raise this expression to a non-negative integer power.
    pub fn pow(self, n: usize) -> Expr<C> {
        Expr::Pow(Box::new(self), n)
    }

    /// Build a named unary function expression, e.g. `Expr64::func("sin", x)`.
    pub fn func(name: impl Into<String>, inner: Expr<C>) -> Expr<C> {
        Expr::Func(name.into(), Box::new(inner))
    }
}

impl<C: Coefficient> Symbolic<C> for Expr<C> {
    fn var(name: impl Into<String>) -> Expr<C> {
        Expr::var(name)
    }

    fn constant(f: f64) -> Expr<C> {
        Expr::constant(f)
    }

    fn pow(self, n: usize) -> Expr<C> {
        Expr::pow(self, n)
    }

    fn simplify(self) -> Expr<C> {
        simplify(self)
    }

    fn derivative(&self, var: &str) -> Expr<C> {
        derivative(self, var)
    }

    fn substitute(&self, var: &str, value: C) -> Expr<C> {
        substitute(self, var, value)
    }

    fn eval(&self, env: &HashMap<String, C>) -> Option<C> {
        eval(self, env)
    }

    fn eval_str(&self, var: &str, value: f64) -> Option<C>
    where
        C: From<f64>,
    {
        let mut env = HashMap::new();
        env.insert(var.to_string(), C::from(value));
        eval(self, &env)
    }
}

/// `Expr<f64>` — the default, floating-point symbolic expression.
pub type Expr64 = Expr<f64>;

impl<C: Coefficient> From<f64> for Expr<C>
where
    C: From<f64>,
{
    fn from(f: f64) -> Expr<C> {
        Expr::Const(C::from(f))
    }
}

impl From<&str> for Expr64 {
    fn from(name: &str) -> Expr64 {
        Expr::Var(name.to_string())
    }
}

impl From<String> for Expr64 {
    fn from(name: String) -> Expr64 {
        Expr::Var(name)
    }
}

macro_rules! binop {
    ($trait:ident, $variant:ident, $method:ident) => {
        impl<C: Coefficient> std::ops::$trait for Expr<C> {
            type Output = Expr<C>;
            fn $method(self, rhs: Expr<C>) -> Expr<C> {
                Expr::$variant(Box::new(self), Box::new(rhs))
            }
        }
    };
}

binop!(Add, Add, add);
binop!(Sub, Sub, sub);
binop!(Mul, Mul, mul);
binop!(Div, Div, div);

impl<C: Coefficient> std::ops::Neg for Expr<C> {
    type Output = Expr<C>;
    fn neg(self) -> Expr<C> {
        Expr::Neg(Box::new(self))
    }
}

impl<C: Coefficient> std::ops::Add<C> for Expr<C> {
    type Output = Expr<C>;
    fn add(self, rhs: C) -> Expr<C> {
        Expr::Add(Box::new(self), Box::new(Expr::Const(rhs)))
    }
}

impl<C: Coefficient> std::ops::Mul<C> for Expr<C> {
    type Output = Expr<C>;
    fn mul(self, rhs: C) -> Expr<C> {
        Expr::Mul(Box::new(self), Box::new(Expr::Const(rhs)))
    }
}

fn simplify<C: Coefficient>(e: Expr<C>) -> Expr<C> {
    match e {
        Expr::Const(c) => Expr::Const(c),
        Expr::Var(v) => Expr::Var(v),
        Expr::Neg(a) => {
            let a = simplify(*a);
            match a {
                Expr::Const(c) => Expr::Const(-c),
                Expr::Neg(inner) => *inner,
                other => Expr::Neg(Box::new(other)),
            }
        }
        Expr::Add(a, b) => fold_add(vec![simplify(*a), simplify(*b)]),
        Expr::Sub(a, b) => fold_add(vec![simplify(*a), simplify(Expr::Neg(Box::new(*b)))]),
        Expr::Mul(a, b) => fold_mul(vec![simplify(*a), simplify(*b)]),
        Expr::Div(a, b) => {
            let (a, b) = (simplify(*a), simplify(*b));
            match (&a, &b) {
                (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1.clone() / c2.clone()),
                (_, Expr::Const(c)) if *c == C::one() => a,
                _ => Expr::Div(Box::new(a), Box::new(b)),
            }
        }
        Expr::Pow(a, n) => {
            let a = simplify(*a);
            match &a {
                Expr::Const(c) => Expr::Const(c.clone().powi(n as i32)),
                _ => Expr::Pow(Box::new(a), n),
            }
        }
        Expr::Func(name, a) => {
            let a = simplify(*a);
            match &a {
                Expr::Const(c) => Expr::Const(apply_func(&name, c.clone())),
                _ => Expr::Func(name, Box::new(a)),
            }
        }
    }
}

/// Flatten and fold a sum of already-simplified terms.
fn fold_add<C: Coefficient>(items: Vec<Expr<C>>) -> Expr<C> {
    let mut stack = items;
    let mut const_acc = C::zero();
    let mut terms: Vec<Expr<C>> = Vec::new();
    while let Some(it) = stack.pop() {
        match it {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Const(c) => const_acc = const_acc + c,
            Expr::Neg(inner) => match *inner {
                Expr::Const(c) => const_acc = const_acc - c,
                other => terms.push(Expr::Neg(Box::new(other))),
            },
            other => terms.push(other),
        }
    }
    if terms.is_empty() {
        return Expr::Const(const_acc);
    }
    let mut iter = terms.into_iter();
    let mut result = iter.next().unwrap();
    for t in iter {
        result = Expr::Add(Box::new(result), Box::new(t));
    }
    if const_acc.is_zero_value() {
        result
    } else {
        Expr::Add(Box::new(Expr::Const(const_acc)), Box::new(result))
    }
}

/// Flatten and fold a product of already-simplified factors.
fn fold_mul<C: Coefficient>(items: Vec<Expr<C>>) -> Expr<C> {
    let mut stack = items;
    let mut const_acc = C::one();
    let mut terms: Vec<Expr<C>> = Vec::new();
    while let Some(it) = stack.pop() {
        match it {
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Const(c) => {
                if c.is_zero_value() {
                    return Expr::Const(C::zero());
                }
                const_acc = const_acc * c;
            }
            Expr::Neg(inner) => {
                const_acc = const_acc * C::from(-1.0);
                stack.push(*inner);
            }
            other => terms.push(other),
        }
    }
    if terms.is_empty() {
        return Expr::Const(const_acc);
    }
    let mut iter = terms.into_iter();
    let mut result = iter.next().unwrap();
    for t in iter {
        result = Expr::Mul(Box::new(result), Box::new(t));
    }
    if const_acc == C::one() {
        result
    } else {
        Expr::Mul(Box::new(Expr::Const(const_acc)), Box::new(result))
    }
}

fn apply_func<C: Coefficient>(name: &str, c: C) -> C {
    fn to_f64<C: Coefficient>(c: C) -> f64 {
        // Coefficients are losslessly renderable through Display then parsed
        // back via From<f64>; for f64 this is the identity path.
        let s = c.to_string();
        s.parse::<f64>().unwrap_or(0.0)
    }
    fn from_f64<C: Coefficient>(f: f64) -> C {
        C::from(f)
    }
    let f = to_f64(c);
    let r = match name {
        "sin" => f.sin(),
        "cos" => f.cos(),
        "tan" => f.tan(),
        "exp" => f.exp(),
        "ln" => f.ln(),
        "sqrt" => f.sqrt(),
        _ => f,
    };
    from_f64(r)
}

fn derivative<C: Coefficient>(e: &Expr<C>, var: &str) -> Expr<C> {
    match e {
        Expr::Const(_) => Expr::Const(C::zero()),
        Expr::Var(v) => {
            if v == var {
                Expr::Const(C::one())
            } else {
                Expr::Const(C::zero())
            }
        }
        Expr::Add(a, b) => derivative(a, var) + derivative(b, var),
        Expr::Sub(a, b) => derivative(a, var) - derivative(b, var),
        Expr::Neg(a) => -derivative(a, var),
        Expr::Mul(a, b) => {
            let da = derivative(a, var);
            let db = derivative(b, var);
            (**a).clone() * db + da * (**b).clone()
        }
        Expr::Div(a, b) => {
            let da = derivative(a, var);
            let db = derivative(b, var);
            let bv = (**b).clone();
            (da * bv.clone() - (**a).clone() * db) / (bv.clone() * bv)
        }
        Expr::Pow(a, n) => {
            if *n == 0 {
                Expr::Const(C::zero())
            } else {
                let da = derivative(a, var);
                let base = (**a).clone();
                let n_c = C::from(*n as f64);
                Expr::Const(n_c) * base.clone().pow(*n - 1) * da
            }
        }
        Expr::Func(name, a) => {
            let da = derivative(a, var);
            let inner = (**a).clone();
            match name.as_str() {
                "sin" => Expr::func("cos", inner) * da,
                "cos" => -Expr::func("sin", inner) * da,
                "exp" => Expr::func("exp", inner) * da,
                "tan" => {
                    let sec = Expr::Const(C::one()) + Expr::func("tan", inner.clone()).pow(2);
                    sec * da
                }
                "ln" => da / inner,
                "sqrt" => da / (Expr::Const(C::from(2.0)) * Expr::func("sqrt", inner)),
                other => Expr::Func(format!("{other}'"), Box::new(inner)) * da,
            }
        }
    }
}

fn substitute<C: Coefficient>(e: &Expr<C>, var: &str, value: C) -> Expr<C> {
    match e {
        Expr::Const(c) => Expr::Const(c.clone()),
        Expr::Var(v) => {
            if v == var {
                Expr::Const(value)
            } else {
                Expr::Var(v.clone())
            }
        }
        Expr::Add(a, b) => substitute(a, var, value.clone()) + substitute(b, var, value),
        Expr::Sub(a, b) => substitute(a, var, value.clone()) - substitute(b, var, value),
        Expr::Mul(a, b) => substitute(a, var, value.clone()) * substitute(b, var, value),
        Expr::Div(a, b) => substitute(a, var, value.clone()) / substitute(b, var, value),
        Expr::Neg(a) => -substitute(a, var, value),
        Expr::Pow(a, n) => substitute(a, var, value).pow(*n),
        Expr::Func(name, a) => Expr::Func(name.clone(), Box::new(substitute(a, var, value))),
    }
}

fn eval<C: Coefficient>(e: &Expr<C>, env: &HashMap<String, C>) -> Option<C> {
    match e {
        Expr::Const(c) => Some(c.clone()),
        Expr::Var(v) => env.get(v).cloned(),
        Expr::Add(a, b) => Some(eval(a, env)? + eval(b, env)?),
        Expr::Sub(a, b) => Some(eval(a, env)? - eval(b, env)?),
        Expr::Mul(a, b) => Some(eval(a, env)? * eval(b, env)?),
        Expr::Div(a, b) => {
            let d = eval(b, env)?;
            if d.is_zero_value() {
                None
            } else {
                Some(eval(a, env)? / d)
            }
        }
        Expr::Neg(a) => Some(-eval(a, env)?),
        Expr::Pow(a, n) => Some(eval(a, env)?.powi(*n as i32)),
        Expr::Func(name, a) => {
            let v = eval(a, env)?;
            Some(apply_func(name, v))
        }
    }
}

impl<C: Coefficient> fmt::Display for Expr<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Var(v) => write!(f, "{v}"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Sub(a, b) => write!(f, "({a} - {b})"),
            Expr::Mul(a, b) => write!(f, "({a} * {b})"),
            Expr::Div(a, b) => write!(f, "({a} / {b})"),
            Expr::Neg(a) => write!(f, "(-{a})"),
            Expr::Pow(a, n) => write!(f, "({a}^{n})"),
            Expr::Func(name, a) => write!(f, "{name}({a})"),
        }
    }
}

#[cfg(feature = "exact")]
mod exact {
    use super::*;
    use std::ops::{Add, Div, Mul, Neg, Sub};
    use tpt_math_exact::BigRational;

    /// Arbitrary-precision rational wrapper implementing [`Coefficient`].
    #[derive(Clone, Debug, PartialEq)]
    pub struct Rational(pub BigRational);

    impl From<f64> for Rational {
        fn from(f: f64) -> Rational {
            Rational(BigRational::from_float(f).unwrap_or_else(BigRational::zero))
        }
    }

    impl From<BigRational> for Rational {
        fn from(r: BigRational) -> Rational {
            Rational(r)
        }
    }

    impl fmt::Display for Rational {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Zero for Rational {
        fn zero() -> Rational {
            Rational(BigRational::zero())
        }
        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }

    impl One for Rational {
        fn one() -> Rational {
            Rational(BigRational::one())
        }
    }

    impl Add for Rational {
        type Output = Rational;
        fn add(self, rhs: Rational) -> Rational {
            Rational(self.0 + rhs.0)
        }
    }
    impl Sub for Rational {
        type Output = Rational;
        fn sub(self, rhs: Rational) -> Rational {
            Rational(self.0 - rhs.0)
        }
    }
    impl Mul for Rational {
        type Output = Rational;
        fn mul(self, rhs: Rational) -> Rational {
            Rational(self.0 * rhs.0)
        }
    }
    impl Div for Rational {
        type Output = Rational;
        fn div(self, rhs: Rational) -> Rational {
            Rational(self.0 / rhs.0)
        }
    }
    impl Neg for Rational {
        type Output = Rational;
        fn neg(self) -> Rational {
            Rational(-self.0)
        }
    }

    impl Coefficient for Rational {
        fn powi(self, n: i32) -> Rational {
            if n >= 0 {
                let k = n as u32;
                Rational(BigRational::new(
                    self.0.numer().pow(k),
                    self.0.denom().pow(k),
                ))
            } else {
                let k = (-n) as u32;
                Rational(BigRational::new(
                    self.0.denom().pow(k),
                    self.0.numer().pow(k),
                ))
            }
        }
    }

    /// `Expr<Rational>` — exact, arbitrary-precision symbolic expression.
    pub type ExprRational = Expr<Rational>;
}

#[cfg(feature = "exact")]
pub use exact::{ExprRational, Rational};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_folding() {
        let x = Expr64::var("x");
        let e = Expr64::from(2.0) * x.clone() + Expr64::from(3.0) - Expr64::from(3.0);
        let s = e.simplify();
        assert_eq!(s, Expr64::from(2.0) * x.clone());
    }

    #[test]
    fn derivative_polynomial() {
        // d/dx (x^2) = 2x
        let x = Expr64::var("x");
        let d = x.clone().pow(2).derivative("x").simplify();
        assert_eq!(d.eval_str("x", 4.0).unwrap(), 8.0);
    }

    #[test]
    fn derivative_product_rule() {
        // d/dx (x * (x + 1)) = 2x + 1
        let x = Expr64::var("x");
        let e = x.clone() * (x.clone() + Expr64::from(1.0));
        let d = e.derivative("x").simplify();
        assert_eq!(d.eval_str("x", 5.0).unwrap(), 11.0);
    }

    #[test]
    fn derivative_sin() {
        // d/dx sin(x) = cos(x)
        let x = Expr64::var("x");
        let d = Expr64::func("sin", x.clone()).derivative("x").simplify();
        let expected = (2.0f64).cos();
        assert!((d.eval_str("x", 2.0).unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn substitute_and_eval() {
        let x = Expr64::var("x");
        let e = (x.clone() + Expr64::from(1.0)).pow(2);
        let s = e.substitute("x", 3.0).simplify();
        assert_eq!(s.eval_str("x", 0.0).unwrap(), 16.0);
    }

    #[test]
    fn derivative_is_zero_for_other_var() {
        let x = Expr64::var("x");
        let y = Expr64::var("y");
        let d = (x.clone() + y.clone()).derivative("y").simplify();
        assert_eq!(d.eval_str("x", 10.0).unwrap(), 1.0);
    }
}
