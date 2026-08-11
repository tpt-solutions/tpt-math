//! Cross-crate example: `tpt-math-symbolic` + `tpt-math-exact`.
//!
//! Differentiates a polynomial symbolically (default `f64` coefficient) and
//! then repeats exact arithmetic with arbitrary-precision rationals
//! (`ExprRational`, backed by `tpt-math-exact`'s `BigRational`).

use std::collections::HashMap;

use tpt_math_exact::BigRational;
use tpt_math_symbolic::{Expr64, ExprRational, Rational, Symbolic};

fn main() {
    // f(x) = (3x + 1)^2  =>  f'(x) = 18x + 6,  f'(2) = 42.
    let x = Expr64::var("x");
    let expr = (Expr64::from(3.0) * x.clone() + Expr64::from(1.0)).pow(2);
    let derivative = expr.derivative("x").simplify();
    let at_two = derivative.eval_str("x", 2.0).unwrap();
    assert!((at_two - 42.0).abs() < 1e-9, "f'(2) = {at_two}");
    println!("symbolic: d/dx (3x+1)^2 at x=2 = {at_two}");

    // Exact arithmetic: 1/3 + 2/3 = 1, with no floating-point error.
    let one_third = ExprRational::Const(Rational(BigRational::new(1u8.into(), 3u8.into())));
    let two_thirds = ExprRational::Const(Rational(BigRational::new(2u8.into(), 3u8.into())));
    let sum = (one_third + two_thirds).simplify();
    let result = sum.eval(&HashMap::new()).unwrap();
    let one = BigRational::new(1u8.into(), 1u8.into());
    assert_eq!(result.0, one, "1/3 + 2/3 should be exactly 1");
    println!("exact: 1/3 + 2/3 = {}", result.0);
}
