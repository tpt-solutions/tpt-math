# tpt-math-symbolic

Permissive-license symbolic math (a small computer-algebra core). It provides an
expression tree `Expr<C>` parameterised over a numeric `Coefficient`, with
algebraic simplification, symbolic differentiation, substitution and evaluation.
It consolidates the prior TPT `tpt-sym` engine into a dependency-light,
dual-licensed crate whose only mandatory dependency is `num-traits`.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
It is a leaf of the symbolic layer: nothing else in the workspace depends on it,
and it depends on nothing except `num-traits` plus, optionally,
`tpt-math-exact` for arbitrary-precision rational coefficients.

## Features

- `std` *(default)* — the crate as shipped. Simplification, differentiation and
  evaluation currently use `std` types (`String`, `Box`, `HashMap` for the
  evaluation environment) and `f64` transcendentals for `sin`/`cos`/`tan`/
  `exp`/`ln`/`sqrt` folding, so **std is required**.
- `alloc` — reserved for a future `no_std` + `alloc` build; it is currently a
  no-op and does not by itself make the crate build without `std`.
- `exact` — pulls in `tpt-math-exact` and adds the `Rational` coefficient (an
  arbitrary-precision `BigRational` wrapper) plus the `ExprRational` alias, for
  exact rational symbolic algebra.

## Quick start

```toml
[dependencies]
tpt-math-symbolic = "0.1"
```

Build `(3x + 1)^2`, differentiate it with respect to `x`, simplify, and evaluate:

```rust
use tpt_math_symbolic::{Expr64, Symbolic};

let x = Expr64::var("x");
let expr = (Expr64::from(3.0) * x.clone() + Expr64::from(1.0)).pow(2);
let d = expr.derivative("x").simplify();

assert_eq!(d.eval_str("x", 2.0).unwrap(), 42.0);
```

`Expr64` is `Expr<f64>`, the default floating-point expression type. Named unary
functions are built with `Expr::func` and differentiate via the chain rule
(`sin`, `cos`, `tan`, `exp`, `ln`, `sqrt` are known; anything else derives to a
formal `name'(...)` factor):

```rust
use tpt_math_symbolic::{Expr64, Symbolic};

let x = Expr64::var("x");
let d = Expr64::func("sin", x).derivative("x").simplify();

assert!((d.eval_str("x", 2.0).unwrap() - 2.0_f64.cos()).abs() < 1e-9);
```

With `features = ["exact"]` the same tree can carry exact coefficients via
`ExprRational` (`Expr<Rational>`), evaluated through `Symbolic::eval` with a
`HashMap<String, Rational>` environment.

## Notes

- The `Coefficient` trait is the extension point: implement it (a field-like
  type with `Clone + PartialEq + Debug + Display + From<f64> + Zero + One`, the
  four arithmetic operations, `Neg` and `powi`) to plug in your own number type.
  `f64` implements it out of the box.
- Function folding (`apply_func`) routes coefficients through `Display` and
  `f64` parsing, so transcendental folding on exact `Rational` coefficients is
  lossy — differentiation and rational arithmetic stay exact, but evaluating a
  `sin`/`ln`/`sqrt` node does not.
- `Expr::eval` returns `None` for an unbound variable or a division by an
  exactly-zero denominator, rather than panicking.
- `simplify` performs constant folding and sum/product flattening; it is not a
  full normal form and does not currently expand or collect like terms.
- Dual-licensed `MIT OR Apache-2.0`, matching the rest of the workspace, which is
  the point of the crate: it replaces copyleft-licensed CAS options. Its only
  mandatory dependency, `num-traits`, is `MIT OR Apache-2.0` too.

## License

Licensed under either of MIT or Apache-2.0 at your option.
