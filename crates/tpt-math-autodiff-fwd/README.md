# tpt-math-autodiff-fwd

Forward-mode automatic differentiation via dual numbers. A `Dual<T, N>` pairs a
value with a vector of `N` first derivatives, one per independent variable, so
evaluating a function on duals yields both its value and its gradient in a
single forward pass. This crate wraps no third-party autodiff library — the
dual-number arithmetic is implemented here directly on top of the numeric
traits from `tpt-math-numeric`.

## Part of tpt-math

This crate is a member of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
workspace. It is the lowest layer of the autodiff stack: it depends only on
`tpt-math-numeric` (`Float`, `One`, `Zero`), underpins the tape-based
`tpt-math-autodiff-rev`, and is re-exported by the `tpt-math-autodiff` umbrella
crate.

## Features

- `std` *(default)* — enables `tpt-math-numeric/std`.
- `alloc` — enables `tpt-math-numeric/alloc`.

The crate itself is `#![no_std]` and allocation-free: derivatives live in a
fixed-size `[T; N]` array inside `Dual`, so it builds and runs unchanged with
`default-features = false`. Transcendental functions come from the `Float`
trait, which resolves to `libm` in `no_std` builds.

## Quick start

```toml
[dependencies]
tpt-math-autodiff-fwd = "0.1"
```

Scalar derivative (the default `N = 1` is the classic dual number):

```rust
use tpt_math_autodiff_fwd::Dual;

// f(x) = x^3 + 2x at x = 3; f(3) = 33, f'(3) = 3x^2 + 2 = 29.
let x = Dual::<f64>::variable(3.0, 0);
let y = x * x * x + x * Dual::constant(2.0);
assert!((y.re() - 33.0).abs() < 1e-12);
assert!((y.du(0) - 29.0).abs() < 1e-12);
```

Full gradient in one pass with `N > 1`:

```rust
use tpt_math_autodiff_fwd::Dual;

// f(x, y) = x*y + x at (2, 3); df/dx = y + 1 = 4, df/dy = x = 2.
let x = Dual::<f64, 2>::variable(2.0, 0);
let y = Dual::<f64, 2>::variable(3.0, 1);
let f = x * y + x;
assert_eq!(f.re(), 8.0);
assert_eq!(f.deriv(), &[4.0, 2.0]);
```

Transcendentals propagate exactly:

```rust
use tpt_math_autodiff_fwd::Dual;

let x = Dual::<f64>::variable(0.0, 0);
let y = x.sin();          // d/dx sin(x) = cos(x) = 1 at x = 0
assert!((y.du(0) - 1.0).abs() < 1e-12);
```

## Notes

- Constructors: `Dual::constant(re)` (zero derivative everywhere),
  `Dual::variable(re, idx)` (unit derivative in direction `idx`; an out-of-range
  `idx` is ignored and yields a constant), and `Dual::new(re, du)` for an
  explicit derivative vector.
- Accessors: `re()` for the primal value, `du(idx)` for one partial, and
  `deriv()` for the whole `&[T; N]`.
- Supported operations: `Add`, `Sub`, `Mul`, `Div` (operator impls), the
  `negate()` method, and `sin`, `cos`, `exp`, `ln` for `T: Float`. There is no
  `Neg` operator impl — use `negate()`. Higher-order derivatives are not
  provided; nest `Dual<Dual<T, N>, M>` only if the inner type satisfies the
  required bounds.
- Cost scales with `N`: forward mode computes one directional derivative per
  variable, so it is the right tool for few-input / many-output problems.
  Prefer `tpt-math-autodiff-rev` for many-input gradients.
- `#![no_std]` with no allocator requirement; `Dual` is `Copy` whenever `T` is.
- Dual-licensed `MIT OR Apache-2.0`, matching the rest of the workspace.

## License

Licensed under either of MIT or Apache-2.0 at your option.
