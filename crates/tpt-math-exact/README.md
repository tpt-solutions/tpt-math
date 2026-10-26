# tpt-math-exact

Exact rational and interval arithmetic for the `tpt-math` substrate. It
builds on [`num-bigint`](https://crates.io/crates/num-bigint) to provide its own
arbitrary-precision `BigRational`, and layers a generic `Interval<T>` on top so
that rigorous bounds can be computed without ever accumulating floating-point
rounding error.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is the "exact"
half of the exact-and-linear-algebra layer, and the crate `tpt-formal` reaches
for when arithmetic has to be provably correct rather than merely fast. It is
not an umbrella crate; inside the workspace it depends only on
`tpt-math-numeric`, which it re-exports as `tpt_math_exact::numeric`.

## Features

- `std` *(default)* — enables the `std` feature of `tpt-math-numeric`,
  `num-bigint`, `num-integer` and `num-traits`.
- `alloc` — enables `tpt-math-numeric/alloc` for `no_std + alloc` consumers.
- `no_std` support: the crate is `#![no_std]`, but it always declares
  `extern crate alloc` because arbitrary-precision integers allocate. An
  allocator is therefore required even with `default-features = false`.

## Quick start

```toml
[dependencies]
tpt-math-exact = "0.1"
```

```rust
use tpt_math_exact::{Interval, Rational};

fn rat(n: i64) -> Rational {
    Rational::new(n.into(), 1.into())
}

// Exact rationals: 1/3 + 1/6 is exactly 1/2, with no rounding.
let third = Rational::new(1.into(), 3.into());
let sixth = Rational::new(1.into(), 6.into());
assert_eq!(third + sixth, Rational::new(1.into(), 2.into()));

// Interval arithmetic over exact endpoints. Multiplication takes the
// min/max over all four corner products, so the result is a true enclosure.
let a = Interval::new(rat(-1), rat(2));
let b = Interval::new(rat(-3), rat(4));
let product = &a * &b;
assert_eq!(product.lo(), &rat(-6));
assert_eq!(product.hi(), &rat(8));

assert!(a.contains(&rat(0)));
assert_eq!(a.width(), rat(3));
assert_eq!(a.hull(&b), Interval::new(rat(-3), rat(4)));
assert_eq!(a.intersect(&b), Some(Interval::new(rat(-1), rat(2))));
```

`Rational` is an alias for `BigRational`; `BigInt` and `BigUint` are re-exported
too, so consumers normally need only this crate.

## Rational numbers

`BigRational` is always held in lowest terms with a positive denominator, so
equality and hashing agree with numeric equality. Beyond the arithmetic
operators it offers exact `from_float`, `pow`, `floor`/`ceil`/`trunc`/`fract`/
`round`, `FromStr` (`"3/4"`, `"-1.25e3"`), directed-rounding `to_f64_down` and
`to_f64_up`, `sqrt_bounds`, continued fractions with `limit_denominator`, and
`to_decimal_string`.

```rust
use tpt_math_exact::Rational;

let x: Rational = "-1.25e1".parse().unwrap();
assert_eq!(x, Rational::new((-25).into(), 2.into()));
assert!(x.to_f64_down() <= -12.5);
```

## Rigorous numerics

The `roots` module and `certify` give enclosures that provably contain the
answer:

- `roots::isolate_roots` bisects on an interval extension and returns boxes that
  provably contain every root. `poly_enclosure` builds a valid extension for a
  polynomial, and `poly_eval` and `has_sign_change` give point evaluation and a
  Bolzano certificate.
- `roots::newton_refine` runs interval Newton refinement of a root enclosure. It
  returns `None` when a box is proven root-free.
- `roots::validated_euler` and `roots::validated_euler_system` enclose the
  solution of a scalar ODE, or of a system, at the end of an interval. Each step
  uses an a-priori (Picard–Lindelöf) box.
- `certify::certified_solve` returns certified enclosures of the solution of a
  linear system with `f64` data, by the Krawczyk operator. The approximate
  inverse is only a preconditioner, and the contraction test and the enclosure
  use exact rational arithmetic. A system the method cannot contract is reported
  as `CertifyError::NotContracting`, which is not a claim that the system has no
  solution.

## Notes

- `Interval<T>` is generic: the endpoints only have to be `Clone + PartialOrd`
  for construction and `contains`, and `Ord` for `hull`, `intersect` and the
  arithmetic impls. It works with `i64` or `BigRational`; `f64` intervals are
  not supported by the `Ord`-bound methods, which is deliberate — this crate is
  about exactness.
- `Interval::new` panics when `lo > hi`, in release builds as well as debug.
- `Add`, `Sub` and `Mul` are implemented for both owned and borrowed intervals.
  Use `div_nonzero` for division. It is defined only when the divisor excludes
  zero, so dividing by an interval that straddles zero is left to the caller.
  Addition and subtraction use two endpoint operations each.
- `midpoint` additionally requires `Div` and `FromPrimitive` on the endpoint
  type, and is exact for rationals (no "average of two bignums" rounding).
- `num-bigint` and `num-integer` are dual-licensed `MIT OR Apache-2.0`, in line
  with the workspace policy.

## License

Licensed under either of MIT or Apache-2.0 at your option.
