# tpt-math-exact

Exact rational and interval arithmetic for the `tpt-math` substrate. It
thin-wraps [`num-bigint`](https://crates.io/crates/num-bigint) and
[`num-rational`](https://crates.io/crates/num-rational) to expose
arbitrary-precision rationals, and layers a generic `Interval<T>` on top so
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
  `num-bigint`, `num-rational` and `num-traits`.
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

## Notes

- `Interval<T>` is generic: the endpoints only have to be `Clone + PartialOrd`
  for construction and `contains`, and `Ord` for `hull`, `intersect` and the
  arithmetic impls. It works with `i64` or `BigRational`; `f64` intervals are
  not supported by the `Ord`-bound methods, which is deliberate — this crate is
  about exactness.
- `Interval::new` only `debug_assert!`s that `lo <= hi`, so a release build will
  silently construct an inverted interval if you hand it one.
- `Add`, `Sub` and `Mul` are implemented for both owned and borrowed intervals.
  There is **no** `Div`: dividing by an interval that straddles zero has no
  single-interval answer, so that decision is left to the caller.
- `midpoint` additionally requires `Div` and `FromPrimitive` on the endpoint
  type, and is exact for rationals (no "average of two bignums" rounding).
- `num-bigint` and `num-rational` are dual-licensed `MIT OR Apache-2.0`, in line
  with the workspace policy.

## License

Licensed under either of MIT or Apache-2.0 at your option.
