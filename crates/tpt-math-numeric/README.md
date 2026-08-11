# tpt-math-numeric

Scalar numeric trait glue for the `tpt-math` substrate. This crate is a thin
wrapper over [`num-traits`](https://crates.io/crates/num-traits) (and,
optionally, [`libm`](https://crates.io/crates/libm)) that re-exports the numeric
traits the rest of the workspace builds on and adds a single `Scalar`
supertrait meaning "a floating-point scalar type we can do math on".

## Part of tpt-math

`tpt-math-numeric` is the bottom layer of
[tpt-math](https://github.com/tpt-solutions/tpt-math): every other crate in the
workspace either depends on it directly or is generic over the traits it
re-exports. It deliberately contains no algorithms — only the trait vocabulary
(`Float`, `Num`, `NumCast`, `FloatConst`, `Real`, the `Checked*` family, …) so
that higher layers agree on one numeric hierarchy instead of each importing
`num-traits` with its own feature set.

## Features

- `std` *(default)* — enables `num-traits`' `std`-dependent items.
- `alloc` — signals that an allocator is available; adds no dependencies.
- `libm` — re-exports `libm` so callers can do transcendental math on `no_std`
  targets.

The crate is `#![no_std]` unconditionally; the `std` feature only widens what
`num-traits` itself offers. `num-traits` is always built with its `libm`
backend, so float math works without `std`.

## Quick start

```toml
[dependencies]
tpt-math-numeric = "0.1"
```

```rust
use tpt_math_numeric::prelude::*;

// Generic over any float: f32, f64, or a downstream scalar newtype.
fn midpoint<T: Scalar>(a: T, b: T) -> T {
    (a + b) / (T::one() + T::one())
}

fn circle_area<T: Scalar>(radius: T) -> T {
    T::PI() * radius * radius
}

assert_eq!(midpoint(2.0_f64, 4.0_f64), 3.0_f64);
assert!((circle_area(1.0_f32) - core::f32::consts::PI).abs() < 1e-6);
```

`Scalar` is a blanket supertrait: `impl<T: Float + NumCast + FloatConst> Scalar
for T {}`, so `f32` and `f64` satisfy it with no extra work. The `prelude`
module re-exports `Float`, `FloatConst`, `Num`, `NumCast`, `One`, `Real`,
`Scalar`, `Signed`, `Unsigned` and `Zero`; the crate root additionally
re-exports the checked-arithmetic traits, `Pow`, `Inv`, `PrimInt`,
`FromPrimitive`/`ToPrimitive`, the `cast` module and `num_traits` itself.

## Notes

- On `no_std` targets, disable default features and enable `libm` if you need
  `libm`'s free functions directly:
  `tpt-math-numeric = { version = "0.1", default-features = false, features = ["libm"] }`.
- `Scalar` intentionally requires `Float`, so integer types do **not** implement
  it; use the re-exported `Num`/`PrimInt` bounds for integer-generic code.
- Upstream `num-traits` is `MIT OR Apache-2.0` and the optional `libm` is MIT,
  both compatible with this crate's dual license.

## License

Licensed under either of MIT or Apache-2.0 at your option.
