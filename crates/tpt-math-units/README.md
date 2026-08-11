# tpt-math-units

Compile-time typed units for the `tpt-math` substrate. This crate is a thin
wrap over [`uom`](https://crates.io/crates/uom): quantities such as `Length`,
`Mass` and `Time` carry their dimension in the type system, so dividing a
`Length` by a `Time` yields a `Velocity` and adding a `Mass` to a `Length` is a
compile error rather than a runtime surprise.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It sits in the
scalar-and-units layer, just above `tpt-math-numeric`: `tpt-math-linalg` uses
its quantity types as the phantom unit tags on vectors and matrices, and
`tpt-math-units-dyn` bridges them to units that are only known at runtime. It
is not an umbrella crate — its only third-party dependencies are `uom` and
`num-traits`.

## Features

- `std` *(default)* — enables `num-traits/std`. It does **not** enable `uom`'s
  `std` feature.
- `alloc` — signals that an allocator is available; adds no dependencies.
- `no_std` support: the crate is `#![no_std]` and `uom` is always built with
  `default-features = false` (features `si`, `f32`, `f64`, `autoconvert`), so
  `default-features = false` gives a clean `no_std` build.

`autoconvert` is on, which is what allows quantities expressed in different
units of the same dimension to be combined directly.

## Quick start

```toml
[dependencies]
tpt-math-units = "0.1"
```

```rust
use tpt_math_units::prelude::*;               // f64-backed SI quantity types
use tpt_math_units::si::length::kilometer;
use tpt_math_units::si::time::second;
use tpt_math_units::si::velocity::kilometer_per_hour;

let distance = Length::new::<kilometer>(3.0);
let duration = Time::new::<second>(90.0);

// Length / Time = Velocity, enforced by the type system.
let speed: Velocity = distance / duration;
assert!((speed.get::<kilometer_per_hour>() - 120.0).abs() < 1e-9);

// Conversion happens on the way in and on the way out, never in between.
assert!((Length::new::<kilometer>(1.0).get::<tpt_math_units::si::length::meter>() - 1000.0).abs() < 1e-9);
```

The public surface is three modules plus a re-export:

- `uom` — the whole upstream API, unmodified.
- `si` — re-export of `uom::si` (systems, quantities, unit markers such as
  `si::length::kilometer`).
- `prelude` — re-export of `uom::si::f64::*`, i.e. the concrete `f64` quantity
  types (`Length`, `Mass`, `Time`, `Velocity`, …).
- `q` — shorthand re-exports of the generic quantity aliases (`Area`, `Length`,
  `Mass`, `Ratio`, `ThermodynamicTemperature`, `Time`, `Velocity`, `Volume`)
  from their `uom::si::*` modules.

## Notes

- `prelude` gives the concrete `f64` types; the aliases in `q` are the generic
  `Quantity<U, V>` forms, so `q::Length` still needs its unit-system and
  storage-type parameters. Use `prelude` unless you are writing code generic
  over the storage type.
- `f32` and `f64` storage are both enabled upstream; other storage types
  (integers, rationals) are not, to keep the dependency surface small.
- Because `uom`'s `std` feature is never enabled, transcendental operations on
  quantities go through the `no_std` math path on every target — behaviour is
  therefore identical with and without this crate's `std` feature.
- `uom` is dual-licensed `MIT OR Apache-2.0`, matching this workspace's policy.

## License

Licensed under either of MIT or Apache-2.0 at your option.
