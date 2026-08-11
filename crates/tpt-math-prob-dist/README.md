# tpt-math-prob-dist

Standard probability distributions for the `tpt-math` stack. This crate wraps
[`rand_distr`](https://crates.io/crates/rand_distr) and bridges it into the
minimal trait ecosystem of `tpt-math-prob-core`, so any `rand_distr`
distribution can be sampled through the workspace's own `Rng`/`Distribution`
traits without pulling in `rand`'s generators.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It sits one layer
above `tpt-math-prob-core` (whose traits it implements) and is re-exported by
the `tpt-math-prob` umbrella. It is not an umbrella crate itself: its only job
is the `rand_distr` bridge plus ergonomic constructors.

## Features

- `std` *(default)* — turns on `alloc` and enables `rand/std`,
  `rand_distr/std` and `tpt-math-prob-core/std`.
- `alloc` — enables `rand/alloc`, `rand_distr/alloc` and
  `tpt-math-prob-core/alloc` for allocator-only targets.
- `no_std` support: the crate is `#![no_std]` and `#![forbid(unsafe_code)]`.
  Build with `default-features = false` for bare-metal targets; float math is
  delegated to `rand_distr` (which falls back to `num-traits`/`libm`).

## Quick start

```toml
[dependencies]
tpt-math-prob-dist = "0.1"
```

```rust
use tpt_math_prob_dist::{normal, poisson, uniform, AsU64, Dist, Distribution, SplitMix64};

let mut rng = SplitMix64::seed_from_u64(0);

// Constructors return `Result<_, &'static str>` on invalid parameters.
// Wrap a `rand_distr` distribution in `Dist` to sample it through our `Rng`.
let gauss = Dist::new(normal((0.0, 1.0)).unwrap());
let x: f64 = gauss.sample(&mut rng);
assert!(x.is_finite());

let unit = Dist::new(uniform(0.0_f64, 1.0).unwrap());
let u: f64 = unit.sample(&mut rng);
assert!((0.0..1.0).contains(&u));

// `AsU64` adapts float-valued count distributions (e.g. Poisson) to `u64`.
let counts = AsU64(poisson(4.0).unwrap());
let n: u64 = counts.sample(&mut rng);
let _ = (x, u, n);
```

Constructors cover `normal`, `lognormal`, `exp`, `gamma`, `beta`,
`chi_squared`, `student_t`, `cauchy`, `poisson` and `uniform`. `Dist` works for
*every* `rand_distr` distribution, not only those with a constructor here, and
the raw `rand_distr` module is re-exported for direct use.

## Notes

- `CoreRng` is the adapter in the other direction: it turns any
  `tpt_math_prob_core::Rng` into a `rand::RngCore`, so `rand_distr`
  distributions can be sampled directly
  (`dist.sample(&mut CoreRng(&mut rng))`). It is zero-cost and stateless.
- The `rand` version matters: `rand_distr` 0.5 is built against `rand` 0.9, and
  `CoreRng` implements *that* `RngCore`. Mixing a different `rand` major
  version in the same graph will not type-check.
- Constructor errors are flattened to `&'static str` to keep the public surface
  allocation-free and `no_std`-friendly; use `rand_distr`'s own `new` if you
  need the typed error.
- The core traits (`Distribution`, `Rng`, `Sampler`, `SplitMix64`, `Standard`)
  are re-exported, so depending on this crate alone is enough.

## License

Licensed under either of MIT or Apache-2.0 at your option.
