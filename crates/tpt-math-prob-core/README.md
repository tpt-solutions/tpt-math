# tpt-math-prob-core

The shared randomness vocabulary of the `tpt-math-prob-*` family. It wraps no
upstream crate: it defines the `Rng`, `Distribution<T>` and `Sampler<T>` traits
that every other probability crate in the workspace implements against, plus a
small deterministic `SplitMix64` generator so those traits are usable even
where `rand` is unavailable.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. This is the base
layer of the probability group: `tpt-math-prob-dist`, `-bayes`, `-markov`,
`-monte-carlo` and `-sampler` all sample through the traits defined here, and
the `tpt-math-prob` umbrella re-exports them. It is not an umbrella crate; it
depends only on `tpt-math-numeric`.

## Features

- `std` *(default)* — enables `tpt-math-numeric/std`. The crate's own code is
  `#![no_std]` regardless of features.
- `alloc` — enables `tpt-math-numeric/alloc` for consumers that have an
  allocator but no `std`.
- `no_std` support: the whole public API (traits, `Standard`, `SplitMix64`)
  works with `default-features = false` — no `std`, no `alloc`, no allocation.

`tpt-math-numeric` is re-exported as `tpt_math_prob_core::numeric` so downstream
crates can reach the scalar traits without adding a second dependency.

## Quick start

```toml
[dependencies]
tpt-math-prob-core = "0.1"
```

```rust
use tpt_math_prob_core::{Distribution, Rng, Sampler, SplitMix64, Standard};

let mut rng = SplitMix64::seed_from_u64(42);

// Draw one value: `Standard` implements `Distribution` for f64, f32,
// u64, u32, i64 and bool.
let x: f64 = Standard.sample(&mut rng);
assert!((0.0..=1.0).contains(&x));

// Every `Distribution` is also a `Sampler`, which adds bulk helpers that
// fill a caller-provided slice (no allocation).
let mut buf = [0u64; 8];
Standard.sample_slice(&mut rng, &mut buf);

// Implementing `Rng` only requires `next_u64`; `next_f64` has a default.
let bits = rng.next_u64();
let _ = (x, buf, bits);
```

Implement `Distribution<T>` for your own type and you get `Sampler<T>`
(`sample_one`, `sample_slice`) for free via the blanket impl.

## Notes

- `SplitMix64` is deterministic and **not** cryptographically secure. It exists
  for reproducible tests, simulations, and `no_std` targets; use a CSPRNG for
  anything security-sensitive.
- `&mut R` implements `Rng` whenever `R` does, so a borrowed generator can be
  handed to `&mut impl Rng` APIs without cloning state.
- The traits are deliberately minimal (one required method) so the same code
  compiles on bare-metal targets; the `rand`/`rand_distr` bridge lives in
  `tpt-math-prob-dist` (`CoreRng`), not here.

## License

Licensed under either of MIT or Apache-2.0 at your option.
