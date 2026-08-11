# tpt-math-prob-sampler

Reusable sampling *strategies* — inverse-transform categorical draws, particle
resampling, streaming reservoirs, and rejection sampling — built on the shared
`tpt-math-prob-core` traits. It consolidates the sampling machinery previously
spread across the `tpt-zero-sampler` / `tpt-zero-rand` crates behind one
`no_std`-friendly, dependency-light API.

## Part of tpt-math

This crate is a member of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
workspace, in the probability layer. It sits directly on top of
`tpt-math-prob-core` (the `Rng` / `Distribution` / `Sampler` traits) and is
re-exported as the `sampler` module of the `tpt-math-prob` umbrella crate. It
depends on no third-party crates.

## Features

- `std` *(default)* — enables `tpt-math-prob-core/std`, the owned
  `alloc::vec::Vec`-backed constructors, and `impl std::error::Error for
  SamplerError`.
- `alloc` — the same owned constructors (`WeightedIndex`, `categorical`,
  `InverseCdfSampler::from_weights`, `SystematicResampler::from_weights`)
  without pulling in `std`.
- No features at all — the crate is `#![no_std]` and the whole allocation-free
  surface (`InverseCdfSampler::from_cdf` over any `AsRef<[f64]>` storage,
  `SystematicResampler::new`, `ReservoirSampler`, `RejectionSampler`,
  `Uniform`, `Bernoulli`, `sample_categorical`, `uniform_index`, `shuffle`)
  remains available.

## Quick start

```toml
[dependencies]
tpt-math-prob-sampler = "0.1"
```

```rust
use tpt_math_prob_sampler::{categorical, ResampleScheme, SystematicResampler};
use tpt_math_prob_sampler::{Distribution, SplitMix64};

let mut rng = SplitMix64::seed_from_u64(0);

// Weighted categorical draw via the inverse-CDF (alias) sampler.
let cat = categorical(&[1.0, 2.0, 3.0]).unwrap();
let i = cat.sample(&mut rng);
assert!(i < 3);

// Low-variance systematic resampling of 5 particle indices.
let resampler = SystematicResampler::new(cat.cdf().to_vec()).unwrap();
let mut out = [0usize; 5];
resampler.sample_indices(&mut rng, ResampleScheme::Systematic, &mut out);
assert!(out.iter().all(|&i| i < 3));
```

Allocation-free, `no_std`-only usage (`default-features = false`):

```rust
use tpt_math_prob_sampler::{sample_categorical, shuffle, ReservoirSampler, SplitMix64};

let mut rng = SplitMix64::seed_from_u64(7);

let i = sample_categorical(&[1.0, 1.0, 2.0], &mut rng).unwrap();
assert!(i < 3);

let mut slots = [0u32; 3];
let mut reservoir = ReservoirSampler::new(&mut slots);
for x in 0u32..100 {
    reservoir.offer(x, &mut rng);
}

let mut deck = [1, 2, 3, 4, 5];
shuffle(&mut deck, &mut rng);
```

## Notes

- The crate is `#![no_std]`; `extern crate alloc` is always declared, but only
  the owned-`Vec` constructors actually require an allocator, so
  `default-features = false` gives a fully allocation-free build.
- Float helpers are hand-rolled (`fabs`, an unbiased rejection-based
  `uniform_index`) rather than taken from `std`, so behaviour is identical on
  and off `std`.
- `Distribution`, `Rng`, `Sampler`, `SplitMix64` and `Standard` are re-exported
  from `tpt-math-prob-core`, so consumers normally need only this crate.
- All sampling is generic over `Rng`; seeding `SplitMix64` makes an entire run
  bit-for-bit reproducible.
- `RejectionSampler`'s `Distribution<f64>` impl is guaranteed to terminate: it
  falls back to a proposal draw after `max_iterations` (default `1024`)
  rejections. Use `try_sample` when you need to observe that failure.
- Dual-licensed `MIT OR Apache-2.0`, matching the rest of the workspace.

## License

Licensed under either of MIT or Apache-2.0 at your option.
