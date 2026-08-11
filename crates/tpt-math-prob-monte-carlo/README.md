# tpt-math-prob-monte-carlo

Monte Carlo estimation routines: crude Monte Carlo integration with a standard
error, mean/variance helpers over drawn samples, and importance sampling for
variance reduction. This crate consolidates the prior TPT
`tpt-zero-monte-carlo` work into the `tpt-math` probability stack; its only
dependency is `tpt-math-prob-core`.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is a leaf crate
in the probability layer: every routine is generic over the `Rng` and
`Distribution` traits from `tpt-math-prob-core`, and it is re-exported by the
`tpt-math-prob` umbrella. It is not an umbrella crate.

## Features

- `default = []` — the crate declares no optional features; the full API is
  always available.
- **`std` required.** This crate is not `no_std`: estimators buffer samples in
  `Vec` and use `std` float math.

## Quick start

```toml
[dependencies]
tpt-math-prob-monte-carlo = "0.1"
tpt-math-prob-core = "0.1"   # for Standard and a generator such as SplitMix64
```

```rust
use tpt_math_prob_core::{SplitMix64, Standard};
use tpt_math_prob_monte_carlo::{estimate_mean, importance, integrate, mean_and_var};

let mut rng = SplitMix64::seed_from_u64(0);

// Crude Monte Carlo: ∫_0^1 x² dx = 1/3, returned as (mean, stderr).
let (mean, stderr) = integrate(&mut rng, |x| x * x, 0.0, 1.0, 100_000);
assert!((mean - 1.0 / 3.0).abs() < 10.0 * stderr);

// Mean of a distribution, with the standard error of the estimate.
let (m, se) = estimate_mean(&Standard, &mut rng, 200_000);
assert!((m - 0.5).abs() < 10.0 * se);

// Importance sampling: identical target and proposal densities give
// weight 1, i.e. a zero-variance estimator of E[1].
let (est, _se) = importance(&mut rng, 50_000, &Standard, |_x| 1.0, |_x| 1.0, |_x| 1.0);
assert!((est - 1.0).abs() < 1e-12);

// Summary statistics over an existing sample (Bessel-corrected variance).
let (mu, var) = mean_and_var(&[1.0, 2.0, 3.0, 4.0, 5.0]);
assert!((mu - 3.0).abs() < 1e-12 && (var - 2.5).abs() < 1e-12);
```

`ImportanceSampling::new(target_density, proposal_density)` plus
`.estimate(rng, proposal, h, n)` is the struct form of the `importance`
convenience function.

## Notes

- Every estimator returns `(estimate, stderr)`. `integrate` uses
  `I = (b - a) · mean f(xᵢ)` with `xᵢ` uniform on `[a, b]` and reports
  `(b - a) · sqrt(var(f) / n)`; the error decays as `1 / sqrt(n)`.
- `mean_and_var` applies Bessel's correction for `n > 1` and returns
  `(0.0, 0.0)` for an empty slice and `(x, 0.0)` for a single sample. `n == 0`
  likewise yields `(0.0, 0.0)` from the estimators.
- Importance sampling reweights by `f(x) / g(x)`, so the proposal must cover the
  target's support and its density must be strictly positive there; a proposal
  concentrated where `h` is large is what buys the variance reduction.
- Sampling is generic over `Rng`, so a seeded `SplitMix64` makes an estimate
  exactly reproducible.

## License

Licensed under either of MIT or Apache-2.0 at your option.
