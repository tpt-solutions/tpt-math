# tpt-math-prob

Umbrella crate for the probability layer of `tpt-math`. It re-exports the five
`tpt-math-prob-*` crates behind Cargo features so downstream code can take one
dependency — `tpt-math-prob` — instead of wiring up distributions, Bayesian
updates, Markov chains, Monte Carlo estimation, and samplers individually.

## Part of tpt-math

This crate is a member of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
workspace and is the top of the probability layer: it contains no logic of its
own, only feature-gated `pub use` re-exports of its constituents.

| Feature | Re-exported as | Source crate |
|---------|----------------|--------------|
| `tpt-math-prob-dist` | `dist` | `tpt-math-prob-dist` — standard distributions bridging `rand_distr` |
| `tpt-math-prob-bayes` | `bayes` | `tpt-math-prob-bayes` — conjugate priors, Metropolis sampling |
| `tpt-math-prob-markov` | `markov` | `tpt-math-prob-markov` — Markov chains and HMMs |
| `tpt-math-prob-monte-carlo` | `monte_carlo` | `tpt-math-prob-monte-carlo` — Monte Carlo integration, importance sampling |
| `tpt-math-prob-sampler` | `sampler` | `tpt-math-prob-sampler` — sampling / resampling strategies |

The shared `Rng` / `Distribution` / `Sampler` traits all constituents implement
live in `tpt-math-prob-core`, which each of them re-exports.

## Features

- `tpt-math-prob-dist` *(default)* — enables the `dist` module.
- `tpt-math-prob-bayes` *(default)* — enables the `bayes` module.
- `tpt-math-prob-markov` *(default)* — enables the `markov` module.
- `tpt-math-prob-monte-carlo` *(default)* — enables the `monte_carlo` module.
- `tpt-math-prob-sampler` *(default)* — enables the `sampler` module.

Each feature name matches its crate name and simply activates the optional
dependency. All five are on by default; disable `default-features` and opt in
to keep the dependency tree small.

This umbrella crate is **not** `no_std`: `bayes`, `markov` and `monte_carlo`
require `std`. If you only need `no_std` pieces, depend on `tpt-math-prob-dist`
or `tpt-math-prob-sampler` directly.

## Quick start

Take only the pieces you need:

```toml
[dependencies]
tpt-math-prob = { version = "0.1", default-features = false, features = [
    "tpt-math-prob-bayes",
    "tpt-math-prob-sampler",
] }
```

```rust
use tpt_math_prob::bayes::Beta;
use tpt_math_prob::sampler::{categorical, Distribution, SplitMix64};

// Bayes: uniform prior over a coin's bias, then 7 heads and 3 tails.
let posterior = Beta::uniform().update(7, 3);
assert_eq!((posterior.alpha(), posterior.beta()), (8.0, 4.0));

// Sampler: a weighted categorical draw from a seeded generator.
let mut rng = SplitMix64::seed_from_u64(0);
let cat = categorical(&[1.0, 2.0, 3.0]).unwrap();
assert!(cat.sample(&mut rng) < 3);
```

With default features every module is present:

```rust
use tpt_math_prob::{bayes, dist, markov, monte_carlo, sampler};
```

## Notes

- The re-exported module names are `snake_case` (`monte_carlo`) while the
  feature names are the `kebab-case` crate names (`tpt-math-prob-monte-carlo`).
- Types are re-exported verbatim, so `tpt_math_prob::sampler::Foo` and
  `tpt_math_prob_sampler::Foo` are the same type; mixing the umbrella and a
  direct dependency is safe.
- Constituent crates keep their own features. Because this crate depends on
  them with their defaults, `tpt-math-prob-dist` and `tpt-math-prob-sampler`
  arrive with `std` enabled.
- Dual-licensed `MIT OR Apache-2.0`, matching the rest of the workspace.

## License

Licensed under either of MIT or Apache-2.0 at your option.
