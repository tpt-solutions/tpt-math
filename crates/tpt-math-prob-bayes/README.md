# tpt-math-prob-bayes

Bayesian inference primitives: exact conjugate updates where conjugacy holds,
and a random-walk Metropolis sampler where it does not. This crate consolidates
the prior TPT `tpt-zero-bayes` work into the `tpt-math` probability stack, with
no dependencies beyond `tpt-math-prob-core`.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is a leaf crate
in the probability layer: it builds directly on the `Rng`/`Distribution` traits
from `tpt-math-prob-core` and is re-exported by the `tpt-math-prob` umbrella.
It is not an umbrella crate.

## Features

- `default = []` — the crate declares no optional features; the full API is
  always available.
- **`std` required.** Unlike `tpt-math-prob-core` and `tpt-math-prob-dist` this
  crate is *not* `no_std`: it uses `Vec` for MCMC traces, `f64` math from
  `std`, and implements `std::error::Error` for `ParameterError`.

## Quick start

```toml
[dependencies]
tpt-math-prob-bayes = "0.1"
tpt-math-prob-core = "0.1"   # for a concrete generator such as SplitMix64
```

```rust
use tpt_math_prob_bayes::{Beta, Metropolis};
use tpt_math_prob_core::SplitMix64;

// Exact conjugate update: uniform prior, then 7 heads and 3 tails.
let posterior = Beta::uniform().update(7, 3);
assert_eq!((posterior.alpha(), posterior.beta()), (8.0, 4.0));
let (lo, hi) = posterior.credible_interval(0.95);
assert!(lo < posterior.mean() && posterior.mean() < hi);

// Same posterior by MCMC, for targets without a closed form.
let prior = Beta::new(2.0, 2.0);
let (successes, failures) = (30, 70);
let target = move |p: f64| prior.log_unnormalized_posterior(p, successes, failures);
let mut sampler = Metropolis::with_gaussian_proposal(target, 0.1);
let mut rng = SplitMix64::seed_from_u64(2024);
let trace = sampler.run_with_burn_in(&mut rng, 0.5, 2_000, 40_000);

let mcmc_mean = trace.iter().sum::<f64>() / trace.len() as f64;
let exact_mean = prior.update(successes, failures).mean();
assert!((mcmc_mean - exact_mean).abs() < 0.01);
```

What is in the box:

- `Beta` — conjugate prior for a Bernoulli/Binomial rate: `update`,
  `update_one`, `update_bernoulli`, `update_binomial`, `pdf`/`log_pdf`, `cdf`,
  `quantile`, `credible_interval`, `log_marginal_likelihood`, sampling.
- `Gaussian` (aliased `Normal`) — density, CDF/quantile, sampling, and the
  Normal–Normal conjugate update for an unknown mean with known variance
  (`update`, `update_summary`, `posterior_predictive`).
- `Gamma` — conjugate prior for a Poisson rate (`update_poisson`,
  `update_poisson_summary`), and the engine behind Beta sampling.
- `Metropolis` — random-walk Metropolis(–Hastings) over any unnormalised scalar
  log-target: `step`, `run`, `run_with_burn_in`, `run_thinned`, plus
  `acceptance_rate` diagnostics.
- Free log-likelihood helpers: `bernoulli_log_likelihood`,
  `binomial_log_likelihood`, `normal_log_likelihood`,
  `poisson_log_likelihood`, `exponential_log_likelihood`, `log_sum_exp`.
- `special` — `ln_gamma`, `ln_beta`, `ln_binomial`, `ln_factorial`, `xlogy`,
  `erf`, `standard_normal_cdf` and friends.

## Notes

- Sampling is generic over `tpt_math_prob_core::Rng`, so seeding a
  deterministic generator such as `SplitMix64` makes an entire inference run
  bit-for-bit reproducible.
- Constructors come in pairs: `new` panics on invalid parameters, `try_new`
  returns `ParameterError` (`NotFinite` / `NotPositive`).
- The `Metropolis` proposal must be **symmetric** over increments (a zero-mean
  `Gaussian` is the default via `with_gaussian_proposal`); that symmetry is what
  reduces the Hastings ratio to the plain target ratio. Candidates whose
  log-target is `-inf` or `NaN` are always rejected, so bounded targets such as
  a `Beta` posterior are safe to sample directly.
- `Distribution`, `Rng` and `Sampler` are re-exported for convenience;
  concrete generators (`SplitMix64`) still come from `tpt-math-prob-core`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
