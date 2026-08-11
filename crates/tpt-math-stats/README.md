# tpt-math-stats

Descriptive statistics, classical hypothesis tests, and least-squares
regression. This crate wraps [`statrs`](https://crates.io/crates/statrs) —
whose distributions, special functions, and `Statistics` traits are
battle-tested — behind a small slice-oriented API that takes plain `&[f64]` and
returns plain tuples, so callers never have to construct a distribution object
just to get a p-value.

## Part of tpt-math

This crate is a member of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
workspace and is its home for "classical" statistics, sitting alongside the
`tpt-math-prob-*` probability layer. It depends on `statrs` and on
`tpt-math-prob-core` for the workspace randomness traits, and re-exports both
in full so downstream crates need no second dependency declaration.

## Features

This crate has no optional features (`default = []`). It requires `std`:
`statrs` is a `std` crate, and the panicking wrappers format their error
messages.

## What's provided

| Area | Functions |
|------|-----------|
| Descriptive | `mean`, `variance`, `std_dev`, `min`, `max`, `median` |
| Hypothesis tests | `one_sample_t_test`, `two_sample_t_test` (Welch), `chi_squared_goodness_of_fit` |
| Regression | `pearson_correlation`, `linear_regression` |

Every function has a panicking short-named form and a checked `try_*` twin
returning `StatsError`; the former is exactly the latter unwrapped.

## Quick start

```toml
[dependencies]
tpt-math-stats = "0.1"
```

```rust
use tpt_math_stats::{
    chi_squared_goodness_of_fit, linear_regression, mean, one_sample_t_test,
    pearson_correlation, try_two_sample_t_test,
};

// Is this sample drawn from a population with mean 100?
let iq = [105.0, 98.0, 110.0, 102.0, 99.0, 107.0, 101.0, 103.0];
let (t, p) = one_sample_t_test(&iq, 100.0);
assert!(mean(&iq) > 100.0 && t > 0.0 && p < 0.10);

// Welch's two-sample test, checked form.
let control = [4.1, 3.9, 4.4, 4.0, 3.8];
let treated = [5.2, 5.5, 5.1, 5.4, 5.3];
let (_t, p) = try_two_sample_t_test(&control, &treated).unwrap();
assert!(p < 0.001);

// Ordinary least squares: returns (slope, intercept).
let dose = [0.0, 1.0, 2.0, 3.0, 4.0];
let response = [1.0, 3.0, 5.0, 7.0, 9.0];
assert_eq!(linear_regression(&dose, &response), (2.0, 1.0));
assert_eq!(pearson_correlation(&dose, &response), 1.0);

// Goodness of fit: 100 draws that should have been split 25/25/25/25.
let (x2, p) = chi_squared_goodness_of_fit(&[22u64, 27, 24, 27], &[25.0; 4]);
assert!(x2 < 1.0 && p > 0.5);
```

Anything not wrapped here is one path away, because `statrs` is re-exported
verbatim:

```rust
use tpt_math_stats::statrs::distribution::{ContinuousCDF, FisherSnedecor};

let f = FisherSnedecor::new(3.0, 16.0).unwrap();
assert!(f.sf(5.29) < 0.05);
```

## Notes

- Conventions: inputs must be finite (`NaN`/infinity are rejected rather than
  silently poisoning a result); sample variance is Bessel-corrected (`n - 1`);
  tests return `(statistic, p_value)`, two-sided for the t-tests and upper-tail
  for chi-squared.
- P-values are evaluated with `statrs` survival functions rather than
  `1 - cdf`, so tiny p-values keep their significant digits.
- Sums use compensated (Neumaier) accumulation and variances use the corrected
  two-pass formula, keeping results accurate for long or badly offset samples.
- Documented non-error results: a degenerate (zero standard error) t-test
  returns `(0.0, 1.0)` or `(±∞, 0.0)`; a constant sample makes
  `pearson_correlation` return `NaN` and `linear_regression` return
  `(NaN, NaN)`.
- `tpt-math-prob-core` is re-exported as `prob_core`, with `Distribution`,
  `Rng`, `Sampler`, `SplitMix64` and `Standard` lifted to the crate root, which
  makes reproducible simulation studies against these tests a two-liner.
- Not `no_std`. `statrs` is `MIT`-licensed, compatible with this workspace's
  `MIT OR Apache-2.0` policy.

## License

Licensed under either of MIT or Apache-2.0 at your option.
