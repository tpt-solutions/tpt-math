# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The special functions now live in `tpt-math-prob-core::special`, shared with `tpt-math-prob-bayes`. This module re-exports them under the existing names.

### Fixed

- `Normal::cdf` is computed as `erfc(-z)/2`, so the lower tail keeps relative precision (`Φ(-10)` is no longer 0).
- `gamma` keeps the sign of `Γ(x)` for negative non-integers (`Γ(-1/2) < 0`).
- `pearson_correlation` rescales the samples when the raw sums overflow, so values near `1e160` no longer produce NaN. Ordinary data keep the exact unscaled arithmetic.
- `erf` is computed from the lower regularized gamma, so small arguments keep relative precision.
- `lgamma` takes the magnitude of `sin` in the reflection step, so negative non-integer arguments return finite values.
- `beta_reg` uses `ln_1p(−x)` for `ln(1 − x)`.

### Added

- `try_bootstrap_ci` / `bootstrap_ci`: percentile bootstrap confidence interval for any statistic `Fn(&[f64]) -> f64`. Deterministic for a given seed (uses `SplitMix64`).
- `try_bonferroni`, `try_holm` and `try_benjamini_hochberg` (plus panicking wrappers): multiple-comparison adjustments for a family of p-values.
- `try_mann_whitney_u` / `mann_whitney_u`: Mann–Whitney U with midranks and the tie-corrected normal approximation.
- `try_wilcoxon_signed_rank` / `wilcoxon_signed_rank`: signed-rank test; zero differences dropped; tie-corrected approximation.
- `try_kolmogorov_smirnov_two_sample` / `kolmogorov_smirnov_two_sample`: two-sample KS with the asymptotic Kolmogorov p-value (Stephens correction).
- `try_one_way_anova` / `one_way_anova`: F-test on `(k−1, N−k)` degrees of freedom; p-value from the regularized incomplete beta.
- `try_chi_squared_independence` / `chi_squared_independence`: Pearson test of independence on an r×c table.
- `quantile`/`try_quantile` (type 7), `skewness`/`try_skewness`, `excess_kurtosis`/`try_excess_kurtosis`.
- `covariance`/`try_covariance`: sample covariance of paired samples.

### Changed

- **Breaking:** `StatsError` gained `ProbabilityOutOfRange`; exhaustive matches need an arm.

## [0.1.0]

- Initial workspace release.
