# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `pearson_correlation` rescales the samples when the raw sums overflow, so values near `1e160` no longer produce NaN. Ordinary data keep the exact unscaled arithmetic.
- `erf` is computed from the lower regularized gamma, so small arguments keep relative precision.
- `lgamma` takes the magnitude of `sin` in the reflection step, so negative non-integer arguments return finite values.
- `beta_reg` uses `ln_1p(−x)` for `ln(1 − x)`.

### Added

- `quantile`/`try_quantile` (type 7), `skewness`/`try_skewness`, `excess_kurtosis`/`try_excess_kurtosis`.
- `covariance`/`try_covariance`: sample covariance of paired samples.

### Changed

- **Breaking:** `StatsError` gained `ProbabilityOutOfRange`; exhaustive matches need an arm.

## [0.1.0]

- Initial workspace release.
