# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `special` module (requires `std`): `ln_gamma`, `gamma`, `ln_beta`, `beta`, `ln_binomial`, `ln_factorial`, `xlogy`, `gamma_p`/`gamma_q`, `regularized_incomplete_beta`, `erf`/`erfc`, `standard_normal_cdf`/`pdf`/`quantile`. Consolidates the two previously duplicated copies in `tpt-math-stats` and `tpt-math-prob-bayes`.

### Fixed

- `erf`/`erfc` are computed from the incomplete gamma function, so small arguments keep relative precision.
- `standard_normal_cdf` uses `erfc`, so both tails keep relative precision.
- `gamma` keeps the sign of `Γ(x)` on negative non-integer intervals.
- Series and continued fractions return `NaN` when they exhaust their iteration budget, instead of an unconverged value.
- `ln_gamma(±∞)` returns `+∞` rather than `NaN`.

## [0.1.0]

- Initial workspace release.
