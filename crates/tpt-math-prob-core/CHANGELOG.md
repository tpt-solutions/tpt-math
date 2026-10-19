# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-10-20

### Added

- `special` module (requires `std`): `ln_gamma`, `gamma`, `ln_beta`, `beta`, `ln_binomial`, `ln_factorial`, `xlogy`, `gamma_p`/`gamma_q`, `regularized_incomplete_beta`, `erf`/`erfc`, `standard_normal_cdf`/`pdf`/`quantile`. Consolidates the two previously duplicated copies in `tpt-math-stats` and `tpt-math-prob-bayes`.

### Fixed

- `erf`/`erfc` are computed from the incomplete gamma function, so small arguments keep relative precision.
- `standard_normal_cdf` uses `erfc`, so both tails keep relative precision.
- `gamma` keeps the sign of `Γ(x)` on negative non-integer intervals.
- Series and continued fractions return `NaN` when they exhaust their iteration budget, instead of an unconverged value.
- `ln_gamma(±∞)` returns `+∞` rather than `NaN`.
- `ln_binomial` no longer cancels for large `n`. It sums the logarithmic ratios directly for small `k`, so `ln C(10¹⁵, 1)` is `34.54`, where the old three-`ln_gamma` difference returned `36`.
- `gamma_p` and `gamma_q` return finite values for large shape parameters (about `10⁷` and above). They used to return NaN, which also broke `poisson_cdf(10⁷, 10⁷)` and `chi_squared_cdf(2·10⁷, 2·10⁷)`. For shape `a ≥ 10` the prefactor comes from the Stirling series, and the series and continued fraction get an iteration budget of `max(10 000, 50·√a)`. This is not Temme's expansion.

## [0.1.0]

- Initial workspace release.
