# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `integrate_par` behind the new `rayon` feature: `integrate` over 64 independent `SplitMix64` streams, merged in a fixed order. The result depends only on the seed and `n`, not on the thread count. It is a different sample from `integrate` with the same seed.


### Fixed

- `integrate` and `ImportanceSampling::estimate` use Welford's one-pass variance; the expanded sum-of-squares formula cancelled catastrophically for large, nearly constant values.
- Removed an unused buffer allocation in `ImportanceSampling::estimate`.

## [0.1.0]

- Initial workspace release.
