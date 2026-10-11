# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `integrate` and `ImportanceSampling::estimate` use Welford's one-pass variance; the expanded sum-of-squares formula cancelled catastrophically for large, nearly constant values.
- Removed an unused buffer allocation in `ImportanceSampling::estimate`.

## [0.1.0]

- Initial workspace release.
