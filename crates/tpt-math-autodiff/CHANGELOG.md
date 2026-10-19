# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-10-20

### Added

- `derivatives::jacobian`: the exact Jacobian, with one reverse pass per output.
- `derivatives::hessian`: the Hessian as a central difference of the exact gradient, symmetrised. Its error is of order `1e-6` at the test points.

## [0.1.0]

- Initial workspace release.
