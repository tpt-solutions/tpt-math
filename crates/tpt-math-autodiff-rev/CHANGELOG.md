# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Variable::abs` (`Op::Abs`), with derivative `sign(x)` and `0` at the kink.
- Scalar operator variants (`Op::AddC`, `SubC`, `CSub`, `MulC`, `DivC`, `CDiv`). `Variable op f64` and `f64 op Variable` now record one node, not two. The enum is public, so exhaustive matches on `Op` need the new arms.

### Changed

- `Variable::recip` records a single node.

### Fixed

- The derivative of `Powi(a, 0)` and `Powf(a, 0)` is `0` at `a = 0` (it was NaN from `0 · 0⁻¹`).

## [0.1.0]

- Initial workspace release.
