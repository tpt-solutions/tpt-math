# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-20

### Added

- `Variable::abs` (`Op::Abs`), with derivative `sign(x)` and `0` at the kink.
- Scalar operator variants (`Op::AddC`, `SubC`, `CSub`, `MulC`, `DivC`, `CDiv`). `Variable op f64` and `f64 op Variable` now record one node, not two. The enum is public, so exhaustive matches on `Op` need the new arms.
- Reverse-mode `tanh`, `sinh`, `cosh`, `atan` and `atan2`. Each is checked against central differences.
- `Op` gained variants for them, so `Op` is a minor-breaking change, as noted above.

### Changed

- `Variable::recip` records a single node.
- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

### Fixed

- The derivative of `Powi(a, 0)` and `Powf(a, 0)` is `0` at `a = 0` (it was NaN from `0 · 0⁻¹`).

## [0.1.0]

- Initial workspace release.
