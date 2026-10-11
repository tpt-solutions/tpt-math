# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Dual::sqrt`, `powi`, `powf`, `tan`, `tanh`, `atan` and `abs`, with derivatives.
- `Dual` arithmetic with a plain scalar (`Add`, `Sub`, `Mul`, `Div` against `T`).

### Fixed

- `powi(0)` and `powf(0)` have zero derivative at `0` (not `0·0⁻¹ = NaN`).

## [0.1.0]

- Initial workspace release.
