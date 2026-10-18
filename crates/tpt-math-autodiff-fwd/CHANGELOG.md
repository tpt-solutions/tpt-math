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

### Added

- `Float` (from `num_traits`, via `tpt-math-numeric`) for `Dual<f64, N>`, with `Zero`, `One`, `Num`, `NumCast`, `ToPrimitive`, `FloatConst` and `Neg`. Generic numerical code written for `Float`, such as the fixed-size ODE solvers in `tpt-math-numeric`, now runs on dual numbers and returns exact derivatives. Piecewise-constant functions (`floor`, `round`, `signum`, …) have zero derivative.

## [0.1.0]

- Initial workspace release.
