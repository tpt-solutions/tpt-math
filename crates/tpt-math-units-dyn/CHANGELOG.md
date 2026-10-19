# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-10-20

### Added

- `solve` module, behind the `ode`, `optimize` and `autodiff` features. `dopri5_dimensioned` integrates `y′ = f(t, y)` with every quantity dimension-checked. `minimize_dimensioned` runs L-BFGS on dimensioned parameters. `gradient_dimensioned` attaches the dimension `cost / parameter` to a reverse-mode gradient.
- `SolveError` reports a dimension mismatch (`Unit`), an ODE failure (`Ode`) or an optimiser failure (`Optimize`).

### Changed

- `DynQuantity::from_str` accepts base-unit expressions such as `m^2·kg·s^-2`, which is what `Display` prints for compound units. This is additive: strings that parsed before still parse.

### Fixed

- `Display` and `FromStr` round-trip compound units. The first fuzzing run found that `Display` printed `m^2·kg·s^-2`, which the parser rejected.

## [0.1.0]

- Initial workspace release.
