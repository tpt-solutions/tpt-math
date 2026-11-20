# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-21

### Changed (breaking)

- The `exact` re-exports follow `tpt-math-exact` 0.2.0: `Rational` and `BigRational` are now defined in that crate, and `BigInt`/`BigUint` are `num-bigint` 0.5 types. Code that mixed `num_rational` values with the `exact` feature must switch to the facade's types.
- The `prob` re-exports follow `tpt-math-prob-dist` 0.2.0, where `CoreRng` implements `rand::TryRng` instead of `rand::RngCore`.

### Added

- `splat` feature, included by `full`, re-exporting `tpt-math-splat` 0.1.0.

## [0.1.0] - 2026-10-20

### Added

- Initial facade: one optional re-export per area, short feature names (`numeric`, `units`, `linalg`, `exact`, `prob`, `stats`, `autodiff`, `optimize`, `signal`, `symbolic`, `geometry`, `spatial`, `graph`, `interpolate`), a `full` feature, and a feature-gated `prelude`.
- Features `units-dyn`, `linalg-sparse`, `linalg-complex`, `linalg-fixed` and `pde-operators`, all included by `full`.
- A `std` feature that passes `std` through to every enabled area that has one.
- Prelude names `Lu`, `Qr`, `Fft`, `Dual`, `GradientTape` and `DynQuantity`.
