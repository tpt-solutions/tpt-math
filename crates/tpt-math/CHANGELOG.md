# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial facade: one optional re-export per area, short feature names (`numeric`, `units`, `linalg`, `exact`, `prob`, `stats`, `autodiff`, `optimize`, `signal`, `symbolic`, `geometry`, `spatial`, `graph`, `interpolate`), a `full` feature, and a feature-gated `prelude`.

### Added

- Features `units-dyn`, `linalg-sparse`, `linalg-complex`, `linalg-fixed` and `pde-operators`, all included by `full`.
- A `std` feature that passes `std` through to every enabled area that has one.
- Prelude names `Lu`, `Qr`, `Fft`, `Dual`, `GradientTape` and `DynQuantity`.
