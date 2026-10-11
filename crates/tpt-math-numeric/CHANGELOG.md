# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `roots` module: `bisect` and `brent` bracketing root finders, with `RootError`.
- `quadrature` module: composite `simpson` rule.
- `ode` module: fixed-step classical Runge–Kutta, `rk4_step` and `rk4_integrate`, on fixed-size state arrays (no allocation).

## [0.1.0]

- Initial workspace release.
