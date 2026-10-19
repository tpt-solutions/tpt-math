# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

## [0.2.0] - 2026-10-20

### Added

- `Grid` for uniform 1–3D structured grids with per-axis `Boundary`
  (`Periodic`, `Dirichlet0`, `Neumann0`).
- `Field` with `Centering::{Cell, Face(axis)}` storage.
- `Operators` with `gradient`, `divergence`, `laplacian` and
  `central_difference`, using a staggered layout.
- Structural contracts via `Operators::verify`: adjointness, discrete
  conservation, skew-symmetry, Laplacian symmetry and sign, and constant kernel.
  `Operators::new` enforces them in debug builds and under the `contracts`
  feature.

### Changed

- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.
