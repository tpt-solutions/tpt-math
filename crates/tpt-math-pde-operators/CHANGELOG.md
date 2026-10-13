# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

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
