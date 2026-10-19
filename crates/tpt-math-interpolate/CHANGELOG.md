# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

### Fixed

- `Pchip::new` accepts two nodes (it used to index out of bounds).
- `Pchip::eval` extrapolates linearly along the endpoint tangent outside the node range, as documented.

### Changed

- **Breaking behaviour:** `Kriging::new` now factors the Kriging matrix and returns `DenseError::Singular` for a singular system (for example duplicate samples with zero nugget). It used to fail only at `predict`.
- `Kriging::predict` reuses the stored factorisation instead of refactoring per call.
- The allocating APIs (RBF, Kriging, PCHIP, B-splines and the new types) are gated behind the `alloc` feature. The `std` feature enables it, so default builds are unaffected.
- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

### Added

- `RbfInterpolatorNd`: radial-basis interpolation of scattered data in any dimension (Euclidean distance). The thin-plate kernel gets an affine polynomial tail, which makes the interpolant unique.
- `KrigingNd`: ordinary Kriging in any dimension, with `predict` returning the mean and the Kriging variance.
- `NaturalCubicSpline`: natural cubic spline in one dimension, with `eval`, `derivative` and linear extrapolation beyond the nodes.
- `MultilinearGrid`: bilinear, trilinear and general multilinear interpolation on a rectilinear grid. Queries outside the grid return `None`.

## [0.1.0] - 2026-08-19

### Added

- `RbfInterpolator` — radial-basis-function interpolation with thin-plate,
  Gaussian and multiquadric kernels, weights solved via the dense linear
  solver. `eval` (single point) and `interpolate` (a `DVector` of points).
- `Kriging` — ordinary Kriging with spherical, exponential, Gaussian or linear
  variogram models; `predict` returns the mean and Kriging variance and is an
  exact interpolator at sample nodes (nugget = 0).
- `Pchip` — shape-preserving piecewise-cubic-Hermite interpolation (generic
  over `T: Scalar`) that preserves monotonicity of the data.
- `bspline_basis` (Cox–de Boor recursion) and `BsplineCurve`, a weighted
  B-spline curve; basis functions satisfy the partition-of-unity property.
- Implemented in-house (no `scirs2-interpolate`/`scipy` dependency) per ADR-0007
  to avoid the Apache-2.0-only license of `scirs2-interpolate`.
- `no_std + alloc` with `std` default feature, mirroring the workspace
  lints (`unsafe_code = "forbid"`).
