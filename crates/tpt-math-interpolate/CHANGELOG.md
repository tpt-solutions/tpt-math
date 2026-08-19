# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

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
