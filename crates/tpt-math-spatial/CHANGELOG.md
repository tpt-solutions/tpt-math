# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [0.1.0] - Unreleased

### Added

- `SpatialVector<K>` family with `MotionVector` / `ForceVector` — 6-D Featherstone
  spatial vectors `[angular (top 3); linear (bottom 3)]` with `crm` / `crf`
  cross products, tagged by a kind marker so the result kind is checked at compile
  time.
- `adjoint_motion` / `adjoint_force` Plücker adjoint transforms over an
  `Isometry3`, plus `transform_by` on the spatial vectors.
- `DualQuaternion` — Hamilton product, conjugate, normalisation, and rigid-transform
  conversion (`from_isometry` / `to_isometry`).
- `Screw` — a twist in `se(3)` with the exponential/logarithm maps `exp` / `log`
  to and from `Isometry3` (Lynch–Park convention, active right-handed rotation).
- Implemented in-house (no `spatial-math` dependency) per ADR-0007 — `spatial-math`
  ships under a non-MIT/Apache "Custom license" and does not cover dual
  quaternions / screw theory.
- `#![no_std]` and allocator-free (stack-allocated fixed-size storage), mirroring
  the workspace lints (`unsafe_code = "forbid"`).

### Fixed

- `cross_motion_force` (`crf`) dropped the `mω×fv` term, giving the wrong sign
  and axis for the linear component of the spatial force cross product;
  corrected.
- Workspace deps (`tpt-math-geometry`, `tpt-math-linalg-fixed`,
  `tpt-math-numeric`) now use `default-features = false` so the crate actually
  builds `no_std`; added the missing `Neg` derive for the `Motion`/`Force`
  markers.
- `Screw::log` clamps its `acos` input to `[-1, 1]` to avoid `NaN` from
  floating-point drift, and now guards the `theta ≈ pi` axis-extraction
  singularity via `(R + I) = 2ww^T` with the sign taken from the skew part,
  with round-trip tests at `theta ≈ 0` and `theta ≈ pi`.
