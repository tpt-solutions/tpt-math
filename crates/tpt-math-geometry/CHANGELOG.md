# Changelog

All notable changes to `tpt-math-geometry` are documented here. This crate
follows [Keep a Changelog](https://keepachangelog.com/) and adheres to
Semantic Versioning.

## [Unreleased]

## [0.1.1] - 2026-08-19

### Added

- `Quaternion<T>` scalar `Mul`, `Add`, `Sub`, and `Neg` impls, added to
  support `DualQuaternion` arithmetic in `tpt-math-spatial`.

## [0.1.0]

### Added

- Initial geometry module, built on `tpt-math-linalg-fixed`.
- `Point<T, D>` (with `Point2`/`Point3`), `Translation<T, D>`,
  `Rotation<T, D>` (with `Rotation2`/`Rotation3`, constructed from angles,
  axis-angle and intrinsic Tait–Bryan Euler angles), `Quaternion<T>` /
  `UnitQuaternion<T>` (Hamilton product, conjugate, `rotate_vector`,
  `slerp`, rotation-matrix round-trip), `Isometry<T, D>`
  (`Isometry2`/`Isometry3`), `Similarity<T, D>`, `Scale<T, D>`, and
  `Perspective3<T>` / `Orthographic3<T>` projection matrices.
- `no_std` + allocator-free build.
