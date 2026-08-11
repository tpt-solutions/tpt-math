# Changelog

All notable changes to `tpt-math-geometry` are documented here. This crate
follows [Keep a Changelog](https://keepachangelog.com/) and the workspace's
`0.1.0` pre-publish versioning (crates are `git`-status in the registry, not
yet published to crates.io).

## [0.1.0] - Unreleased

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
