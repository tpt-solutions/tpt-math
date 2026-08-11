# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- **Storage backend swapped from `nalgebra` to `tpt-math-linalg-dense` (faer).**
  `nalgebra` is Apache-2.0-only and disqualified as a wrap target by the
  workspace license policy (ADR-0007). The unit-tagging API (`Vec<U, T>` /
  `Mat<U, V, T>`, `from_raw`, `raw`, the arithmetic/transpose ops, the
  `no_std` story) is unchanged; the dense `DVector`/`DMatrix` types now back
  the phantom-unit wrappers. Derives are `PartialEq` only (faer's storage is
  not `Eq`).

## [0.1.0]

- Initial workspace release.
