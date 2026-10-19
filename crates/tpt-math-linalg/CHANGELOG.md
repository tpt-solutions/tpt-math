# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-20

- `Mat<U, V>` scalar `*` and `/`.
- `Vec::norm` and `Vec::dot`. Both return a bare scalar: unit algebra for `U * U` is not modelled.
- `Mat::solve` (`A x = b`, `x: Vec<V>`) and `Mat::inverse` (`Mat<V, U>`) for `f64`, with the dense backend's error on singular input.
- By-reference `+`, `-` and `*` (`&Vec`, `&Mat`). These clone their operands.
- **Storage backend swapped from `nalgebra` to `tpt-math-linalg-dense`.**
  `nalgebra` is Apache-2.0-only and disqualified as a wrap target by the
  workspace license policy (ADR-0007). `tpt-math-linalg-dense` is implemented
  in-house (column-major `Vec<T>` storage, no external backend), so there is no
  license exposure. The unit-tagging API (`Vec<U, T>` / `Mat<U, V, T>`,
  `from_raw`, `raw`, the arithmetic/transpose ops, the `no_std` story) is
  unchanged; the dense `DVector`/`DMatrix` types now back the phantom-unit
  wrappers.

### Changed

- **Breaking (0.2.0):** the crate is bumped to 0.2.0 for the breaking changes listed above. It has no error or status enums, so this entry is not about `#[non_exhaustive]`.

## [0.1.0]

- Initial workspace release.
