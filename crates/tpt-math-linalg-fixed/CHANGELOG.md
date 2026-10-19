# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

## [0.2.0] - 2026-10-20

### Added

- `Matrix::inverse_with_tolerance` and `inverse_with_relative_tolerance`. `inverse()` keeps its `1e-12` default.

### Changed

- **Breaking:** `Matrix::from_columns` takes `C` columns, each of length `R`. The old signature had the dimensions swapped, so it only worked for square matrices.
- **Breaking (0.2.0):** the crate is bumped to 0.2.0 for the breaking changes listed above. It has no error or status enums, so this entry is not about `#[non_exhaustive]`.

### Fixed

- `from_columns` no longer indexes out of bounds for non-square matrices.

## [0.1.0] - Unreleased

### Added

- `Vector<T, N>` / `Matrix<T, R, C>` const-generic, fixed-size (stack-allocated)
  linear-algebra types, with size aliases `Vector2`/`Vector3`/`Vector4`/`Vector6`
  and `Matrix2`/`Matrix3`/`Matrix4`/`Matrix2x3`/`Matrix3x4`/`Matrix4x3`.
- Construction (`new`, `from_fn`, `from_array`, `from_columns`), indexing,
  elementwise + scalar arithmetic, dot/cross/`perp_dot`, matrix×matrix and
  matrix×vector multiply, transpose.
- Closed-form `determinant` / `inverse` via Gauss elimination with partial
  pivoting for square matrices.
- `#![no_std]`, `#![forbid(unsafe_code)]`, allocator-free.
