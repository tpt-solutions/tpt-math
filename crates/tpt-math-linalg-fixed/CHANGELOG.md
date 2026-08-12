# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

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
