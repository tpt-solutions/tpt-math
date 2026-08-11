# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [0.1.0] - Unreleased

### Added

- `DVector<T>` / `DMatrix<T>` dense linear-algebra types wrapping `faer`.
- Construction (`zeros`, `from_vec`, `from_row_slice`, `from_fn`, `from_diagonal`),
  indexing, elementwise + scalar arithmetic, matrix×matrix and matrix×vector
  multiply, transpose, `dot`, `norm`.
- Fallible dense `solve` / `inverse` via partial-pivot LU.
- `argmin` feature with `ArgminMath`-family trait impls for `DVector<f64>` /
  `DMatrix<f64>` (replacing the `nalgebra` backend of `argmin-math`).
