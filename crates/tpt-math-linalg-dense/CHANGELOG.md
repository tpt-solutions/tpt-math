# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [0.1.0] - Unreleased

### Added

- `as_slice` / `as_mut_slice` on `DVector` and `DMatrix` (column-major order for
  matrices).
- Optional `simd` feature (off by default): `DMatrix * DMatrix`,
  `DMatrix * DVector`, `DVector::dot` and `DVector::norm` use `tpt-simd-blas` for
  `f32`/`f64` (other scalars keep the generic loops). Sums reassociate, so
  results can differ from the scalar loops by a few ulps. Requires `T: 'static`
  for those operations and Rust 1.85+ (tpt-simd is edition 2024). `simd-runtime`
  additionally detects AVX2+FMA at run time (no `-C target-cpu` needed).

- `DVector<T>` / `DMatrix<T>` dense linear-algebra types, implemented in-house
  (column-major `Vec<T>` storage, no external backend).
- Construction (`zeros`, `from_vec`, `from_row_slice`, `from_fn`, `from_diagonal`),
  indexing, elementwise + scalar arithmetic, matrix×matrix and matrix×vector
  multiply, transpose, `dot`, `norm`.
- Fallible dense `solve` / `inverse` via partial-pivot LU.
- `argmin` feature with `ArgminMath`-family trait impls for `DVector<f64>` /
  `DMatrix<f64>` (replacing the `nalgebra` backend of `argmin-math`).
