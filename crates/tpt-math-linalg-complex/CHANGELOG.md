# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to semantic versioning.

## [Unreleased]

### Added

- `ComplexDMatrix::lu_with_tolerance` (absolute pivot threshold) and `lu_with_relative_tolerance` (scale-invariant). `lu()` keeps its `1e-12` default. A zero matrix is always rejected.

### Added

- `ComplexDMatrix::try_eigenvalues`, the fallible form of `eigenvalues`.
- `ComplexDenseError::NotConverged { iterations }`.

### Changed

- **Breaking:** `ComplexDenseError` gained a variant; exhaustive matches need an arm.
- `eigenvalues` now uses Hessenberg reduction and Wilkinson-shifted QR. It panics on a non-square matrix or non-convergence (it used to return an empty vector for non-square input and silently deflate at the iteration cap).
- Matrix multiply, matrix-vector product, `dot`, and the `from_vec`/`from_row_slice`/`from_real_row_slice` constructors assert their dimensions; indexing checks row bounds.
- `ComplexDMatrix` constructors `from_vec`, `from_row_slice` and `from_real_row_slice` assert the data length.

### Fixed

- Division, `inv` and `norm` no longer overflow for large magnitudes (Smith's algorithm and a scaled hypot).

### Added

- Optional `simd` feature (needs Rust 1.95 or later, as `tpt-simd-blas` does): `solve`, `inverse` and `lu` use `getrf_c64`/`getrs_c64` for matrices of order 64 and above. `simd-runtime` adds run-time AVX2 detection and turns on the dispatch.

### Changed

- **Breaking:** `zeros`, `transpose`, `solve` and `inverse` need the `alloc` feature. `std` enables it, so default builds keep the same API.

## [0.1.0] - 2026-08-19

### Added

- Initial release: `Complex<T>`, `ComplexDVector<T>`, `ComplexDMatrix<T>`,
  complex LU (`lu`/`solve`/`inverse`), Cholesky for Hermitian PD matrices
  (`cholesky` with `Cholesky::solve`), and a shifted-QR `eigenvalues` solver.
- `no_std` + `alloc` support, mirroring `tpt-math-linalg-dense`.

### Fixed

- `householder_qr` used the inherent `f64::sqrt` instead of `Float::sqrt`,
  which doesn't exist in `no_std` builds; switched to `Float::sqrt` so the
  crate actually builds without `std`.
