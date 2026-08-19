# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to semantic versioning.

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
