# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [0.1.0] - Unreleased

### Added

- `CooMatrix<T>` coordinate (triplet) storage with `push` / `from_triplets` and
  duplicate-summing `to_csr` / `to_csc` conversion (matching `tpt-fem-sparse`'s
  accumulate-and-scatter semantics).
- `CsrMatrix<T>` (compressed-sparse-row) and `CscMatrix<T>`
  (compressed-sparse-column) storage, each with `nrows` / `ncols` / `nnz`,
  `iter()` over stored entries, `transpose()` (CSR ↔ CSC), and `matvec`.
- Sparse matrix–vector product `CsrMatrix * &DVector<T>` operator form.
- Iterative solvers: `conjugate_gradient` (SPD systems) and `bicgstab`
  (general systems), with `SparseError` (`DimensionMismatch`, `NotConverged`).
- `#![forbid(unsafe_code)]`; `no_std` + `alloc` support.
