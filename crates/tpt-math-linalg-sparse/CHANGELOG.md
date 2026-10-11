# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

### Added

- `SparseError::Breakdown { iterations }` for a numerical breakdown.

### Changed

- **Breaking:** `SparseError` gained a variant.
- CG and BiCGSTAB stop when `‖r‖ ≤ tol·‖b‖` (absolute `tol` when `b = 0`). Previously `tol` was absolute, so a small right-hand side was reported as solved without being solved.
- Breakdowns (non-positive `pᵀAp`, zero `ρ`, `ω = 0`, non-finite values) report `Breakdown` instead of `NotConverged`.

### Fixed

- BiCGSTAB exits early when the half-step residual `‖s‖` is already within tolerance.

## [0.1.0] - 2026-08-19

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
