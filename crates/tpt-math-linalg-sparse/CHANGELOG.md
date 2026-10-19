# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

### Added

- `gmres_right_preconditioned_with_report`: GMRES returning a `SolveReport` for every outcome. The plain `gmres` and `gmres_right_preconditioned` report the same cases as errors, as before.
- Benchmarks for CG and IC(0)-preconditioned CG on a 40x40 Laplacian.

### Added

- `conjugate_gradient_with_report` and `bicgstab_with_report`: return `(x, SolveReport)` for every outcome, including breakdown and non-convergence, which the plain solvers report as errors. Only a dimension mismatch is an `Err`.
- `SparseLu`: left-looking sparse LU with partial pivoting (`P A = L U`), reusable across right-hand sides. Reports `Breakdown` on a singular matrix.
- `Ic0` and `ic0_preconditioned_cg`: incomplete Cholesky with zero fill-in for SPD systems.
- `preconditioned_conjugate_gradient`: CG under any caller-supplied SPD preconditioner. `jacobi_preconditioned_cg` is now a wrapper around it, with unchanged behaviour.

### Changed

- `conjugate_gradient` and `bicgstab` update their vectors in place and reuse work buffers, so their iteration loops no longer allocate.
- COO→CSR/CSC conversion uses a two-pass stable counting sort (`O(nnz + n)`) instead of a comparison sort. Results are unchanged.

### Added

- `gmres_right_preconditioned`: restarted GMRES with a caller-supplied right preconditioner `M⁻¹`. `gmres` is now this function with the identity preconditioner, so its behaviour is unchanged.
- `gmres_ilu0`: GMRES preconditioned by the ILU(0) factors of the matrix itself.
- `Ilu0`: incomplete LU factorisation with zero fill-in (`factor`, `apply`), usable as a preconditioner. Exact when the factors have no fill-in.
- `SparseCholesky`: sparse Cholesky factorisation `A = L Lᵀ` of a symmetric positive-definite matrix, with `factor`, `solve` and fill-in statistics. No fill-reducing ordering is applied. Indefinite input is reported as `Breakdown`.
- `jacobi_preconditioned_cg`: conjugate gradients preconditioned by `diag(A)`. Errors on a zero diagonal entry.
- `gmres`: restarted GMRES(m) for general (non-symmetric) square systems, with Arnoldi basis by modified Gram–Schmidt and Givens-rotation least squares. Takes an optional initial guess, a relative tolerance, the restart length and a total step budget.
- `CsrMatrix::matvec_into(&[T], &mut [T])`: writes `A x` into a caller-owned buffer. `matvec` now delegates to it.
- `SparseError::Breakdown { iterations }` for a numerical breakdown.

### Changed

- **Breaking:** `SparseError` gained a variant.
- CG and BiCGSTAB stop when `‖r‖ ≤ tol·‖b‖` (absolute `tol` when `b = 0`). Previously `tol` was absolute, so a small right-hand side was reported as solved without being solved.
- Breakdowns (non-positive `pᵀAp`, zero `ρ`, `ω = 0`, non-finite values) report `Breakdown` instead of `NotConverged`.

### Fixed

- BiCGSTAB exits early when the half-step residual `‖s‖` is already within tolerance.

### Added

- `LinearOperator` trait (`nrows`, `ncols`, `apply`), implemented for `CsrMatrix`. `matrix_free` wraps a closure as a square operator with no stored matrix.
- `conjugate_gradient_op`: conjugate gradients for any `LinearOperator`, with the same relative-tolerance rule as `conjugate_gradient`.
- `lanczos`, `Which` and `LanczosEigenpairs`: Lanczos eigensolver for symmetric operators. It uses full reorthogonalization, returns Ritz values and vectors with residual estimates, and picks the largest or smallest end of the spectrum.
- `reverse_cuthill_mckee` (bandwidth-reducing ordering), `permute_symmetric` and `ReorderedCholesky` (Cholesky factorisation in an RCM order, with the permutation applied and undone internally).

### Changed

- The scalar conjugate-gradient loop is shared between `conjugate_gradient` and the operator form. Results are unchanged.
- **Fixed:** the `x0` length is checked before the SIMD dispatch, so a wrong-length guess returns `DimensionMismatch` rather than panicking.
- `to_csr`, `to_csc` and the transposes are gated behind `alloc`, as the other sparse containers are.

### Added

- `arnoldi` and `ArnoldiRitz`: Arnoldi eigensolver for general operators. It returns the Ritz values of largest modulus, with residual estimates for the real ones (`None` for complex pairs, whose eigenvectors are not formed).
- `bicgstab_op` and `gmres_op`: BiCGSTAB and restarted GMRES for any `LinearOperator`. The CSR forms now share their loops with these, so results are unchanged.
- `minimum_degree_ordering` and `FillOrdering`: a minimum-degree fill-reducing ordering, with exact degrees on the elimination graph. This is the idea AMD approximates; it is not AMD, and it costs more on very large matrices.
- `ReorderedLu`: sparse LU after a symmetric fill-reducing permutation, solving in the original numbering. `ReorderedCholesky::factor_with` takes a `FillOrdering`.

### Changed

- **Breaking (0.2.0):** `SparseError::DimensionMismatch` carries `what: &'static str`, describing the incompatible operands, and the enum is `#[non_exhaustive]`.

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
