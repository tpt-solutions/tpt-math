# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

## [0.2.0] - 2026-10-20

### Added

- `DMatrix::<f64>::par_mul` behind the new `rayon` feature: the product with output columns computed in parallel. The result is bitwise equal to the serial scalar product (without the `simd` feature, that is `*`).
- `Lu::with_pivot_tolerance(m, tol)`: LU with an explicit absolute pivot threshold. `tol = 0` accepts any non-zero finite pivot. The default `Lu::new` keeps its `1e-12` threshold.
- `Lu::with_relative_tolerance(m, rel)`: scale-invariant singularity test (threshold `rel` times the largest entry). A zero matrix is always rejected.
- In-place operators: `+=`/`-=` (with `&DVector`/`&DMatrix`), `*=`/`/=` by a scalar, and `axpy(α, x)` (`self ← self + α·x`) for `DVector` and `DMatrix`.
- `DMatrix::mul_vector_into` and `DMatrix::mul_matrix_into`: products written into a caller-owned buffer, with no allocation.
- `DMatrix::svd` (thin one-sided Jacobi SVD), `singular_values`, `rank` (numerical rank with the `max(m, n) · σ_max · ε` tolerance), `cond` (2-norm condition number; `∞` when rank-deficient).
- `DMatrix::matrix_exp`: matrix exponential by scaling and squaring with a Taylor series. Errors on non-square input.
- `Lu`: a reusable LU factorisation (`Lu::new`, `solve`, `dim`) for many right-hand sides.
- `Qr` (Householder) and `DMatrix::least_squares` for overdetermined least-squares problems.
- `DMatrix::det` (from LU), `DMatrix::cholesky` (real `A = L Lᵀ`), and `DMatrix::symmetric_eigen` (cyclic Jacobi; ascending eigenvalues, orthonormal eigenvectors as columns).
- `solve_banded`: direct solver for banded systems with partial pivoting. Elimination stays inside the band, and an entry outside the declared band is reported as an error.
- `DMatrix::eigenvector_for_eigenvalue`: eigenvector of a square matrix for a known real eigenvalue, by inverse iteration, returned with its residual. The crate does not yet compute the eigenvalues of a general real matrix.
- `Cholesky` (reusable factor), `solve_tridiagonal`, `pseudo_inverse`, `solve_min_norm` and `Lu::solve_matrix` (multi-right-hand-side solve).
- With `simd-runtime`: LU factorisation, `solve_matrix` (through `getrs`) and Cholesky (through `potrf`) use `tpt-simd-blas` for matrices of order 64 and above.
- `DMatrix::eigenvalues`: all eigenvalues of a general real matrix, by Householder reduction to Hessenberg form and Francis double-shift QR. Returns `Eigenvalue { re, im }` values, complex ones in conjugate pairs. Use `symmetric_eigen` for symmetric matrices, which gives sorted values and eigenvectors.
- `DMatrix::solve_with_report`: solves `A x = b` and returns `SolveDiagnostics` with the residual (as a `SolveReport`), the normwise backward error, the 1-norm condition number (exact, from the inverse), and an `ill_conditioned` flag when `cond · ε` exceeds `1e-8`.
- `solve_banded`: banded LU with partial pivoting. An entry outside the declared band is an error, not silently ignored.

### Changed

- `DMatrix::from_vec`/`from_row_slice` panic when the data length is not `nrows * ncols`; `dot` panics on mismatched lengths; the generic matrix and matrix-vector products assert their inner dimensions.
- **Breaking:** `DenseError` gained `NoConvergence`. It is returned when an iterative routine (the SVD sweeps and the eigenvector iteration) runs out of budget. Exhaustive matches need an arm.
- `svd` returns `NoConvergence` when its sweep budget runs out, instead of the unconverged factors.
- `matrix_exp` uses the Padé(13) approximant with scaling and squaring. For a non-finite input it returns a matrix of NaNs, where the old loop never terminated.
- LU factors are stored as one flat buffer, and `Qr` keeps its factor flat too. Both are private, so the public API is unchanged.
- Measured on the default (scalar) build, from the Phase I benchmarks: LU factorisation of order 128 went from 524 µs to 148 µs, and of order 512 solves dropped from 30.9 ms to 13.6 ms. Inverse of order 128 went from 1.63 ms to 1.00 ms.
- Measured with `simd-runtime` against the scalar build on the same machine: solve of order 512 12.8 ms → 5.2 ms, Cholesky of order 512 10.8 ms → 2.8 ms, inverse of order 128 1.04 ms → 0.45 ms, and the 64 × 64 product 42 µs → 15 µs. `matrix_exp` of order 32 was about 2.5% slower in the same run; it is below the SIMD cutoff, so that difference is within the noise of this measurement.
- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

### Fixed

- `alloc`-only (`no_std`) builds compile again: the f64 square roots in the QR, Cholesky, eigen and SVD code go through `num-traits`, which uses `libm` without `std`. `matrix_exp` scales by repeated halving instead of `powi`.
- `Index<(i, j)>` checks each axis, so a row index past `nrows` no longer aliases another element.

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
