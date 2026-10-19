# tpt-math-linalg-dense

Dense linear algebra implemented **entirely in-house** (no external backend), the
storage backend for [`tpt-math-linalg`](../tpt-math-linalg) and
[`tpt-math-optimize`](../tpt-math-optimize).

The `DVector`/`DMatrix` types are stored column-major in a plain `Vec<T>`, so
there is no `faer`/`nalgebra`/`clarabel` dependency and therefore no license
exposure. The arithmetic, norms, and the partial-pivot-LU `solve`/`inverse` are
all hand-rolled. (The crate was originally prototyped over `faer`, but the final
implementation dropped that dependency for a zero-license-risk, in-repo design.)

## Types

* `DVector<T>` — a dynamically-sized column vector.
* `DMatrix<T>` — a dynamically-sized, column-major matrix.

## Operations

* Arithmetic: `+`, `-`, `*` and scalar `*` / `/`, plus in-place `+=`, `-=`,
  `*=`, `/=`, `axpy(α, x)`, and the allocation-free `mul_vector_into` and
  `mul_matrix_into`. `DMatrix::<f64>::par_mul` is the multi-threaded product
  under `rayon`.
* Dense solvers: `solve` and `inverse` (partial-pivot LU). `Lu` is a reusable
  factorisation, with `with_pivot_tolerance` and `with_relative_tolerance` for
  the singularity test, and `solve_matrix` for many right-hand sides.
  `solve_with_report` returns a `SolveDiagnostics`, with the residual, the
  backward error, the 1-norm condition number and an `ill_conditioned` flag.
* Factorisations: `Cholesky` (reusable, with `solve`, `det`) and `Qr`
  (Householder). `DMatrix::cholesky`, `det`, `symmetric_eigen` (cyclic Jacobi,
  ascending eigenvalues) and `eigenvalues` (Hessenberg reduction with Francis
  double-shift QR, returning `Eigenvalue { re, im }`) are the matrix-level
  entry points.
* Least squares and SVD: `least_squares`, `solve_min_norm`, `pseudo_inverse`,
  `svd` (thin one-sided Jacobi), `singular_values`, `rank` and `cond`.
* Other: `matrix_exp` (scaling and squaring with Padé(13)),
  `eigenvector_for_eigenvalue` (inverse iteration for a known real eigenvalue),
  `solve_banded` (banded LU with partial pivoting) and `solve_tridiagonal`.

Iterative routines (SVD sweeps, eigenvector iteration) return
`DenseError::NoConvergence` when they run out of budget. `DenseError` is
`#[non_exhaustive]`, so match it with a wildcard arm.

## Features

| Feature | Enables | Notes |
|---|---|---|
| `std` | `["tpt-math-numeric/std", "alloc"]` | `std` pulls in the allocator (dynamic vectors need a heap). |
| `alloc` | `["tpt-math-numeric/alloc"]` | Dynamic vectors/matrices always need an allocator. |
| `simd` | `["alloc", "dep:tpt-simd-blas"]` | Route `DMatrix * DMatrix`, `DMatrix * DVector`, `DVector::dot` and `norm` to tpt-simd-blas for `f32`/`f64` (other scalar types use the generic loops). Off by default: results reassociate sums, so they can differ from the scalar loops by a few ulps (tpt-simd ADR 0003, tier 2). Adds a `T: 'static` bound to those operations. Enable `simd-runtime` to detect AVX2+FMA at run time. |
| `rayon` | `["std", "dep:rayon"]` | Multi-threaded `DMatrix::<f64>::par_mul`. Results are bitwise equal to `*`. |
| `simd-runtime` | `["simd", "tpt-simd-blas/runtime-dispatch"]` | As `simd`, with AVX2+FMA detected at run time. The `simd` features need Rust 1.95 or later; the default build stays at 1.85. |

## Example

```rust
use tpt_math_linalg_dense::{DMatrix, DVector};

let m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
let v = DVector::from_vec(vec![1.0_f64, 1.0]);
let mv = m * v;
assert_eq!(mv, DVector::from_vec(vec![3.0, 7.0]));

let inv = m.inverse().unwrap();
let got = m * inv;
assert!((got[(0, 0)] - 1.0).abs() < 1e-12);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
