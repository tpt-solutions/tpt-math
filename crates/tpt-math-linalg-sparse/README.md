# tpt-math-linalg-sparse

Sparse linear algebra implemented **entirely in-house** (no external sparse
backend), part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
numeric substrate. It provides general-purpose sparse storage — coordinate
(triplet, [`CooMatrix`]), compressed-sparse-row ([`CsrMatrix`]) and
compressed-sparse-column ([`CscMatrix`]) — plus two iterative solvers:
[`conjugate_gradient`] (symmetric positive-definite systems) and [`bicgstab`]
(general / non-symmetric systems), with GMRES, several preconditioners, and
direct LU and Cholesky factorisations.

It deliberately carries **no `sprs` / `nalgebra` / `faer` dependency**, so there
is no license exposure and the implementation is fully under workspace control.
This complements — but does not duplicate — [`tpt-fem-sparse`]
(https://crates.io/crates/tpt-fem-sparse), which is a separate FEM-assembly
adapter (element scatter + duplicate-summing triplet accumulation) living in
the `tpt-fem` repo. The duplicate-summing `CooMatrix` conversion here matches
`tpt-fem-sparse`'s `push`-and-accumulate semantics, so assembled triplets drop
straight into either crate.

## Part of tpt-math

This crate owns the sparse-matrix half of the `tpt-math` linear-algebra layer.
Dense right-hand sides and solutions are `tpt-math-linalg-dense`'s `DVector<T>`
(the same type `tpt-math-linalg` / `tpt-math-optimize` use), so values move
between the sparse and dense worlds without conversion.

## Features

* `std` (default) — enable the allocator and the `std` support of deps.
* `alloc` — signal allocator availability (sparse containers need it). There is
  no `std`-only API surface, so a `no_std` build just needs
  `default-features = false, features = ["alloc"]` and an allocator.
* `simd` — `f32`/`f64` CG and BiCGSTAB use `tpt-simd-sparse` for fused vector
  updates and SpMV. Results agree with the scalar loops to rounding, and the
  scalar path stays the default. Needs Rust 1.95 or later.

## Quick start

```rust
use tpt_math_linalg_dense::DVector;
use tpt_math_linalg_sparse::{CooMatrix, conjugate_gradient};

// 2x2 SPD system: A = [[4,1],[1,3]], b = [1,2].
let mut coo = CooMatrix::<f64>::new(2, 2);
coo.push(0, 0, 4.0);
coo.push(0, 1, 1.0);
coo.push(1, 0, 1.0);
coo.push(1, 1, 3.0);

let a = coo.to_csr();
let b = DVector::from_vec(vec![1.0, 2.0]);
let x = conjugate_gradient(&a, &b, None, 1e-12, 100).unwrap();
// x = [1/11, 7/11].
```

## Supported operations

* `CooMatrix`: `push` / `from_triplets`; `to_csr` / `to_csc` (duplicate-summing).
* `CsrMatrix` / `CscMatrix`: `nrows` / `ncols` / `nnz`, `iter()` over stored
  entries, `transpose()` (CSR ↔ CSC), and `matvec` (`A * x`). The `*` operator
  form `csr * &x` is also provided for CSR.
* Iterative solvers: `conjugate_gradient` and `bicgstab`, each taking
  `(a, b, x0, tol, max_iter)`. Restarted `gmres` takes `(a, b, x0, tol, restart,
  max_iter)`, and `gmres_right_preconditioned` adds a caller-supplied `M⁻¹`.
  All return `Result<DVector<T>, SparseError>`.
* Report variants: `conjugate_gradient_with_report`, `bicgstab_with_report` and
  `gmres_right_preconditioned_with_report` return `(x, SolveReport)` for every
  outcome, including breakdown and non-convergence. Only a dimension mismatch is
  an `Err`.
* Operator form: the `LinearOperator` trait (`nrows`, `ncols`, `apply`) is
  implemented for `CsrMatrix`. `matrix_free` wraps a closure as a square
  operator, and `conjugate_gradient_op`, `bicgstab_op` and `gmres_op` accept any
  operator. `CsrMatrix::matvec_into` writes into a caller-owned buffer.
* Preconditioners: `jacobi_preconditioned_cg`, `ic0_preconditioned_cg` (IC(0)),
  `gmres_ilu0` (ILU(0)), and `preconditioned_conjugate_gradient` for any SPD
  preconditioner. The factors `Ilu0` and `Ic0` are also public.
* Eigensolvers: `lanczos` (symmetric operators, full reorthogonalisation, Ritz
  pairs with residual estimates) and `arnoldi` (general operators, Ritz values
  of largest modulus).
* Direct factorisations, reusable for many right-hand sides: `SparseLu` (LU with
  partial pivoting, left-looking) and `SparseCholesky` (SPD). Neither applies a
  fill-reducing ordering on its own.
* Fill-reducing orderings: `reverse_cuthill_mckee` (bandwidth reduction),
  `minimum_degree_ordering` (exact-degree minimum degree, not AMD), and
  `permute_symmetric`. `ReorderedCholesky` and `ReorderedLu` apply an ordering
  and undo it internally, so they solve in the original numbering.

## License

Licensed under either of MIT or Apache-2.0 at your option.
