# tpt-math-linalg-sparse

Sparse linear algebra implemented **entirely in-house** (no external sparse
backend), part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
numeric substrate. It provides general-purpose sparse storage — coordinate
(triplet, [`CooMatrix`]), compressed-sparse-row ([`CsrMatrix`]) and
compressed-sparse-column ([`CscMatrix`]) — plus two iterative solvers:
[`conjugate_gradient`] (symmetric positive-definite systems) and [`bicgstab`]
(general / non-symmetric systems).

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
* Solvers: `conjugate_gradient(a, b, x0, tol, max_iter)` and
  `bicgstab(a, b, x0, tol, max_iter)`, both returning
  `Result<DVector<T>, SparseError>`.

Direct sparse factorization (LU / Cholesky) is intentionally **out of scope**
— only iterative solvers are provided.

## License

Licensed under either of MIT or Apache-2.0 at your option.
