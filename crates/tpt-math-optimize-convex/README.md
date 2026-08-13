# tpt-math-optimize-convex

Convex / quadratic-programme optimisation: a small, self-contained dense
**primal-dual interior-point** (Mehrotra predictor-corrector) QP solver. It
exposes a small API for convex QPs with equality constraints, linear inequality
constraints and per-variable bounds, and needs no external solver crate.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
It is the convex half of the optimisation layer (the general nonlinear half is
`tpt-math-optimize-general`), and is re-exported by the `tpt-math-optimize`
umbrella crate. Dense inputs are `tpt-math-linalg-dense` `DMatrix<f64>` /
`DVector<f64>` (in-house), taken from the same storage that
`tpt-math-linalg` wraps.

## Features

This crate has no optional features: `default = []` and the whole API is always
available. It is **std-only** — the `std::error::Error` impl on `ConvexError`
requires `std`, so there is no `no_std` build.

The `tpt-math-linalg-dense` (in-house) types are re-exported via
`tpt_math_optimize_convex::tpt_math_linalg_dense` for building the dense inputs.

## Quick start

```toml
[dependencies]
tpt-math-optimize-convex = "0.1"
```

`solve_qp` covers the common form — quadratic cost, equality constraints and
per-variable bounds:

```rust
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_optimize_convex::solve_qp;

// minimize x² + y²  subject to  x + y = 1   ->   (0.5, 0.5)
let p = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
let q = DVector::from_vec(vec![0.0, 0.0]);
let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
let b_eq = DVector::from_vec(vec![1.0]);

let x = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap();

assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6);
```

`QuadraticProgram` is the builder for the full form, and returns a `QpSolution`
with the primal vector, the objective value and a `QpStatus`:

```rust
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_optimize_convex::QuadraticProgram;

// minimize x² + y²  subject to  x + y = 1,  x ≥ 0,  y ≥ 0
let qp = QuadraticProgram::new(DVector::from_vec(vec![0.0, 0.0]))
    .objective(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]))
    .equality(DMatrix::from_row_slice(1, 2, &[1.0, 1.0]), DVector::from_vec(vec![1.0]))
    .bounds(&[(0.0, f64::INFINITY), (0.0, f64::INFINITY)]);

let sol = qp.solve().unwrap();

assert!((sol.x[0] - 0.5).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6);
```

The builder also accepts `inequality(a_ineq, b_ineq)` for `A_ineq x ≤ b_ineq`
and `linear_cost(q)`.

## Notes

- Problems are stated as `minimize ½ xᵀ P x + qᵀ x` subject to `A_eq x = b_eq`,
  `A_ineq x ≤ b_ineq` and `l ≤ x ≤ u`, then converted to conic form
  `minimize qᵀ x + ½ xᵀ P x  s.t.  A_eq x = b_eq,  A x + s = b, s ≥ 0`:
  equalities use the zero cone (no slack), inequalities and bounds the
  nonnegative cone (each bound contributes one row, `x - l ≥ 0` and/or
  `u - x ≥ 0`). The KKT linear systems are inverted with
  `tpt-math-linalg-dense`'s in-house dense `DMatrix::solve`.
- `P` is symmetrised as `(P + Pᵀ) / 2` internally.
- Bounds may be one-sided: a non-finite bound (`f64::INFINITY` /
  `f64::NEG_INFINITY`) means "unbounded on that side" and simply contributes no
  row. Pass `&[]` for no bounds at all; otherwise the slice length must equal
  the number of variables.
- `ConvexError` distinguishes `DimensionMismatch`, `NotFinite` and `Solver`
  failures; `QpStatus::Solved` and `AlmostSolved` count as success, everything
  else is reported as a `Solver` error. The solver also reports infeasible or
  unbounded problems as `Solver` errors.
- This crate is dual-licensed `MIT OR Apache-2.0`, and depends only on
  `tpt-math-linalg-dense` (in-house, no external backend) — the old `clarabel`
  (Apache-2.0-only) backend was removed to satisfy the workspace license policy
  (ADR-0007).
- `#![forbid(unsafe_code)]`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
