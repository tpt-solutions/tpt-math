# tpt-math-optimize-convex

Convex / quadratic-programme optimisation: a thin wrapper around the
[`clarabel`](https://docs.rs/clarabel) conic interior-point solver. It exposes a
small dense API for convex QPs with equality constraints, linear inequality
constraints and per-variable bounds, and handles the translation into clarabel's
sparse conic standard form for you.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
It is the convex half of the optimisation layer (the general nonlinear half is
`tpt-math-optimize-general`), and is re-exported by the `tpt-math-optimize`
umbrella crate. Dense inputs are `nalgebra` `DMatrix<f64>`/`DVector<f64>`, taken
from the same `nalgebra` that `tpt-math-linalg` wraps.

## Features

This crate has no optional features: `default = []` and the whole API is always
available. It is **std-only** — `clarabel` and the `std::error::Error` impl on
`ConvexError` both require `std`, so there is no `no_std` build.

Two re-exports are always present: `clarabel` (for custom cones, settings or
lower-level solution inspection) and `nalgebra` (via `tpt-math-linalg`, for
building the dense inputs).

## Quick start

```toml
[dependencies]
tpt-math-optimize-convex = "0.1"
nalgebra = "0.33"
```

`solve_qp` covers the common form — quadratic cost, equality constraints and
per-variable bounds:

```rust
use nalgebra::{dmatrix, dvector};
use tpt_math_optimize_convex::solve_qp;

// minimize x² + y²  subject to  x + y = 1   ->   (0.5, 0.5)
let p = dmatrix![2.0, 0.0; 0.0, 2.0];
let q = dvector![0.0, 0.0];
let a_eq = dmatrix![1.0, 1.0];
let b_eq = dvector![1.0];

let x = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap();

assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6);
```

`QuadraticProgram` is the builder for the full form, and returns a `QpSolution`
with the primal vector, the objective value and clarabel's `SolverStatus`:

```rust
use nalgebra::{dmatrix, dvector};
use tpt_math_optimize_convex::QuadraticProgram;

// minimize x² + y²  subject to  x + y = 1,  x ≥ 0,  y ≥ 0
let qp = QuadraticProgram::new(dvector![0.0, 0.0])
    .objective(dmatrix![2.0, 0.0; 0.0, 2.0])
    .equality(dmatrix![1.0, 1.0], dvector![1.0])
    .bounds(&[(0.0, f64::INFINITY), (0.0, f64::INFINITY)]);

let sol = qp.solve().unwrap();

assert!((sol.x[0] - 0.5).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6);
```

The builder also accepts `inequality(a_ineq, b_ineq)` for `A_ineq x ≤ b_ineq`
and `linear_cost(q)`.

## Notes

- Problems are stated as `minimize ½ xᵀ P x + qᵀ x` subject to `A_eq x = b_eq`,
  `A_ineq x ≤ b_ineq` and `l ≤ x ≤ u`, then converted to clarabel's
  `minimize qᵀ x  s.t.  A x + s = b, s ∈ K`: equalities become a `ZeroConeT`,
  inequalities and bounds a `NonnegativeConeT` (each bound contributes one row,
  `x - l ≥ 0` and/or `u - x ≥ 0`).
- `P` is symmetrised as `(P + Pᵀ) / 2` and only its upper triangle is passed on,
  matching clarabel's convention — done explicitly here for deterministic
  behaviour.
- Bounds may be one-sided: a non-finite bound (`f64::INFINITY` /
  `f64::NEG_INFINITY`) means "unbounded on that side" and simply contributes no
  row. Pass `&[]` for no bounds at all; otherwise the slice length must equal
  the number of variables.
- Inputs are dense and converted to sparse CSC internally (exact zeros are
  dropped), so this API suits small to medium problems; use the re-exported
  `clarabel` directly for large sparse models.
- `ConvexError` distinguishes `DimensionMismatch`, `NotFinite` and `Solver`
  failures; `SolverStatus::Solved` and `AlmostSolved` count as success,
  everything else is reported as an error.
- The `dmatrix!`/`dvector!` macros need `nalgebra` as a direct dependency, as
  above. To avoid that, build inputs through the re-export instead — e.g.
  `tpt_math_optimize_convex::nalgebra::DMatrix::from_row_slice(1, 2, &[1.0, 1.0])`.
- `#![forbid(unsafe_code)]`.
- This crate is dual-licensed `MIT OR Apache-2.0`, matching the rest of the
  workspace; upstream `clarabel` is Apache-2.0. Review its licence when
  redistributing.

## License

Licensed under either of MIT or Apache-2.0 at your option.
