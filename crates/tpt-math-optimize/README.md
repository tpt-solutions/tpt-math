# tpt-math-optimize

Umbrella crate for the `tpt-math` optimisation layer: it re-exports
`tpt-math-optimize-general` (closure-driven wrappers over the
[`argmin`](https://crates.io/crates/argmin) framework) and
`tpt-math-optimize-convex` (a convex QP interface over an in-house dense
primal-dual interior-point solver) behind Cargo features. Depend on this crate
when you want both smooth unconstrained minimisation and constrained convex
programming from a single dependency line.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
This crate sits at the top of the optimisation layer and contains no algorithms
of its own; it is purely a re-export facade over the two leaf crates below it:

| Feature | Re-exported as | Source crate |
|---|---|---|
| `tpt-math-optimize-general` | `general` | [`tpt-math-optimize-general`](https://docs.rs/tpt-math-optimize-general) — closure-driven wrappers over `argmin` |
| `tpt-math-optimize-convex` | `convex` | [`tpt-math-optimize-convex`](https://docs.rs/tpt-math-optimize-convex) — convex QPs via an in-house dense interior-point solver |

Both leaf crates take their parameter vectors as `tpt-math-linalg-dense`
`DVector<f64>` / `DMatrix<f64>` (faer-backed, MIT-only), so values move
between the two solver families (and the rest of the workspace) without
conversion. The old `clarabel` (Apache-2.0-only) and `nalgebra` backends were
removed to satisfy the workspace license policy (ADR-0007); `tpt-math-linalg`
wraps `faer` (via `tpt-math-linalg-dense`) rather than `nalgebra`.

## Features

- `tpt-math-optimize-general` *(default)* — pulls in `tpt-math-optimize-general`
  and exposes it as the `general` module: `minimize_gradient_descent`,
  `minimize_conjugate_gradient`, `minimize_newton`, their `*_with` variants
  taking `Options` and returning a `Solution`, plus the unchanged `argmin` and
  `argmin_math` re-exports.
- `tpt-math-optimize-convex` *(default)* — pulls in `tpt-math-optimize-convex`
  and exposes it as the `convex` module: `solve_qp`, the `QuadraticProgram`
  builder, `QpSolution`, `ConvexError`, plus the unchanged
  `tpt_math_linalg_dense` re-export.
- Default: both features are on. Disable default features to take only the half
  you need — that also drops the corresponding upstream solver from the
  dependency graph.

This crate requires `std`; both `argmin` and the in-house QP solver are
`std`-only, so there is no `no_std` configuration of this umbrella.

## Quick start

```toml
[dependencies]
tpt-math-optimize = "0.1"
```

Unconstrained minimisation through the `general` module:

```rust
use tpt_math_optimize::general::tpt_math_linalg_dense::DVector;
use tpt_math_optimize::general::minimize_gradient_descent;

// f(x, y) = (x - 3)^2 + (y - 2)^2, minimised at (3, 2).
let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] - 2.0).powi(2);
let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 2.0)]);

let best = minimize_gradient_descent(cost, grad, DVector::zeros(2), 100).unwrap();

assert!((best[0] - 3.0).abs() < 1e-6);
assert!((best[1] - 2.0).abs() < 1e-6);
```

A constrained convex QP through the `convex` module:

```rust
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_optimize::convex::solve_qp;

// minimize x^2 + y^2  subject to  x + y = 1  ->  (0.5, 0.5)
let p = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
let q = DVector::zeros(2);
let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
let b_eq = DVector::from_vec(vec![1.0]);

let x = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap();

assert!((x[0] - 0.5).abs() < 1e-6);
assert!((x[1] - 0.5).abs() < 1e-6);
```

Taking only one half:

```toml
[dependencies]
tpt-math-optimize = { version = "0.1", default-features = false, features = ["tpt-math-optimize-convex"] }
```

## Notes

- The feature names deliberately match the crate names, so
  `--features tpt-math-optimize-convex` enables exactly the
  `tpt-math-optimize-convex` dependency and nothing else.
- Nothing is hidden: `general` re-exports `argmin` (and `argmin::core`,
  `argmin::solver`, `argmin_math`) and `convex` re-exports
  `tpt_math_linalg_dense`, so you can drop to the upstream APIs without adding a
  second, possibly version-skewed, dependency on them.
- `general` flattens `argmin` errors to `String`; `convex` returns a typed
  `ConvexError`. The two leaf crates are independent and do not share an error
  type.
- Upstream licences differ from this crate's: `argmin` is `MIT OR Apache-2.0`
  and `tpt-math-linalg-dense` is `MIT`-only (faer). The old `clarabel`
  (Apache-2.0-only) backend was removed; review the remaining upstream licences
  when redistributing.

## License

Licensed under either of MIT or Apache-2.0 at your option.
