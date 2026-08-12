# tpt-math-optimize-general

General numerical optimisation: in-house closure-driven minimisers for the
common smooth, unconstrained cases (steepest descent, nonlinear conjugate
gradient, Newton), built directly on `tpt-math-linalg-dense`'s
`DVector`/`DMatrix`. A cost function and a starting point are enough to get
an answer, with no external optimisation framework in the dependency graph.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
It is the general-purpose half of the optimisation layer (the convex/QP half is
`tpt-math-optimize-convex`), and is re-exported by the `tpt-math-optimize`
umbrella crate. Parameters are plain `tpt-math-linalg-dense` `DVector<f64>`s from the same
faer-backed storage that `tpt-math-linalg` wraps.

## Features

This crate has no optional features: `default = []` and everything described
here is always available. It is **std-only** — `tpt-math-linalg-dense`'s
default (allocator-backed, `faer`) storage requires `std`, so there is no
`no_std` build.

The dependency set is fixed rather than feature-gated: `tpt-math-linalg` and
`tpt-math-linalg-dense`.

## Quick start

```toml
[dependencies]
tpt-math-optimize-general = "0.1"
```

Minimise `f(x, y) = (x - 3)² + (y - 2)²` by gradient descent:

```rust
use tpt_math_optimize_general::tpt_math_linalg_dense::DVector;
use tpt_math_optimize_general::minimize_gradient_descent;

let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] - 2.0).powi(2);
let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 2.0)]);

let best = minimize_gradient_descent(cost, grad, DVector::zeros(2), 100).unwrap();

assert!((best[0] - 3.0).abs() < 1e-6);
assert!((best[1] - 2.0).abs() < 1e-6);
```

The three convenience solvers:

| Function | Method | Needs |
|---|---|---|
| `minimize_gradient_descent` | steepest descent + More-Thuente line search | cost, gradient |
| `minimize_conjugate_gradient` | nonlinear CG (Polak–Ribière+, with periodic restarts) + More-Thuente line search | cost, gradient |
| `minimize_newton` | Newton's method (full step, analytic Hessian) | cost, gradient, Hessian |

Each has a `*_with` variant that takes `Options` and returns a `Solution`
reporting the parameter vector, the cost re-evaluated there, the iteration
count, whether the gradient tolerance was met, and a human-readable
termination reason:

```rust
use tpt_math_optimize_general::tpt_math_linalg_dense::DVector;
use tpt_math_optimize_general::{minimize_conjugate_gradient_with, Options};

let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] + 1.0).powi(2);
let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 1.0)]);

let options = Options::new(50).with_gradient_tolerance(1e-10);
let solution = minimize_conjugate_gradient_with(cost, grad, DVector::zeros(2), &options).unwrap();

assert!(solution.converged);
assert!(solution.cost < 1e-12);
```

## Notes

- Nothing is hidden: `tpt_math_linalg` and `tpt_math_linalg_dense` are both
  re-exported, so you can build on the same parameter types without a second,
  possibly version-skewed, dependency on them.
- Errors are flattened to a `String` message rather than a typed error type.
- Every run stops at `max_iters` or earlier once the gradient's L2 norm falls to
  `Options::gradient_tolerance` (default `1e-9`). That early stop matters: line
  searches fail outright on a numerically zero gradient, so without it an
  already-converged run would report an error instead of its answer.
- Optimisation is deliberately unit-less — a cost mixes every unit in the
  problem — but unit-tagged `tpt_math_linalg::Vec` values can be moved in and
  out with `point_from_tagged` / `point_to_tagged` (`TaggedVec<U>`).
- `minimize_newton` takes no line search; the step is the full Newton step.
- Empty, non-finite initial points and negative gradient tolerances are rejected
  up front with a clear message.
- `#![forbid(unsafe_code)]`.
- This crate is dual-licensed `MIT OR Apache-2.0`, matching the rest of the
  workspace, and depends only on `tpt-math-linalg`/`tpt-math-linalg-dense`
  (both also part of this workspace) — no third-party optimisation framework.

## License

Licensed under either of MIT or Apache-2.0 at your option.
