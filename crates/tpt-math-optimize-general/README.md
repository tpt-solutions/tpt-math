# tpt-math-optimize-general

General numerical optimisation: a thin wrapper around
[`argmin`](https://docs.rs/argmin). It re-exports argmin unchanged and adds
closure-driven entry points for the common smooth, unconstrained minimisation
cases, so a cost function and a starting point are enough to get an answer
without hand-rolling argmin's problem traits, executor and state plumbing.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
It is the general-purpose half of the optimisation layer (the convex/QP half is
`tpt-math-optimize-convex`), and is re-exported by the `tpt-math-optimize`
umbrella crate. Parameters are plain `nalgebra` `DVector<f64>`s from the very
same `nalgebra` that `tpt-math-linalg` wraps.

## Features

This crate has no optional features: `default = []` and everything described
here is always available. It is **std-only** — `argmin` and its `Executor`
require `std`, so there is no `no_std` build.

The dependency set is fixed rather than feature-gated: `argmin`, `argmin-math`
with the `primitives` and `nalgebra_v0_33` backends (the `ArgminMath` impls
needed for `DVector`/`DMatrix` parameters, pinned to the nalgebra version used
by `tpt-math-linalg`), and `tpt-math-linalg` itself.

## Quick start

```toml
[dependencies]
tpt-math-optimize-general = "0.1"
```

Minimise `f(x, y) = (x - 3)² + (y - 2)²` by gradient descent:

```rust
use tpt_math_optimize_general::{minimize_gradient_descent, nalgebra::DVector};

let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] - 2.0).powi(2);
let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 2.0)]);

let best = minimize_gradient_descent(cost, grad, DVector::zeros(2), 100).unwrap();

assert!((best[0] - 3.0).abs() < 1e-6);
assert!((best[1] - 2.0).abs() < 1e-6);
```

The three convenience solvers:

| Function | argmin solver | Needs |
|---|---|---|
| `minimize_gradient_descent` | `SteepestDescent` + `MoreThuenteLineSearch` | cost, gradient |
| `minimize_conjugate_gradient` | `NonlinearConjugateGradient` (Polak–Ribière+, with restarts) + `MoreThuenteLineSearch` | cost, gradient |
| `minimize_newton` | `Newton` | cost, gradient, Hessian |

Each has a `*_with` variant that takes `Options` and returns a `Solution`
reporting the parameter vector, the cost re-evaluated there, the iteration
count, whether the gradient tolerance was met, and argmin's termination reason:

```rust
use tpt_math_optimize_general::{minimize_conjugate_gradient_with, Options, nalgebra::DVector};

let cost = |p: &DVector<f64>| (p[0] - 3.0).powi(2) + (p[1] + 1.0).powi(2);
let grad = |p: &DVector<f64>| DVector::from_vec(vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 1.0)]);

let options = Options::new(50).with_gradient_tolerance(1e-10);
let solution = minimize_conjugate_gradient_with(cost, grad, DVector::zeros(2), &options).unwrap();

assert!(solution.converged);
assert!(solution.cost < 1e-12);
```

## Notes

- Nothing is hidden: `argmin`, `argmin::core`, `argmin::solver`, `argmin_math`,
  `tpt_math_linalg` and `nalgebra` are all re-exported, so you can drop down to
  a hand-built `Executor` whenever the conveniences are too narrow.
- Errors are flattened to `String` so callers need not depend on `argmin` to
  handle them; use the re-exported `argmin` directly if you need a typed
  `argmin::core::Error`.
- Every run stops at `max_iters` or earlier once the gradient's L2 norm falls to
  `Options::gradient_tolerance` (default `1e-9`). That early stop matters: line
  searches fail outright on a numerically zero gradient, so without it an
  already-converged run would report an error instead of its answer.
- Optimisation is deliberately unit-less — a cost mixes every unit in the
  problem — but unit-tagged `tpt_math_linalg::Vec` values can be moved in and
  out with `point_from_tagged` / `point_to_tagged` (`TaggedVec<U>`).
- argmin 0.11's `Newton` takes no line search; the step is the full Newton step
  scaled by a fixed `gamma`. Build `Newton::with_gamma` via the re-exported
  `argmin` for a damped variant.
- Empty, non-finite initial points and negative gradient tolerances are rejected
  up front with a clear message.
- `#![forbid(unsafe_code)]`.
- This crate is dual-licensed `MIT OR Apache-2.0`, matching the rest of the
  workspace; upstream `argmin` and `argmin-math` are also `MIT OR Apache-2.0`.
  Review the upstream licences when redistributing.

## License

Licensed under either of MIT or Apache-2.0 at your option.
