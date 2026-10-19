# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-20

### Fixed

- `numerical_gradient` scales its step with the coordinate, `h · max(1, |xᵢ|)`. An absolute step rounded to zero far from the origin and reported a flat gradient, which made the optimiser declare false convergence. Results are unchanged for `|xᵢ| ≤ 1`.
- The line search brackets its zoom on `[a_lo, a]` (it used a degenerate `[a, a]` bracket after the first expansion), and it rejects non-finite steps.
- Gradient descent and conjugate gradients stop with termination `LineSearchStalled` when no decrease can be found, instead of running to `max_iters`.

### Added

- `Solution::report()`, returning a `tpt_math_numeric::SolveReport`: `Converged`, `Breakdown` for a stalled line search, or `NotConverged`.
- `Solution::residual`: the gradient norm at the answer, or the cost spread over the simplex for Nelder–Mead.
- `linear_program::maximize`: two-phase simplex for `max cᵀx` subject to `A x ≤ b, x ≥ 0`, with Bland's rule against cycling. Returns `LpSolution` with `Optimal`, `Infeasible` or `Unbounded` status. Minimise by negating `c`.
- `minimize_nelder_mead` and `minimize_nelder_mead_with`: derivative-free simplex minimisation.
- `numerical_gradient`: central finite-difference gradients.
- `minimize_lbfgs` and `minimize_lbfgs_with`: limited-memory BFGS with the two-loop recursion and the strong-Wolfe line search.
- `levenberg_marquardt`: nonlinear least squares by Levenberg–Marquardt with Nielsen's damping update. It takes a residual function and its Jacobian, and returns a `LeastSquaresSolution` with the termination reason (`LsTermination`).
- `curve_fit`: fits `y ≈ f(x; p)` with a central-difference Jacobian.
- `LeastSquaresOptions::with_bounds`: componentwise box constraints. The solver projects each trial point onto the box and tests convergence on the projected gradient. This is a projected method, not an active-set one like L-BFGS-B.
- `minimize_lbfgs_autodiff`: L-BFGS with the gradient from reverse-mode autodiff. The cost is written once on the tape. The last evaluated point is cached, so each iterate is taped once. This adds a dependency on `tpt-math-autodiff-rev`.

### Changed

- `Solution` has a new public field `residual`. Code that builds a `Solution` with a struct literal must set it.
- **`argmin`/`argmin-math` dependency removed entirely.** The solvers behind
  `minimize_gradient_descent`, `minimize_conjugate_gradient` and
  `minimize_newton` (steepest descent, nonlinear CG, Newton's method) are now
  implemented in-house directly against `tpt-math-linalg-dense`'s
  `DVector`/`DMatrix`, with no external optimisation framework in the
  dependency graph. All public APIs (`minimize_*`, `Options`, `Solution`,
  `point_from_tagged`, `point_to_tagged`, `TaggedVec`, the
  `tpt_math_linalg`/`tpt_math_linalg_dense` re-exports) are unchanged in
  signature.
- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

## [0.1.0]

- Initial workspace release.
