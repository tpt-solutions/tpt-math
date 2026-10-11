# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- **`argmin`/`argmin-math` dependency removed entirely.** The solvers behind
  `minimize_gradient_descent`, `minimize_conjugate_gradient` and
  `minimize_newton` (steepest descent, nonlinear CG, Newton's method) are now
  implemented in-house directly against `tpt-math-linalg-dense`'s
  `DVector`/`DMatrix`, with no external optimisation framework in the
  dependency graph. All public APIs (`minimize_*`, `Options`, `Solution`,
  `point_from_tagged`, `point_to_tagged`, `TaggedVec`, the
  `tpt_math_linalg`/`tpt_math_linalg_dense` re-exports) are unchanged in
  signature.

### Fixed

- The line search brackets its zoom on `[a_lo, a]` (it used a degenerate `[a, a]` bracket after the first expansion), and it rejects non-finite steps.
- Gradient descent and conjugate gradients stop with termination `LineSearchStalled` when no decrease can be found, instead of running to `max_iters`.

### Added

- `linear_program::maximize`: two-phase simplex for `max cᵀx` subject to `A x ≤ b, x ≥ 0`, with Bland's rule against cycling. Returns `LpSolution` with `Optimal`, `Infeasible` or `Unbounded` status. Minimise by negating `c`.
- `minimize_nelder_mead` and `minimize_nelder_mead_with`: derivative-free simplex minimisation.
- `numerical_gradient`: central finite-difference gradients.
- `minimize_lbfgs` and `minimize_lbfgs_with`: limited-memory BFGS with the two-loop recursion and the strong-Wolfe line search.

## [0.1.0]

- Initial workspace release.
