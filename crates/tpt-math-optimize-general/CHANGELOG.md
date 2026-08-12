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

## [0.1.0]

- Initial workspace release.
