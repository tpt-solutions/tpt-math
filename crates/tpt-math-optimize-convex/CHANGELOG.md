# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `QpSolution::iterations`: interior-point iterations performed before the solve was accepted. Adding this field breaks code that builds `QpSolution` with a struct literal.


- **`clarabel` backend replaced by an in-house dense primal-dual interior-point
  (Mehrotra predictor-corrector) QP solver.** `clarabel` is Apache-2.0-only and
  disqualified as a wrap target by the workspace license policy (ADR-0007). The
  new solver depends only on `tpt-math-linalg-dense` (in-house, no external
  backend); the KKT systems are inverted with the in-house `DMatrix::solve`. The public API is
  unchanged in signature except for `QpSolution::status`, which is now the local
  `QpStatus` enum (replacing clarabel's `SolverStatus`). Infeasible/unbounded
  problems are reported as `ConvexError::Solver`.

### Changed

- Residual tolerances scale with the problem data, so a uniformly scaled QP converges like the original. The complementarity gap is compared with the objective magnitude.
- `solve_qp` documents that a solution accepted as `QpStatus::AlmostSolved` is returned as `Ok`. Use `solve_qp_internal` to read the status.

## [0.1.0]

- Initial workspace release.
