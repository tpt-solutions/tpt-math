# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `report` module: `SolveStatus` (`Converged`, `NotConverged`, `Breakdown`) and `SolveReport<T>` { status, iterations, residual }, shared by the solver crates.
- `quadrature::gauss_kronrod_15` (15-point Gauss–Kronrod pair, exact to degree 22, returning the embedded error estimate) and `quadrature::adaptive_gauss_kronrod` (bisection on that estimate, `QuadratureError::MaxDepth` on failure). Handles integrable endpoint singularities such as `1/√x`.
- `quadrature::gauss_legendre` (fixed `N`-point rule, nodes by Newton iteration, exact to degree `2N − 1`) and `quadrature::adaptive_gauss_legendre` (bisection on a panel-difference error test, returning `QuadratureError::MaxDepth` on failure).
- `ode::dopri5_integrate`: adaptive Dormand–Prince 5(4) integrator with relative and absolute tolerances and a step-attempt budget. Integrates backwards when `t1 < t0`. Returns `OdeError::{MaxStepsExceeded, StepSizeUnderflow}` on failure.
- `roots` module: `bisect` and `brent` bracketing root finders, with `RootError`.
- `quadrature` module: composite `simpson` rule.
- `ode` module: fixed-step classical Runge–Kutta, `rk4_step` and `rk4_integrate`, on fixed-size state arrays (no allocation).

## [0.1.0]

- Initial workspace release.
