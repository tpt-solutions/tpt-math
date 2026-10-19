# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-20

### Added

- `report` module: `SolveStatus` (`Converged`, `NotConverged`, `Breakdown`) and `SolveReport<T>` { status, iterations, residual }, shared by the solver crates.
- `quadrature::gauss_kronrod_15` (15-point Gauss–Kronrod pair, exact to degree 22, returning the embedded error estimate) and `quadrature::adaptive_gauss_kronrod` (bisection on that estimate, `QuadratureError::MaxDepth` on failure). Handles integrable endpoint singularities such as `1/√x`.
- `quadrature::gauss_legendre` (fixed `N`-point rule, nodes by Newton iteration, exact to degree `2N − 1`) and `quadrature::adaptive_gauss_legendre` (bisection on a panel-difference error test, returning `QuadratureError::MaxDepth` on failure).
- `ode::dopri5_integrate`: adaptive Dormand–Prince 5(4) integrator with relative and absolute tolerances and a step-attempt budget. Integrates backwards when `t1 < t0`. Returns `OdeError::{MaxStepsExceeded, StepSizeUnderflow}` on failure.
- `roots` module: `bisect` and `brent` bracketing root finders, with `RootError`.
- `quadrature` module: composite `simpson` rule.
- `ode` module: fixed-step classical Runge–Kutta, `rk4_step` and `rk4_integrate`, on fixed-size state arrays (no allocation).
- `ode::dopri5_dense` and `DenseSolution`: adaptive Dormand–Prince 5(4) with cubic Hermite dense output, `eval(t)` at any time within the span.
- `ode::dopri5_events`, `Crossing`, `Event` and `EventRun`: locates sign changes of an event function `g(t, y)` by bisection on the dense output, with optional termination at the first event.
- `stiff::rosenbrock2_integrate`: stiff solver, the L-stable second-order Rosenbrock method ROS2 with step-doubling error control. It takes the Jacobian and handles autonomous systems. Non-autonomous systems are made autonomous by adding `t` as a state.
- The `alloc` feature now enables `extern crate alloc`, which the dense-output and event code needs.
- `sensitivity::forward_sensitivity`, with `Sensitivities`: forward sensitivity analysis, integrating `y' = f(t, y, p)` and `∂y/∂p` together as the augmented variational system. It takes explicit Jacobians with respect to the state and the parameters, and uses a runtime-sized Dormand–Prince (`dopri5_vec`). It is not generic over a dual-number scalar, because `Dual` has no `Float` implementation.

### Changed

- The Dormand–Prince step is shared by `dopri5_integrate`, `dopri5_dense` and `dopri5_events`. `dopri5_integrate` behaves as before.
- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

## [0.1.0]

- Initial workspace release.
