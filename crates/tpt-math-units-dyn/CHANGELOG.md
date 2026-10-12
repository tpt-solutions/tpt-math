# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `solve` module, behind the `ode`, `optimize` and `autodiff` features. `dopri5_dimensioned` integrates `y′ = f(t, y)` with every quantity dimension-checked. `minimize_dimensioned` runs L-BFGS on dimensioned parameters. `gradient_dimensioned` attaches the dimension `cost / parameter` to a reverse-mode gradient.
- `SolveError` reports a dimension mismatch (`Unit`), an ODE failure (`Ode`) or an optimiser failure (`Optimize`).


## [0.1.0]

- Initial workspace release.
