# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `adaptive` module: `run_adaptive_metropolis`, adaptive random-walk Metropolis with the Haario proposal covariance (`2.38²/d` scaling), adapted until a caller-chosen iteration and frozen afterwards.
- `hmc` module: `run_hmc`, Hamiltonian Monte Carlo with a leapfrog integrator, unit mass matrix and a caller-supplied gradient.
- `diagnostics` module: `split_r_hat` (split-R̂ across chains), `effective_sample_size` (Geyer initial positive sequence) and `importance_estimate` (self-normalised importance sampling with its ESS).

### Changed

- `special` is re-exported from `tpt-math-prob-core::special`, replacing the local copy. The public paths are unchanged. `erf`, `erfc` and `standard_normal_cdf` now keep relative precision in the tails and near zero.
- The MCMC run methods (`run`, `run_with_burn_in`, `run_thinned`) panic when the starting state has a NaN log-density, instead of returning a chain that never moves.
- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

## [0.1.0]

- Initial workspace release.
