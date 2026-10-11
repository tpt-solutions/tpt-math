# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The MCMC run methods (`run`, `run_with_burn_in`, `run_thinned`) panic when the starting state has a NaN log-density, instead of returning a chain that never moves.

## [0.1.0]

- Initial workspace release.
