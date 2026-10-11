# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- The derivative of `Powi(a, 0)` and `Powf(a, 0)` is `0` at `a = 0` (it was NaN from `0 · 0⁻¹`).

## [0.1.0]

- Initial workspace release.
