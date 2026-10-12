# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Sums and products are ordered by their printed form. `simplify` is now a normal form: a second pass returns the first pass's output. The order of terms in a simplified sum or product can differ from earlier versions.


### Added

- Named functions `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`: folded numerically by `simplify` and differentiated by the chain rule. Checked against central differences.

### Changed

- Unknown function names stay symbolic: `simplify` leaves them unfolded, and `eval` returns `None`. They used to be treated as the identity.
- `to_f64` parses `p/q` rationals and returns NaN for unparseable input instead of 0.

### Added

- Like terms are merged in sums: `2x + 3x` simplifies to `5x`, `x − x` to `0`, and `x·y + 2(x·y)` to `3(x·y)`.

## [0.1.0]

- Initial workspace release.
