# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Unknown function names stay symbolic: `simplify` leaves them unfolded, and `eval` returns `None`. They used to be treated as the identity.
- `to_f64` parses `p/q` rationals and returns NaN for unparseable input instead of 0.

## [0.1.0]

- Initial workspace release.
