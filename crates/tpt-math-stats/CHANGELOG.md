# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `pearson_correlation` rescales the samples when the raw sums overflow, so values near `1e160` no longer produce NaN. Ordinary data keep the exact unscaled arithmetic.

## [0.1.0]

- Initial workspace release.
