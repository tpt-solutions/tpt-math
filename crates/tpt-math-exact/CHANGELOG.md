# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- `Interval` addition and subtraction use two endpoint operations each (`[a+c, b+d]` and `[a-d, b-c]`) instead of the four-corner min/max used for products.
- `Interval::new` panics on `lo > hi` in release builds as well as debug builds.

### Fixed

- `num-bigint` is pinned to the 0.4 line that `num-rational` uses. The direct 0.5 dependency made the re-exported `BigInt` a different type from the `BigInt` inside `BigRational`, so `BigRational::from_integer(BigInt::from(n))` did not compile.

## [0.1.0]

- Initial workspace release.
