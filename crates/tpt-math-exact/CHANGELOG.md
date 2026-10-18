# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Interval::div_nonzero`: interval division, defined only when the divisor excludes zero.
- `roots::newton_refine`: interval Newton refinement of a root enclosure, returning `None` when a box is proven root-free.
- `roots::validated_euler` and `roots::validated_euler_system`: rigorous enclosures of the solution of a scalar ODE, or of a system, at the end of an interval, using an a-priori (Picard–Lindelöf) box per step.

### Added

- `roots` module: `isolate_roots` bisects on an interval extension and returns boxes that provably contain every root. `poly_enclosure` builds a valid extension for a polynomial. `poly_eval` and `has_sign_change` give point evaluation and a Bolzano certificate.

### Changed

- `Interval` addition and subtraction use two endpoint operations each (`[a+c, b+d]` and `[a-d, b-c]`) instead of the four-corner min/max used for products.
- `Interval::new` panics on `lo > hi` in release builds as well as debug builds.

### Fixed

- `num-bigint` is pinned to the 0.4 line that `num-rational` uses. The direct 0.5 dependency made the re-exported `BigInt` a different type from the `BigInt` inside `BigRational`, so `BigRational::from_integer(BigInt::from(n))` did not compile.

### Added

- `certified_solve` and `CertifyError`: certified enclosures of the solution of a linear system with `f64` data, by the Krawczyk operator. The approximate inverse is `f64` and is only a preconditioner. The contraction test and the enclosure use exact rational arithmetic, so each returned interval contains the exact solution of the given system. A system the method cannot contract is reported with `NotContracting`, which is not a claim that the system has no solution.

## [0.1.0]

- Initial workspace release.
