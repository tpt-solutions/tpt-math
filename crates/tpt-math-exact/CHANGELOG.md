# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- `BigRational` is now implemented in this crate instead of re-exported from `num-rational`. It is always stored in lowest terms with a positive denominator. The `num-rational` dependency is removed and `num-integer` 0.1 is added; `num-bigint` stays on the 0.4 line.

### Added

- `BigRational` methods beyond the `num-rational` surface: `from_float` (exact conversion of an `f64`), `to_f64_down` and `to_f64_up` (directed-rounding conversion to `f64`), `sqrt_bounds` (rational bounds on a square root), `continued_fraction`, `from_continued_fraction` and `limit_denominator` (best rational approximations), and `to_decimal_string`.
- `BigRational::pow`, `floor`, `ceil`, `trunc`, `fract`, `round`, `recip` and `abs`, plus `numer`, `denom` and `is_integer`.
- `FromStr` for `BigRational`, accepting `p/q` fractions and decimals such as `-1.25e3`, with `ParseRationalError` (`Invalid`, `ZeroDenominator`) as the error type.
- `Zero`, `One`, `ToPrimitive`, `FromPrimitive`, `Ord`, `Hash` and `Display` for `BigRational`, and the usual arithmetic operators (owned and borrowed, including the assigning forms).

## [0.1.1] - 2026-10-20

### Added

- `Interval::div_nonzero`: interval division, defined only when the divisor excludes zero.
- `roots::newton_refine`: interval Newton refinement of a root enclosure, returning `None` when a box is proven root-free.
- `roots::validated_euler` and `roots::validated_euler_system`: rigorous enclosures of the solution of a scalar ODE, or of a system, at the end of an interval, using an a-priori (Picard–Lindelöf) box per step.
- `roots` module: `isolate_roots` bisects on an interval extension and returns boxes that provably contain every root. `poly_enclosure` builds a valid extension for a polynomial. `poly_eval` and `has_sign_change` give point evaluation and a Bolzano certificate.
- `certified_solve` and `CertifyError`: certified enclosures of the solution of a linear system with `f64` data, by the Krawczyk operator. The approximate inverse is `f64` and is only a preconditioner. The contraction test and the enclosure use exact rational arithmetic, so each returned interval contains the exact solution of the given system. A system the method cannot contract is reported with `NotContracting`, which is not a claim that the system has no solution.

### Changed

- `Interval` addition and subtraction use two endpoint operations each (`[a+c, b+d]` and `[a-d, b-c]`) instead of the four-corner min/max used for products.
- `Interval::new` panics on `lo > hi` in release builds as well as debug builds.

### Fixed

- `num-bigint` is pinned to the 0.4 line that `num-rational` uses. The direct 0.5 dependency made the re-exported `BigInt` a different type from the `BigInt` inside `BigRational`, so `BigRational::from_integer(BigInt::from(n))` did not compile.

## [0.1.0]

- Initial workspace release.
