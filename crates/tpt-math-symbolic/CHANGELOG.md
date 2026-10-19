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

### Added

- `parse_expr`: parses infix text such as `x^2 + sin(x)` into an `Expr`, with positioned `ParseError`s. Exponents must be non-negative integer literals.
- `Expr::to_latex`: LaTeX output for an expression.
- `compile`, `Expr::compile`, `Compiled` and `CompileError`: compiles an expression to a closure over an ordered variable list. Variable and function names are checked at compile time, and the closure returns the same values as `eval`, including `None` on division by zero.

### Changed

- **Breaking (0.2.0):** the public error and status enums are `#[non_exhaustive]`, so matches in other crates need a wildcard arm. The crate is bumped to 0.2.0 for this and for the other breaking changes listed above.

## [0.1.0]

- Initial workspace release.
