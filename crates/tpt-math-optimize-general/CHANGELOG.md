# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- **`ArgminMath` backend swapped from `nalgebra` to `tpt-math-linalg-dense`
  (faer).** `nalgebra` is Apache-2.0-only and disqualified as a wrap target by
  the workspace license policy (ADR-0007). The `argmin-math` `nalgebra_v0_33`
  backend feature is dropped; the `ArgminMath`-family impls for
  `DVector<f64>`/`DMatrix<f64>` now come from `tpt-math-linalg-dense`'s `argmin`
  feature, which also resolves the orphan-rule issue. All public APIs
  (`minimize_*`, `point_from_tagged`, `point_to_tagged`, `TaggedVec`, the
  re-exports) are unchanged in signature.

## [0.1.0]

- Initial workspace release.
