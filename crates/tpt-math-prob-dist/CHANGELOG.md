# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `density` families: Student-t (`student_t_pdf`, `student_t_cdf`, `student_t_moments`), Cauchy (`cauchy_pdf`, `cauchy_cdf`, `cauchy_moments`) and chi-squared (`chi_squared_pdf`, `chi_squared_cdf`, `chi_squared_moments`).
- `density` moments: `normal_moments`, `exponential_moments`, `gamma_moments`, `beta_moments`, `uniform_moments` and `poisson_moments`, each returning `(mean, variance)`.
- `density` module (requires `std`): PDF/PMF, CDF and quantile functions for Normal, Exponential, Gamma (plus survival), Beta, Uniform and Poisson. Invalid parameters return `NaN`.

## [0.1.0]

- Initial workspace release.
