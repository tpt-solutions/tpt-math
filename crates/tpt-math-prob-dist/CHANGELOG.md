# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `student_t_cdf` lost the centre of the distribution for small `|x|`, because it evaluated the incomplete beta at an argument that rounds to 1. It now uses the `P(|T| ≤ |x|)` identity in whichever form is accurate for that `x`.

### Added

- `density` quantiles for the remaining families: `gamma_quantile`, `beta_quantile`, `chi_squared_quantile`, `student_t_quantile` and `cauchy_quantile`, plus the discrete `poisson_quantile` (smallest `k` with `P(X ≤ k) ≥ p`, returned as a float). The continuous ones invert their CDFs by bisection, except Cauchy, which is closed form. Quantiles now cover every family in the module.
- `density` families: Student-t (`student_t_pdf`, `student_t_cdf`, `student_t_moments`), Cauchy (`cauchy_pdf`, `cauchy_cdf`, `cauchy_moments`) and chi-squared (`chi_squared_pdf`, `chi_squared_cdf`, `chi_squared_moments`).
- `density` moments: `normal_moments`, `exponential_moments`, `gamma_moments`, `beta_moments`, `uniform_moments` and `poisson_moments`, each returning `(mean, variance)`.
- `density` module (requires `std`): PDF/PMF, CDF and quantile functions for Normal, Exponential, Gamma (plus survival), Beta, Uniform and Poisson. Invalid parameters return `NaN`.

### Added

- `lognormal_pdf`, `lognormal_cdf`, `lognormal_quantile` and `lognormal_moments`.
- `weibull_pdf`, `weibull_cdf`, `weibull_quantile` and `weibull_moments`.
- `binomial_pmf`, `binomial_cdf` and `binomial_moments`.
- `negative_binomial_pmf`, `negative_binomial_cdf` and `negative_binomial_moments`.
- Optional `multivariate` feature (needs `std`): the multivariate normal in `prob_dist::multivariate`, using the dense Cholesky factor.

## [0.1.0]

- Initial workspace release.
