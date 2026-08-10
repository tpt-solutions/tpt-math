//! Bayesian inference primitives.
//!
//! Small, dependency-free building blocks for exact conjugate updates and for
//! approximate inference when conjugacy runs out:
//!
//! * [`Beta`] — conjugate prior for a Bernoulli/Binomial success rate.
//! * [`Gaussian`] (aliased [`Normal`]) — density, sampling, and the
//!   Normal–Normal conjugate update for an unknown mean with known variance.
//! * [`Gamma`] — conjugate prior for a Poisson rate, and the engine behind
//!   Beta sampling.
//! * [`Metropolis`] — a random-walk Metropolis(–Hastings) sampler for any
//!   unnormalised scalar log-target.
//! * [`special`] — the log-gamma/log-beta/normal-CDF machinery underneath.
//! * Free log-likelihood helpers such as [`bernoulli_log_likelihood`],
//!   [`normal_log_likelihood`], and [`log_sum_exp`].
//!
//! Every distribution samples through the [`Rng`] and [`Distribution`] traits
//! from `tpt-math-prob-core`, so a deterministic generator such as
//! `SplitMix64` makes any inference run exactly reproducible.
//!
//! # Conjugate updates
//!
//! ```
//! use tpt_math_prob_bayes::Beta;
//!
//! // Uniform prior over a coin's bias, then 7 heads and 3 tails.
//! let posterior = Beta::uniform().update(7, 3);
//! assert_eq!((posterior.alpha(), posterior.beta()), (8.0, 4.0));
//! assert!((posterior.mean() - 2.0 / 3.0).abs() < 1e-12);
//!
//! // Posterior belief with 95% probability mass.
//! let (lo, hi) = posterior.credible_interval(0.95);
//! assert!(lo < posterior.mean() && posterior.mean() < hi);
//! ```
//!
//! # Approximate inference
//!
//! ```
//! use tpt_math_prob_bayes::{Beta, Metropolis};
//! use tpt_math_prob_core::SplitMix64;
//!
//! let prior = Beta::new(2.0, 2.0);
//! let (successes, failures) = (30, 70);
//!
//! // Sample the same posterior with MCMC instead of the conjugate formula.
//! let target = move |p: f64| prior.log_unnormalized_posterior(p, successes, failures);
//! let mut sampler = Metropolis::with_gaussian_proposal(target, 0.1);
//! let mut rng = SplitMix64::seed_from_u64(2024);
//! let trace = sampler.run_with_burn_in(&mut rng, 0.5, 2_000, 40_000);
//!
//! let mcmc_mean = trace.iter().sum::<f64>() / trace.len() as f64;
//! let exact_mean = prior.update(successes, failures).mean();
//! assert!((mcmc_mean - exact_mean).abs() < 0.01);
//! ```

use core::fmt;

mod beta;
mod gamma;
mod likelihood;
mod mcmc;
mod normal;
pub mod special;

pub use beta::Beta;
pub use gamma::Gamma;
pub use likelihood::{
    bernoulli_log_likelihood, binomial_log_likelihood, exponential_log_likelihood, log_sum_exp,
    normal_log_likelihood, poisson_log_likelihood,
};
pub use mcmc::Metropolis;
pub use normal::{Gaussian, Normal};

/// The randomness traits this crate is built on, re-exported for convenience.
pub use tpt_math_prob_core::{Distribution, Rng, Sampler};

/// An invalid distribution parameter.
///
/// Returned by the `try_new` constructors; the panicking `new` constructors
/// report the same condition in their panic message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterError {
    /// The parameter was `NaN` or infinite.
    NotFinite {
        /// Name of the offending parameter.
        parameter: &'static str,
    },
    /// The parameter must be strictly positive but was zero or negative.
    NotPositive {
        /// Name of the offending parameter.
        parameter: &'static str,
    },
}

impl fmt::Display for ParameterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParameterError::NotFinite { parameter } => {
                write!(f, "`{parameter}` must be finite")
            }
            ParameterError::NotPositive { parameter } => {
                write!(f, "`{parameter}` must be strictly positive")
            }
        }
    }
}

impl std::error::Error for ParameterError {}

/// Validate that `value` is finite.
pub(crate) fn check_finite(parameter: &'static str, value: f64) -> Result<(), ParameterError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ParameterError::NotFinite { parameter })
    }
}

/// Validate that `value` is finite and strictly positive.
pub(crate) fn check_positive(parameter: &'static str, value: f64) -> Result<(), ParameterError> {
    check_finite(parameter, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ParameterError::NotPositive { parameter })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_error_displays_the_parameter_name() {
        let err = check_positive("alpha", -1.0).unwrap_err();
        assert_eq!(err, ParameterError::NotPositive { parameter: "alpha" });
        assert_eq!(err.to_string(), "`alpha` must be strictly positive");

        let err = check_finite("mean", f64::NAN).unwrap_err();
        assert_eq!(err, ParameterError::NotFinite { parameter: "mean" });
        assert_eq!(err.to_string(), "`mean` must be finite");

        assert!(check_positive("rate", 1.0).is_ok());
        assert!(check_finite("mean", -3.0).is_ok());
        assert!(check_positive("rate", f64::INFINITY).is_err());
    }

    #[test]
    fn parameter_error_is_a_std_error() {
        fn as_error(e: impl std::error::Error) -> String {
            e.to_string()
        }
        assert!(as_error(ParameterError::NotFinite { parameter: "std" }).contains("std"));
    }
}
