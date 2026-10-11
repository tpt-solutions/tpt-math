//! Special functions backing the conjugate-prior distributions.
//!
//! The implementations live in [`tpt_math_prob_core::special`], shared with
//! `tpt-math-stats`. They are re-exported here so this crate's public path
//! (`tpt_math_prob_bayes::special`) is unchanged.

pub use tpt_math_prob_core::special::{
    erf, erfc, ln_beta, ln_binomial, ln_factorial, ln_gamma, regularized_incomplete_beta,
    standard_normal_cdf, standard_normal_pdf, standard_normal_quantile, xlogy, LN_2PI,
};
