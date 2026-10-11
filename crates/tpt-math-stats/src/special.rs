//! Special functions used by the distributions and tests in this crate.
//!
//! The implementations live in [`tpt_math_prob_core::special`], shared with
//! `tpt-math-prob-bayes`. This module keeps the names this crate has always
//! used (`lgamma`, `beta_reg`).

pub use tpt_math_prob_core::special::{
    beta, erf, erfc, gamma, gamma_p, gamma_q, ln_gamma as lgamma,
    regularized_incomplete_beta as beta_reg,
};
