//! Umbrella crate re-exporting the `tpt-math-prob-*` family behind Cargo
//! features.
//!
//! # Feature matrix
//!
//! | Feature                     | Re-exported as | Source crate            |
//! |-----------------------------|---------------|------------------------|
//! | `tpt-math-prob-dist`        | `dist`        | `tpt-math-prob-dist`   |
//! | `tpt-math-prob-bayes`      | `bayes`       | `tpt-math-prob-bayes`  |
//! | `tpt-math-prob-markov`     | `markov`      | `tpt-math-prob-markov` |
//! | `tpt-math-prob-monte-carlo`| `monte_carlo` | `tpt-math-prob-monte-carlo` |
//! | `tpt-math-prob-sampler`    | `sampler`     | `tpt-math-prob-sampler`|
//!
//! All features are enabled by default. Disable `default-features` and opt in
//! to only the constituents you need to keep your dependency tree small.

#[cfg(feature = "tpt-math-prob-dist")]
pub use tpt_math_prob_dist as dist;

#[cfg(feature = "tpt-math-prob-bayes")]
pub use tpt_math_prob_bayes as bayes;

#[cfg(feature = "tpt-math-prob-markov")]
pub use tpt_math_prob_markov as markov;

#[cfg(feature = "tpt-math-prob-monte-carlo")]
pub use tpt_math_prob_monte_carlo as monte_carlo;

#[cfg(feature = "tpt-math-prob-sampler")]
pub use tpt_math_prob_sampler as sampler;
