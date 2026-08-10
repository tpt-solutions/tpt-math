//! Umbrella crate re-exporting the autodiff family behind Cargo features.
//!
//! # Feature matrix
//!
//! | Feature                   | Re-exported as | Source crate              |
//! |---------------------------|---------------|--------------------------|
//! | `tpt-math-autodiff-fwd`   | `fwd`         | `tpt-math-autodiff-fwd`   |
//! | `tpt-math-autodiff-rev`   | `rev`         | `tpt-math-autodiff-rev`   |
//!
//! Both features are enabled by default.

#[cfg(feature = "tpt-math-autodiff-fwd")]
pub use tpt_math_autodiff_fwd as fwd;

#[cfg(feature = "tpt-math-autodiff-rev")]
pub use tpt_math_autodiff_rev as rev;
