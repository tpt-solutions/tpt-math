//! Umbrella crate re-exporting the `tpt-math-optimize-*` family behind Cargo
//! features.
//!
//! # Feature matrix
//!
//! | Feature                      | Re-exported as | Source crate                 |
//! |------------------------------|---------------|-----------------------------|
//! | `tpt-math-optimize-general`  | `general`     | `tpt-math-optimize-general` |
//! | `tpt-math-optimize-convex`   | `convex`      | `tpt-math-optimize-convex`  |
//!
//! Both features are enabled by default.

#[cfg(feature = "tpt-math-optimize-general")]
pub use tpt_math_optimize_general as general;

#[cfg(feature = "tpt-math-optimize-convex")]
pub use tpt_math_optimize_convex as convex;
