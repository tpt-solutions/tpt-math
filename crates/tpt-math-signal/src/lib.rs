//! Umbrella crate re-exporting the `tpt-math-signal-*` family behind Cargo
//! features.
//!
//! # Feature matrix
//!
//! | Feature                   | Re-exported as | Source crate               |
//! |---------------------------|---------------|---------------------------|
//! | `tpt-math-signal-fft`     | `fft`         | `tpt-math-signal-fft`     |
//! | `tpt-math-signal-filter`  | `filter`      | `tpt-math-signal-filter`  |
//!
//! Both features are enabled by default.

#[cfg(feature = "tpt-math-signal-fft")]
pub use tpt_math_signal_fft as fft;

#[cfg(feature = "tpt-math-signal-filter")]
pub use tpt_math_signal_filter as filter;
