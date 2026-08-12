//! Shim crate for `tpt-math`.
//!
//! The workspace depends transitively on `argmin` and `faer`/`gemm`, both of
//! which invoke the `paste!` macro from the now-archived `paste` crate
//! (RUSTSEC-2024-0436). This crate is named `paste` and simply re-exports the
//! public API of [`pastey`] — the maintained, drop-in successor — so dependents
//! that write `paste::paste! { .. }` keep working while the archived crate is
//! kept entirely out of the dependency graph.
//!
//! It is wired in through the `[patch.crates-io]` entry in the workspace root
//! `Cargo.toml`. There is no crate named `paste` published by us; this exists
//! only to satisfy the name that `argmin`/`gemm` require.

pub use pastey::*;
