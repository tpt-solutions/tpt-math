# tpt-math workspace tasks.
#
# Thin wrappers around `cargo xtask` so the same commands work locally and in
# CI. Run `just` (or `just --list`) to see everything.

default:
    @just --list

# Run the full verification gate: fmt --check + clippy + cargo-deny.
check:
    cargo xtask check

# Format the whole workspace.
fmt:
    cargo xtask fmt

# Run clippy with -D warnings across all targets/features.
clippy:
    cargo xtask clippy

# Run the workspace test suite (all features).
test:
    cargo xtask test

# Run `cargo-deny` checks.
deny:
    cargo xtask deny

# Build the `no_std = true` crates for thumbv6m-none-eabi.
no-std:
    cargo xtask no-std

# Run all four cross-crate examples.
examples:
    cargo run --quiet -p tpt-math-examples --bin units_linalg
    cargo run --quiet -p tpt-math-examples --bin prob_stats
    cargo run --quiet -p tpt-math-examples --bin autodiff_optimize
    cargo run --quiet -p tpt-math-examples --bin symbolic_exact
