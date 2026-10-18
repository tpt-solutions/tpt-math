# Contributing to `tpt-math`

`tpt-math` is the internal numeric substrate for the TPT Solutions
`tpt-science` / `tpt-engineering` / `tpt-formal` stack. This document
describes how to propose changes.

## Issues only — no external pull requests

`tpt-math` follows an **issues-only** workflow. There is no external pull-request
review process:

- To report a bug, request a feature, or propose an API change, **open an issue**
  on the repository's issue tracker.
- For suspected vulnerabilities or soundness bugs, see `SECURITY.md` first (do
  **not** open a public PR for a security fix).
- Maintainers triage issues, make the change on a branch, and land it via the
  internal review process.

Please do not open a pull request from a fork expecting it to be merged; it will
be closed in favour of an issue.

## Per-crate checklist

Every crate in this workspace is built to the same shape (see `todo.md` for the
full template). When adding or modifying a crate, confirm:

- [ ] The crate builds `no_std` where `registry.toml` marks `no_std = true`
      (verified by `cargo xtask no-std`).
- [ ] `cargo fmt --check` is clean (`cargo xtask fmt` to fix).
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` is
      clean (run via `cargo xtask clippy`).
- [ ] `cargo test --workspace --all-features` passes, including doctests.
- [ ] `cargo-deny` (`cargo xtask deny`) is clean.
- [ ] Public items have rustdoc; panicking operators document `# Panics`.
- [ ] `no_std` crates opt into the workspace lints (`[lints] workspace = true`)
      and never use `unsafe` (the workspace sets `unsafe_code = "forbid"`).
- [ ] `registry.toml` is updated: a new crate reads `status = "git"`, and its
      `no_std` flag is set correctly.

The one-stop command is `cargo xtask check` (or `just check`).

## License policy (`deny.toml`)

All workspace crates are `MIT OR Apache-2.0`. `deny.toml` enforces:

- `advisories.yanked = "deny"` — yanked dependency versions are rejected.
- `sources.unknown-registry = "deny"` and `sources.unknown-git = "deny"` —
  every dependency must come from crates.io or a declared git source.
- A curated `licenses` allow-list (permissive licenses only). New wrap targets
  must be re-checked against this list; note that some upstreams are
  Apache-2.0-only, which is fine as a *dependency* of a dual-licensed crate but
  must not be copied into the workspace source under a different license.

When adding a new external wrap, confirm its license is on the allow-list before
merging, and update the note in `deny.toml` if you add a new upstream family.

## Tolerance conventions

Solvers use one of two conventions. Pick the one the routine's family already
uses, and say which one in its rustdoc.

- **Relative to the data.** The default for new code. Dense LU, `det`, `solve`
  and `inverse` treat a pivot as zero when it is below `1e-12 · max|a_ij|`.
  The QR and the sparse CG and BiCGSTAB solvers stop when `‖r‖ ≤ tol · ‖b‖`
  (absolute when `b = 0`). Results do not depend on the units of the input.
- **Absolute.** Only where a caller needs a fixed threshold. `Lu::with_pivot_tolerance`
  takes one, and the convex interior-point KKT solve pins `1e-12` through it.

## External `registry.toml` dependency

Crate status and the `no_std` marking come from `../tpt-rust-map/registry.toml`,
a file in a sibling repository. This repo does not vendor it. Two things depend
on it:

- `cargo xtask no-std` builds the crates in its `NO_STD_CRATES` list. That list
  mirrors the `no_std = true` entries in `registry.toml`.
- `cargo xtask drift` checks that `NO_STD_CRATES`, the crates declaring
  `#![no_std]` in their `src/lib.rs`, and the README `no_std` column all agree.
  It runs as part of `cargo xtask check`.

`xtask` does not read `registry.toml` itself. The list is a hand-maintained
mirror, so when you change a crate's `no_std` status, update the crate's
`#![no_std]`, the `NO_STD_CRATES` entry, the README row and `registry.toml`
together. `cargo xtask drift` catches the in-repo parts if one is missed.

## Adding a new crate

1. Scaffold under `crates/<name>/` following the existing crates' layout.
2. Inherit workspace fields in `Cargo.toml` and add `[lints] workspace = true`.
3. Register the crate in the root `Cargo.toml` `members` array **and** in
   `../tpt-rust-map/registry.toml` with `status = "git"`.
4. Run the per-crate checklist above.
