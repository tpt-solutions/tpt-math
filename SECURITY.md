# Security Policy

This document describes the security posture and conventions of the
`tpt-math` workspace. It applies to all crates under `crates/`, the `xtask`
tooling crate, and the `examples/` workspace member.

## No-`unsafe` policy

Every crate in this workspace opts into workspace lints
(`[lints] workspace = true`) and the workspace sets
`unsafe_code = "forbid"`. **No crate may introduce `unsafe` code.** A handful
of crates additionally assert this with a crate-level `#![forbid(unsafe_code)]`
inner attribute as defence in depth.

If a future feature genuinely requires `unsafe`, it must:

1. Be isolated in a dedicated module with a `// SAFETY:` justification on every
   `unsafe` block.
2. Be gated behind a feature flag off by default.
3. Be reviewed by a maintainer and the rationale recorded in the crate's
   `CHANGELOG.md`.

## Dependency & licensing hygiene (`deny.toml`)

Dependency hygiene is enforced with [`cargo-deny`]:

- **Advisories:** `yanked = "deny"` — pulling a yanked crate version is a hard
  failure.
- **Sources:** `unknown-registry = "deny"` and `unknown-git = "deny"` — every
  dependency must come from crates.io or a git source explicitly declared in the
  workspace; arbitrary unknown sources are rejected.
- **Licenses:** all workspace crates are `MIT OR Apache-2.0`. The allow-list in
  `deny.toml` also permits the permissive licenses of transitive dependencies.
  New wrap targets must be re-checked against this list (see the note in
  `deny.toml` about Apache-2.0-only upstreams).

Run `cargo xtask deny` (or `just deny`) in CI and locally to verify.

## Panic vs `try_*` convention

`tpt-math` favours total, non-panicking APIs wherever the failure is
recoverable, and uses panics only for programmer errors (preconditions that
should have been caught at compile time or via documented invariants).

- **Dimension / shape mismatches** (e.g. `tpt-math-linalg` vector/matrix
  operators) panic, because they indicate a logic error in the caller; the
  precise panic conditions are documented in each operator's `# Panics`
  section.
- **Recoverable failures** (parsing, out-of-domain evaluation, allocation)
  should be surfaced through `Result`-returning `try_*` variants where one
  exists.
- Prefer `try_*` / `Result` over panicking constructors when designing new APIs.

## Symbolic recursion caveat (`tpt-math-symbolic`)

`tpt-math-symbolic`'s [`simplify`](crates/tpt-math-symbolic/src/lib.rs) is
structural recursion with **no guard against cyclic input**. Acyclic trees built
through the public `Expr` API are safe; only hand-built or externally-parsed,
potentially self-referential expressions can overflow the stack. Validate or
trust the source of any expression tree before simplifying untrusted input.

Additionally, transcendental function evaluation in `tpt-math-symbolic` coerces
coefficients through an `f64` round-trip (see `apply_func`), so exact
coefficients lose precision for `sin`/`cos`/`exp`/etc. Algebraic simplification
stays exact.

## Vulnerability disclosure

`tpt-math` is an internal foundation dependency and is **not published to
crates.io** in this pass (crates stop at `status = "git"` in
`../tpt-rust-map/registry.toml`).

To report a suspected vulnerability or a serious soundness bug (e.g. an
unsound `unsafe`, a numeric correctness error, or a panic reachable from
safe code on valid input):

- Open a **security issue** via the repository's issue tracker, or
- Email the maintainers at **security@tpt.solutions**.

Please do **not** open a public pull request for a security fix; the project is
issues-only (see `CONTRIBUTING.md`). A maintainer will triage, confirm, and
coordinate a fix and disclosure timeline with you.
