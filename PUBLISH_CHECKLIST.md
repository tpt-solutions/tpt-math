# crates.io Publish Checklist — next update release

Status as of 2026-10-20. Built from the `[Unreleased]` section of each crate's
`CHANGELOG.md`, `todo.md`, `release-plz.toml` and the crate manifests. The repo
has no git tags until section 6, so the CHANGELOGs were the source of truth.

**Release status: published.** All 28 changed crates are on crates.io at the
versions below, and each has a git tag. Items still open are in section 6.

Commits on master:

- `3187a22` — version bumps and the CHANGELOG heading merge
- `2e8c74b` — CHANGELOG stamping and the linalg-dense README fix (the published commit)
- CI commit — the semver job no longer excludes the facade or `pde-operators`

Tags: `tpt-math-<crate>-v<version>` for each of the 27 crates, and `tpt-math-v0.1.0`
for the facade, all on `2e8c74b`.

Version bumps follow pre-1.0 SemVer: a breaking change bumps the minor, anything
else the patch.

---

## 1. Blockers — resolve before publishing

- [x] **Stale `release-plz.toml` comment.** Updated: the comment now says the
      simd crates are published on crates.io.
- [x] **`tpt-math-pde-operators` registry entry.** Present in `tpt-rust-map/registry.toml`
      with `domain = "math.pde"`, `no_std = true`, `status = "git"`.
- [x] **Refresh `Cargo.lock`** for `tpt-math-pde-operators`. Lockfile has it at 0.2.0.
- [x] **Stray tracked file** named `- [ ] Dense QR, least squares`. Not in the repo
      root and not tracked by git. Nothing to delete.
- [x] **`tpt-math` facade.** Published in this release (0.1.0). The name was free
      on crates.io (404) when rechecked before publishing.
- [x] **Confirm current crates.io versions.** Checked. Two decisions followed from
      it: `geometry` moved to 0.2.0 (not 0.1.2) because `from_matrix` now rejects
      reflections, and `prob-markov` moved to 0.1.2 (not 0.2.0) because the
      behaviour change does not change any signature.

## 2. Per-crate release plan

Released versions. The earlier suggested table was out of date: it predated the
0.2.0 bumps. Now that every crate is published, this table records what went out.

| Crate | Was on crates.io | Released | Unreleased changes |
|---|---|---|---|
| `tpt-math-numeric` | 0.1.0 | **0.2.0** | `report`, Gauss–Kronrod 15, adaptive GK, ODE, stiff, sensitivity; breaking (`non_exhaustive`) |
| `tpt-math-prob-core` | 0.1.0 | 0.1.1 | `special` module; precision fixes for `erf`, `gamma`, `ln_binomial` |
| `tpt-math-prob-dist` | 0.1.0 | 0.1.1 | `student_t_cdf` fix; density, quantile and moment families |
| `tpt-math-prob-bayes` | 0.1.0 | **0.2.0** | `adaptive`, `hmc`, diagnostics; breaking |
| `tpt-math-prob-markov` | 0.1.1 | 0.1.2 | `stationary` solves `πP = π` for periodic chains; Viterbi precompute |
| `tpt-math-prob-monte-carlo` | 0.1.0 | 0.1.1 | `integrate_par` behind `rayon`; Welford variance |
| `tpt-math-stats` | 0.1.0 | **0.2.0** | KDE, corrections, bootstrap, multiple testing; breaking |
| `tpt-math-autodiff-fwd` | 0.1.0 | 0.1.1 | `Float` for `Dual`; `sqrt`, `powi`, `powf`, `tan`, `tanh`, `atan`, `abs` |
| `tpt-math-autodiff-rev` | 0.1.0 | **0.2.0** | `Op` enum gained variants; breaking |
| `tpt-math-autodiff` | 0.1.0 | 0.1.1 | `jacobian`, `hessian` (this crate had changes, so the earlier "no change" note was wrong) |
| `tpt-math-symbolic` | 0.1.0 | **0.2.0** | `compile`, syntax; term-order behaviour change; breaking |
| `tpt-math-optimize-general` | 0.1.0 | **0.2.0** | least squares, autodiff L-BFGS; breaking |
| `tpt-math-optimize-convex` | 0.1.0 | **0.2.0** | `QpSolution` is `non_exhaustive`; breaking |
| `tpt-math-signal-fft` | 0.1.0 | 0.1.1 | `rfft`/`irfft`, `stft`, `welch`, `fft_convolve` |
| `tpt-math-signal-filter` | 0.1.0 | 0.1.1 | `filter_par` behind `rayon`; biquads, Butterworth, zero-phase |
| `tpt-math-exact` | 0.1.0 | 0.1.1 | interval division, `newton_refine`, validated Euler, Krawczyk `certified_solve` |
| `tpt-math-graph` | 0.1.0 | **0.2.0** | `try_dijkstra`, `try_astar`, `GraphError`, connectivity; breaking |
| `tpt-math-interpolate` | 0.1.0 | **0.2.0** | `NaturalCubicSpline`, `MultilinearGrid`, RBF and Kriging in n-D; breaking |
| `tpt-math-linalg-complex` | 0.1.0 | **0.2.0** | `simd_lu`, `det`, `product`; breaking |
| `tpt-math-linalg-dense` | 0.1.0 | **0.2.0** | flat LU, Hessenberg/eigenvalues, banded, diagnostics; breaking |
| `tpt-math-linalg-fixed` | 0.1.0 | **0.2.0** | `non_exhaustive` error types; breaking |
| `tpt-math-linalg-sparse` | 0.1.0 | **0.2.0** | `SparseError::DimensionMismatch { what }`; operators, ordering, Lanczos/Arnoldi; breaking |
| `tpt-math-linalg` | 0.1.0 | **0.2.0** | breaking (follows the linalg family) |
| `tpt-math-geometry` | 0.1.1 | **0.2.0** | `from_matrix` returns `None` for reflections; breaking |
| `tpt-math-spatial` | 0.1.0 | **0.2.0** | depends on geometry 0.2.0 (in its public API); `Screw::log` fix |
| `tpt-math-units-dyn` | 0.1.0 | 0.1.1 | `solve` module behind `ode`, `optimize`, `autodiff`; `FromStr` accepts compound units |
| `tpt-math-pde-operators` | — | **0.2.0** | **New crate** |
| `tpt-math` (facade) | — | **0.1.0** | **New crate** |
| `tpt-math-optimize`, `tpt-math-prob`, `tpt-math-prob-sampler`, `tpt-math-signal`, `tpt-math-units` | 0.1.0 | not released | No source change since 0.1.0 |

Breaking releases (0.2.0): 17 of the 28. That is the 15 crates with breaking
changes from the earlier session, plus `geometry` and `spatial`. Behaviour-only
changes that did not need a minor bump: `geometry`'s `from_matrix` (listed as
breaking in its changelog, see section 7), `prob-markov`, and `exact` (interval
`new` now panics in release builds too).

## 3. Pre-publish verification

- [x] `cargo fmt --check`
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [x] `cargo test --workspace --all-features` (including doctests). 87 result blocks, all passing.
- [x] `cargo xtask check` (includes fmt, clippy, tests, doctests, rustdoc, no_std, drift, cargo-deny). Exit 0 on the release tree.
- [x] `cargo xtask no-std` (includes `thumbv6m-none-eabi`)
- [x] `cargo doc --workspace --all-features` with `RUSTDOCFLAGS=-D warnings`
- [x] `cargo deny check` (run through `cargo xtask check`)
- [x] Simd features pass: `-p tpt-math-linalg-dense --features simd` and
      `-p tpt-math-linalg-sparse --features simd` (toolchain 1.98, above the 1.95 minimum)
- [x] `cargo package -p <crate> --list` for each crate. No stray files.
- [~] `cargo publish --dry-run` for each crate, in dependency order. Not possible
      standalone: a dry-run fails until the crate's workspace dependencies are on
      crates.io. Each real `cargo publish` ran the same packaging and verification
      step in order, and all 28 succeeded.
- [x] **cargo-semver-checks.** Breaking crates passed in the earlier session;
      the 12 crates with a published baseline and changes passed this session. No
      baseline for `pde-operators` or the facade.
- [x] `[package.metadata.docs.rs]` check. `linalg-dense` already uses
      `all-features = true`, so no change was needed.

## 4. Docs and metadata

- [x] Each published crate's `## [Unreleased]` entries moved under `## [<version>] - 2026-10-20`, with a fresh empty `## [Unreleased]` above.
- [x] Feature tables in the READMEs cover `rayon`, `simd`, `simd-runtime`, and `ode` / `optimize` / `autodiff` on `units-dyn`. The `simd-runtime` row in `linalg-dense` was missing its description; fixed.
- [x] READMEs' `no_std` column matches `NO_STD_CRATES` (checked by `cargo xtask drift`, part of `xtask check`).
- [x] `rust-version`: the default build stays at 1.85. The simd rows in `linalg-dense` and `linalg-complex` and `linalg-sparse` state that they need 1.95.

## 5. Publish

- [x] Published in dependency order: leaves first, the facade last (order computed from `cargo metadata`, 28 crates).
- [x] Each crate was published only after its workspace dependencies appeared in the index. `cargo publish` waits for this by default.
- [x] After each publish, the version was confirmed on crates.io (all 28 match).
- [x] docs.rs builds. All 28 are green: `docs.rs/crate/<name>/<version>/status.json` reports `doc_status: true` for every published version, including the facade.

## 6. After publishing

- [x] Tagged the release. Annotated tags `tpt-math-<crate>-v<version>` for each of the 27 crates and `tpt-math-v0.1.0` for the facade, all on `2e8c74b`. Pushed to origin (56 entries from `ls-remote`, which includes peeled tags).
- [x] Removed the `--exclude tpt-math --exclude tpt-math-pde-operators` flags from the semver job in `.github/workflows/ci.yml`. Pushed.
- [x] Update `todo.md`: release-plz (7.5) is unchecked until it can publish. Done in the commit that adds this checklist.
- [ ] Re-enable `publish = true` in `release-plz.toml`. Left as `false` on purpose: the file's comment says to wait until the first release has been reviewed by hand.
- [x] Updated `Cargo.lock` and committed.

## 7. Open follow-ups

- **Confirmed:** `geometry` stays at 0.2.0. `from_matrix` changed behaviour
  without a signature change, so semver-checks does not catch it. It is listed as
  breaking in the changelog and the version reflects that.
- `todo.md` item 7.5 (release-plz and automation) is reopened until release-plz can
  publish. The rest of that item is done.
- `release-plz` publishing stays off until the release has been reviewed by hand.
