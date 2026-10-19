# crates.io Publish Checklist — next update release

Status as of 2026-10-18. Built from the `[Unreleased]` section of each crate's
`CHANGELOG.md`, `todo.md`, `release-plz.toml` and the crate manifests. The repo
has no git tags, so the CHANGELOGs are the source of truth.

Version bumps are **suggestions** (pre-1.0 SemVer: a breaking change bumps the
minor, anything else the patch). Confirm each one before publishing.

---

## 1. Blockers — resolve before publishing

- [ ] **Stale `release-plz.toml` comment.** It says the simd crates are path
      dependencies on a sibling checkout, which `todo.md` (item 7.5) says was
      fixed. Update the comment, then decide whether `publish = false` can go.
- [ ] **`tpt-math-pde-operators` registry entry.** Add its `tpt-rust-map/registry.toml`
      entry (`domain = "math.pde"`, `no_std = true`, `wraps = []`,
      `status = "planned"` → `"git"`). That file is in a sibling repo, so this
      needs the other checkout.
- [ ] **Refresh `Cargo.lock`** for `tpt-math-pde-operators` (`cargo update -p tpt-math-pde-operators`).
      Needs the sibling checkout, per `todo.md`.
- [ ] **Stray tracked file in the repo root**, named `- [ ] Dense QR, least squares`
      (60 KB, tracked by git). It looks like a mis-saved todo paste. Confirm it is
      not wanted, then delete it.
- [ ] **Decide on the `tpt-math` facade crate.** It is not published yet. The name
      check returned 404 (free) at the time, so recheck it right before publishing.
      Publish it in this release, or hold it back.
- [ ] **Confirm current crates.io versions** of every crate below. Some (e.g.
      `tpt-math-geometry`, `tpt-math-prob-markov`) are already at 0.1.1, so
      the bumps below may need adjusting.

## 2. Per-crate release plan

Dependency bumps follow the version bump: update `[workspace.dependencies]` in
the root `Cargo.toml` (currently `version = "0.1.0"` for most) **and** each
crate's own `version`, or the published crates will pull in the old releases.

| Crate | Now | Suggested | Unreleased changes |
|---|---|---|---|
| `tpt-math-numeric` | 0.1.0 | 0.1.1 | `report` module (`SolveReport`); Gauss–Kronrod 15 and adaptive GK |
| `tpt-math-prob-core` | 0.1.0 | 0.1.1 | `special` module (gamma, beta, erf, normal CDF/quantile) |
| `tpt-math-prob-dist` | 0.1.0 | 0.1.1 | Fix: `student_t_cdf` near the centre |
| `tpt-math-prob-bayes` | 0.1.0 | 0.1.1 | `adaptive` (adaptive Metropolis), `hmc` (HMC); uses new `prob-core::special` |
| `tpt-math-prob-markov` | 0.1.1 | 0.2.0 | **Behaviour change:** `stationary` now solves directly for periodic chains |
| `tpt-math-prob-monte-carlo` | 0.1.0 | 0.1.1 | `integrate_par` behind `rayon` |
| `tpt-math-stats` | 0.1.0 | 0.1.1 | Special functions re-exported from `prob-core` |
| `tpt-math-autodiff-fwd` | 0.1.0 | 0.1.1 | `sqrt`, `powi`, `powf`, `tan`, `tanh`, `atan`, `abs`, `Dual` ∘ scalar |
| `tpt-math-autodiff-rev` | 0.1.0 | 0.2.0 | **Breaking:** `Op` enum gained variants (exhaustive matches break); `abs`, `Powi(a,0)` fix |
| `tpt-math-symbolic` | 0.1.0 | 0.2.0 | **Behaviour change:** sum/product term order changes; new functions and rules |
| `tpt-math-optimize-general` | 0.1.0 | 0.1.1 | Fix: `numerical_gradient` step scales with `\|x\|` |
| `tpt-math-optimize-convex` | 0.1.0 | 0.2.0 | **Breaking:** `QpSolution` gained `iterations` (struct literals break) |
| `tpt-math-signal-fft` | 0.1.0 | 0.1.1 | `rfft` / `irfft` |
| `tpt-math-signal-filter` | 0.1.0 | 0.1.1 | `filter_par` behind `rayon` |
| `tpt-math-exact` | 0.1.0 | 0.1.1 | `Interval::div_nonzero`, `roots::newton_refine` |
| `tpt-math-graph` | 0.1.0 | 0.1.1 | `try_dijkstra`, `try_astar`, `GraphError` |
| `tpt-math-interpolate` | 0.1.0 | 0.1.1 | Fix: `Pchip::new` with two nodes |
| `tpt-math-linalg-complex` | 0.1.0 | 0.1.1 | `lu_with_tolerance`, `lu_with_relative_tolerance` |
| `tpt-math-linalg-dense` | 0.1.0 | 0.1.1 | `par_mul` behind `rayon`; new `simd` deps on `tpt-simd-blas` 0.1.1 |
| `tpt-math-linalg-fixed` | 0.1.0 | 0.1.1 | `inverse_with_tolerance`, `inverse_with_relative_tolerance` |
| `tpt-math-linalg-sparse` | 0.1.0 | 0.1.1 | `gmres_right_preconditioned_with_report`; new `simd` dep on `tpt-simd-sparse` 0.1.1 |
| `tpt-math-linalg` | 0.1.0 | 0.1.1 | Scalar `*`/`/`, `norm`, `dot`, `solve`, `inverse`, by-ref operators |
| `tpt-math-geometry` | 0.1.1 | 0.1.2 or 0.2.0 | `from_matrix` rejects reflections; quaternion renormalises. **Check:** if `from_matrix` already returned `Option`, patch; otherwise minor |
| `tpt-math-spatial` | 0.1.0 | 0.1.1 | `Screw::log` uses `atan2` |
| `tpt-math-units-dyn` | 0.1.0 | 0.1.1 | New `solve` module behind `ode`, `optimize`, `autodiff` features |
| `tpt-math-pde-operators` | — | 0.1.0 | **New crate, first publish** (blocker above) |
| `tpt-math-autodiff`, `tpt-math-optimize`, `tpt-math-prob`, `tpt-math-signal`, `tpt-math-units`, `tpt-math-prob-sampler` | 0.1.0 | no change | No unreleased entries. Republish only if a dependency bump requires it |
| `tpt-math` (facade) | — | 0.1.0 | Not published. See blocker above |

Totals: 3 breaking (`autodiff-rev`, `optimize-convex`, `symbolic`), plus
`prob-markov` behaviour change. Confirm each one.

## 3. Pre-publish verification

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features` (including doctests)
- [ ] `cargo xtask check` (matches CI: fmt, clippy, tests, doctests, rustdoc,
      no_std build, drift check, cargo-deny)
- [ ] `cargo xtask no-std` (includes `thumbv6m-none-eabi`)
- [ ] `cargo doc --workspace --all-features` with `RUSTDOCFLAGS=-D warnings`
- [ ] `cargo deny check`
- [ ] Simd features pass: `cargo test -p tpt-math-linalg-dense --features simd`
      and `-p tpt-math-linalg-sparse --features simd` (needs Rust 1.95+)
- [ ] `cargo package -p <crate> --list` for each crate, to confirm no stray files ship
- [ ] `cargo publish --dry-run` for each crate, in dependency order
- [ ] **cargo-semver-checks** on the crates with breaking changes (not yet in CI, per `todo.md`)
- [ ] `[package.metadata.docs.rs]` sanity check on every crate. `linalg-dense` still
      uses `features = ["alloc"]`, not `all-features`. Decide whether to switch it now.

## 4. Docs and metadata

- [ ] Move each crate's `## [Unreleased]` entries under `## [<version>] - 2026-MM-DD`
      in its `CHANGELOG.md`, and add a fresh empty `## [Unreleased]` above it
- [ ] Update each crate's README feature table for the new features
      (`rayon`, `simd`, `simd-runtime`, `ode` / `optimize` / `autodiff` on `units-dyn`)
- [ ] Confirm READMEs' `no_std` column matches `NO_STD_CRATES` in `xtask`
      (`cargo xtask drift` checks this)
- [ ] Check the workspace `rust-version`. The default build stays at 1.85; the
      `simd` feature needs 1.95 and the docs should say so

## 5. Publish

- [ ] Publish in dependency order, leaf crates first:
      `numeric` → `prob-core` → `linalg-dense`, `linalg-complex`, `linalg-fixed`,
      `prob-dist`, `interpolate`, … → umbrella and facade crates last
      (generate the exact order from `cargo metadata`)
- [ ] Wait for each crate to appear in the crates.io index before publishing its dependents
- [ ] After each publish, confirm the version on crates.io and that docs.rs builds

## 6. After publishing

- [ ] Tag the release in git (none exist yet). Suggest `v<version>` per crate, or
      one tag for the workspace release
- [ ] Update `todo.md`: release-plz (7.5) stays unchecked until it can publish
- [ ] Re-enable `publish = true` in `release-plz.toml` once the first release
      has been reviewed by hand (per the file's own comment)
- [ ] Update `Cargo.lock` and commit
