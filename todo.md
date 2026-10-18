# tpt-math — Build Todo

> Tracks bootstrap + full 23-crate build-out for the tpt-math foundation repo,
> per `spec.txt` and `tpt-rust-map/registry.toml`. The workspace has since been
> published to crates.io in full (see `spec2.txt` / per-crate `CHANGELOG.md`).
> Crates were stopped at `status = "git"` in the registry during the initial
> build-out. License for every crate: `MIT OR Apache-2.0`. Author: TPT
> Solutions. Toolchain MSRV: Rust 1.85 (workspace `rust-version`).

## Phase 0 — Repo Bootstrap

(one-time, seed from `tpt-rust-map/template/`)

- [x] Copy `template/Cargo.toml` → root `Cargo.toml` (workspace `resolver = "2"`,
      `[workspace.package]`: `edition = "2021"`, `rust-version = "1.75"`,
      `license = "MIT OR Apache-2.0"`, `authors = ["TPT Solutions"]`; replace
      placeholder `homepage`/`repository` URLs)
- [x] Copy `template/rust-toolchain.toml`
- [x] Copy `template/rustfmt.toml`
- [x] Copy `template/deny.toml`
- [x] Copy `template/.github/workflows/ci.yml`
- [x] Copy `template/LICENSE-MIT` and `template/LICENSE-APACHE`
- [x] Create empty `crates/` directory
- [x] Add a Rust `.gitignore` (`/target`, etc.)
- [x] Write root `README.md` stub — tpt-math's role as the numeric substrate
      for tpt-science/tpt-engineering/tpt-formal; link to `spec.txt` and
      `tpt-rust-map`
- [x] `git init` (local only — no GitHub remote/push)
- [x] Initial commit
- [x] Sanity check: `cargo build` succeeds on the empty workspace

## Per-Crate Checklist Template

Every phase below repeats this shape. Umbrella crates (`tpt-math-prob`,
`tpt-math-autodiff`, `tpt-math-optimize`, `tpt-math-signal`) use the umbrella
variant instead of steps 2-4.

**Standard crate:**
1. Scaffold `crates/<name>/` (Cargo.toml inheriting workspace fields, `lib.rs` stub)
2. Wire dependencies (internal `tpt-math-*` + external wraps), `default = ["std"]` with additive `alloc`/`libm`/`serde` features (ADR 0001)
3. Implement scope
4. Unit tests + doctests
5. Rustdoc (crate-level + public API)
6. `cargo fmt --check` / `cargo clippy --all-targets --all-features -- -D warnings` clean
7. `cargo deny check` clean
8. no_std target verification (`thumbv6m-none-eabi`) — only if `no_std = true`
9. Update `tpt-rust-map/registry.toml`: `status = "planned"` → `"git"`

**Umbrella crate:**
1. Scaffold `crates/<name>/` (Cargo.toml with Cargo features gating each constituent re-export)
2. Wire optional deps + matching feature flags per constituent crate
3. Re-export each constituent's public API behind its feature
4. Rustdoc documenting the feature matrix
5. `cargo fmt` / `clippy` / `deny` clean across feature combinations
6. Update `tpt-rust-map/registry.toml`: `status = "planned"` → `"git"`

---

## Phase 1 — tpt-math-numeric

*Wrap num-traits + libm. Scalar numeric trait glue. no_std. No internal deps.*

- [x] Scaffold `crates/tpt-math-numeric/`
- [x] Wire deps: `num-traits`, `libm`; `default = ["std"]` + `alloc`/`libm` features
- [x] Implement scalar numeric trait glue
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-numeric` → `"git"`

## Phase 2 — tpt-math-units

*Wrap uom (disable default std feature). Compile-time typed units. no_std. No internal deps.*

- [x] Scaffold `crates/tpt-math-units/`
- [x] Wire deps: `uom` (std feature disabled by default)
- [x] Implement compile-time typed units wrapper
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-units` → `"git"`

## Phase 3 — tpt-math-units-dyn ⚠️ assumption flagged

*Runtime dimension-checked units for dynamic/config-driven payloads.
Consolidates tpt-rust2's tpt-units-runtime. Depends on: tpt-math-units.*

> **Note:** neither `spec.txt`'s BUILD ORDER nor `tpt-rust-map/TODO.md`
> (identical wording) mentions this crate's position. Its only declared
> dependency is `tpt-math-units` and nothing downstream depends on it, so
> it's placed here as the lowest-risk slot. Re-sequence freely if that
> assumption turns out wrong.

- [x] Scaffold `crates/tpt-math-units-dyn/`
- [x] Wire deps: `tpt-math-units`
- [x] Implement runtime dimension-checked unit type for dynamic/config payloads
- [x] Port/consolidate logic from `tpt-rust2`'s `tpt-units-runtime`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-units-dyn` → `"git"`

(no no_std step — `registry.toml` has `no_std = false` for this crate)

## Phase 4 — tpt-math-exact

*Thin-wrap num-bigint/num-rational; interval arithmetic layer built on top.
Arbitrary-precision rational + interval arithmetic. Consolidates
tpt-formal-lab's tpt-exact-math. no_std+alloc. Depends on: tpt-math-numeric.*

- [x] Scaffold `crates/tpt-math-exact/`
- [x] Wire deps: `num-bigint`, `num-rational`, `tpt-math-numeric`
- [x] Implement exact-rational wrapper + interval arithmetic layer on top
- [x] Port/consolidate logic from `tpt-formal-lab`'s `tpt-exact-math`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std+alloc verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-exact` → `"git"`

## Phase 5 — tpt-math-linalg

*Wrap nalgebra. Pair with tpt-math-units for dimensionally-checked
vectors/matrices — nalgebra only, not a dual nalgebra+faer facade.
Consolidates tpt-zero-formal's tpt-zero-linalg and tpt-zero-text's
tpt-zero-matrix. no_std. Depends on: tpt-math-units.*

- [x] Scaffold `crates/tpt-math-linalg/`
- [x] Wire deps: `nalgebra`, `tpt-math-units`
- [x] Implement dimensionally-checked vector/matrix types pairing nalgebra with tpt-math-units
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-linalg` and `tpt-zero-text`'s `tpt-zero-matrix`
- [x] Unit tests + doctests
- [x] Rustdoc (note the nalgebra-only backend decision — no faer facade, per spec.txt rationale)
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-linalg` → `"git"`

## Phase 6 — tpt-math-prob-core

*Shared Distribution/Sampler traits that the tpt-math-prob-* crates
implement against. no_std. Depends on: tpt-math-numeric.*

- [x] Scaffold `crates/tpt-math-prob-core/`
- [x] Wire deps: `tpt-math-numeric`
- [x] Implement shared `Distribution`/`Sampler` traits
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-prob-core` → `"git"`

## Phase 7 — tpt-math-prob-* (any order / parallelizable)

All five depend only on `tpt-math-prob-core` and can be built in any order or in parallel.

### 7a — tpt-math-prob-dist

*Wrap rand_distr. Standard distributions. no_std. Consolidates
tpt-zero-formal's tpt-zero-dist.*

- [x] Scaffold `crates/tpt-math-prob-dist/`
- [x] Wire deps: `rand_distr`, `tpt-math-prob-core`
- [x] Implement standard distributions wrapper
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-dist`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-prob-dist` → `"git"`

### 7b — tpt-math-prob-bayes

*Bayesian inference primitives. Consolidates tpt-zero-formal's tpt-zero-bayes.*

- [x] Scaffold `crates/tpt-math-prob-bayes/`
- [x] Wire deps: `tpt-math-prob-core`
- [x] Implement Bayesian inference primitives
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-bayes`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-prob-bayes` → `"git"`

### 7c — tpt-math-prob-markov

*Markov chains / HMM. Consolidates tpt-zero-formal's tpt-zero-markov.*

- [x] Scaffold `crates/tpt-math-prob-markov/`
- [x] Wire deps: `tpt-math-prob-core`
- [x] Implement Markov chain / HMM primitives
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-markov`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-prob-markov` → `"git"`

### 7d — tpt-math-prob-monte-carlo

*Monte Carlo methods. Consolidates tpt-zero-formal's tpt-zero-monte-carlo.*

- [x] Scaffold `crates/tpt-math-prob-monte-carlo/`
- [x] Wire deps: `tpt-math-prob-core`
- [x] Implement Monte Carlo methods
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-monte-carlo`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-prob-monte-carlo` → `"git"`

### 7e — tpt-math-prob-sampler

*Sampling strategies. no_std. Consolidates tpt-zero-formal's
tpt-zero-sampler and tpt-zero-rand.*

- [x] Scaffold `crates/tpt-math-prob-sampler/`
- [x] Wire deps: `tpt-math-prob-core`
- [x] Implement sampling strategies
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-sampler` and `tpt-zero-rand`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-prob-sampler` → `"git"`

## Phase 8 — tpt-math-prob (umbrella)

*Re-exports the five tpt-math-prob-* crates behind Cargo features.*

- [x] Scaffold `crates/tpt-math-prob/`
- [x] Wire optional deps + feature flags for dist/bayes/markov/monte-carlo/sampler
- [x] Re-export each constituent's public API behind its feature
- [x] Rustdoc documenting the feature matrix
- [x] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [x] registry.toml: `tpt-math-prob` → `"git"`

## Phase 9 — tpt-math-stats

*In-house hypothesis tests / regression (no `statrs` — see Phase D). Consolidates
tpt-zero-formal's tpt-zero-stats and tpt-rust6's tpt-stat. Depends on:
tpt-math-prob-core.*

- [x] Scaffold `crates/tpt-math-stats/`
- [x] Wire deps: `tpt-math-prob-core` (statrs replaced by in-house `special`/`dist`)
- [x] Implement hypothesis tests / regression wrapper
- [x] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-stats` and `tpt-rust6`'s `tpt-stat`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-stats` → `"git"`

## Phase 10 — tpt-math-autodiff-fwd

*Dual numbers over tpt-math-numeric. Forward-mode autodiff. no_std. Depends
on: tpt-math-numeric.*

- [x] Scaffold `crates/tpt-math-autodiff-fwd/`
- [x] Wire deps: `tpt-math-numeric`
- [x] Implement dual-number forward-mode autodiff
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] registry.toml: `tpt-math-autodiff-fwd` → `"git"`

## Phase 11 — tpt-math-autodiff-rev

*Reverse-mode/tape autodiff. Consolidates tpt-rust6's
tpt-grad/tpt-grad-macro and tpt-zero-formal's tpt-zero-grad. Depends on:
tpt-math-autodiff-fwd.*

- [x] Scaffold `crates/tpt-math-autodiff-rev/`
- [x] Wire deps: `tpt-math-autodiff-fwd`
- [x] Implement reverse-mode/tape autodiff
- [x] Port/consolidate logic from `tpt-rust6`'s `tpt-grad`/`tpt-grad-macro` and `tpt-zero-formal`'s `tpt-zero-grad`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-autodiff-rev` → `"git"`

## Phase 12 — tpt-math-autodiff (umbrella)

*Re-exports fwd + rev.*

- [x] Scaffold `crates/tpt-math-autodiff/`
- [x] Wire optional deps + feature flags for fwd/rev
- [x] Re-export each constituent's public API behind its feature
- [x] Rustdoc documenting the feature matrix
- [x] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [x] registry.toml: `tpt-math-autodiff` → `"git"`

## Phase 13 — tpt-math-symbolic

*Permissive-license symbolic math (CAS) — genuine, unfilled ecosystem gap.
Consolidates tpt-rust6's tpt-sym. Optionally depends on: tpt-math-exact.*

- [x] **Design decision (open call per spec.txt, resolve now):** does
      `tpt-math-symbolic` default to f64 via a generic `Coefficient` trait,
      or default to `tpt-math-exact`-backed exact rationals?
- [x] Scaffold `crates/tpt-math-symbolic/`
- [x] Wire deps per the decision above (optionally `tpt-math-exact`)
- [x] Implement CAS core (expression representation, simplification, etc.)
- [x] Port/consolidate logic from `tpt-rust6`'s `tpt-sym`
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-symbolic` → `"git"`

## Phase 14 — tpt-math-optimize-* (either order relative to each other)

### 14a — tpt-math-optimize-general

*Wrap argmin. General numerical optimization. Depends on: tpt-math-linalg.*

- [x] Scaffold `crates/tpt-math-optimize-general/`
- [x] Wire deps: `argmin`, `tpt-math-linalg`
- [x] Implement general numerical optimization wrapper
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-optimize-general` → `"git"`

### 14b — tpt-math-optimize-convex

*Wrap clarabel. Convex / QP optimization. Depends on: tpt-math-linalg.*

- [x] Scaffold `crates/tpt-math-optimize-convex/`
- [x] Wire deps: `clarabel`, `tpt-math-linalg`
- [x] Implement convex/QP optimization wrapper
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-optimize-convex` → `"git"`

## Phase 15 — tpt-math-optimize (umbrella)

*Re-exports general + convex.*

- [x] Scaffold `crates/tpt-math-optimize/`
- [x] Wire optional deps + feature flags for general/convex
- [x] Re-export each constituent's public API behind its feature
- [x] Rustdoc documenting the feature matrix
- [x] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [x] registry.toml: `tpt-math-optimize` → `"git"`

## Phase 16 — tpt-math-signal-fft

*Wrap rustfft. FFT. Depends on: tpt-math-numeric.*

- [x] Scaffold `crates/tpt-math-signal-fft/`
- [x] Wire deps: `rustfft`, `tpt-math-numeric`
- [x] Implement FFT wrapper
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-signal-fft` → `"git"`

## Phase 17 — tpt-math-signal-filter

*FIR/IIR filters, windowing. Depends on: tpt-math-signal-fft.*

- [x] Scaffold `crates/tpt-math-signal-filter/`
- [x] Wire deps: `tpt-math-signal-fft`
- [x] Implement FIR/IIR filters + windowing
- [x] Unit tests + doctests
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] registry.toml: `tpt-math-signal-filter` → `"git"`

## Phase 18 — tpt-math-signal (umbrella)

*Re-exports fft + filter.*

- [x] Scaffold `crates/tpt-math-signal/`
- [x] Wire optional deps + feature flags for fft/filter
- [x] Re-export each constituent's public API behind its feature
- [x] Rustdoc documenting the feature matrix
- [x] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [x] registry.toml: `tpt-math-signal` → `"git"`

## Final Phase — Workspace Closeout

*No crates.io publishing in this pass — crates stop at `status = "git"`. (Since
superseded: the full workspace has been published to crates.io.)*

- [x] `cargo test --workspace --all-features` passes
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [x] `cargo deny check` clean workspace-wide
- [x] no_std matrix passes for all 8 `no_std = true` crates (numeric, units, exact, linalg, prob-core, prob-dist, prob-sampler, autodiff-fwd)
- [x] Root `README.md` documents the full crate map, build order, and how `tpt-science`/`tpt-engineering`/`tpt-formal` are expected to consume `tpt-math`
- [x] Confirm every `tpt-math-*` entry in `tpt-rust-map/registry.toml` reads `status = "git"`
- [x] Update `tpt-rust-map/TODO.md`: mark the `tpt-math` repo-creation line done; note `tpt-science`/`tpt-engineering`/`tpt-formal` can now proceed in parallel

## Post-Build Hardening — Packaging, Security, Tooling

*Follow-up pass after a full review: the code itself had no stubs, but
packaging metadata, CI, and adoption tooling did. Tracked here the same way
as the build phases above.*

### Packaging + CI bugs

- [x] Add `CHANGELOG.md` (Keep-a-Changelog template) to the 18 crates
      missing it: autodiff-fwd, autodiff-rev, autodiff, exact, linalg,
      optimize-general, optimize-convex, optimize, prob-monte-carlo,
      prob-sampler, prob, signal-fft, signal-filter, signal, stats,
      symbolic, units-dyn, units
- [x] Add `README.md` to the 4 crates missing it: exact, linalg, units-dyn, units
- [x] Fix `.github/workflows/ci.yml`'s `no_std` job — replace the
      `echo "no crates yet"` placeholder with a real build of the 8
      `no_std = true` crates

### Security hardening

- [x] Add `unsafe_code = "forbid"` to `[workspace.lints.rust]` and
      `[lints]\nworkspace = true` to all 22 crate `Cargo.toml`s (currently
      zero crates opt into `[workspace.lints]` at all)
- [x] Tighten `deny.toml`: `advisories.yanked = "deny"`,
      `sources.unknown-registry = "deny"`, `sources.unknown-git = "deny"`
- [x] Add `# Panics` docs to `tpt-math-linalg`'s `Index`/operator impls
- [x] Document `tpt-math-symbolic`'s unbounded-recursion hazard and the
      `f64` round-trip that breaks exactness for transcendental functions;
      comment the two invariant-guarded `unwrap()`s in `fold_add`/`fold_mul`
- [x] Add root `SECURITY.md` (no-`unsafe` policy, `deny.toml` posture,
      panic/`try_*` convention, symbolic recursion caveat, disclosure contact)

### Adoption tooling

- [x] Add `xtask` crate (`fmt`/`clippy`/`test`/`deny`/`no-std`/`check`
      subcommands) + `.cargo/config.toml` alias; CI's `no_std` job calls it
- [x] Add root `justfile` with recipes shelling out to `cargo xtask *`
- [x] Add `examples/` workspace member (`tpt-math-examples`, unpublished)
      with 4 runnable cross-crate programs (units+linalg, prob+stats,
      autodiff+optimize, symbolic+exact)
- [x] Add `cargo-hack` feature-powerset CI job for the 4 umbrella crates;
      swap `test` job to `cargo nextest run` + `cargo test --doc`
- [x] Add root README section: depending on `tpt-math` pre-publish (git-dep
      snippet), pointers to `examples/`, `cargo xtask check`/`just check`,
      `SECURITY.md`/`CONTRIBUTING.md`
- [x] Add root `CONTRIBUTING.md` — issues-only (no external PR workflow):
      how to file an issue, the per-crate checklist, `deny.toml` license policy

### Benchmarks

- [x] Add `criterion` benches (`benches/`, `harness = false`) to linalg,
      signal-fft, optimize-convex, exact
- [x] Add `bench-smoke` CI job: `cargo bench --no-run` across those 4 crates
      (compile-only, not run-for-pass/fail)

### Verification

- [x] `cargo build`/`test --workspace --all-features`,
      `cargo fmt --check`, `cargo clippy -D warnings`, `cargo deny check`
- [x] `cargo xtask no-std` and `cargo xtask check` / `just check`
- [x] Run all 4 new examples; `cargo bench --no-run` on all 4 new bench targets
- [x] `cargo package -p <each newly-fixed crate> --list` — confirms the
      README/CHANGELOG packaging bug is actually fixed

## License-Compliance Fix — nalgebra + clarabel are Apache-2.0-only

*`spec.txt` claims nalgebra, clarabel, and faer are all "dual MIT/Apache" or
"permissively licensed" — verified false for all three (nalgebra: Apache-2.0
only since v0.24.1; clarabel: Apache-2.0 only; faer: MIT only, not dual, but
not disqualifying since only Apache-2.0-ONLY is disqualified per ADR-0007).
nalgebra and clarabel are both disqualified as wrap targets under the
workspace's own no-exceptions rule. Fix (Phase A, blocking): new
`tpt-math-linalg-dense` crate wraps `faer` and owns its own `DVector`/
`DMatrix` types (also solves an orphan-rule problem for `ArgminMath` impls);
`tpt-math-optimize-convex` replaces `clarabel` with an in-repo dense
primal-dual interior-point QP solver. Scope was then deliberately expanded
(Phases B/C, not blocking, no existing consumer) into a fuller
nalgebra-equivalent: `tpt-math-linalg-fixed` (const-generic fixed-size
vectors/matrices, hand-rolled) and `tpt-math-geometry` (full geometry module:
Point/Rotation/Translation/Isometry/Similarity/Scale/Quaternion/
Perspective/Orthographic). Plan: `nalgebra-is-apache-only-unified-giraffe.md`.*

- [x] Bump workspace `rust-version` `1.75` → `1.84` (faer's MSRV)

### Phase A — resolve the license violation (blocking)

### tpt-math-linalg-dense (new crate)

- [x] Scaffold `crates/tpt-math-linalg-dense/` (Cargo.toml, `src/lib.rs`,
      `benches/`)
- [x] Wire deps: `faer` (no_std+alloc feature set), `tpt-math-numeric` (reuse
      its `Scalar` trait instead of a new bound)
- [x] Implement `DVector<T>`/`DMatrix<T>` wrapping `faer::Col<T>`/`Mat<T>`:
      construction (`zeros`, `from_vec`, `from_row_slice`, `from_fn`,
      `from_diagonal`), indexing, elementwise `Add`/`Sub`/`Neg`, scalar
      `Mul`/`Div`, matrix\*matrix and matrix\*vector `Mul`, `transpose()`,
      `norm()`, `dot()`, fallible dense solve/inverse (faer's partial-pivot LU)
- [x] Optional `argmin` feature: `ArgminMath`-family trait impls for
      `DVector<f64>`/`DMatrix<f64>` (compiler-driven — implement whatever the
      four `tpt-math-optimize-general` solvers actually require)
- [x] Unit tests: construction, indexing, arithmetic, transpose, solve/inverse
      (including a deliberately singular matrix), norm/dot
- [x] Rustdoc
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`;
      add `faer` to `[workspace.dependencies]`

### tpt-math-linalg: nalgebra → tpt-math-linalg-dense

- [x] `Cargo.toml`: drop `nalgebra`, add `tpt-math-linalg-dense`
- [x] `src/lib.rs`: `raw` field + all method bodies retargeted at the new
      crate's types; `pub use tpt_math_linalg_dense;` replaces
      `pub use nalgebra;`; `T: nalgebra::Scalar` → `T: tpt_math_numeric::Scalar`
- [x] Update the 4 existing unit tests to the new construction calls
- [x] `benches/linalg_bench.rs`: update construction calls
- [x] `README.md`: remove the "nalgebra is Apache-2.0 only, but permitted"
      paragraph; update quick-start example and "Available operations"
- [x] `CHANGELOG.md`: note the backend swap

### tpt-math-optimize-general: nalgebra/argmin-math → tpt-math-linalg-dense

- [x] `Cargo.toml`: drop `argmin-math`'s `nalgebra_v0_33` feature; add
      `tpt-math-linalg-dense` (with `argmin` feature enabled)
- [x] `src/lib.rs`: `Param`/`Gradient`/`Hessian` types →
      `tpt_math_linalg_dense::{DVector, DMatrix}`; solver logic
      (`ClosureProblem`, `WithGradientTolerance`, `run()`, `validate()`)
      unchanged
- [x] Update doctests + `tests` module to the new constructors
- [x] `README.md`: update code examples, drop `nalgebra_v0_33` mention

### tpt-math-optimize-convex: clarabel → in-house dense IPM QP solver

- [x] `Cargo.toml`: drop `clarabel`, add `tpt-math-linalg-dense`
- [x] Drop `dense_to_csc`/`symmetric_upper_csc` sparse-conversion helpers
- [x] Implement dense primal-dual interior-point method (Mehrotra
      predictor-corrector) against the existing `A x + s = b, s ∈ K`
      formulation; KKT solves via `tpt-math-linalg-dense`'s faer-backed solve
- [x] Map solver outcomes (non-convergence, infeasible, unbounded) to
      `ConvexError::Solver`; replace clarabel's `SolverStatus` with a local enum
      or `String` status
- [x] Keep public API frozen: `solve_qp`, `QuadraticProgram` builder,
      `ConvexError`, `QpSolution`
- [x] All 7 existing tests still pass at comparable tolerances
- [x] New tests: larger random QPs cross-checked against a brute-force/known
      optimum, an infeasible QP, an unbounded QP
- [x] `benches/optimize_convex_bench.rs`: update construction calls
- [x] `README.md`: rewrite clarabel description as the in-house IPM

### Phase B — tpt-math-linalg-fixed (new crate; scope expansion, not blocking)

*Const-generic fixed-size dense linalg (nalgebra's Vector3/Matrix4 layer).
No allocator needed. Depends only on tpt-math-numeric.*

- [x] Scaffold `crates/tpt-math-linalg-fixed/`
- [x] Implement `Vector<T, const N: usize>` / `Matrix<T, const R, const C>`
      + nalgebra-style aliases (`Vector2/3/4/6`, `Matrix2/3/4`, `Matrix3x4`, ...)
- [x] Implement ops: elementwise `Add`/`Sub`/`Neg`, scalar `Mul`/`Div`,
      componentwise `Mul`/`Div`, indexing + `.x()/.y()/.z()/.w()` accessors,
      `dot()`, 3D `cross()` (+ 2D perp-dot), `norm()`/`normalize()`,
      matrix\*matrix / matrix\*vector `Mul`, `transpose()`, `identity()`,
      `from_fn`/`from_array`/`from_columns`
- [x] Implement closed-form determinant/inverse for 2×2/3×3/4×4
- [x] Unit tests: exact known-matrix inverse checks + property-based
      random-invertible-matrix checks, per type/dimension combination
- [x] Rustdoc
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (no `alloc` feature needed — confirm genuinely
      allocator-free)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`

### Phase C — tpt-math-geometry (new crate; scope expansion, not blocking)

*Full geometry module, matching nalgebra's actual breadth. Built on
tpt-math-linalg-fixed.*

- [x] Scaffold `crates/tpt-math-geometry/`
- [x] Implement `Point<T, const D>` (`Point2`/`Point3` aliases)
- [x] Implement `Translation<T, const D>`
- [x] Implement `Rotation<T, const D>` (2D angle constructor; 3D axis-angle +
      Euler constructors; orthogonality guaranteed by construction)
- [x] Implement `Quaternion<T>` / `UnitQuaternion<T>` (Hamilton product,
      conjugate, normalize, `Rotation3` conversion, `slerp`)
- [x] Implement `Isometry<T, const D>` (composition, inverse, point/vector
      action; document the composition convention explicitly)
- [x] Implement `Similarity<T, const D>` and `Scale<T, const D>`
- [x] Implement `Perspective3<T>` / `Orthographic3<T>` (pick and document a
      concrete handedness/depth-range convention)
- [x] Unit tests: known-value rotations (90°/180° per axis vs. textbook
      matrices), round-trip `Rotation3 ↔ UnitQuaternion`, composition/inverse
      identities (`t.inverse() * t ≈ identity`)
- [x] Rustdoc, with every convention (handedness, active/passive rotation,
      row/column vectors) stated explicitly
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `clippy` clean
- [x] `cargo deny check` clean
- [x] no_std verify (no `alloc` feature needed)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`

### Workspace / docs cleanup

- [x] `spec.txt`: fix nalgebra/faer/clarabel license claims (~lines 30-33);
      add dense QP solving to "Genuine gaps" (~lines 41-52) and note
      fixed-size linalg/geometry as deliberate scope expansion, not a
      license-forced gap; rewrite `tpt-math-linalg` crate inventory entry
      (~lines 83-93); add `tpt-math-linalg-dense`/`tpt-math-linalg-fixed`/
      `tpt-math-geometry` inventory entries; rewrite `tpt-math-optimize-convex`
      inventory entry
- [x] `crates/tpt-math-optimize/README.md` (umbrella): update mentions
- [x] `examples/src/bin/units_linalg.rs`, `examples/src/bin/autodiff_optimize.rs`:
      update construction calls
- [x] `deny.toml`: `RUSTSEC-2024-0436` (`paste`) ignore entry — `statrs` is now
      gone and `nalgebra`/`simba` no longer appear in the graph (verified in
      `Cargo.lock`). The archived `paste` itself is now swapped for `pastey`
      (its maintained, drop-in successor) via a local `vendor/paste` shim wired
      through `[patch.crates-io]` in the root `Cargo.toml`, so **no crate
      named `paste` remains in the graph** and the advisory ignore has been
      **removed** from `deny.toml`.

### Phase D — tpt-math-stats: statrs → in-crate MIT implementations (new task)

*statrs (Apache-2.0 only) was the last crate pulling `nalgebra` into tpt-math
(via statrs → `simba` → `paste`/`nalgebra`), keeping the `RUSTSEC-2024-0436`
ignore alive. Replace the statrs surface actually used — error function, gamma,
beta, regularized incomplete gamma/beta, ChiSquared/StudentsT/Normal
distributions — with in-crate `special`/`dist` modules (no external dep). Plan:
`1786420651815-tpt-math-stats-statrs-replacement.md`.*

- [x] Add `src/special.rs` (no external deps): `erf`/`erfc` (via `gamma_q(0.5,
      x²)`), `lgamma` (Lanczos g=7, n=9), `gamma`, `beta`, `gamma_p`/`gamma_q`
      (series + continued fraction), `beta_reg`/`beta_cf` (Lentz)
- [x] Add `src/dist.rs`: `ContinuousCDF` trait, `ChiSquared` (`gamma_p`/`gamma_q`),
      `StudentsT` (`beta_reg` cdf/sf), `Normal` (`erf`-based cdf/sf + pdf/mean/var)
- [x] `src/lib.rs`: add `mod special; mod dist;`, public re-exports
      (`ChiSquared`/`ContinuousCDF`/`Normal`/`StudentsT`, `beta`/`beta_reg`/`erf`/
      `erfc`/`gamma`/`gamma_p`/`gamma_q`/`lgamma`); remove `pub use statrs;`
- [x] `src/lib.rs`: rewrite "Reaching through to statrs" doc section + the
      `statrs_reexport_is_usable` test → `distributions_and_special_functions_are_usable`
- [x] `src/hypothesis.rs`: imports retargeted to `crate::dist`; doc comments
      de-statrs'd; df=1 chi-squared test cross-checks against `crate::special::erfc`
- [x] `src/descriptive.rs`: `agrees_with_statrs_statistics` test replaced with an
      internal variance-identity check (no statrs)
- [x] `Cargo.toml`: drop `statrs = "0.18.0"` dependency; update description
- [x] `tests/stats.rs`: remove `statrs` import + `statrs_stays_reachable...` test;
      cross-checks now use the public `StudentsT`/`ChiSquared` re-exports
- [x] FIX `beta_reg`/`beta_cf` — root-caused to the Lentz continued fraction
      seeding `c = 1.0` (not `1/TINY`) and corrected in `src/special.rs`; the
      gamma path was already correct. Student's t p-values now match closed forms
      (df=1 Cauchy `atan` to 1e-12; df=4 two-sided 5% critical 2.776445 to 1e-6).
- [x] `cargo test -p tpt-math-stats` → all pass (incl. the df=1/2/4 t-test
      closed-form and chi-squared df=1/2 closed forms)
- [x] Update `crates/tpt-math-stats/README.md`: drop "wraps statrs" language
- [x] Update root `README.md` table: `tpt-math-stats` dependency column `statrs` →
      `in-house (special/dist)`
- [x] `deny.toml`: `RUSTSEC-2024-0436` (`paste`) ignore **removed** — replaced
      the archived `paste` crate with `pastey` (maintained successor) via a
      `vendor/paste` shim + `[patch.crates-io]` (see root `Cargo.toml`), so no
      `paste` crate remains in the graph; `cargo deny check` now passes with the
      ignore gone.
- [x] `spec.txt` + Phase 9 line 255 ("Wire deps: `statrs`") — de-statrs
- [x] `cargo deny check licenses` — confirm `statrs` and `nalgebra` no longer
      appear anywhere in the tpt-math dependency graph (verified: both absent
      from `Cargo.lock`)
- [x] `cargo test --workspace` (tpt-math) green

### Verification

- [x] `cargo test --workspace`
- [x] `cargo build -p tpt-math-linalg-dense -p tpt-math-linalg
      --no-default-features --features alloc`
- [x] `cargo build -p tpt-math-linalg-fixed -p tpt-math-geometry
      --no-default-features` (confirm allocator-free)
- [x] `cargo bench -p tpt-math-linalg -p tpt-math-optimize-convex` (sanity run)
- [x] `cargo doc --workspace --no-deps`
- [x] `cargo deny check licenses` — confirm nalgebra/clarabel no longer appear
      in the dependency tree

## Phase E — tpt-math-linalg-sparse (new crate) + dense-crate cleanup

*User asked whether tpt-math has anything for sparse matrices — it doesn't.
The only "sparse" entry in `tpt-rust-map/registry.toml` is `tpt-fem-sparse`
(separate `tpt-fem` repo, FEM-assembly-specific: element scatter +
duplicate-summing triplet accumulation). `tpt-math-linalg-sparse` is
additive: general-purpose sparse matrix types + iterative solvers, hand-rolled
with no external backend (matches the user's no-`faer` license preference and
`tpt-math-linalg-dense`'s actual current design). Plan:
`we-dont-have-anything-atomic-hare.md`.*

### Prerequisite — fix pre-existing issues found in `tpt-math-linalg-dense`

- [x] Fix unresolved merge conflict in
      `crates/tpt-math-linalg-dense/src/lib.rs` (`git status` shows `UU`,
      leftover from an unresolved `git stash pop` — literal `<<<<<<< Updated
      upstream` / `>>>>>>> Stashed changes` markers at `from_row_slice` and
      `from_fn`). Keep the "Updated upstream" branch in both cases — the
      "Stashed changes" branch iterates `0..ncols` instead of
      `0..nrows*ncols`, silently truncating/misindexing non-square (and even
      square) matrices.
- [x] `cargo build --workspace` / `cargo test -p tpt-math-linalg-dense` pass
      after the fix
- [x] Correct stale "faer-backed" documentation left over from before the
      hand-rolled swap (no crate actually depends on `faer` — confirmed
      absent from `Cargo.lock` and every `Cargo.toml`):
      `crates/tpt-math-linalg-dense/{README.md,CHANGELOG.md}`,
      `crates/tpt-math-linalg/{Cargo.toml,src/lib.rs,README.md,CHANGELOG.md}`,
      `crates/tpt-math-optimize-general/src/lib.rs`,
      `crates/tpt-math-optimize-convex/{src/lib.rs,README.md,CHANGELOG.md}`,
      `crates/tpt-math-optimize/README.md`, root `README.md`/`todo.md`/
      `spec.txt`, and `deny.toml`'s `faer`/`gemm` comment
- [x] `grep -ri faer` across the repo — confirm no remaining reference
      describes it as a live dependency

### tpt-math-linalg-sparse (new crate)

- [x] Scaffold `crates/tpt-math-linalg-sparse/` (Cargo.toml mirroring
      `tpt-math-linalg-dense`'s skeleton, `src/lib.rs`, `benches/`)
- [x] Wire deps: `tpt-math-numeric`, `tpt-math-linalg-dense` (reuse `DVector`
      as the dense RHS/solution type); `default = ["std"]` + `alloc` features;
      `[lints] workspace = true`
- [x] Implement `CooMatrix<T>` (triplet list, `push`/`from_triplets`,
      duplicate-summing conversion matching `tpt-fem-sparse`'s semantics),
      `CsrMatrix<T>` (`row_ptr`/`col_idx`/`values`), `CscMatrix<T>`
      (`col_ptr`/`row_idx`/`values`); conversions `Coo::to_csr`/`to_csc`,
      `Csr::transpose`/`Csc::transpose`
- [x] Implement ops: sparse `matvec` (`CsrMatrix<T> * &DVector<T> ->
      DVector<T>`), `nnz()`/`nrows()`/`ncols()`, iteration over stored entries
- [x] Implement iterative solvers only (no hand-rolled direct sparse
      LU/Cholesky — out of scope): `conjugate_gradient` (SPD systems) and
      `bicgstab` (general systems), tolerance + max-iter, `Result<DVector<T>,
      SparseError>`; `SparseError` enum (`DimensionMismatch`,
      `NotConverged { iterations }`)
- [x] Unit tests: COO→CSR/CSC round-trip with duplicate summing, SpMV against
      a known small matrix, CG against a hand-verified SPD system (e.g. 2D
      Laplacian stencil) with known solution, BiCGSTAB against a small
      non-symmetric system, a deliberately non-converging case
      (`SparseError::NotConverged`)
- [x] Rustdoc (crate-level + public API)
- [x] README.md + CHANGELOG.md (rationale: no external sparse backend,
      hand-rolled to avoid license exposure; complements but does not
      duplicate `tpt-fem-sparse`'s FEM-assembly-specific adapter)
- [x] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D
      warnings` clean
- [x] `cargo deny check` clean
- [x] no_std+alloc verify (`cargo build --no-default-features --features
      alloc`)
- [x] Add to root `Cargo.toml` `[workspace] members` +
      `[workspace.dependencies]` (same pattern as the other
      `tpt-math-linalg-*` entries)
- [x] Add `tpt-rust-map/registry.toml` entry: `tpt-math-linalg-sparse`,
      `domain = "math.linalg"`, `no_std = true`, `wraps = []`, description
      drawing the boundary vs. `tpt-fem-sparse`; `status = "planned"` →
      `"git"` once this checklist is done

### Verification

- [x] `cargo build --workspace` / `cargo test --workspace --all-features`
- [x] `cargo fmt --check` / `cargo clippy --workspace --all-targets
      --all-features -- -D warnings` / `cargo deny check` — workspace-wide
- [x] `cargo build -p tpt-math-linalg-sparse --no-default-features --features
      alloc`
- [x] `cargo doc --workspace --no-deps`
- [x] `git status` clean (no leftover conflict markers anywhere)

## Phase F — spec2.txt expanded vision: 4 new crates

*`spec2.txt` extends scope with graph theory, spatial kinematics,
interpolation, and complex-valued linear algebra. Reviewed against ADR-0007's
default-to-wrap policy: `tpt-math-linalg-complex`/`tpt-math-spatial`/
`tpt-math-interpolate` are correctly in-house (no compliant-license wrap
target covers their combined scope — `nalgebra`/`faer` precedent for
linalg-complex, `spatial-math`'s non-MIT/Apache "Custom license" for
spatial, `scirs2-interpolate`'s confirmed Apache-2.0-only license for
interpolate). `tpt-math-graph`'s "in-house" call does not hold up:
`petgraph` is dual `Apache-2.0 OR MIT`, `no_std`-feature-gateable, and
covers adjacency structures/toposort/Dijkstra/A* directly — wrap it, and
build max-flow in-house on top (same "thin-wrap + build the gap" shape as
`tpt-math-exact` over `num-bigint`/`num-rational`).*

### Prerequisite — spec2.txt corrections

- [x] `tpt-math-graph` row: change "Wraps / consolidates" from
      "— (in-house)" to `petgraph` (thin wrap for adjacency/toposort/
      Dijkstra/A*; max-flow stays in-house on top); note the license
      verification in the Notes column
- [x] Add missing `tpt-math-linalg-sparse` row back into spec2.txt's
      "Exact & Linear Algebra" table (it already exists in the workspace
      but is entirely absent from spec2.txt)
- [x] Add `tpt-math-linalg-sparse` to spec2.txt's Section 4 build order
      (alongside `tpt-math-linalg-dense`)
- [x] Add `tpt-math-linalg-sparse` to spec2.txt's Section 5 downstream
      consumption (at minimum `tpt-fem`)

### Phase F1 — tpt-math-linalg-complex

*Complex-valued matrices, LU/Cholesky, QR eigenvalue solvers for
EM/Quantum. In-house, extends `tpt-math-linalg-dense`. no_std+alloc.
Depends on: `tpt-math-linalg-dense`, `tpt-math-numeric`.*

- [x] Scaffold `crates/tpt-math-linalg-complex/`
- [x] Wire deps: `tpt-math-linalg-dense`, `tpt-math-numeric`;
      `default = ["std"]` + `alloc` features; `[lints] workspace = true`
- [x] Implement complex-valued `DVector`/`DMatrix` (reuse
      `tpt-math-linalg-dense`'s storage pattern for a `Complex<T>` scalar)
- [x] Implement complex LU decomposition + solve/inverse
- [x] Implement complex Cholesky decomposition (Hermitian positive-definite)
- [x] Implement QR-algorithm eigenvalue solver for complex matrices
- [x] Unit tests: construction, arithmetic, LU/Cholesky against known
      matrices, QR eigenvalues against known spectra (incl. a Hermitian case)
- [x] Rustdoc (crate-level + public API)
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] `cargo deny check` clean
- [x] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`
- [x] Add `tpt-rust-map/registry.toml` entry: `domain = "math.linalg"`,
      `no_std = true`, `wraps = []`; `status = "planned"` → `"git"` once done

### Phase F2 — tpt-math-graph

*Adjacency structures, topological sort, shortest-path (Dijkstra/A*),
max-flow. Wraps `petgraph` (dual `Apache-2.0 OR MIT`) for adjacency/
toposort/Dijkstra/A*; max-flow built in-house on top (petgraph has no
built-in max-flow — confirmed, upstream feature request open since 2021).
no_std (alloc). Depends on: `tpt-math-numeric`.*

- [x] Scaffold `crates/tpt-math-graph/`
- [x] Wire deps: `petgraph` (`default-features = false`, `no_std`-compatible
      feature set), `tpt-math-numeric`; `default = ["std"]` + `alloc`
      features; `[lints] workspace = true`
- [x] Thin-wrap `petgraph`'s `Graph`/`StableGraph`/`GraphMap` adjacency
      types, `toposort`, `dijkstra`, `astar` behind `tpt-math-graph`'s API
- [x] Implement max-flow in-house (Edmonds-Karp or a Dinic's-algorithm
      variant) on top of the wrapped graph types
- [x] Unit tests: toposort on a known DAG, Dijkstra/A* against hand-verified
      shortest paths, max-flow against a known small flow network
- [x] Rustdoc (crate-level + public API; document the wrap-vs-in-house
      split explicitly)
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] `cargo deny check` clean
- [x] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`;
      add `petgraph` to `[workspace.dependencies]`
- [x] Add `tpt-rust-map/registry.toml` entry: `domain = "math.graph"`,
      `no_std = true`, `wraps = ["petgraph"]`; `status = "planned"` →
      `"git"` once done

### Phase F3 — tpt-math-spatial

*Featherstone spatial vector algebra (6D), dual quaternions, screw theory.
In-house — no compliant-license crate covers the combined scope
(`spatial-math` ships under a non-MIT/Apache "Custom license" and doesn't
cover dual quaternions/screw theory anyway). no_std. Depends on:
`tpt-math-geometry`, `tpt-math-linalg-fixed`, `tpt-math-numeric`.*

- [x] Scaffold `crates/tpt-math-spatial/`
- [x] Wire deps: `tpt-math-geometry`, `tpt-math-linalg-fixed`,
      `tpt-math-numeric`; `[lints] workspace = true`
- [x] Implement 6D spatial vector types (motion/force vectors), Plücker
      coordinate transforms, spatial cross products
- [x] Implement dual quaternions (construction, multiplication, conjugate,
      normalize, rigid-transform conversion)
- [x] Implement screw theory primitives (screw axis, twist, wrench,
      exponential/logarithmic maps to/from `Isometry`)
- [x] Unit tests: known-value spatial transforms, dual-quaternion round-trip
      vs. `tpt-math-geometry::Isometry`, screw motion identities
- [x] Rustdoc, with conventions (frame, handedness) stated explicitly
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] `cargo deny check` clean
- [x] no_std verify (`thumbv6m-none-eabi`)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`
- [x] Add `tpt-rust-map/registry.toml` entry: `domain = "math.spatial"`,
      `no_std = true`, `wraps = []`; `status = "planned"` → `"git"` once done

### Phase F4 — tpt-math-interpolate

*RBF, Kriging, PCHIP, B-spline basis evaluation for scattered data/
surrogates. In-house — no single compliant-license crate covers the full
combined scope (`scirs2-interpolate` covers it all but is confirmed
Apache-2.0-only; piecing together `rbf-interp`/`kriging-rs`/a hand-rolled
PCHIP would fragment the crate behind inconsistent upstream APIs).
no_std+alloc. Depends on: `tpt-math-linalg-dense`, `tpt-math-numeric`.*

- [x] Scaffold `crates/tpt-math-interpolate/`
- [x] Wire deps: `tpt-math-linalg-dense`, `tpt-math-numeric`;
      `default = ["std"]` + `alloc` features; `[lints] workspace = true`
- [x] Implement RBF interpolation (thin-plate, Gaussian, multiquadric
      kernels)
- [x] Implement ordinary Kriging (variogram fitting + prediction)
- [x] Implement PCHIP (shape-preserving piecewise cubic Hermite)
- [x] Implement B-spline basis evaluation (Cox-de Boor)
- [x] Unit tests: known-function interpolation accuracy per method,
      PCHIP monotonicity preservation, B-spline partition-of-unity check
- [x] Rustdoc (crate-level + public API)
- [x] README.md + CHANGELOG.md
- [x] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] `cargo deny check` clean
- [x] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`
- [x] Add `tpt-rust-map/registry.toml` entry: `domain = "math.interpolate"`,
      `no_std = true`, `wraps = []`; `status = "planned"` → `"git"` once done

### Verification

- [x] `cargo build --workspace` / `cargo test --workspace --all-features`
- [x] `cargo fmt --check` / `cargo clippy --workspace --all-targets
      --all-features -- -D warnings` / `cargo deny check` — workspace-wide
- [x] no_std matrix passes for the newly `no_std = true` crates
      (linalg-complex, graph, spatial, interpolate)
- [x] `cargo doc --workspace --no-deps`
- [x] `cargo deny check licenses` — confirm `petgraph` resolves as
      `Apache-2.0 OR MIT` in the dependency graph
- [x] Update root `README.md`'s crate map + `tpt-science`/`tpt-engineering`/
      `tpt-fem`/`tpt-physics` downstream-consumption notes for the 4 new
      crates

## Phase G — Platform review follow-ups (2026-08-19)

*Findings from a workspace-wide review for bugs/todos/missing features.
`todo.md` itself was fully checked off with no stub markers anywhere in the
code; these are the concrete hardening items the review surfaced, plus
committing the in-progress no_std-hygiene fix already sitting in the working
tree.*

- [x] Commit the pending working-tree changes (no_std hygiene +
      `default-features = false` cleanup): root `Cargo.toml`,
      `crates/tpt-math-spatial/{Cargo.toml,src/lib.rs}`,
      `crates/tpt-math-geometry/src/lib.rs`,
      `crates/tpt-math-linalg-complex/src/lib.rs`, and this `todo.md`'s
      Phase F checkboxes
      (already committed in `5bab272` — working tree was clean)
- [x] `tpt-math-spatial::Screw::log`: clamp `cos_t = (tr - 1) / 2` to
      `[-1, 1]` before calling `.acos()` — floating-point rounding near
      identity or 180° rotations can otherwise push `cos_t` out of domain
      and silently propagate `NaN`
- [x] `tpt-math-spatial::Screw::log`: document (or guard) the second
      singularity in axis extraction near `theta ≈ π` (division by
      `2*sin(theta)` — guarded by taking the direction from `(R + I) = 2 w wᵀ`
      with the sign recovered from the skew part; doc-commented)
- [x] Add `tpt-math-spatial` test coverage for the `theta ≈ 0` and
      `theta ≈ π` edge cases of `Screw::log`
- [x] Confirm `tpt-math-prob-markov`'s panic-on-out-of-range-state-index
      API (`chain.rs`, `hmm.rs`) is the intended contract vs. a `Result`-
      returning `try_*` variant, consistent with the `try_X`/`X` pattern
      used elsewhere in the workspace (stats, prob-bayes, units-dyn,
      linalg-sparse): `chain.rs` already had `try_*` pairs; `hmm.rs` setters
      (`set_initial`/`set_transition`/`set_emission`) were missing `try_*`
      counterparts — added `try_set_initial`/`try_set_transition`/
      `try_set_emission` and refactored the panicking setters to delegate
- [x] Audit `tpt-math-prob-sampler`'s `as usize`/`as i32` float-to-int
      casts (highest concentration in the workspace) for unvalidated
      truncation on out-of-range input: no float→int casts exist in production
      code (only `u64`→`usize` on a stream-count that is bounded by the index
      space); the only float→int casts were in tests and are range-bounded by
      construction — no fix required
- [x] Review thin test coverage in `tpt-math-symbolic` (6→16 tests),
      `tpt-math-exact` (5→8 tests), and `tpt-math-linalg` (4→10 tests)
      relative to their surface area; added tests for the gaps (interval
      `Sub`/`is_point`, `Vec`/`Mat` add-sub-neg-scalar-index-zeros, symbolic
      Div-by-one/Func-of-const/Pow-of-const/double-neg/`eval` `None`
      cases and `cos`/`ln`/quotient/`pow(0)` derivatives)

---

## Phase 7 — Platform Review (2026-10-12)

> Findings from a read-only audit of numerics, prob/stats/optimize and
> tooling. Items marked **(V)** were spot-checked against the code; the rest
> come from the audit and need a failing regression test before the fix.
> Delivery rule: test-first — every bug fix lands with a test that fails
> before the fix. Plan: `.claude/plans/review-platform-for-bugs-abundant-pike.md`.

### 7.1 Correctness bugs

- [x] **(V)** `tpt-math-linalg-sparse` `to_csr_impl`: `ncols` inferred as
      `max(col)+1`, so empty / trailing-empty-column matrices get the wrong
      shape and break matvec/CG. Pass `ncols` through (CSR, CSC, transposes)
- [x] **(V)** `tpt-math-interpolate` `Pchip::new` panics for n=2 (`h[1]`
      OOB); `Pchip::eval` does not extrapolate linearly despite the docs
- [x] **(V)** `tpt-math-optimize-general` line search (~603-621): after the
      first iteration `a_hi == a_lo`, so `zoom` gets a degenerate bracket and
      `cubic_min` returns NaN. Use `zoom(a_lo, a)`; add a finiteness guard on
      the step and a stagnation exit (alpha = 0 spins to `max_iters`)
- [x] **(V)** `tpt-math-linalg-dense` / `-complex` `Index<(i,j)>` has no row
      bound check; generic matmul/matvec have no dimension asserts (silent
      wrong results); `from_vec`/`from_row_slice` length checks
- [x] **(V)** `tpt-math-symbolic` `apply_func`: unknown function names
      evaluate as the identity; `to_f64` parses `"3/4"` as 0.0. Return
      `None`/error for unknown functions, convert rationals properly
- [x] `tpt-math-linalg-complex` eigenvalue QR (838-962): Q accumulated as Qᴴ
      (not a similarity transform), phase-less Householder, no Hessenberg
      reduction, silent deflation at `max_iter`, only n=2 tested. Rewrite
      with Hessenberg + Wilkinson-shifted QR, `Err` on non-convergence,
      add n>=3 tests
- [x] `tpt-math-linalg-fixed` `from_columns` wrong for R != C (OOB)
- [x] `tpt-math-graph` `astar`: infinite loop when all open f are inf/NaN;
      O(V²) open-set scan → binary heap. Validate negative weights in
      Dijkstra/A*. Max-flow absolute eps drops small capacities
- [x] Special functions (`tpt-math-stats/special.rs`,
      `tpt-math-prob-bayes/special.rs`): `erf = 1 - erfc` cancellation;
      `Normal::cdf`/`gamma_q` lose tails; `lgamma` NaN for negative args;
      fixed `MAXIT=400` silently unconverged; `ln(1-x)` → `ln_1p`. Dedupe the
      two copies into one shared implementation
      Status: Done: the two copies are now one `tpt-math-prob-core::special`
      module, re-exported by both crates under their existing names. Decision:
      it lives in `tpt-math-prob-core` because both consumers already depend on
      it, and `tpt-math-numeric` stays free of transcendental code. `erf`/`erfc`
      use the incomplete gamma, `Normal::cdf` uses `erfc` in both tails, `gamma`
      keeps its sign, and non-convergence returns `NaN`.
- [x] `tpt-math-geometry`: `slerp`/`Rotation3::powf` `acos` not clamped
      (NaN); `Rotation::from_matrix` accepts det = -1; `UnitQuaternion`
      product never renormalises. `tpt-math-spatial` `Screw::log` uses
      `acos` (use `atan2`); near-π axis extraction precision
      Status: Done: `powf` and `Screw::log` use `atan2`, `from_matrix` rejects det =
      -1, and the quaternion product renormalises. The near-π axis precision
      in `Screw::log` was guarded in Phase G and was not re-audited here.
- [x] `tpt-math-linalg-sparse` CG/BiCGSTAB: absolute (not relative)
      tolerance; no NaN / negative-curvature / `omega == 0` breakdown
      detection; no half-step `‖s‖` check
- [x] `tpt-math-optimize-convex`: `AlmostSolved` returned at 10×tol and
      dropped by `solve_qp`; absolute tolerances, no scaling; separate
      primal/dual step lengths for P ≠ 0
      Status: Done: tolerances are scaled by the problem data, and the
      complementarity gap is objective-relative. `QpSolution.status` reports
      `Solved` or `AlmostSolved`; `solve_qp` returns `Ok` for both and documents it.
      Primal/dual step lengths were already separate.
- [x] `tpt-math-autodiff-rev`: `Powi(a,0)` at 0 → NaN; extra `Const` node
      per scalar op doubles tape growth; no subgradient handling at 0
      Status: Done: `Powi(a, 0)` and `Powf(a, 0)` are 0 at 0. Scalar operators
      record one node (`AddC`, `SubC`, `CSub`, `MulC`, `DivC`, `CDiv`). `abs` uses
      the subgradient 0 at 0. The `Op` enum gained variants, so it is a minor-breaking change.
- [x] `tpt-math-prob-monte-carlo`: naive one-pass variance (cancellation);
      unused `buf` allocation in importance sampling
- [x] Relative singularity thresholds in dense/complex/fixed LU (currently
      absolute 1e-12); overflow-safe norms (scaled / hypot); complex
      `Div`/`inv`/`sqrt` stability (Smith's algorithm)
      Status: Done: complex `Div`/`inv`/`norm` are overflow-safe (Smith, scaled
      hypot), and dense `norm` scales by the largest magnitude. Dense, complex and
      fixed-size LU/inverse each have an absolute and a scale-invariant threshold
      (`with_pivot_tolerance`/`with_relative_tolerance`, `lu_with_*`,
      `inverse_with_*`). The default `1e-12` is unchanged, because the convex IPM
      factors KKT systems with legitimately small pivots.
- [x] `tpt-math-prob-bayes` MCMC: validate initial log-density (NaN / -inf
      leaves chain stuck); `tpt-math-stats` `pearson_correlation`
      `sxx*syy` overflow; `tpt-math-exact` `Interval::new` only
      `debug_assert`s `lo <= hi`
      Status: Done: a NaN starting log-density now panics in the run loops (a `-inf`
      start is not stuck, since any finite candidate is accepted).
      `pearson_correlation` rescales only when the raw sums overflow, so
      exact fits still return exactly ±1. `Interval::new` checks in release
      builds.
- [x] `tpt-math-prob-markov`: power-iteration stationary distribution can
      oscillate for periodic chains and ignores the residual — use a direct
      solve
      Status: Done: `stationary`/`try_stationary` fall back to a direct Gaussian
      solve when power iteration has not settled; reducible chains still
      report `NotConverged`.

### 7.2 Performance

- [x] Dense: j-k-i matmul (contiguous columns); reusable `Lu` struct
      (factor once, solve many RHS); flat storage instead of `Vec<Vec<_>>`;
      in-place `+=` / `axpy` / `*_into` operators
      Status: Done: reusable `Lu`, the j-k-i matmul, in-place `+=`/`-=`/`*=`/`/=`
      and `axpy`, and allocation-free `mul_vector_into`/`mul_matrix_into` in
      `tpt-math-linalg-dense`. Dense storage is a flat column-major `Vec`.
- [x] Sparse: `matvec_into`, in-place axpy in CG/BiCGSTAB, residual norm
      computed once per iteration, counting-sort transpose/convert
      Status: Done: `CsrMatrix::matvec_into`, residual norm once per iteration,
      in-place axpy/xpay updates in CG and BiCGSTAB (slice-based, since `DVector`
      has no `IndexMut`), and counting-sort COO→CSR/CSC conversion.
- [x] `Kriging::predict`: factor once in `new` (currently O(n³) per call)
- [x] HMM Viterbi: precompute log transition/emission matrices
- [x] `tpt-math-exact` intervals: add/sub in 2 ops, not 12 BigRational clones
      Status: Done for add/sub (two endpoint operations each). Multiplication still
      uses the four-corner rule, which is needed for mixed signs.
- [x] `tpt-math-signal-filter`: FFT overlap-add for long FIR kernels; real-FFT
      path in `signal-fft`
      Status: Partial: `FirFilter::filter_overlap_add` is done, and matches the direct
      `filter` to 1e-10. The real-FFT path is `Fft::rfft`/`irfft` in `signal-fft`,
      and the filter uses it.
- [x] Optional `rayon` feature: Monte Carlo, FIR, GEMM
      Status: Done: `rayon` features for Monte Carlo (`integrate_par`, seeded and
      thread-count independent), f64 GEMM (`DMatrix::par_mul`, bitwise equal to the
      serial scalar product), and FIR (`FirFilter::filter_par`, bitwise identical to
      `filter`). Overlap-add stays serial, since it is already fast.
- [x] Criterion benches for every numeric crate (only 5 have any); scheduled
      regression run in CI
      Status: Done: every crate with a numeric kernel has a bench, and `bench.yml`
      runs them weekly, restoring the previous Criterion results from cache so each
      run compares with the one before. The workflow reports and does not fail on a
      slowdown, because shared runners are too noisy to gate on.

### 7.3 Missing features

- [x] Dense QR, least squares, symmetric eigen, SVD, real Cholesky,
      det / rank / cond, matrix exp
      Status: Done: Householder QR, least squares, real Cholesky, symmetric eigen,
      det, thin one-sided Jacobi SVD, `rank`, `cond` and scaling-and-squaring
      `matrix_exp` in `tpt-math-linalg-dense`.
- [x] Optimisation: L-BFGS, Nelder-Mead, damped Newton + line search,
      simplex LP, finite-difference gradients
      Status: Done: gradient descent, nonlinear CG, Newton, L-BFGS, Nelder-Mead,
      finite-difference gradients (`numerical_gradient`), and a two-phase simplex
      LP (`linear_program::maximize`, Bland's rule).
- [x] Stats: quantiles, skew/kurtosis, covariance, pooled/paired t,
      Mann-Whitney, Wilcoxon, KS, ANOVA/F, chi-squared independence,
      bootstrap, multiple-comparison corrections
      Status: Done: quantiles (type 7), skewness, excess kurtosis, covariance,
      one/two-sample/pooled/paired t-tests, chi-squared goodness-of-fit and
      independence, one-way ANOVA, Mann-Whitney, Wilcoxon signed-rank,
      two-sample KS, Bonferroni/Holm/Benjamini-Hochberg corrections, and a
      percentile bootstrap CI (`try_bootstrap_ci`).
- [x] Sampling: HMC, multivariate/adaptive Metropolis, R-hat/ESS
      diagnostics, self-normalised importance sampling + ESS
      Status: Done: `Metropolis`, HMC (`run_hmc`, unit mass) and adaptive Metropolis
      (`run_adaptive_metropolis`, which is multivariate) in `tpt-math-prob-bayes`,
      with split-R̂, effective sample size and self-normalised importance sampling
      (`diagnostics`).
- [x] ODE solvers (RK4/RK45), quadrature, root finding
      Status: Done: fixed-step RK4, adaptive Dormand–Prince 5(4) (`dopri5_integrate`),
      composite Simpson, fixed and adaptive Gauss–Legendre, 15-point Gauss–Kronrod
      (`gauss_kronrod_15`, `adaptive_gauss_kronrod`), and bisection and Brent root
      finding in `tpt-math-numeric`.
- [x] IIR / biquad / Butterworth filters in `signal-filter`
      Status: Done: RBJ biquads for lowpass, highpass, bandpass (constant 0 dB
      peak) and notch, plus Butterworth lowpass, highpass and bandpass cascades.
      The bandpass is a high-pass stage cascaded with a low-pass stage, not an
      LP→BP prototype transform, so its band edges are only approximately −3 dB
      for narrow bands (documented).
- [x] Sparse LU/Cholesky, Jacobi/ILU/IC preconditioners, GMRES
      Status: Done: restarted GMRES (`gmres`), Jacobi-preconditioned CG
      (`jacobi_preconditioned_cg`), IC(0)-preconditioned CG (`ic0_preconditioned_cg`),
      ILU(0) (`Ilu0`) and ILU-preconditioned GMRES (`gmres_ilu0`), sparse Cholesky
      (`SparseCholesky`, no fill-reducing ordering), and sparse LU with partial
      pivoting (`SparseLu`, left-looking, no fill-reducing ordering).
- [x] `autodiff-fwd`: sqrt, powi/powf, tan, tanh, atan, abs, `Dual op scalar`
- [x] `prob-dist`: pdf / cdf / quantile / moments (currently sampling-only)
      Status: Done: `density` gives pdf/pmf, cdf, moments and quantiles for Normal,
      Exponential, Gamma, Beta, Uniform, Poisson, Student-t, Cauchy and chi-squared
      (requires `std`). Continuous quantiles without a closed form invert the CDF
      by bisection; Poisson returns the smallest `k` with `P(X ≤ k) ≥ p`.
- [x] `symbolic`: like-term collection, more functions
      Status: Done: `x/x` (taken as 1, undefined at 0), `x^0`, `x^1`, nested
      powers, fixpoint simplification, like-term collection, and the functions
      `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `asin`, `acos`, `atan`, `sinh`,
      `cosh`, `tanh`, all with derivatives checked against finite differences.
      Canonical form: sums and products are ordered by their printed form, so a
      second `simplify` pass returns the first pass's output. The idempotency property
      is checked by proptest (5000 cases) and by the value-preservation tests.
- [x] `tpt-math-linalg` wrappers: `Mat * scalar`, `Vec` dot/norm, solve,
      by-reference operator forms
      Status: Done: `Mat` scalar `*`/`/`, `Vec::norm`/`Vec::dot` (bare-scalar
      results, units not modelled), `Mat::solve`/`Mat::inverse` (f64, unit-typed),
      and `&`-operand forms for `+`/`-`/`*`. By-reference forms clone their operands.

### 7.4 Innovation (pick one or two)

- [x] Cross-crate differential testing harness (proptest): exact vs f64
      solvers; autodiff-fwd vs autodiff-rev vs symbolic vs finite
      differences
      Status: Done as a deterministic grid-based harness in
      `tpt-math-autodiff/tests/differential.rs` (no new proptest dependency).
      It found and fixed a `num-bigint` version split in `tpt-math-exact`.
- [x] Interval / error-bounded solvers built on `tpt-math-exact::Interval`
      Status: Done: `tpt-math-exact::roots` isolates roots by interval bisection,
      refines them by interval Newton (with `Interval::div_nonzero`), certifies sign
      changes, and gives rigorous Euler enclosures for scalar ODEs and systems
      (`validated_euler`, `validated_euler_system`). The enclosure uses a-priori
      (Picard–Lindelöf) boxes. It is first order, so widths grow with the step count.
      (for `tpt-formal`)
- [x] Units-aware optimise / ODE / autodiff via `units-dyn` + generic scalars
      Status: Done as a boundary layer, not as generic scalars. `tpt-math-units-dyn`
      `solve` checks every user function against the dimension its output must have
      (`dopri5_dimensioned`, `minimize_dimensioned`, `gradient_dimensioned`). The
      numerical cores still run on SI values. Making `Dual` or the solvers generic
      over a dimensioned scalar would need the whole `Float` trait; that was judged
      not worth it. The gradient path checks its output dimensions but not the
      function written on the tape.
- [x] Common `SolveReport { status, iters, residual }` across sparse, convex
      and general solvers
      Status: Partial: `tpt-math-numeric::report` defines `SolveStatus` and
      `SolveReport<T>`. Sparse `conjugate_gradient_with_report`,
      `bicgstab_with_report` and `gmres_right_preconditioned_with_report` return it
      for every outcome. The convex solver reports `iterations` on `QpSolution` but
      not as a `SolveReport`, because its failures carry a reason string that does
      not fit `SolveStatus`, so that is a deliberate exception. The general optimizers
      have `Solution::report()`, with a `residual` field.
- [x] Top-level `tpt-math` facade crate with `prelude` and short feature names
      Status: Done: `crates/tpt-math` re-exports each area behind a short feature
      (nothing on by default, `full` for all), with a feature-gated prelude. It is
      not published. The crates.io name `tpt-math` returned 404 (free) on the check made
      in this session. Recheck just before publishing.

### 7.5 Repo hygiene, CI and automation

- [x] Delete tracked `pkg_tpt-math-*.log` (6 files) and stale
      `history/todo 1260819.md`; add `pkg_*.log` to `.gitignore`
      Status: Done: the six `pkg_*.log` files and `history/todo 1260819.md` were deleted from the working tree. Not committed.
- [x] Fix the header of this file (MSRV says 1.75, workspace is 1.84;
      "publishing out of scope" contradicts "since published")
- [x] Commit `Cargo.lock` (workspace has binaries); remove from `.gitignore`
      Status: Done: `Cargo.lock` is tracked and is no longer ignored.
- [x] CI MSRV job pinned to 1.84; resolve `simd` feature needing Rust 1.85+
      and the out-of-repo `tpt-simd-blas` path dependency that breaks
      `--all-features` and `cargo package`
      Status: Done. `tpt-simd-blas` and `tpt-simd-sparse` are crates.io dependencies
      (0.1.1) instead of path dependencies on the sibling checkout. The sibling
      checkout step is removed from every workflow. The `simd` feature needs Rust
      1.95 (the published crates' rust-version), so the workspace MSRV stays at 1.84
      for the default build; no MSRV CI job exists, so nothing else changes.
      `simd` tests pass for `linalg-dense` and `linalg-sparse`.
- [x] README: `tpt-math-geometry` listed twice; `no_std` column disagrees
      with `NO_STD_CRATES` in `xtask` (8 crates) — extend the CI list or
      correct the README; add an xtask check that validates the README
      tables and the no-std list against `cargo metadata`
      Status: Done: `NO_STD_CRATES` now lists all 16 crates that declare `#![no_std]`
      (the 8 missing ones build for `thumbv6m-none-eabi`). The README column matches.
      `cargo xtask drift` compares the crate sources, the list and the README column
      (it reads the sources and README directly rather than `cargo metadata`).
- [x] CI additions: rust-cache, `concurrency` cancel, `cargo doc` with
      `-D warnings`, cargo-semver-checks, coverage (llvm-cov), scheduled
      `cargo deny` / audit, wasm32 build, cargo-hack feature powerset for
      all crates, benches for every crate with a bench
      Status: Partial: rust-cache, `concurrency` cancel, and a `docs` job with
      `RUSTDOCFLAGS=-D warnings` are in. Doing that surfaced broken intra-doc links,
      now fixed. Not done: cargo-semver-checks, coverage, scheduled audit, wasm32
      build, powerset for all crates (it covers the 4 umbrellas only), benches for
      every crate.
- [x] Make `xtask check` equal CI; add `just ci`
      Status: Done, with one exception: `cargo xtask check` runs fmt, clippy, tests,
      doctests, rustdoc, the no_std build, the drift check and cargo-deny, matching
      the CI jobs. The feature-powerset and bench-smoke jobs need `cargo-hack` and
      are CI-only. `just ci` runs `check`.
- [x] Automation: release-plz (31-crate ordered publishing), Dependabot
      (group `tpt-*`), proptest, cargo-fuzz (symbolic, exact), typos, taplo
      Status: Done, with release-plz manual until item 2 is published. Dependabot is
      configured (weekly, cargo and actions, `tpt-*` grouped). `typos` is pinned to
      `crate-ci/typos@v1.51.1`. `proptest` covers `exact` and `symbolic`. Fuzz
      targets for `symbolic` and `exact` are in `fuzz/`, run weekly on Linux by
      `fuzz.yml`. libFuzzer cannot run on Windows MSVC here: the nightly toolchain
      has no sanitizer runtime and the coverage symbols do not link. `taplo.toml`
      formats the TOML, and CI checks it. `release-plz.toml` opens release PRs with
      publishing off, from a manual workflow, because `cargo package` cannot run
      while the simd path dependency is unpublished.
- [x] `[package.metadata.docs.rs]` (`all-features`, `docsrs` cfg) in each
      crate; feature tables in READMEs; examples for geometry/spatial,
      graph, signal, sparse/complex linalg, `no_std`
      Status: Done: the docs.rs table is in all 31 crates, and `tpt-math-linalg-dense` now uses `all-features`. Its `simd` dependency is on crates.io, so the old path blocker is gone. Feature tables are in the READMEs, and examples cover geometry/spatial, graph, signal and sparse/complex, with a `no_std` snippet in the `linalg-fixed` README.
- [x] Move hard-coded external dep versions (`num-traits`, `uom`, `rand`,
      `criterion`) into `[workspace.dependencies]`; make `default-features`
      consistent
      Status: Done: `criterion`, `num-traits`, `rand`, `rand_distr`, `rustfft` and
      `uom` are inherited from `[workspace.dependencies]`. `symbolic` now uses the
      workspace `num-traits` entry too. It only needs `One` and `Zero`, and its tests
      and no-default-features build pass.
- [x] `deny.toml`: set `[graph]` targets, tighten `wildcards`, enforce
      `multiple-versions`; document the `registry.toml` external dependency
      in CONTRIBUTING.md
      Status: Done: `[graph]` targets set; `wildcards = "deny"`; `multiple-versions = "deny"`
      with one documented skip for `hashbrown` (`petgraph` and `indexmap` need
      different majors); CONTRIBUTING has a section on the `registry.toml` dependency.


### Phase PDE — tpt-math-pde-operators

*Discrete gradient, divergence, Laplacian and central difference on uniform
structured grids (1–3D), staggered (MAC) layout. Structural contracts
(adjointness, discrete conservation, skew-symmetry, Laplacian sign/symmetry,
constant kernel) are checked in debug builds, and in release with the
`contracts` feature. no_std + alloc. Depends on: `tpt-math-numeric`.*

- [x] Scaffold `crates/tpt-math-pde-operators/` (`Grid`, `Field`, `Boundary`,
      `Operators`, contracts)
- [x] Wire deps: `tpt-math-numeric`; `default = ["std"]`, `std`/`alloc`,
      `contracts` features; `[lints] workspace = true`
- [x] Unit tests (interior Dirichlet Laplacian exact on quadratics, Neumann
      constant kernel, contract-report tolerance) and integration tests
      (periodic second-order convergence, mixed 3D `verify`)
- [x] Property tests (proptest) over random 1–3D shapes and boundary mixes
- [x] Rustdoc (crate-level + public API) and README.md + CHANGELOG.md
- [x] `cargo fmt --check` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] no_std+alloc verify (`thumbv6m-none-eabi`)
- [x] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`;
      `NO_STD_CRATES` entry in `xtask`; README crate-map row
- [x] Add `tpt-rust-map/registry.toml` entry (sibling repo, not available here):
      `domain = "math.pde"`, `no_std = true`, `wraps = []`; `status = "planned"` → `"git"`
      Status: Done in the sibling repo (`tpt-rust-map`, commit b4ad23f). The statuses of the built `tpt-math` crates are `git`, and entries were added for linalg-dense, linalg-sparse, linalg-complex, linalg-fixed, geometry, spatial, graph, interpolate and pde-operators. The sibling checkout is at `../tpt-rust-map`; earlier notes that it was missing were wrong.
- [x] Refresh `Cargo.lock` with `cargo update -p tpt-math-pde-operators` once the
      `tpt-simd` sibling checkout is available (CI runs against it)
      Status: Done. `Cargo.lock` lists `tpt-math-pde-operators`, and the `tpt-simd-*` crates
      resolve from crates.io, so no sibling checkout is needed.

---

## Phase H — Platform Review (2026-10-18)

> Findings from a review of the whole workspace. Clippy (`-D warnings`) and all
> 1,065 tests passed with `--all-features` at the time. Items marked **(V)** were
> reproduced with a scratch program or a real build. Delivery rule as in
> Phase 7: every bug fix lands with a test that fails before the fix.

### H.1 Correctness bugs

- [x] **(V)** `tpt-math-linalg-dense` `matrix_exp`: never returns when an entry
      is infinite. `while scaled_norm > 0.5` never exits for `inf`. Return an
      error or a NaN matrix for non-finite input
- [x] **(V)** `tpt-math-linalg-dense` `symmetric_eigen`: returns `[0, 0]` for
      `[[0, 1e-16], [1e-16, 0]]` (want ±1e-16). The stop test
      `off <= 1e-30·(1+scale²)` has an absolute floor. Use `off <= ε²·‖A‖²_F`.
      The symmetry check `1e-12·(1+scale)` has the same flaw
- [x] **(V)** `tpt-math-linalg-dense` `det`/`solve`/`inverse`: the absolute
      `1e-12` pivot default rejects well-conditioned small-scale matrices.
      `det(1e-13·I)` returns `Ok(0.0)`, and `solve` fails. 7.1 kept the default
      because the convex IPM factors KKT systems with small pivots. Have the IPM
      call `Lu::with_pivot_tolerance` explicitly, and make the public defaults
      relative (`Qr` already is). At minimum, `det` should not report 0 for a
      non-singular matrix. `RbfInterpolator::new` and `Kriging::new` inherit this
- [x] **(V)** `tpt-math-linalg-dense` does not compile without `alloc`
      (`cargo check -p tpt-math-linalg-dense --no-default-features`: 18 errors).
      The `#[cfg(feature = "alloc")]` above `DEFAULT_PIVOT_TOLERANCE` (~line 1584)
      was meant for `partial_pivot_lu`, and `partial_pivot_lu_with` has no gate.
      Either gate them or make `alloc` mandatory. CI misses it because
      `xtask no-std` always passes `alloc` (see H.4)
- [x] **(V)** `tpt-math-prob-core::special` `gamma_p`/`gamma_q`: NaN for shape
      ≥ ~1e7 (the series needs more than `MAX_ITER = 10_000` terms near
      `x ≈ a`). This also breaks `poisson_cdf(1e7, 1e7)` and
      `chi_squared_cdf(2e7, 2e7)`. Add Temme's uniform asymptotic expansion for
      large `a`
      Status: Done with a different method from the one above. Temme's expansion was not
      used. For `a ≥ 10` the prefactor `x^a e^-x / Γ(a)` is computed from the Stirling
      series (no cancellation), and the series and continued fraction get an iteration
      budget of `max(10_000, 50·√a)`. `gamma_p(1e7, 1e7)` is now accurate; the old NaN is gone.
- [x] **(V)** `tpt-math-prob-core::special` `ln_binomial`: `ln C(1e15, 1)`
      returns 36 (want 34.5388). The three-`ln_gamma` difference cancels for
      large `n`. Sum `ln((n-i)/(i+1))` for small `k`, or subtract the Stirling
      remainders
- [x] **(V)** `tpt-math-linalg-sparse` with `simd`: a wrong-length `x0` panics
      in CG/BiCGSTAB, while the scalar path returns `Err(DimensionMismatch)`.
      Check `x0.len()` before the `simd::applies` dispatch
- [x] **(V)** Declared MSRV 1.84 does not build: `petgraph → indexmap 2.14 →
      hashbrown 0.17` needs edition 2024. Rust 1.85 builds the workspace
      cleanly. Raise `rust-version` to 1.85 (and fix the README and this file's
      header), or cap `indexmap`. 7.5 marked the MSRV item done, but no MSRV job
      exists
      Status: Done: `rust-version` is 1.85 in `Cargo.toml`, README and PUBLISH_CHECKLIST are
      updated, and the workspace builds on the 1.85 toolchain. Building on 1.84 fails
      at dependency resolution (`clap_lex` needs edition 2024), not only at `indexmap`.
- [x] `tpt-math-linalg-dense` `svd`: hitting `SVD_MAX_SWEEPS` returns the
      unconverged factors silently. Return an error, or a `SolveReport`

### H.2 Docs and hygiene

- [x] `conjugate_gradient` rustdoc says `tol` is an absolute tolerance; the code
      (since 7.1) treats it as relative (`‖r‖ ≤ tol·‖b‖`)
- [x] `ci.yml` `no_std` job comment says it builds 8 crates; `NO_STD_CRATES`
      lists 17
- [x] `release-plz.toml` comment still says simd is a path dependency (also in
      `PUBLISH_CHECKLIST.md`)
- [x] Delete the stray tracked file `- [ ] Dense QR, least squares` in the repo
      root (an older copy of this file)
- [ ] Mark public error and status enums `#[non_exhaustive]` (only 6 of 28
      are), and `QpSolution`. Do it in the release that already bumps
      `autodiff-rev`, `optimize-convex` and `symbolic` to 0.2.0, so future
      variants (like this release's `Op` change) are not breaking
      Status: Not done. The enums are not `#[non_exhaustive]` yet. Adding it breaks downstream exhaustive matches, so it goes with the 0.2.0 bumps, which are not decided.
- [ ] Unify error detail: `SparseError::DimensionMismatch` carries nothing,
      `DenseError::DimensionMismatch` carries a message
      Status: Not done. Changing `SparseError::DimensionMismatch` to carry a message is a breaking change, so it goes with the same version bumps.
- [x] One tolerance convention across solvers: dense LU absolute, QR and the
      sparse solvers relative. Document the rule in CONTRIBUTING

### H.3 Missing features

- [x] `tpt-math` facade: add features for `units-dyn`, `linalg-sparse`,
      `linalg-complex`, `linalg-fixed` and `pde-operators`. Add a `std`/no_std
      passthrough and a richer prelude (`Lu`, `Qr`, `Fft`, `Dual`,
      `GradientTape`, `DynQuantity`). Do this before the facade's first publish.
      Status: Done: features `units-dyn`, `linalg-sparse`, `linalg-complex`, `linalg-fixed`
      and `pde-operators` (all in `full`); a `std` passthrough; and prelude names `Lu`, `Qr`,
      `Fft`, `Dual`, `GradientTape` and `DynQuantity`.
- [x] ODE: a stiff solver (BDF or Rosenbrock), event detection and dense output
      in `tpt-math-numeric`
      Status: Done: `ode::dopri5_dense` (dense output), `ode::dopri5_events` (event location by bisection, optional termination) and `stiff::rosenbrock2_integrate`, a second-order L-stable Rosenbrock method (ROS2) with step-doubling error control. The Robertson kinetics benchmark matches the reference at t = 40.
- [x] Optimisation: nonlinear least squares (Levenberg–Marquardt, a
      `curve_fit`-style helper) and bound constraints (L-BFGS-B or projected)
      Status: Done: `levenberg_marquardt`, `curve_fit` and box bounds (`LeastSquaresOptions::with_bounds`). The bounds are handled by projection, not by an active-set method such as L-BFGS-B.
- [x] Dense linalg: pseudo-inverse / minimum-norm least squares via SVD (`Qr`
      rejects rank-deficient input), a tridiagonal/banded solver, a reusable
      Cholesky solve object, and general real eigenvectors
      Status: Done: `pseudo_inverse` and `solve_min_norm` (SVD), `Cholesky` (reusable factor), `solve_tridiagonal` (Thomas), `solve_banded` (banded LU with partial pivoting), eigenvalues of a general real matrix (Householder Hessenberg reduction and Francis double-shift QR), and the eigenvector for a known real eigenvalue by inverse iteration.
- [x] Sparse linalg: a Lanczos/Arnoldi eigensolver, a fill-reducing ordering
      (AMD/RCM) for `SparseLu`/`SparseCholesky`, and a matrix-free
      `LinearOperator` trait for the Krylov solvers
      Status: Done: `lanczos` (symmetric; Ritz pairs and residuals), `arnoldi` (general; Ritz values, residuals for real ones), `minimum_degree_ordering` with `FillOrdering`, `ReorderedCholesky` and `ReorderedLu`, and `LinearOperator` with operator forms of CG, BiCGSTAB and GMRES. Deviation: the ordering is exact minimum degree on the elimination graph, not AMD, which approximates the degrees. It is fine for thousands of unknowns, not for very large problems.
- [x] Interpolation: N-D `RbfInterpolator` and `Kriging` (they are 1-D only),
      RBF polynomial tail (thin-plate), natural cubic spline, bilinear/trilinear
      grid interpolation
      Status: Done: `RbfInterpolatorNd` (with the thin-plate affine tail), `KrigingNd`, `NaturalCubicSpline` and `MultilinearGrid`.
- [x] Distributions: binomial, negative binomial, lognormal, Weibull,
      multivariate normal
      Status: Done. Binomial, negative binomial (pmf, cdf, moments) and lognormal and Weibull
      (pdf, cdf, quantile, moments) are in `density`. The multivariate normal is in
      `prob_dist::multivariate` behind the `multivariate` feature, which needs `std`. It uses the
      dense Cholesky factor, and its tests check a bivariate closed form and the 1-D case.
- [x] Stats: multiple and logistic regression, Spearman and Kendall
      correlation, kernel density estimation
      Status: Done: Spearman (tie-corrected), Kendall τ-b, multiple regression (`try_multiple_regression`, QR), logistic regression (Newton, `try_logistic_regression`) and Gaussian kernel density estimation (`try_kernel_density`, Silverman bandwidth).
- [x] Signal: STFT/spectrogram, Welch PSD, FFT convolution, zero-phase
      `filtfilt`
      Status: Done: `stft` (complex frames, so `|·|²` gives the spectrogram), `welch`
      (one-sided PSD), `fft_convolve` in `tpt-math-signal-fft`, and `zero_phase_filter`
      (odd-extension filtfilt over a biquad cascade) in `tpt-math-signal-filter`.
- [x] `autodiff-rev`: `atan`, `tanh`, `atan2`, `sinh`/`cosh` (forward mode has
      `atan` and `tanh`); Jacobian and Hessian helpers in the umbrella crate
      Status: Done. The Jacobian is exact, with one reverse pass per output (`autodiff::derivatives::jacobian`). The Hessian is a central difference of the exact gradient, symmetrised, with an error of order 1e-6 at the test points (`hessian`). The reverse-mode operators are done (`tanh`, `sinh`, `cosh`, `atan`,
      `atan2`), each checked against central differences.
- [x] `symbolic`: string parser (`"x^2 + sin(x)"`), compile an expression to a
      closure, LaTeX output
      Status: Done: `parse_expr` (infix grammar with positioned errors), `to_latex`, and `compile`, which turns an expression into a closure over an ordered variable list.
- [x] `graph`: path reconstruction for `dijkstra`; re-export or wrap the
      petgraph algorithms (BFS/DFS, topological sort, MST, components)
      Status: Done: `dijkstra_path` returns the route as well as its cost. BFS/DFS, reachability, cycle detection, connected and strongly connected components, and minimum spanning forest wrap `petgraph`. Topological sort was already re-exported.

### H.4 CI and automation

- [x] Run `cargo hack check --each-feature --no-dev-deps` for every crate.
      This catches H.1's `alloc` build failure; the powerset job covers 10 crates
- [x] Add an MSRV job (`cargo +<msrv> check --workspace`, or
      `cargo hack --rust-version`)
- [x] Run `cargo-semver-checks` on push to master, against the crates.io
      baseline. It only runs on pull requests now, and work is pushed straight to
      master, so it never runs
- [x] Add Windows and macOS to the test job (CI is Ubuntu-only)
- [x] `bench-smoke`: `cargo bench --no-run --workspace` (it builds 4 crates'
      benches; every crate has one)
- [x] More fuzz targets: the `DynQuantity` `FromStr` parser, `from_triplets`,
      and special-function invariants (no NaN on valid input, monotone CDFs,
      `P + Q = 1`)
      Status: Done: `units_dyn_parse`, `sparse_triplets` and `special_functions` are in `fuzz/`,
      and all three build. The first run of `units_dyn_parse` found a real bug, which is fixed:
      `Display` printed compound units such as `m^2·kg·s^-2`, and the parser rejected them. The
      parser now reads canonical base-unit expressions. The runs here are short smoke tests
      without coverage feedback on Windows, because libFuzzer cannot run coverage-guided on MSVC.
      A longer run on Linux is still needed.
- [x] Golden-value accuracy suite: a script that generates reference tables
      from mpmath/SciPy and a test that checks them. It would have caught the
      large-shape gamma and `ln_binomial` bugs
- [x] Extend `xtask drift` to check that the facade exposes every library
      crate and that the README crate count matches `cargo metadata`
      Status: Done: the facade check walks the `[dependencies]` of every crate from `tpt-math` and fails on any crate it cannot reach. The README crate count is compared with `cargo metadata`.
- [x] Occasional `cargo-mutants` run to measure test strength
      Status: Done for `tpt-math-graph` (cargo-mutants 27.1): 134 mutants, 75 caught, 16 missed, 6 timeouts, 37 unviable. The misses cluster in the comparisons of `try_dijkstra`, `dijkstra_path` and `try_astar`, which is where the next tests should go. Other crates have not been run; each run takes minutes, so it stays an occasional check.

### H.5 Innovation (pick one or two)

- [x] Certified linear solves and interval Newton (Krawczyk) for systems,
      building on `exact::roots` and `Interval`, returning guaranteed error
      bounds (for `tpt-formal`)
      Status: Done: `tpt-math-exact::certified_solve` runs the Krawczyk operator with an f64 approximate inverse as preconditioner. The contraction test and the enclosure are exact rational arithmetic, so the enclosure is rigorous. Interval Newton proper (with bisection) is not implemented.
- [x] Diagnostics from dense solvers: `solve_with_report` with a condition
      estimate, backward error and a warning when `cond·ε` exceeds the
      tolerance, so every solver family returns a `SolveReport`
      Status: Done for dense LU: `DMatrix::solve_with_report` returns the residual, normwise backward error, 1-norm condition estimate, and an ill-conditioning flag when cond·ε exceeds 1e-8. The condition number is exact, not estimated, so the call costs about n³ more. Other solver families are not yet wrapped.
- [x] Autodiff-driven optimisers: `minimize_lbfgs_autodiff(|x| …)` takes the
      gradient from `autodiff-rev`, so callers never write one
      Status: Done: `minimize_lbfgs_autodiff` takes the cost written on the reverse-mode tape. It caches the last point, so each iterate is taped once.
- [x] ODE solvers generic over `Dual` for forward sensitivity analysis
      Status: Done: `autodiff-fwd` implements `Float` for `Dual<f64, N>`, so `dopri5_integrate` and the other fixed-size solvers run on dual numbers and return exact parameter sensitivities (tested against the closed form). `numeric::sensitivity::forward_sensitivity` is the explicit-Jacobian alternative for runtime-sized systems.
- [ ] Interactive wasm docs playground (the `no_std` crates already build for
      `wasm32`); Python bindings as a later reach
      Status: Not started, and out of scope for this repository. The `no_std` crates build for `wasm32` in CI. A playground or Python bindings are product decisions for a separate project.

---

## Phase I — Performance, using tpt-simd 0.1.1 (2026-10-18)

> tpt-math already depends on `tpt-simd-blas`/`tpt-simd-sparse` 0.1.1 but uses only
> `gemm`/`gemv`/`dot`/`nrm` and the CG/BiCGSTAB kernels. 0.1.1 also ships blocked
> `getrf`/`getrs`, `potrf`/`potrs`, `trsm`, complex `gemm_c64`/`getrf_c64`, and
> `tpt-simd-reduce` has deterministic f64 reductions. `tpt-simd-math`, `-convolve`
> and `-window` are f32/i16 only, so they don't fit the f64 APIs. Plan:
> `~/.claude/plans/also-do-you-think-memoized-falcon.md`.
>
> Rules: benchmark first and keep a change only if it measurably wins (use a size
> cutoff where SIMD loses on small `n`); keep today's singularity semantics (`1e-12`
> default, `with_*_tolerance`); additive public API only; no result changes beyond
> the documented few-ulp `simd` reassociation.

### I.0 Benchmarks and baseline

- [x] `linalg-dense` benches: `solve`/`Lu::new` at 32/128/512, `inverse` 128,
      `det` 128, `cholesky` 128/512, `svd` 64, `matrix_exp` 32
      Status: Done, default features. Baseline saved as `before` (criterion, bench target
      `linalg_dense_bench`). Medians: solve 32 / 128 / 512 = 12.5 µs / 558 µs / 30.9 ms;
      inverse 128 = 1.63 ms; det 128 = 518 µs; cholesky 128 / 512 = 178 µs / 11.0 ms;
      svd 64 = 2.28 ms; matrix_exp 32 = 938 µs.
- [x] `linalg-complex` benches: `mul`/`solve` 128. `stats` benches:
      `median`/`quantile`/`variance` at 1e6
      Status: Done: `linalg_complex_bench` is now Criterion, with LU, solve, inverse and product at orders 64, 128 and 256. `linalg` gains `Mat solve 32` and `Mat inverse 32`. Results are in the CHANGELOG notes.
- [x] Save Criterion baselines (`-- --save-baseline before`), with and without
      `--features simd`
      Status: Done: the `scalar` (default features) and `simd_runtime` baselines are saved for `linalg-dense`. The `simd-runtime` run was compared against `scalar`. The older `before` baseline predates several bench cases.

### I.1 Default path: flat column-major dense LU

- [x] Rewrite `partial_pivot_lu_with` to factor in place over one flat buffer, with same pivot-tolerance check
      Status: Done, but row-major rather than column-major: the update walks two contiguous row slices, which vectorises just as well and needs no transpose. Same pivot-tolerance check. Result: 128 factor 524 µs -> 148 µs (-72%); 512 solve 30.9 ms -> 13.6 ms (-56%); inverse 128 1.63 ms -> 1.00 ms (-38%); det 128 518 µs -> 151 µs.
- [x] `Lu` internals become `{ n, lu: Vec<f64>, piv }` (private); flat `solve_with_lu`
      Status: Done. The dimension is kept as `piv.len()`, so `n` is not stored separately.
- [x] Add `Lu::solve_matrix(&DMatrix<f64>)` (multi-RHS); `inverse` uses it instead
      of n allocating single solves
      Status: Done. `inverse` factors once and calls `solve_matrix` on the identity. The
      speed-up over the earlier loop is in the noise (inverse 128 is about 1.0 ms both
      before and after this step); the gain was from the flat layout in I.1.
- [x] Fix the misplaced `#[cfg(feature = "alloc")]` on the LU helpers (H.1)
- [x] Inherited by `solve`/`inverse`/`det`, `linalg` `Mat::solve`/`inverse`, the
      convex IPM, the Newton step, and RBF/Kriging. Confirm with their benches
      Status: Done: benches cover the dense `solve`, `inverse` and `det`; `linalg` `Mat::solve`/`inverse`; the convex IPM (`qp 2x2` and `qp 20 variables`); Newton (`Newton Rosenbrock 4-D`); and Kriging fit and predict. The RBF bench already existed.

### I.2 `simd` path: LU and Cholesky through `tpt-simd-blas`

- [x] For `n >= CUTOFF` (from I.0), LU via `getrf_f64`. Apply the existing tolerance
      check to `|U_kk|` afterwards (ignore its exact-zero `Err`); non-finite pivot is
      singular
      Status: Done under `simd-runtime`, not `simd`. Cutoff 64. The `simd` build alone was
      slower than scalar at every size measured (factor 128: 148 us -> 262 us; 512: 13.5 ms ->
      15.2 ms), so the dispatch needs the run-time AVX2 detection. With `simd-runtime`: factor 64
      -7%, 128 -17% (148 us -> 128 us), 512 -65% (13.5 ms -> 4.8 ms). The kernel is a
      column-major getrf, so the result is converted to the row-major layout, and the sequential
      swaps are turned into the permutation. Tests check the factors and permutation against the
      scalar path at n = 5, 33 and 200, and singular input.
- [x] Handle the LAPACK swap-sequence pivots: `getrs_f64` for solves, `det` sign
      from `#{i : ipiv[i] != i}`
      Status: Done: `Lu` factors are converted to LAPACK sequential swaps for `getrs_f64`, and `det` keeps the sign from the permutation. A test checks the sign after 65 row swaps.
- [x] `inverse` / `Lu::solve_matrix`: one `getrs_f64` call with `nrhs = n`
      Status: Done: `Lu::solve_matrix`, and so `inverse`, makes one `getrs_f64` call with `nrhs = n` above the cutoff.
- [x] `cholesky` via `potrf_f64`; zero the strictly upper triangle to match today's
      output
      Status: Done: `DMatrix::cholesky` uses `potrf_f64` above the cutoff. The result has a zero upper triangle, as before.
- [x] Tests: simd vs scalar agree to ~1e-12 relative at n = 5, 33, 200; singular
      and near-singular inputs give the same `Err` on both paths; `det` sign under
      many swaps
      Status: Done: LU factors, permutations, `getrs`-based solves and Cholesky agree with the scalar path at n = 5, 33 and 200. `det` sign is tested after 65 row swaps. Singular input and inputs on both sides of the relative threshold are classified the same way on both paths.

### I.3 Default path: other dense routines

- [x] `matrix_exp`: drop the Index-based `matmul_square` for `mul_matrix_into` with
      reused buffers (picks up SIMD `gemm` under `simd`); replace the 30-term Taylor
      series with Higham's Padé(13) scaling and squaring; add the non-finite guard
      (H.1 hang). Test against closed forms and the old result
      Status: Done: the products go through `mul_matrix_into`, with one scratch buffer for the intermediate products and the squaring loop. `u` and `v` are still owned values, since both are needed at the end.
- [x] `svd`: carry squared column norms between rotations
      Status: Done. Norms are recomputed exactly at the start of each sweep and carried through the rotations, so each pair costs one dot product instead of three. At n = 64 the time drops from 2.28 ms to 1.78 ms (-22%). Tests reconstruct an ill-conditioned Hilbert matrix and a tall Vandermonde matrix.
      (`alpha_p -= t·gamma`, `alpha_q += t·gamma`), recomputed exactly each sweep.
      That's one dot product per pair instead of three. Test `‖A − UΣVᵀ‖` on
      ill-conditioned input
- [x] `Qr::new` and default-path `cholesky` on flat column-major storage, same
      operation order
      Status: Done: `Qr::new` works on flat row-major storage, and so does `cholesky`. The todo asked for column-major; row-major was chosen to match the LU and keep dot products on contiguous slices, so this is a deliberate deviation.

### I.4 `linalg-complex` `simd` feature

- [x] Add `simd = ["alloc", "dep:tpt-simd-blas"]` (`tpt-simd-blas = "0.1.1"`; needs
      Rust 1.95, noted in Cargo.toml and README)
      Status: Done: the `simd` and `simd-runtime` features. The LU path dispatches above a cutoff of 64.
- [x] For `n >= CUTOFF`: matmul via split re/im planes and `gemm_c64`;
      `solve`/`inverse`/`det` via `getrf_c64`/`getrs_c64`. Public `lu()` unchanged
      Status: Done: `ComplexDMatrix<f64>::product` uses `gemm_c64` above the cutoff (with `simd-runtime`), and `det` uses `getrf_c64`. `solve`, `inverse` and `lu` were already done. The `*` operator stays the scalar generic product, because it is generic over `T`.
- [ ] CI `--exclude-features simd` handling and the `simd` test commands in
      `PUBLISH_CHECKLIST.md`
      Status: Open: the CI handling for the `simd` feature, and the simd test commands in `PUBLISH_CHECKLIST.md` (an untracked file), are not done.

### I.5 Algorithmic wins

- [x] `stats` `try_median`/`try_quantile`: `select_nth_unstable_by(f64::total_cmp)`
      instead of a full sort (O(n), identical results). Test against a sort-based
      reference
      Status: Done. Results are bitwise identical to the sort-based reference in
      `selection_tests`. Against the `before` baseline at 1e6: median 26.4 ms -> 4.0 ms
      (-85%), quantile 0.9 26.6 ms -> 3.5 ms (-87%).
- [x] Optional `stats` `simd` feature with `tpt-simd-reduce = "0.1.1"` for mean,
      variance and covariance. Keep only if I.0 shows a win at large `n`
      Status: Done as an opt-in `simd` feature. The variance uses `tpt-simd-reduce` and runs about 7× faster at 10⁶ samples (3.9 ms → 0.56 ms). It keeps pairwise sums instead of compensated ones, so it is off by default.

### I.6 Wrap-up

- [x] Re-run benches against the `before` baselines (with and without `simd`);
      revert any step that does not win; record speedups in each CHANGELOG
      `[Unreleased]`
      Status: Done (see the entry above): the `linalg-dense` benches were re-run against the scalar and `simd-runtime` builds, and the numbers are in the `linalg-dense` CHANGELOG.
- [x] `cargo test --workspace` (default and `--all-features`), `--features simd`
      for dense and complex, clippy `-D warnings`, `cargo xtask check`
      Status: Done: `cargo test --workspace` passes in both default features (1,216 tests) and `--all-features` (1,250 tests). `cargo clippy --workspace --all-targets --all-features -- -D warnings` and `cargo xtask check` pass.

### I.7 Deferred (each needs its own design pass)

- [x] AMD/RCM fill-reducing ordering for `SparseLu`/`SparseCholesky`
      Status: Done: RCM (earlier) and minimum degree (`minimum_degree_ordering`, `FillOrdering`), used by `ReorderedCholesky` and `ReorderedLu`. Deviation: exact minimum degree instead of AMD's approximate degrees.
- [ ] `tpt-simd-rng` lane-parallel sampling in `prob-monte-carlo` (changes the
      random streams, so decide on stream compatibility first)
      Status: Deferred, as the item says: it needs its own design pass first.

---

## Phase J — Changelogs, CI verification, breaking changes

> Gaps found by reviewing the work since the Phase H review. Phase I and H.3 recorded
> their results in `todo.md` only; these items bring the public record up to date.

- [x] CHANGELOG `[Unreleased]` entries for the new public APIs, one per touched crate:
      `tpt-math-linalg-dense` (`Cholesky`, `solve_tridiagonal`, `pseudo_inverse`,
      `solve_min_norm`, `Lu::solve_matrix`, `NoConvergence`, the Padé `matrix_exp`, the
      flat LU, the `simd-runtime` LU dispatch); `tpt-math-linalg-sparse` (length check on
      `x0`, the `to_csr`/`to_csc`/`transpose` alloc gating); `tpt-math-linalg`,
      `tpt-math-linalg-complex`, `tpt-math-interpolate` (alloc-gated APIs, `std` enables
      `alloc`); `tpt-math-stats` (`try_multiple_regression`, `try_logistic_regression`,
      `try_spearman_correlation`, `try_kendall_tau`, `RankDeficientDesign`,
      `DidNotConverge`, selection-based median and quantile); `tpt-math-prob-dist`
      (lognormal, Weibull, binomial, negative binomial, `multivariate` feature);
      `tpt-math-prob-core` (`ln_binomial` fix, large-shape `gamma_p`/`gamma_q`);
      `tpt-math-graph` (`dijkstra_path`); `tpt-math-signal-fft` (`stft`, `welch`,
      `fft_convolve`); `tpt-math-signal-filter` (`zero_phase_filter`);
      `tpt-math-symbolic` (`parse_expr`, `to_latex`); `tpt-math-units-dyn` (`FromStr`
      accepts base-unit expressions); `tpt-math-autodiff-rev` (`tanh`, `sinh`, `cosh`,
      `atan`, `atan2`); `tpt-math-autodiff` (`derivatives::jacobian`, `hessian`);
      `tpt-math` (new area features and prelude names); `xtask` is not published.
      Status: Done: the `[Unreleased]` sections of 17 crates now list the new APIs, and breaking changes are marked `**Breaking:**`.
- [ ] Verify the first GitHub Actions run of `ci.yml` after the changes in `b362e15`:
      the Windows and macOS legs of `test` (including `--all-features`, which pulls in
      the simd dependencies), the `msrv` job on 1.85, `each-feature`, `feature-powerset`
      with the corrected exclude list, `bench-smoke` over the whole workspace, and
      `semver-checks` on push against crates.io. Fix whatever fails, and record here what
      passed.
      Status: Run 39122793847 on `d884adb` passed `test` on Ubuntu, Windows and macOS, `msrv` (1.85), `feature-powerset`, `each-feature`, `clippy`, `docs`, `fmt`, `no_std`, `wasm`, `cargo-deny`, `coverage` and `bench-smoke`. It failed two jobs. `typos` flagged two words in new code. Run 39123538331 on `74d38e3` passes `typos`, and `semver-checks` is the only job that still fails. `semver-checks` fails as expected, because the breaking changes are recorded in the CHANGELOGs but the versions are not bumped, so it waits on the version decision below.
- [ ] Record the breaking changes and decide the version bumps before publishing:
      `StatsError` gained `RankDeficientDesign` and `DidNotConverge`, and it is not
      `#[non_exhaustive]`; `DenseError` gained `NoConvergence`; `tpt-math-linalg-dense`
      gained `Lu`'s flat storage (private, so not breaking); `tpt-math-linalg` and
      `tpt-math-linalg-complex` now need `alloc` for their `zeros`, `transpose`, `solve`
      and `inverse` APIs (default builds keep them via `std`). Each goes in the CHANGELOG
      with its crate's bump, and the list is checked against `cargo semver-checks` once a
      baseline exists. Additive changes, such as `DynQuantity::from_str` accepting base-unit
      expressions, go in the CHANGELOG only.
      Status: Partly done: each breaking change is marked in its crate's CHANGELOG. The version bumps are not decided, and no crate has been bumped.
