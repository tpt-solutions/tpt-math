# tpt-math — Build Todo

> Tracks bootstrap + full 23-crate build-out for the tpt-math foundation repo,
> per `spec.txt` and `tpt-rust-map/registry.toml`. Crates.io publishing is
> intentionally **out of scope** for this pass — crates stop at
> `status = "git"` in the registry, not `"published"`. License for every
> crate: `MIT OR Apache-2.0`. Author: TPT Solutions.

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

*No crates.io publishing in this pass — crates stop at `status = "git"`.*

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

- [ ] `tpt-math-graph` row: change "Wraps / consolidates" from
      "— (in-house)" to `petgraph` (thin wrap for adjacency/toposort/
      Dijkstra/A*; max-flow stays in-house on top); note the license
      verification in the Notes column
- [ ] Add missing `tpt-math-linalg-sparse` row back into spec2.txt's
      "Exact & Linear Algebra" table (it already exists in the workspace
      but is entirely absent from spec2.txt)
- [ ] Add `tpt-math-linalg-sparse` to spec2.txt's Section 4 build order
      (alongside `tpt-math-linalg-dense`)
- [ ] Add `tpt-math-linalg-sparse` to spec2.txt's Section 5 downstream
      consumption (at minimum `tpt-fem`)

### Phase F1 — tpt-math-linalg-complex

*Complex-valued matrices, LU/Cholesky, QR eigenvalue solvers for
EM/Quantum. In-house, extends `tpt-math-linalg-dense`. no_std+alloc.
Depends on: `tpt-math-linalg-dense`, `tpt-math-numeric`.*

- [ ] Scaffold `crates/tpt-math-linalg-complex/`
- [ ] Wire deps: `tpt-math-linalg-dense`, `tpt-math-numeric`;
      `default = ["std"]` + `alloc` features; `[lints] workspace = true`
- [ ] Implement complex-valued `DVector`/`DMatrix` (reuse
      `tpt-math-linalg-dense`'s storage pattern for a `Complex<T>` scalar)
- [ ] Implement complex LU decomposition + solve/inverse
- [ ] Implement complex Cholesky decomposition (Hermitian positive-definite)
- [ ] Implement QR-algorithm eigenvalue solver for complex matrices
- [ ] Unit tests: construction, arithmetic, LU/Cholesky against known
      matrices, QR eigenvalues against known spectra (incl. a Hermitian case)
- [ ] Rustdoc (crate-level + public API)
- [ ] README.md + CHANGELOG.md
- [ ] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo deny check` clean
- [ ] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [ ] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`
- [ ] Add `tpt-rust-map/registry.toml` entry: `domain = "math.linalg"`,
      `no_std = true`, `wraps = []`; `status = "planned"` → `"git"` once done

### Phase F2 — tpt-math-graph

*Adjacency structures, topological sort, shortest-path (Dijkstra/A*),
max-flow. Wraps `petgraph` (dual `Apache-2.0 OR MIT`) for adjacency/
toposort/Dijkstra/A*; max-flow built in-house on top (petgraph has no
built-in max-flow — confirmed, upstream feature request open since 2021).
no_std (alloc). Depends on: `tpt-math-numeric`.*

- [ ] Scaffold `crates/tpt-math-graph/`
- [ ] Wire deps: `petgraph` (`default-features = false`, `no_std`-compatible
      feature set), `tpt-math-numeric`; `default = ["std"]` + `alloc`
      features; `[lints] workspace = true`
- [ ] Thin-wrap `petgraph`'s `Graph`/`StableGraph`/`GraphMap` adjacency
      types, `toposort`, `dijkstra`, `astar` behind `tpt-math-graph`'s API
- [ ] Implement max-flow in-house (Edmonds-Karp or a Dinic's-algorithm
      variant) on top of the wrapped graph types
- [ ] Unit tests: toposort on a known DAG, Dijkstra/A* against hand-verified
      shortest paths, max-flow against a known small flow network
- [ ] Rustdoc (crate-level + public API; document the wrap-vs-in-house
      split explicitly)
- [ ] README.md + CHANGELOG.md
- [ ] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo deny check` clean
- [ ] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [ ] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`;
      add `petgraph` to `[workspace.dependencies]`
- [ ] Add `tpt-rust-map/registry.toml` entry: `domain = "math.graph"`,
      `no_std = true`, `wraps = ["petgraph"]`; `status = "planned"` →
      `"git"` once done

### Phase F3 — tpt-math-spatial

*Featherstone spatial vector algebra (6D), dual quaternions, screw theory.
In-house — no compliant-license crate covers the combined scope
(`spatial-math` ships under a non-MIT/Apache "Custom license" and doesn't
cover dual quaternions/screw theory anyway). no_std. Depends on:
`tpt-math-geometry`, `tpt-math-linalg-fixed`, `tpt-math-numeric`.*

- [ ] Scaffold `crates/tpt-math-spatial/`
- [ ] Wire deps: `tpt-math-geometry`, `tpt-math-linalg-fixed`,
      `tpt-math-numeric`; `[lints] workspace = true`
- [ ] Implement 6D spatial vector types (motion/force vectors), Plücker
      coordinate transforms, spatial cross products
- [ ] Implement dual quaternions (construction, multiplication, conjugate,
      normalize, rigid-transform conversion)
- [ ] Implement screw theory primitives (screw axis, twist, wrench,
      exponential/logarithmic maps to/from `Isometry`)
- [ ] Unit tests: known-value spatial transforms, dual-quaternion round-trip
      vs. `tpt-math-geometry::Isometry`, screw motion identities
- [ ] Rustdoc, with conventions (frame, handedness) stated explicitly
- [ ] README.md + CHANGELOG.md
- [ ] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`
- [ ] Add `tpt-rust-map/registry.toml` entry: `domain = "math.spatial"`,
      `no_std = true`, `wraps = []`; `status = "planned"` → `"git"` once done

### Phase F4 — tpt-math-interpolate

*RBF, Kriging, PCHIP, B-spline basis evaluation for scattered data/
surrogates. In-house — no single compliant-license crate covers the full
combined scope (`scirs2-interpolate` covers it all but is confirmed
Apache-2.0-only; piecing together `rbf-interp`/`kriging-rs`/a hand-rolled
PCHIP would fragment the crate behind inconsistent upstream APIs).
no_std+alloc. Depends on: `tpt-math-linalg-dense`, `tpt-math-numeric`.*

- [ ] Scaffold `crates/tpt-math-interpolate/`
- [ ] Wire deps: `tpt-math-linalg-dense`, `tpt-math-numeric`;
      `default = ["std"]` + `alloc` features; `[lints] workspace = true`
- [ ] Implement RBF interpolation (thin-plate, Gaussian, multiquadric
      kernels)
- [ ] Implement ordinary Kriging (variogram fitting + prediction)
- [ ] Implement PCHIP (shape-preserving piecewise cubic Hermite)
- [ ] Implement B-spline basis evaluation (Cox-de Boor)
- [ ] Unit tests: known-function interpolation accuracy per method,
      PCHIP monotonicity preservation, B-spline partition-of-unity check
- [ ] Rustdoc (crate-level + public API)
- [ ] README.md + CHANGELOG.md
- [ ] `cargo fmt` / `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo deny check` clean
- [ ] no_std+alloc verify (`cargo build --no-default-features --features alloc`)
- [ ] Add to root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]`
- [ ] Add `tpt-rust-map/registry.toml` entry: `domain = "math.interpolate"`,
      `no_std = true`, `wraps = []`; `status = "planned"` → `"git"` once done

### Verification

- [ ] `cargo build --workspace` / `cargo test --workspace --all-features`
- [ ] `cargo fmt --check` / `cargo clippy --workspace --all-targets
      --all-features -- -D warnings` / `cargo deny check` — workspace-wide
- [ ] no_std matrix passes for the newly `no_std = true` crates
      (linalg-complex, graph, spatial, interpolate)
- [ ] `cargo doc --workspace --no-deps`
- [ ] `cargo deny check licenses` — confirm `petgraph` resolves as
      `Apache-2.0 OR MIT` in the dependency graph
- [ ] Update root `README.md`'s crate map + `tpt-science`/`tpt-engineering`/
      `tpt-fem`/`tpt-physics` downstream-consumption notes for the 4 new
      crates

