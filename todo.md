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

*Wrap statrs for hypothesis tests / regression. Consolidates
tpt-zero-formal's tpt-zero-stats and tpt-rust6's tpt-stat. Depends on:
tpt-math-prob-core.*

- [x] Scaffold `crates/tpt-math-stats/`
- [x] Wire deps: `statrs`, `tpt-math-prob-core`
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
- [ ] `deny.toml`: remove the `RUSTSEC-2024-0436` (`paste`/nalgebra→simba)
      ignore entry once nalgebra is gone from the tree
      > **BLOCKED:** `nalgebra` is *not* gone — `statrs` (which
      > `tpt-math-stats` wraps) depends on `nalgebra` → `simba` → `paste`, so the
      > advisory still fires. The ignore was kept (comment updated to note
      > `statrs`). It can only be removed by dropping/replacing `statrs`.

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

