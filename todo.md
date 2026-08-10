# tpt-math — Build Todo

> Tracks bootstrap + full 23-crate build-out for the tpt-math foundation repo,
> per `spec.txt` and `tpt-rust-map/registry.toml`. Crates.io publishing is
> intentionally **out of scope** for this pass — crates stop at
> `status = "git"` in the registry, not `"published"`. License for every
> crate: `MIT OR Apache-2.0`. Author: TPT Solutions.

## Phase 0 — Repo Bootstrap

(one-time, seed from `tpt-rust-map/template/`)

- [ ] Copy `template/Cargo.toml` → root `Cargo.toml` (workspace `resolver = "2"`,
      `[workspace.package]`: `edition = "2021"`, `rust-version = "1.75"`,
      `license = "MIT OR Apache-2.0"`, `authors = ["TPT Solutions"]`; replace
      placeholder `homepage`/`repository` URLs)
- [ ] Copy `template/rust-toolchain.toml`
- [ ] Copy `template/rustfmt.toml`
- [ ] Copy `template/deny.toml`
- [ ] Copy `template/.github/workflows/ci.yml`
- [ ] Copy `template/LICENSE-MIT` and `template/LICENSE-APACHE`
- [ ] Create empty `crates/` directory
- [ ] Add a Rust `.gitignore` (`/target`, etc.)
- [ ] Write root `README.md` stub — tpt-math's role as the numeric substrate
      for tpt-science/tpt-engineering/tpt-formal; link to `spec.txt` and
      `tpt-rust-map`
- [ ] `git init` (local only — no GitHub remote/push)
- [ ] Initial commit
- [ ] Sanity check: `cargo build` succeeds on the empty workspace

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

- [ ] Scaffold `crates/tpt-math-numeric/`
- [ ] Wire deps: `num-traits`, `libm`; `default = ["std"]` + `alloc`/`libm` features
- [ ] Implement scalar numeric trait glue
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-numeric` → `"git"`

## Phase 2 — tpt-math-units

*Wrap uom (disable default std feature). Compile-time typed units. no_std. No internal deps.*

- [ ] Scaffold `crates/tpt-math-units/`
- [ ] Wire deps: `uom` (std feature disabled by default)
- [ ] Implement compile-time typed units wrapper
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-units` → `"git"`

## Phase 3 — tpt-math-units-dyn ⚠️ assumption flagged

*Runtime dimension-checked units for dynamic/config-driven payloads.
Consolidates tpt-rust2's tpt-units-runtime. Depends on: tpt-math-units.*

> **Note:** neither `spec.txt`'s BUILD ORDER nor `tpt-rust-map/TODO.md`
> (identical wording) mentions this crate's position. Its only declared
> dependency is `tpt-math-units` and nothing downstream depends on it, so
> it's placed here as the lowest-risk slot. Re-sequence freely if that
> assumption turns out wrong.

- [ ] Scaffold `crates/tpt-math-units-dyn/`
- [ ] Wire deps: `tpt-math-units`
- [ ] Implement runtime dimension-checked unit type for dynamic/config payloads
- [ ] Port/consolidate logic from `tpt-rust2`'s `tpt-units-runtime`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-units-dyn` → `"git"`

(no no_std step — `registry.toml` has `no_std = false` for this crate)

## Phase 4 — tpt-math-exact

*Thin-wrap num-bigint/num-rational; interval arithmetic layer built on top.
Arbitrary-precision rational + interval arithmetic. Consolidates
tpt-formal-lab's tpt-exact-math. no_std+alloc. Depends on: tpt-math-numeric.*

- [ ] Scaffold `crates/tpt-math-exact/`
- [ ] Wire deps: `num-bigint`, `num-rational`, `tpt-math-numeric`
- [ ] Implement exact-rational wrapper + interval arithmetic layer on top
- [ ] Port/consolidate logic from `tpt-formal-lab`'s `tpt-exact-math`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std+alloc verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-exact` → `"git"`

## Phase 5 — tpt-math-linalg

*Wrap nalgebra. Pair with tpt-math-units for dimensionally-checked
vectors/matrices — nalgebra only, not a dual nalgebra+faer facade.
Consolidates tpt-zero-formal's tpt-zero-linalg and tpt-zero-text's
tpt-zero-matrix. no_std. Depends on: tpt-math-units.*

- [ ] Scaffold `crates/tpt-math-linalg/`
- [ ] Wire deps: `nalgebra`, `tpt-math-units`
- [ ] Implement dimensionally-checked vector/matrix types pairing nalgebra with tpt-math-units
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-linalg` and `tpt-zero-text`'s `tpt-zero-matrix`
- [ ] Unit tests + doctests
- [ ] Rustdoc (note the nalgebra-only backend decision — no faer facade, per spec.txt rationale)
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-linalg` → `"git"`

## Phase 6 — tpt-math-prob-core

*Shared Distribution/Sampler traits that the tpt-math-prob-* crates
implement against. no_std. Depends on: tpt-math-numeric.*

- [ ] Scaffold `crates/tpt-math-prob-core/`
- [ ] Wire deps: `tpt-math-numeric`
- [ ] Implement shared `Distribution`/`Sampler` traits
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-prob-core` → `"git"`

## Phase 7 — tpt-math-prob-* (any order / parallelizable)

All five depend only on `tpt-math-prob-core` and can be built in any order or in parallel.

### 7a — tpt-math-prob-dist

*Wrap rand_distr. Standard distributions. no_std. Consolidates
tpt-zero-formal's tpt-zero-dist.*

- [ ] Scaffold `crates/tpt-math-prob-dist/`
- [ ] Wire deps: `rand_distr`, `tpt-math-prob-core`
- [ ] Implement standard distributions wrapper
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-dist`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-prob-dist` → `"git"`

### 7b — tpt-math-prob-bayes

*Bayesian inference primitives. Consolidates tpt-zero-formal's tpt-zero-bayes.*

- [ ] Scaffold `crates/tpt-math-prob-bayes/`
- [ ] Wire deps: `tpt-math-prob-core`
- [ ] Implement Bayesian inference primitives
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-bayes`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-prob-bayes` → `"git"`

### 7c — tpt-math-prob-markov

*Markov chains / HMM. Consolidates tpt-zero-formal's tpt-zero-markov.*

- [ ] Scaffold `crates/tpt-math-prob-markov/`
- [ ] Wire deps: `tpt-math-prob-core`
- [ ] Implement Markov chain / HMM primitives
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-markov`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-prob-markov` → `"git"`

### 7d — tpt-math-prob-monte-carlo

*Monte Carlo methods. Consolidates tpt-zero-formal's tpt-zero-monte-carlo.*

- [ ] Scaffold `crates/tpt-math-prob-monte-carlo/`
- [ ] Wire deps: `tpt-math-prob-core`
- [ ] Implement Monte Carlo methods
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-monte-carlo`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-prob-monte-carlo` → `"git"`

### 7e — tpt-math-prob-sampler

*Sampling strategies. no_std. Consolidates tpt-zero-formal's
tpt-zero-sampler and tpt-zero-rand.*

- [ ] Scaffold `crates/tpt-math-prob-sampler/`
- [ ] Wire deps: `tpt-math-prob-core`
- [ ] Implement sampling strategies
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-sampler` and `tpt-zero-rand`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-prob-sampler` → `"git"`

## Phase 8 — tpt-math-prob (umbrella)

*Re-exports the five tpt-math-prob-* crates behind Cargo features.*

- [ ] Scaffold `crates/tpt-math-prob/`
- [ ] Wire optional deps + feature flags for dist/bayes/markov/monte-carlo/sampler
- [ ] Re-export each constituent's public API behind its feature
- [ ] Rustdoc documenting the feature matrix
- [ ] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [ ] registry.toml: `tpt-math-prob` → `"git"`

## Phase 9 — tpt-math-stats

*Wrap statrs for hypothesis tests / regression. Consolidates
tpt-zero-formal's tpt-zero-stats and tpt-rust6's tpt-stat. Depends on:
tpt-math-prob-core.*

- [ ] Scaffold `crates/tpt-math-stats/`
- [ ] Wire deps: `statrs`, `tpt-math-prob-core`
- [ ] Implement hypothesis tests / regression wrapper
- [ ] Port/consolidate logic from `tpt-zero-formal`'s `tpt-zero-stats` and `tpt-rust6`'s `tpt-stat`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-stats` → `"git"`

## Phase 10 — tpt-math-autodiff-fwd

*Dual numbers over tpt-math-numeric. Forward-mode autodiff. no_std. Depends
on: tpt-math-numeric.*

- [ ] Scaffold `crates/tpt-math-autodiff-fwd/`
- [ ] Wire deps: `tpt-math-numeric`
- [ ] Implement dual-number forward-mode autodiff
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std verify (`thumbv6m-none-eabi`)
- [ ] registry.toml: `tpt-math-autodiff-fwd` → `"git"`

## Phase 11 — tpt-math-autodiff-rev

*Reverse-mode/tape autodiff. Consolidates tpt-rust6's
tpt-grad/tpt-grad-macro and tpt-zero-formal's tpt-zero-grad. Depends on:
tpt-math-autodiff-fwd.*

- [ ] Scaffold `crates/tpt-math-autodiff-rev/`
- [ ] Wire deps: `tpt-math-autodiff-fwd`
- [ ] Implement reverse-mode/tape autodiff
- [ ] Port/consolidate logic from `tpt-rust6`'s `tpt-grad`/`tpt-grad-macro` and `tpt-zero-formal`'s `tpt-zero-grad`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-autodiff-rev` → `"git"`

## Phase 12 — tpt-math-autodiff (umbrella)

*Re-exports fwd + rev.*

- [ ] Scaffold `crates/tpt-math-autodiff/`
- [ ] Wire optional deps + feature flags for fwd/rev
- [ ] Re-export each constituent's public API behind its feature
- [ ] Rustdoc documenting the feature matrix
- [ ] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [ ] registry.toml: `tpt-math-autodiff` → `"git"`

## Phase 13 — tpt-math-symbolic

*Permissive-license symbolic math (CAS) — genuine, unfilled ecosystem gap.
Consolidates tpt-rust6's tpt-sym. Optionally depends on: tpt-math-exact.*

- [ ] **Design decision (open call per spec.txt, resolve now):** does
      `tpt-math-symbolic` default to f64 via a generic `Coefficient` trait,
      or default to `tpt-math-exact`-backed exact rationals?
- [ ] Scaffold `crates/tpt-math-symbolic/`
- [ ] Wire deps per the decision above (optionally `tpt-math-exact`)
- [ ] Implement CAS core (expression representation, simplification, etc.)
- [ ] Port/consolidate logic from `tpt-rust6`'s `tpt-sym`
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-symbolic` → `"git"`

## Phase 14 — tpt-math-optimize-* (either order relative to each other)

### 14a — tpt-math-optimize-general

*Wrap argmin. General numerical optimization. Depends on: tpt-math-linalg.*

- [ ] Scaffold `crates/tpt-math-optimize-general/`
- [ ] Wire deps: `argmin`, `tpt-math-linalg`
- [ ] Implement general numerical optimization wrapper
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-optimize-general` → `"git"`

### 14b — tpt-math-optimize-convex

*Wrap clarabel. Convex / QP optimization. Depends on: tpt-math-linalg.*

- [ ] Scaffold `crates/tpt-math-optimize-convex/`
- [ ] Wire deps: `clarabel`, `tpt-math-linalg`
- [ ] Implement convex/QP optimization wrapper
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-optimize-convex` → `"git"`

## Phase 15 — tpt-math-optimize (umbrella)

*Re-exports general + convex.*

- [ ] Scaffold `crates/tpt-math-optimize/`
- [ ] Wire optional deps + feature flags for general/convex
- [ ] Re-export each constituent's public API behind its feature
- [ ] Rustdoc documenting the feature matrix
- [ ] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [ ] registry.toml: `tpt-math-optimize` → `"git"`

## Phase 16 — tpt-math-signal-fft

*Wrap rustfft. FFT. Depends on: tpt-math-numeric.*

- [ ] Scaffold `crates/tpt-math-signal-fft/`
- [ ] Wire deps: `rustfft`, `tpt-math-numeric`
- [ ] Implement FFT wrapper
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-signal-fft` → `"git"`

## Phase 17 — tpt-math-signal-filter

*FIR/IIR filters, windowing. Depends on: tpt-math-signal-fft.*

- [ ] Scaffold `crates/tpt-math-signal-filter/`
- [ ] Wire deps: `tpt-math-signal-fft`
- [ ] Implement FIR/IIR filters + windowing
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] registry.toml: `tpt-math-signal-filter` → `"git"`

## Phase 18 — tpt-math-signal (umbrella)

*Re-exports fft + filter.*

- [ ] Scaffold `crates/tpt-math-signal/`
- [ ] Wire optional deps + feature flags for fft/filter
- [ ] Re-export each constituent's public API behind its feature
- [ ] Rustdoc documenting the feature matrix
- [ ] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [ ] registry.toml: `tpt-math-signal` → `"git"`

## Final Phase — Workspace Closeout

*No crates.io publishing in this pass — crates stop at `status = "git"`.*

- [ ] `cargo test --workspace --all-features` passes
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] `cargo deny check` clean workspace-wide
- [ ] no_std matrix passes for all 8 `no_std = true` crates (numeric, units, exact, linalg, prob-core, prob-dist, prob-sampler, autodiff-fwd)
- [ ] Root `README.md` documents the full crate map, build order, and how `tpt-science`/`tpt-engineering`/`tpt-formal` are expected to consume `tpt-math`
- [ ] Confirm every `tpt-math-*` entry in `tpt-rust-map/registry.toml` reads `status = "git"`
- [ ] Update `tpt-rust-map/TODO.md`: mark the `tpt-math` repo-creation line done; note `tpt-science`/`tpt-engineering`/`tpt-formal` can now proceed in parallel
