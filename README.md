# tpt-math

The numeric substrate for the TPT Solutions science / engineering / formal
verification stack. `tpt-math` is a workspace of small, dependency-light,
dual-licensed (`MIT OR Apache-2.0`) crates that wrap the best-of-breed pure-Rust
math ecosystem (and, where no dual-licensed option exists, consolidate prior
TPT crates) behind a coherent, `no_std`-friendly API.

It exists so that `tpt-science`, `tpt-engineering`, and `tpt-formal` can depend
on one consistent set of math primitives instead of each re-wrapping
`nalgebra` / `statrs` / `rand_distr` / etc. ad hoc.

## Crate map

Crates are organised into layers; lower layers never depend on higher ones.

### Scalar & units
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-numeric` | `num-traits`, `libm` | yes | Scalar numeric trait glue. |
| `tpt-math-units` | `uom` (std disabled) | yes | Compile-time typed units. |
| `tpt-math-units-dyn` | `tpt-units-runtime` | no | Runtime dimension-checked units. |

### Exact & linear algebra
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-exact` | `num-bigint`, `num-rational` | yes (alloc) | Exact rational + interval arithmetic. |
| `tpt-math-linalg` | `nalgebra` | yes | Dimensionally-checked vectors/matrices. nalgebra-only (no faer facade). |

### Probability
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-prob-core` | — | yes | Shared `Distribution`/`Sampler` traits. |
| `tpt-math-prob-dist` | `rand_distr` | yes | Standard distributions. |
| `tpt-math-prob-bayes` | `tpt-zero-bayes` | no | Bayesian inference primitives. |
| `tpt-math-prob-markov` | `tpt-zero-markov` | no | Markov chains / HMM. |
| `tpt-math-prob-monte-carlo` | `tpt-zero-monte-carlo` | no | Monte Carlo methods. |
| `tpt-math-prob-sampler` | `tpt-zero-sampler`, `tpt-zero-rand` | yes | Sampling strategies. |
| `tpt-math-prob` | (umbrella) | no | Re-exports the five `prob-*` crates behind features. |

### Statistics, autodiff, symbolic
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-stats` | in-house (`special`/`dist`) | no | Hypothesis tests / regression + in-house distributions & special functions. |
| `tpt-math-autodiff-fwd` | — | yes | Dual-number forward-mode autodiff. |
| `tpt-math-autodiff-rev` | `tpt-grad`/`tpt-grad-macro`/`tpt-zero-grad` | no | Reverse-mode / tape autodiff. |
| `tpt-math-autodiff` | (umbrella) | no | Re-exports fwd + rev. |
| `tpt-math-symbolic` | `tpt-sym` | no | Permissive-license CAS; generic `Coefficient`, default `f64`, optional exact `BigRational`. |

### Optimisation & signal
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-optimize-general` | `argmin` | no | General numerical optimisation. |
| `tpt-math-optimize-convex` | `clarabel` | no | Convex / QP optimisation. |
| `tpt-math-optimize` | (umbrella) | no | Re-exports general + convex. |
| `tpt-math-signal-fft` | `rustfft` | no | FFT. |
| `tpt-math-signal-filter` | — | no | FIR/IIR filters, windowing. |
| `tpt-math-signal` | (umbrella) | no | Re-exports fft + filter. |

## Build order

```
tpt-math-numeric
  -> tpt-math-units
       -> tpt-math-exact
            -> tpt-math-linalg
                 -> tpt-math-prob-core
                      -> tpt-math-prob-dist / -bayes / -markov / -monte-carlo / -sampler
                      -> tpt-math-prob            (umbrella)
                 -> tpt-math-stats
       -> tpt-math-units-dyn
  -> tpt-math-autodiff-fwd
       -> tpt-math-autodiff-rev
            -> tpt-math-autodiff (umbrella)
  -> tpt-math-symbolic
  -> tpt-math-optimize-general / -convex
       -> tpt-math-optimize      (umbrella)
  -> tpt-math-signal-fft
       -> tpt-math-signal-filter
            -> tpt-math-signal    (umbrella)
```

## Consuming `tpt-math`

Downstream repos depend only on the leaf / umbrella crates they need:

- **`tpt-science`** — reach for `tpt-math-prob` + `tpt-math-stats`
  (inference, Monte Carlo, hypothesis tests), `tpt-math-autodiff`
  (gradient-based solvers), `tpt-math-optimize` (parameter fitting), and
  `tpt-math-signal` (time-series / spectral analysis).
- **`tpt-engineering`** — reach for `tpt-math-linalg` + `tpt-math-units`
  (physically-dimensioned models), `tpt-math-optimize-convex` (QP/control
  problems), `tpt-math-prob` (uncertainty propagation).
- **`tpt-formal`** — reach for `tpt-math-exact` (provably-exact arithmetic
  under verification), `tpt-math-numeric` / `tpt-math-units` (trusted
  scalar/unit primitives).

## Repository layout

- `crates/*` — the 23 library workspace members.
- `xtask/` — the developer-tooling crate (`cargo xtask …`).
- `examples/` — `tpt-math-examples`, four unpublished cross-crate demos.
- `Cargo.toml` — workspace manifest (`resolver = "2"`, shared `[workspace.package]`).
- `deny.toml` — license / advisory / source hygiene (dual-license policy, ADR 0007).
- `spec.txt` — the original build spec.
- `todo.md` — the per-phase build checklist.

## Depending on `tpt-math` before crates.io publication

This repository is **not published to crates.io** in this build pass — every
crate stops at `status = "git"` in `../tpt-rust-map/registry.toml`. Until the
crates are published you can depend on them directly from git:

```toml
[dependencies]
tpt-math-linalg = { git = "https://github.com/tpt-solutions/tpt-math", package = "tpt-math-linalg" }
tpt-math-prob    = { git = "https://github.com/tpt-solutions/tpt-math", package = "tpt-math-prob" }
```

Pin to a commit or tag for reproducibility. The workspace also ships local
tooling so you can sanity-check an integration before publishing:

- **Examples** — `examples/` (`tpt-math-examples`, unpublished) holds four
  runnable cross-crate programs (`units+linalg`, `prob+stats`,
  `autodiff+optimize`, `symbolic+exact`). Run them all with `just examples`
  or `cargo run -p tpt-math-examples --bin <name>`.
- **Verification** — `cargo xtask check` (or `just check`) runs
  `cargo fmt --check`, clippy with `-D warnings`, and `cargo-deny`. The
  `no_std` crates are built for `thumbv6m-none-eabi` via `cargo xtask no-std`.
- **Policy docs** — `SECURITY.md` (no-`unsafe` policy, `deny.toml` posture,
  panic/`try_*` convention, symbolic recursion caveat) and `CONTRIBUTING.md`
  (issues-only workflow, per-crate checklist, `deny.toml` license policy).

## License

Every crate is `MIT OR Apache-2.0`. Author: TPT Solutions.
