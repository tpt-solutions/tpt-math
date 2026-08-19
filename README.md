# tpt-math

The numeric substrate for the TPT Solutions science / engineering / formal
verification stack. `tpt-math` is a workspace of small, dependency-light,
dual-licensed (`MIT OR Apache-2.0`) crates that wrap the best-of-breed pure-Rust
math ecosystem (and, where no dual-licensed option exists, consolidate prior
TPT crates) behind a coherent, `no_std`-friendly API.

It exists so that `tpt-science`, `tpt-engineering`, and `tpt-formal` can depend
on one consistent set of math primitives instead of each re-wrapping the Rust
math ecosystem ad hoc.

**License posture.** Per ADR-0007 every external dependency must be
dual-licensed (`MIT OR Apache-2.0`) or more permissive. The crates that were
Apache-2.0-*only* — `nalgebra`, `clarabel`, `statrs`, `argmin`, and `faer` —
have been replaced with in-house implementations (see the crate map), so the
workspace carries **no Apache-2.0-only dependency**. The remaining wrapped
ecosystem crates (`uom`, `rustfft`, `rand`/`rand_distr`, `num-*`) are all
dual-licensed.

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
| `tpt-math-linalg-dense` | — (in-house; was `faer`) | yes (alloc) | Dense `DVector`/`DMatrix`, column-major `Vec` storage; storage backend for `tpt-math-linalg` / `-optimize`. |
| `tpt-math-linalg` | `tpt-math-linalg-dense` (in-house) | yes | Dimensionally-checked vectors/matrices over the in-house dense backend. |
| `tpt-math-linalg-fixed` | — (in-house) | yes | Const-generic fixed-size vectors/matrices + closed-form det/inverse; no allocator. |
| `tpt-math-linalg-complex` | — (in-house) | yes (alloc) | Complex-valued matrices, complex LU/Cholesky, QR eigenvalue solver for EM/quantum. |
| `tpt-math-linalg-sparse` | — (in-house) | yes (alloc) | Sparse COO/CSR/CSC + iterative CG/BiCGSTAB solvers; hand-rolled, complements `tpt-fem-sparse`. |
| `tpt-math-geometry` | `tpt-math-linalg-fixed` (in-house) | yes | Points, rotations, quaternions, isometries, projections; no `nalgebra`. |

### Geometry & spatial
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-geometry` | `tpt-math-linalg-fixed` (in-house) | yes | Points, rotations, quaternions, isometries, projections; no `nalgebra`. |
| `tpt-math-spatial` | — (in-house) | yes | Featherstone 6-D spatial vectors, dual quaternions, screw theory; built on `tpt-math-geometry` / `-linalg-fixed`. |

### Graph & interpolation
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-graph` | `petgraph` (dual `Apache-2.0 OR MIT`) | yes (alloc) | Thin wrap for adjacency / toposort / Dijkstra / A*; in-house max-flow on top. |
| `tpt-math-interpolate` | — (in-house) | yes (alloc) | RBF, ordinary Kriging, PCHIP, B-spline basis for scattered data / surrogates. |

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
| `tpt-math-stats` | in-house (`special`/`dist`); no `statrs` | no | Hypothesis tests / regression + in-house distributions & special functions. |
| `tpt-math-autodiff-fwd` | — | yes | Dual-number forward-mode autodiff. |
| `tpt-math-autodiff-rev` | — (in-house; was `tpt-grad`) | no | Reverse-mode / tape autodiff, built on `tpt-math-autodiff-fwd`. |
| `tpt-math-autodiff` | (umbrella) | no | Re-exports fwd + rev. |
| `tpt-math-symbolic` | `tpt-sym` | no | Permissive-license CAS; generic `Coefficient`, default `f64`, optional exact `BigRational`. |

### Optimisation & signal
| Crate | Wraps / consolidates | `no_std` | Notes |
|-------|----------------------|----------|-------|
| `tpt-math-optimize-general` | — (in-house; was `argmin`) | no | Steepest descent, nonlinear conjugate gradient, Newton. |
| `tpt-math-optimize-convex` | — (in-house; was `clarabel`) | no | Dense primal-dual interior-point (Mehrotra) QP solver. |
| `tpt-math-optimize` | (umbrella) | no | Re-exports general + convex. |
| `tpt-math-signal-fft` | `rustfft` | no | FFT. |
| `tpt-math-signal-filter` | — | no | FIR/IIR filters, windowing. |
| `tpt-math-signal` | (umbrella) | no | Re-exports fft + filter. |

## Build order

```
tpt-math-numeric
  -> tpt-math-linalg-dense          (in-house dense storage; was faer)
  -> tpt-math-linalg-fixed          (in-house fixed-size)
       -> tpt-math-geometry
  -> tpt-math-units
       -> tpt-math-linalg           (wraps linalg-dense)
       -> tpt-math-units-dyn
       -> tpt-math-exact
            -> tpt-math-prob-core
                 -> tpt-math-prob-dist / -bayes / -markov / -monte-carlo / -sampler
                 -> tpt-math-prob            (umbrella)
            -> tpt-math-stats               (in-house; no statrs)
       -> tpt-math-linalg-sparse    (in-house; needs linalg-dense)
       -> tpt-math-linalg-complex   (in-house; needs linalg-dense)
  -> tpt-math-graph                 (wraps petgraph; needs numeric)
  -> tpt-math-interpolate           (in-house; needs linalg-dense)
  -> tpt-math-autodiff-fwd
       -> tpt-math-autodiff-rev
            -> tpt-math-autodiff            (umbrella)
  -> tpt-math-symbolic
  -> tpt-math-optimize-general      (in-house; was argmin)
       -> tpt-math-optimize-convex   (in-house QP; needs linalg-dense)
            -> tpt-math-optimize      (umbrella)
  -> tpt-math-signal-fft
       -> tpt-math-signal-filter
            -> tpt-math-signal        (umbrella)
```

## Consuming `tpt-math`

Downstream repos depend only on the leaf / umbrella crates they need:

- **`tpt-science`** — reach for `tpt-math-prob` + `tpt-math-stats`
  (inference, Monte Carlo, hypothesis tests), `tpt-math-autodiff`
  (gradient-based solvers), `tpt-math-optimize` (parameter fitting), and
  `tpt-math-signal` (time-series / spectral analysis).
- **`tpt-engineering`** — reach for `tpt-math-linalg` + `tpt-math-units`
  (physically-dimensioned models), `tpt-math-optimize-convex` (QP/control
  problems), `tpt-math-prob` (uncertainty propagation), `tpt-math-spatial`
  (rigid-body / kinematics), `tpt-math-linalg-sparse` (FEM / large systems).
- **`tpt-formal`** — reach for `tpt-math-exact` (provably-exact arithmetic
  under verification), `tpt-math-numeric` / `tpt-math-units` (trusted
  scalar/unit primitives).
- **`tpt-fem`** — reach for `tpt-math-linalg` / `-linalg-dense` /
  `-linalg-sparse` / `-linalg-complex` and `tpt-math-optimize` for assembly
  and sparse solves.
- **`tpt-physics`** — reach for `tpt-math-linalg`, `tpt-math-graph`,
  `tpt-math-spatial`, and `tpt-math-interpolate` (surrogate modelling).

## Repository layout

- `crates/*` — the 31 library workspace members.
- `xtask/` — the developer-tooling crate (`cargo xtask …`).
- `examples/` — `tpt-math-examples`, four unpublished cross-crate demos.
- `Cargo.toml` — workspace manifest (`resolver = "2"`, shared `[workspace.package]`).
- `deny.toml` — license / advisory / source hygiene (dual-license policy, ADR 0007).
- `spec.txt` — the original build spec.
- `todo.md` — the per-phase build checklist.

## Depending on `tpt-math`

Every crate in `crates/*` is published to crates.io — depend on them the
usual way:

```toml
[dependencies]
tpt-math-linalg = "0.1"
tpt-math-prob    = "0.1"
```

To track `main` ahead of a release instead, depend on them directly from git:

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
