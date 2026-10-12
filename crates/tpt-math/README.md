# tpt-math

Facade for the `tpt-math` numeric substrate. Depend on one crate, turn on the
areas you need by short feature name, and use the `prelude` for the most common
names.

```toml
[dependencies]
tpt-math = { version = "0.1", features = ["linalg", "exact"] }
```

| Feature       | Module          | Area                                            |
|---------------|-----------------|-------------------------------------------------|
| `numeric`     | `numeric`       | scalar traits, quadrature, ODE, roots           |
| `units`       | `units`         | compile-time typed units                        |
| `linalg`      | `linalg`        | dense, sparse, complex and fixed linear algebra |
| `exact`       | `exact`         | rationals, intervals, root isolation            |
| `prob`        | `prob`          | distributions, sampling, Bayes, Markov          |
| `stats`       | `stats`         | hypothesis tests and regression                 |
| `autodiff`    | `autodiff`      | forward and reverse mode                        |
| `optimize`    | `optimize`      | general and convex optimisation                 |
| `signal`      | `signal`        | FFT and filters                                 |
| `symbolic`    | `symbolic`      | symbolic expressions                            |
| `geometry`    | `geometry`      | points, rotations, isometries                   |
| `spatial`     | `spatial`       | spatial vectors and screw theory                |
| `graph`       | `graph`         | graphs, shortest paths, max-flow                |
| `interpolate` | `interpolate`   | RBF, kriging, PCHIP, B-splines                  |

`full` enables every area. Nothing is enabled by default.

The facade re-exports the existing crates. Each area is still published and
versioned on its own, so this crate adds no numerical code of its own.
