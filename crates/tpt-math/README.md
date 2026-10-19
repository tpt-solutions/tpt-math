# tpt-math

Facade for the `tpt-math` numeric substrate. Depend on one crate, turn on the
areas you need by short feature name, and use the `prelude` for the most common
names.

```toml
[dependencies]
tpt-math = { git = "https://github.com/tpt-solutions/tpt-math", package = "tpt-math", features = ["linalg", "exact"] }
```

| Feature        | Module          | Area                                                  |
|----------------|-----------------|-------------------------------------------------------|
| `numeric`      | `numeric`       | scalar traits, quadrature, ODE, roots                 |
| `units`        | `units`         | compile-time typed units                              |
| `units-dyn`    | `units_dyn`     | runtime dimension-checked units                       |
| `linalg`       | `linalg`        | dimensionally-checked dense vectors and matrices      |
| `linalg-sparse`| `linalg_sparse` | sparse storage, iterative solvers, factorisations     |
| `linalg-complex`| `linalg_complex` | complex dense matrices, LU, Cholesky, eigenvalues    |
| `linalg-fixed` | `linalg_fixed`  | const-generic fixed-size vectors and matrices         |
| `exact`        | `exact`         | rationals, intervals, rigorous root and ODE enclosures |
| `prob`         | `prob`          | distributions, sampling, Bayes, Markov                |
| `stats`        | `stats`         | hypothesis tests, correlation and regression          |
| `autodiff`     | `autodiff`      | forward and reverse mode                              |
| `optimize`     | `optimize`      | general and convex optimisation                       |
| `signal`       | `signal`        | FFT, spectra and filters                              |
| `symbolic`     | `symbolic`      | symbolic expressions, parsing and compilation         |
| `geometry`     | `geometry`      | points, rotations, isometries                         |
| `spatial`      | `spatial`       | spatial vectors and screw theory                      |
| `graph`        | `graph`         | graphs, shortest paths, max-flow                      |
| `interpolate`  | `interpolate`   | RBF, kriging, PCHIP, splines, multilinear grids       |
| `pde-operators`| `pde_operators` | gradient, divergence and Laplacian on structured grids |

`full` enables every area. Nothing is enabled by default. The `std` feature
passes `std` through to each enabled area that has one.

The `prelude` brings in the most common names from the enabled areas: the scalar
traits from `numeric`, `DMatrix`, `DVector`, `Lu` and `Qr`, `Interval`,
`Rational`, `Expr64`, `Symbolic`, `Fft`, `Dual`, `GradientTape` and
`DynQuantity`.

The facade re-exports the existing crates. Each area is still released and
versioned on its own, so this crate adds no numerical code of its own.
