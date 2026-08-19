# tpt-math-interpolate

In-house interpolation & approximation for the `tpt-math` substrate, built on
[`tpt-math-linalg-dense`](../tpt-math-linalg-dense) storage and the
[`tpt-math-numeric`](../tpt-math-numeric) `Scalar` trait. **No external
interpolation backend** — everything is implemented from scratch, so there is
no `scipy`/`scirs2` dependency and therefore no license exposure.

## Why in-house? (ADR-0007)

The obvious off-the-shelf choice, `scirs2-interpolate`, is **Apache-2.0-only**.
This substrate is dual-licensed MIT/Apache-2.0 and its policy (ADR-0007)
disqualifies copyleft-leaning / single-license dependencies that do not permit
the MIT option. Rather than vendor or wrap a disqualified crate, the four
routines below are re-implemented in-repo against the existing dense linear
algebra backend.

## Implemented methods

* **RBF interpolation** — `RbfInterpolator` with thin-plate (`r²·ln r`),
  Gaussian (`exp(-(εr)²)`) and multiquadric (`sqrt(1+(εr)²)`) kernels. Weights
  are the solution of the dense kernel system via `DMatrix::solve`.
* **Ordinary Kriging** — `Kriging` with a configurable variogram model
  (spherical, exponential, Gaussian or linear). Returns the prediction and the
  Kriging variance; exact at sample nodes when the nugget is zero.
* **PCHIP** — `Pchip`, shape-preserving piecewise-cubic-Hermite interpolation
  using the Fritsch–Carlson derivative-limiting rule; preserves monotonicity of
  the data.
* **B-spline basis** — `bspline_basis` (Cox–de Boor recursion) and
  `BsplineCurve`, a weighted sum of basis functions. Basis functions satisfy
  the partition-of-unity property.

## Features

* `std` (default) — the allocator plus `std` support of deps.
* `alloc` — signal allocator availability (dynamic vectors need it).

## Example

```rust
use tpt_math_interpolate::{RbfInterpolator, RbfKernel};
use tpt_math_linalg_dense::DVector;

let xs = DVector::from_vec(vec![0.0_f64, 1.0, 2.0]);
let ys = DVector::from_vec(vec![0.0_f64, 1.0, 4.0]);
let rbf = RbfInterpolator::new(xs, ys, RbfKernel::ThinPlate { epsilon: 1.0 }).unwrap();
// RBF interpolation is exact at the sample nodes.
assert!((rbf.eval(1.0) - 1.0).abs() < 1e-6);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
