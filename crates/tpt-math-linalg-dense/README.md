# tpt-math-linalg-dense

Dense linear algebra implemented **entirely in-house** (no external backend), the
storage backend for [`tpt-math-linalg`](../tpt-math-linalg) and
[`tpt-math-optimize`](../tpt-math-optimize).

The `DVector`/`DMatrix` types are stored column-major in a plain `Vec<T>`, so
there is no `faer`/`nalgebra`/`clarabel` dependency and therefore no license
exposure. The arithmetic, norms, and the partial-pivot-LU `solve`/`inverse` are
all hand-rolled. (The crate was originally prototyped over `faer`, but the final
implementation dropped that dependency for a zero-license-risk, in-repo design.)

## Types

* `DVector<T>` — a dynamically-sized column vector.
* `DMatrix<T>` — a dynamically-sized, column-major matrix.

## Features

* `std` (default) — the allocator plus `std` support of deps.
* `alloc` — signal allocator availability (dynamic vectors need it).
* `argmin` — `ArgminMath`-family trait impls for `DVector<f64>` / `DMatrix<f64>`,
  so they can drive `argmin` solvers without the `nalgebra` backend.

## Example

```rust
use tpt_math_linalg_dense::{DMatrix, DVector};

let m = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
let v = DVector::from_vec(vec![1.0_f64, 1.0]);
let mv = m * v;
assert_eq!(mv, DVector::from_vec(vec![3.0, 7.0]));

let inv = m.inverse().unwrap();
let got = m * inv;
assert!((got[(0, 0)] - 1.0).abs() < 1e-12);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
