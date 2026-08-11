# tpt-math-linalg-dense

Dense linear algebra over [`faer`](https://crates.io/crates/faer) (MIT-only), the
storage backend for [`tpt-math-linalg`](../tpt-math-linalg) and
[`tpt-math-optimize`](../tpt-math-optimize).

`faer` is chosen deliberately: `nalgebra` and `clarabel` are Apache-2.0-only and
are disqualified as wrap targets under this workspace's no-exceptions license
policy (ADR-0007). `faer` is MIT-only, so it is the dense-linalg backend.

## Types

* `DVector<T>` — a dynamically-sized column vector (wraps `faer::Col`).
* `DMatrix<T>` — a dynamically-sized, column-major matrix (wraps `faer::Mat`).

## Features

* `std` (default) — faer's `std` support plus the allocator.
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
