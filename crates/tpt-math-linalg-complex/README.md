# tpt-math-linalg-complex

Complex-valued linear algebra for the `tpt-math` substrate: complex matrices and
vectors with LU / Cholesky decompositions and a shifted-QR eigenvalue solver,
implemented entirely in-house (no `nalgebra` / `faer` dependency, no license
exposure). Built on `tpt-math-linalg-dense`'s storage pattern.

## Scope

* `Complex<T>` — a complex scalar with the usual arithmetic, conjugate, modulus
  and complex square root.
* `ComplexDVector<T>` / `ComplexDMatrix<T>` — dynamically-sized complex vectors
  and matrices, stored column-major (same layout as `tpt-math-linalg-dense`).
* `ComplexDMatrix::lu` / `solve` / `inverse` — partial-pivot complex LU.
* `ComplexDMatrix::cholesky` — Cholesky factorisation of a Hermitian
  positive-definite matrix, with `Cholesky::solve`.
* `ComplexDMatrix::eigenvalues` — shifted-QR eigenvalue solver for general
  complex matrices.

## Usage

```rust
use tpt_math_linalg_complex::{Complex, ComplexDMatrix};

let a = ComplexDMatrix::from_real_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
let inv = a.inverse().unwrap();
let z = Complex::new(0.0, 1.0);
assert_eq!(z * z, Complex::new(-1.0, 0.0));
```

## Features

* `std` (default) — enable the allocator and `std` support of dependencies.
* `alloc` — signal allocator availability (dynamic vectors need it).

## License

Dual-licensed under either of `MIT` or `Apache-2.0` at your option.
