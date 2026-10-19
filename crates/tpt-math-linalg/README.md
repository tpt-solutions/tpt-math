# tpt-math-linalg

Dimensionally-checked vectors and matrices. It wraps
[`tpt-math-linalg-dense`](https://crates.io/crates/tpt-math-linalg-dense)
storage and tags it with a phantom unit type (typically a `uom` quantity from
`tpt-math-units`), so that adding a length-vector to a time-vector is a compile
error and matrix multiplication propagates units:
`Mat<U, V> * Mat<V, W> = Mat<U, W>`.

`tpt-math-linalg-dense` is implemented in-house (column-major `Vec<T>` storage)
with no external backend, so there is no `faer`/`nalgebra` license exposure.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is the linear
algebra half of the exact-and-linear-algebra layer and the bridge between two
crates that otherwise cannot meet: the dense backend has vectors and matrices
but no units, `uom` has units but no vectors or matrices. It is not an umbrella
crate; it depends on `tpt-math-linalg-dense`, `tpt-math-units` and
`tpt-math-numeric`, and re-exports the first and last so downstream code usually
needs this crate alone.

## Features

- `std` *(default)* — enables `tpt-math-linalg-dense/std`,
  `tpt-math-units/std` and `tpt-math-numeric/std`.
- `alloc` — enables `tpt-math-linalg-dense/alloc` and
  `tpt-math-units/alloc`.
- `no_std` support: the crate itself is `#![no_std]`. Because `Vec` and `Mat`
  are backed by `tpt-math-linalg-dense`'s `DVector`/`DMatrix` (heap-allocated,
  dynamically sized), a `no_std` build needs
  `default-features = false, features = ["alloc"]` and an allocator.

## Quick start

```toml
[dependencies]
tpt-math-linalg = "0.2"
```

```rust
use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_linalg::tpt_math_units::prelude::{Length, Time};
use tpt_math_linalg::{Mat, Vec as UnitVec};

// Two position vectors, both tagged `Length`.
let a = UnitVec::<Length>::from_raw(DVector::from_row_slice(&[1.0, 2.0]));
let b = UnitVec::<Length>::from_raw(DVector::from_row_slice(&[3.0, 4.0]));
let sum = a + b;
assert_eq!(sum.raw(), &DVector::from_row_slice(&[4.0, 6.0]));

// Mat<U, V> * Vec<V> = Vec<U>: the column tag has to match the vector's.
let m = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]));
let t = UnitVec::<Time>::from_raw(DVector::from_row_slice(&[1.0, 2.0]));
let scaled: UnitVec<Length> = m * t;
assert_eq!(scaled[0], 2.0);
assert_eq!(scaled.len(), 2);

// `let bad = sum + t;` would not compile: Length vs Time.
```

Available operations: `Vec + Vec`, `Vec - Vec`, `-Vec`, `Vec * scalar`,
`Vec / scalar`, indexing, `len`/`is_empty`, `Vec::zeros`, `Vec::norm` and
`Vec::dot`; `Mat + Mat`, `Mat - Mat`, `Mat::negate`, `Mat * Mat`, `Mat * Vec`,
`Mat * scalar`, `Mat / scalar`, `Mat::transpose` (which swaps the row/column
tags), `nrows`/`ncols`, `Mat::zeros`, and `Mat::solve` and `Mat::inverse` for
`f64`. `+`, `-` and `*` also work on references (`&Vec`, `&Mat`), at the cost
of cloning the operands. `from_raw` wraps a raw dense value and `raw()` borrows
it back, so the full upstream API is always one call away.

`Vec::norm` and `Vec::dot` return a bare scalar, because unit algebra for
`U * U` is not modelled.

## Notes

- The unit parameter is a `PhantomData` tag: it costs nothing at runtime, and
  the elements stay plain scalars (`f64` by default). No unit conversion is
  performed — a `Vec<Length>` is "these numbers are lengths in whichever unit
  you chose", not a vector of `uom` quantities.
- Only *units* are checked at compile time. Shape mismatches are still runtime
  panics coming from the dense backend, because `DVector`/`DMatrix` are
  dynamically sized.
- `tpt_math_linalg::Vec` shadows the standard `Vec` if imported unqualified;
  import it as an alias (`Vec as UnitVec`) in code that also uses
  `alloc::vec::Vec`.
- `Mat` uses an inherent `negate()` method rather than a `Neg` impl.
- **Backend decision:** this crate wraps `tpt-math-linalg-dense` only, never
  `nalgebra`. `nalgebra` is Apache-2.0-only and is disqualified as a wrap target
  by the workspace license policy (ADR-0007); `tpt-math-linalg-dense` is
  implemented in-house without any such dependency, so there is no license
  exposure.

## License

Licensed under either of MIT or Apache-2.0 at your option.
