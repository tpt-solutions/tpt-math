# tpt-math-linalg

Dimensionally-checked vectors and matrices. It wraps
[`nalgebra`](https://crates.io/crates/nalgebra) storage and tags it with a
phantom unit type (typically a `uom` quantity from `tpt-math-units`), so that
adding a length-vector to a time-vector is a compile error and matrix
multiplication propagates units: `Mat<U, V> * Mat<V, W> = Mat<U, W>`.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is the linear
algebra half of the exact-and-linear-algebra layer and the bridge between two
crates that otherwise cannot meet: `nalgebra` has vectors and matrices but no
units, `uom` has units but no vectors or matrices. It is not an umbrella crate;
it depends on `nalgebra`, `num-traits` and `tpt-math-units`, and re-exports the
first and last so downstream code usually needs this crate alone.

## Features

- `std` *(default)* — enables `nalgebra/std`, `tpt-math-units/std` and
  `num-traits/std`.
- `alloc` — enables `nalgebra/alloc` and `tpt-math-units/alloc`.
- `no_std` support: the crate itself is `#![no_std]`. Because `Vec` and `Mat`
  are backed by `nalgebra::DVector`/`DMatrix` (heap-allocated, dynamically
  sized), a `no_std` build needs `default-features = false, features =
  ["alloc"]` and an allocator.

## Quick start

```toml
[dependencies]
tpt-math-linalg = "0.1"
```

```rust
use tpt_math_linalg::nalgebra::{DMatrix, DVector};
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
`Vec / scalar`, indexing, `len`/`is_empty`, `Vec::zeros`; `Mat + Mat`,
`Mat - Mat`, `Mat::negate`, `Mat * Mat`, `Mat * Vec`, `Mat::transpose` (which
swaps the row/column tags), `nrows`/`ncols`, `Mat::zeros`. `from_raw` wraps a
raw nalgebra value and `raw()` borrows it back, so the full upstream API is
always one call away.

## Notes

- The unit parameter is a `PhantomData` tag: it costs nothing at runtime, and
  the elements stay plain scalars (`f64` by default). No unit conversion is
  performed — a `Vec<Length>` is "these numbers are lengths in whichever unit
  you chose", not a vector of `uom` quantities.
- Only *units* are checked at compile time. Shape mismatches are still runtime
  panics coming from nalgebra, because `DVector`/`DMatrix` are dynamically
  sized.
- `tpt_math_linalg::Vec` shadows the standard `Vec` if imported unqualified;
  import it as an alias (`Vec as UnitVec`) in code that also uses
  `alloc::vec::Vec`.
- `Mat` uses an inherent `negate()` method rather than a `Neg` impl, and scalar
  `Mul`/`Div` are implemented for `Vec` only.
- **Backend decision:** this crate wraps `nalgebra` only. A dual
  `nalgebra` + `faer` facade was explicitly rejected — the two share no trait
  surface, and rebuilding dense-linalg kernels is wasted effort. Code with
  proven extreme dense-matrix performance needs should depend on `faer`
  directly.
- This crate is `MIT OR Apache-2.0`, but note that `nalgebra` itself is
  **Apache-2.0 only** (`uom` is `MIT OR Apache-2.0`). That is permitted by the
  workspace policy — an Apache-2.0 dependency of a dual-licensed crate — but
  downstream users who pick the MIT arm still have to honour Apache-2.0 for
  `nalgebra`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
