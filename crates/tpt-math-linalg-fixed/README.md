# tpt-math-linalg-fixed

Const-generic, fixed-size (stack-allocated) linear algebra. No allocator.

This crate mirrors the `Vector3`/`Matrix4` layer of `nalgebra` but is
implemented from scratch with const generics, so it carries no Apache-2.0
dependency and needs no allocator (ADR-0007). Storage is plain fixed-size
arrays (`[T; N]` / `[[T; C]; R]`), so every type here is `#![no_std]` and
allocator-free.

## Types

* `Vector<T, N>` — a fixed-size column vector of `N` elements, with aliases
  `Vector2`, `Vector3`, `Vector4`, `Vector6`.
* `Matrix<T, R, C>` — a fixed-size, row-major `R × C` matrix, with aliases
  `Matrix2`, `Matrix3`, `Matrix4`, `Matrix2x3`, `Matrix3x4`, `Matrix4x3`.

All arithmetic is generic over `tpt_math_numeric::Scalar` (i.e. any float
type). Elementwise, scalar, dot/cross and matrix-product ops are implemented.
Square matrices get closed-form `determinant`/`inverse` via Gauss elimination
with partial pivoting (exact for 2×2 / 3×3 / 4×4, and works for any size).

## Features

* `std` (default) — enables `tpt-math-numeric/std`.
* `alloc` — enables `tpt-math-numeric/alloc`.

Neither feature is required: the crate is usable in a plain `#![no_std]`,
allocator-free context with `default-features = false`.

## Example

```rust
use tpt_math_linalg_fixed::{Vector3, Matrix3};

let a = Vector3::new([1.0_f64, 2.0, 3.0]);
let b = Vector3::new([4.0_f64, 5.0, 6.0]);
assert_eq!(a.dot(&b), 32.0);

let m = Matrix3::identity();
assert_eq!(m * a, a);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
