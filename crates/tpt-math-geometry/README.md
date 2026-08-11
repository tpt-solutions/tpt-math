# tpt-math-geometry

A from-scratch, const-generic **geometry** module built on
[`tpt-math-linalg-fixed`](https://docs.rs/tpt-math-linalg-fixed): points,
translations, rotations, unit quaternions, isometries, similarities, uniform
per-axis scaling, and the perspective / orthographic projection matrices.

It deliberately carries **no `nalgebra` dependency** and **no allocator** —
everything is stack-allocated fixed-size storage — so it can drop into any
`no_std` target. It exists because `nalgebra` (the usual home of this module)
is Apache-2.0-only and disqualified as a wrap target by the workspace license
policy (ADR-0007); `faer` is MIT-only but has no geometry layer, so this crate
fills the gap from scratch.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It sits at the top
of the linear-algebra layer and depends only on `tpt-math-linalg-fixed` and
`tpt-math-numeric`.

## Conventions (stated explicitly)

* **Active (alibi) rotations** — a rotation acts on a vector `v` as `R * v`
  (matrix–vector product, vector on the right).
* **Column vectors** — transformations apply as `M * v`.
* **Right-handed coordinates** — positive 2-D angle is counter-clockwise;
  `x × y = z`.
* **Euler angles** (3-D) use the intrinsic Tait–Bryan order
  `Rz(yaw) · Ry(pitch) · Rx(roll)`.
* **Hamilton quaternions** `q = w + x i + y j + z k` (scalar `w` last).
* **Projection matrices** are right-handed, look down `-z`, NDC depth in
  `[-1, 1]` (OpenGL-style).
* **Isometry composition** `B * A` means "apply `A`, then `B`".

## Quick start

```toml
[dependencies]
tpt-math-geometry = "0.1"
```

```rust
use tpt_math_geometry::{Isometry3, Point3, Rotation3, Translation3, Vector3};
use tpt_math_linalg_fixed::Vector3 as FVector3;

// Rotate 90° about Z, then translate by (1, 2, 3).
let iso = Isometry3::new(
    Translation3::new(Vector3::new([1.0_f64, 2.0, 3.0])),
    Rotation3::from_axis_angle(&FVector3::new([0.0, 0.0, 1.0]), std::f64::consts::FRAC_PI_2),
);
let p = Point3::new(FVector3::new([1.0, 0.0, 0.0]));
let q = iso.transform_point(&p);
```

## Features

- `std` *(default)* — enables `tpt-math-linalg-fixed/std` and
  `tpt-math-numeric/std`.
- `alloc` — allocator signal (this crate is fully allocator-free; enabling it
  only forwards the feature to its dependencies).
- `no_std` support: the crate is `#![no_std]` and needs no allocator, so it
  builds with `--no-default-features` directly.

## License

Licensed under either of MIT or Apache-2.0 at your option.
