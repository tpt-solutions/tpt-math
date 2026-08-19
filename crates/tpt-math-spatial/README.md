# tpt-math-spatial

In-house (no external-crate) spatial kinematics for the `tpt-math` substrate:
Featherstone 6-D spatial vectors, Plücker / adjoint transforms, dual quaternions
and screw theory (exponential / logarithmic maps).

Everything is built on [`tpt-math-geometry`](../tpt-math-geometry)
(`Isometry3`, `Quaternion`, `UnitQuaternion`, …) and
[`tpt-math-linalg-fixed`](../tpt-math-linalg-fixed). It is `#![no_std]` and
allocator-free (stack-allocated fixed-size storage), and carries **no `nalgebra`
dependency**, matching the workspace license policy (ADR-0007).

## Why in-house? (ADR-0007)

The obvious off-the-shelf choice, `spatial-math`, ships under a non-MIT/Apache
"Custom license" and does not cover dual quaternions / screw theory anyway.
This crate re-implements the combined scope in-repo against the existing
fixed-size linear-algebra and geometry backends, so there is no license
exposure.

## Implemented primitives

* **Spatial (6-D) vectors** — `MotionVector` / `ForceVector` (the
  `SpatialVector<K>` family) with Featherstone's `crm` / `crf` cross products,
  tagged by a kind marker so the result kind is type-checked at compile time.
* **Adjoint (Plücker) transforms** — `adjoint_motion` / `adjoint_force` over an
  `Isometry3`, plus `transform_by` on the spatial vectors.
* **Dual quaternions** — `DualQuaternion` with Hamilton product, conjugate,
  normalisation, and rigid-transform conversion (`from_isometry` /
  `to_isometry`).
* **Screw theory** — `Screw` (a twist in `se(3)`) with the exponential/logarithm
  maps `exp` / `log` to and from `Isometry3`.

## Conventions (stated explicitly)

* **Active (alibi) rotations** — a transform acts on a vector `v` as `R * v`;
  the point/vector is rotated, not the coordinate frame.
* **Column vectors** — transformations apply as `M * v`.
* **Right-handed coordinates** — positive angle about `+z` rotates `x → y`;
  `x × y = z`.
* **Spatial (6-D) vectors** are stored `[angular (top 3); linear (bottom 3)]`.
* **Screw exponential** follows the standard Lynch–Park `se(3)` exponential with
  an active, right-handed rotation.

## Example

```rust
use tpt_math_spatial::{MotionVector, Screw};
use tpt_math_geometry::{Isometry3, Rotation3, Translation};
use tpt_math_linalg_fixed::Vector3;

// A 90° active rotation about +Z.
let rot = Rotation3::from_axis_angle(
    &Vector3::new([0.0_f64, 0.0, 1.0]),
    std::f64::consts::FRAC_PI_2,
);
let iso = Isometry3::new(Translation::new(Vector3::new([0.0, 0.0, 0.0])), rot);

// A pure-linear motion (1,0,0): after the rotation its linear part is (0,1,0).
let m = MotionVector::new(
    Vector3::new([0.0, 0.0, 0.0]),
    Vector3::new([1.0, 0.0, 0.0]),
);
let m2 = m.transform_by(&iso);
assert!((m2.linear().x()).abs() < 1e-12);
assert!((m2.linear().y() - 1.0).abs() < 1e-12);

// A screw twist exponentiates to the same isometry.
let screw = Screw::new(
    Vector3::new([0.0, 0.0, std::f64::consts::FRAC_PI_2]),
    Vector3::new([0.0, 0.0, 0.0]),
);
let iso2 = screw.exp();
assert!((iso2.rotation.matrix().data[0][1] - (-1.0)).abs() < 1e-12);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
