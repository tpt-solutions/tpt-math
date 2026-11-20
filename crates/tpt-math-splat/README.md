# tpt-math-splat

`no_std`, allocator-free math primitives for 3D and 4D Gaussian splatting.

| Module | Contents |
|---|---|
| `sym` | Packed `Sym3` / `Sym4`; Cholesky, LDLᵀ, SPD inverse/determinant (all return `None` for non-positive-definite input); Jacobi eigensolver for 3×3. |
| `gaussian` | `Σ = R S Sᵀ Rᵀ` in 3D and 4D (4D rotation from two quaternions), conditioning a 4D Gaussian on time (`condition_on_time`), EWA projection with conic and screen radius. |
| `sh` | Real spherical harmonics, degrees 0–3, 3DGS sign convention; colour evaluation with the `+0.5` offset and clamp. |
| `sfc` | Morton keys (2D/3D/4D), Hilbert keys in any dimension (Skilling), quantisation, allocation-free Hilbert sort. |

## Correctness

Floating-point results are not claimed bit-exact. What is guaranteed and tested:
covariances are positive semi-definite *by construction*; factorisations refuse
bad input instead of returning garbage; the Morton/Hilbert code is pure integer
arithmetic with bijection and unit-step adjacency verified exhaustively on small
grids; 4D conditioning is cross-checked against an independent precision-matrix
formulation; the SH basis is checked for orthonormality by quadrature.

Per-frame depth sorting for alpha blending is view-dependent and intentionally
out of scope; Morton/Hilbert give *spatial* coherence for caching and
compression.

## Example

```rust
use tpt_math_splat::gaussian::{condition_on_time, cov4_from_rot_scale};

let id = [1.0_f64, 0.0, 0.0, 0.0];
let cov = cov4_from_rot_scale(id, id, [1.0, 1.0, 1.0, 0.5]).unwrap();
let g = condition_on_time([0.0; 4], &cov, 0.0).unwrap();
assert!((g.weight - 1.0).abs() < 1e-12);
```

Licensed under MIT OR Apache-2.0.
