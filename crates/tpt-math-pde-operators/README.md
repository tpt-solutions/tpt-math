# tpt-math-pde-operators

Discrete differential operators on uniform structured grids: gradient,
divergence, Laplacian and periodic central difference, in 1 to 3 dimensions.
The first building block for a PDE solver in the `tpt-math` substrate.

Scalars sit at cell centres and gradients on faces (a staggered, or MAC,
layout). Under this layout the gradient and divergence are exact negative
adjoints, so the discrete Laplacian inherits the properties of the continuous
one.

## Structural contracts

Operators are checked against the identities they must satisfy, not only
against known solutions:

| Contract | Identity |
|---|---|
| Adjointness | `<grad u, F> = -<u, div F>` |
| Discrete conservation | `sum(vol * div F)` equals the net wall flux |
| Skew-symmetry (periodic axes) | `<Dx, y> = -<x, Dy>` |
| Laplacian symmetry | `<u, Lv> = <Lu, v>` |
| Laplacian sign | `<u, Lu> <= 0` |
| Constant kernel (no Dirichlet walls) | `L(1) = 0` |

`Operators::new` verifies these in debug builds, and in any build with the
`contracts` feature. A broken stencil or boundary treatment therefore fails at
construction. `Operators::verify` returns the individual reports at any time.

## Boundary conditions

- `Periodic`: the axis wraps.
- `Dirichlet0`: the field is zero on the wall; the wall gradient is taken half a
  cell from the boundary cell.
- `Neumann0`: zero flux and zero normal derivative on the wall.

## Example

```rust
use tpt_math_pde_operators::{Boundary, Centering, Field, Grid, Operators};

let grid = Grid::new([64, 64], [1.0 / 64.0; 2], [Boundary::Dirichlet0; 2]).unwrap();
let ops = Operators::new(grid);
let u = Field::from_fn(Centering::Cell, [64, 64], |c| (c[0] * c[1]) as f64);
let lap = ops.laplacian(&u);
```

## Features

- `std` (default): enables `std` support in dependencies.
- `alloc`: enables the allocating operators (required).
- `contracts`: keeps structural contracts active in release builds.

## Scope

Not yet included: curl, higher-order stencils, non-uniform or curvilinear grids,
and any time integration or solver.
