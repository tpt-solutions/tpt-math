#![no_std]
#![forbid(unsafe_code)]
// Stencil loops read most clearly with explicit indexing.
#![allow(clippy::needless_range_loop)]
//! Discrete differential operators on structured grids.
//!
//! This crate is the mathematical groundwork for a future PDE solver. It
//! provides gradient, divergence, Laplacian and central-difference operators
//! on uniform structured grids in 1 to 3 dimensions, built on the
//! [`tpt_math_numeric`] `Scalar` trait with **no external backend**.
//!
//! Beyond returning numbers, every [`Operators`] value is checked against the
//! identities a discrete operator must satisfy:
//!
//! * **Adjointness** — `<grad u, F> = -<u, div F>` with the grid's quadrature
//!   weights, so the gradient and divergence are exact negative adjoints.
//! * **Discrete conservation** — `sum(vol * div F)` equals the net boundary
//!   flux, so the divergence telescopes.
//! * **Skew-symmetry** — the periodic central difference satisfies
//!   `<Dx, y> = -<x, Dy>`.
//! * **Laplacian sign and symmetry** — `<u, Lv> = <Lu, v>` and
//!   `<u, Lu> <= 0`; constants lie in the kernel when no wall is Dirichlet.
//!
//! [`Operators::new`] runs these checks in debug builds (and whenever the
//! `contracts` feature is enabled), so a broken stencil or boundary treatment
//! fails at construction rather than silently corrupting a solver. Use
//! [`Operators::verify`] to inspect the individual reports at any time.
//!
//! # Layout
//!
//! Scalars live at cell centres. Gradients live on faces, one field per axis
//! (a staggered, or MAC, layout). Walls with [`Boundary::Dirichlet0`] or
//! [`Boundary::Neumann0`] store their boundary faces explicitly, so a
//! non-periodic axis of `n` cells has `n + 1` faces and a periodic axis has `n`.
//!
//! # Examples
//!
//! ```
//! use tpt_math_pde_operators::{Boundary, Centering, Field, Grid, Operators};
//!
//! let grid = Grid::new([8], [0.125_f64], [Boundary::Neumann0]).unwrap();
//! let ops = Operators::new(grid);
//!
//! // The Laplacian of a constant under zero-flux walls is zero.
//! let one = Field::from_fn(Centering::Cell, [8], |_| 1.0);
//! assert!(ops.laplacian(&one).as_slice().iter().all(|&x| x == 0.0));
//! ```

extern crate alloc;

/// Asserts a structural contract.
///
/// Active in debug builds and with the `contracts` feature; compiles to a
/// constant-false branch otherwise, so release builds pay nothing.
macro_rules! contract {
    ($cond:expr, $($arg:tt)+) => {
        if cfg!(any(debug_assertions, feature = "contracts")) {
            assert!($cond, $($arg)+);
        }
    };
}

mod boundary;
mod contracts;
mod field;
mod grid;
mod ops;

pub use boundary::Boundary;
pub use contracts::ContractReport;
pub use field::{Centering, Field};
pub use grid::{Grid, GridError};
pub use ops::Operators;
