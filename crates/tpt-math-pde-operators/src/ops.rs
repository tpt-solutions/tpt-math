//! Discrete gradient, divergence, Laplacian and central difference.

use alloc::vec::Vec;

use tpt_math_numeric::Scalar;

use crate::grid::with_axis;
use crate::{Boundary, Centering, Field, Grid};

/// Discrete differential operators on a structured [`Grid`].
///
/// Gradients are face-centred and divergences cell-centred, so on a
/// non-periodic grid [`Operators::gradient`] and [`Operators::divergence`] are
/// exact negative adjoints under the grid's quadrature weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operators<T, const D: usize> {
    grid: Grid<T, D>,
}

impl<T: Scalar, const D: usize> Operators<T, D> {
    /// Builds the operators on `grid`.
    ///
    /// In debug builds, or with the `contracts` feature, every structural
    /// contract is verified first (see [`Operators::verify`]).
    ///
    /// # Panics
    ///
    /// When contracts are active and one of them fails.
    pub fn new(grid: Grid<T, D>) -> Self {
        let ops = Self { grid };
        if cfg!(any(debug_assertions, feature = "contracts")) {
            let failed = ops.verify().into_iter().find(|report| !report.passed);
            contract!(failed.is_none(), "structural contract violated: {failed:?}");
        }
        ops
    }

    /// The grid these operators act on.
    pub fn grid(&self) -> &Grid<T, D> {
        &self.grid
    }

    /// Gradient: a cell-centred field to one face-centred field per axis.
    ///
    /// Interior faces take the two-point difference. A
    /// [`Boundary::Dirichlet0`] wall takes the half-cell difference against
    /// zero, and a [`Boundary::Neumann0`] wall has zero gradient. Periodic
    /// axes wrap around.
    ///
    /// # Panics
    ///
    /// If `u` is not a cell-centred field with the grid's shape.
    pub fn gradient(&self, u: &Field<T, D>) -> Vec<Field<T, D>> {
        self.assert_cell_field(u);
        let grid = &self.grid;
        let shape = grid.shape();
        let two = T::one() + T::one();
        (0..D)
            .map(|axis| {
                let n = shape[axis];
                let inv_h = T::one() / grid.spacing()[axis];
                let boundary = grid.boundary()[axis];
                Field::from_fn(Centering::Face(axis), grid.face_shape(axis), |face| {
                    let i = face[axis];
                    if grid.is_wall_face(axis, i) {
                        match boundary {
                            Boundary::Dirichlet0 => {
                                // The wall is half a cell outside the boundary cell.
                                let (cell, sign) = if i == 0 {
                                    (0, T::one())
                                } else {
                                    (n - 1, -T::one())
                                };
                                sign * two * inv_h * u.get(with_axis(face, axis, cell))
                            }
                            _ => T::zero(),
                        }
                    } else {
                        let left = if i == 0 { n - 1 } else { i - 1 };
                        (u.get(face) - u.get(with_axis(face, axis, left))) * inv_h
                    }
                })
            })
            .collect()
    }

    /// Divergence: one face-centred flux per axis to a cell-centred field.
    ///
    /// A [`Boundary::Neumann0`] wall carries no flux, so any value stored on
    /// it is ignored. Other walls use the stored value.
    ///
    /// # Panics
    ///
    /// If `flux` does not hold one face field per axis, each with the shape
    /// [`Grid::face_shape`] gives for that axis.
    pub fn divergence(&self, flux: &[Field<T, D>]) -> Field<T, D> {
        assert_eq!(
            flux.len(),
            D,
            "divergence needs one flux component per axis"
        );
        for (axis, component) in flux.iter().enumerate() {
            assert_eq!(
                component.centering(),
                Centering::Face(axis),
                "flux component {axis} is not a face field on that axis"
            );
            assert_eq!(
                component.shape(),
                self.grid.face_shape(axis),
                "flux component {axis} has the wrong shape"
            );
        }
        let grid = &self.grid;
        let shape = grid.shape();
        let flux_at = |axis: usize, face: [usize; D]| {
            if grid.is_wall_face(axis, face[axis]) && grid.boundary()[axis] == Boundary::Neumann0 {
                T::zero()
            } else {
                flux[axis].get(face)
            }
        };
        Field::from_fn(Centering::Cell, shape, |cell| {
            (0..D).fold(T::zero(), |acc, axis| {
                let i = cell[axis];
                let upper = if grid.boundary()[axis].is_periodic() {
                    (i + 1) % shape[axis]
                } else {
                    i + 1
                };
                let inv_h = T::one() / grid.spacing()[axis];
                let net = flux_at(axis, with_axis(cell, axis, upper))
                    - flux_at(axis, with_axis(cell, axis, i));
                acc + net * inv_h
            })
        })
    }

    /// Laplacian, assembled as `divergence(gradient(u))`.
    ///
    /// # Panics
    ///
    /// If `u` is not a cell-centred field with the grid's shape.
    pub fn laplacian(&self, u: &Field<T, D>) -> Field<T, D> {
        let flux = self.gradient(u);
        self.divergence(&flux)
    }

    /// Second-order central difference along a periodic `axis`:
    /// `(u[i+1] - u[i-1]) / (2h)`.
    ///
    /// # Panics
    ///
    /// If `axis` is not periodic, or `u` is not a cell-centred field with the
    /// grid's shape.
    pub fn central_difference(&self, u: &Field<T, D>, axis: usize) -> Field<T, D> {
        assert!(
            self.grid.boundary()[axis].is_periodic(),
            "central difference requires a periodic axis"
        );
        self.assert_cell_field(u);
        let shape = self.grid.shape();
        let n = shape[axis];
        let two = T::one() + T::one();
        let inv = T::one() / (two * self.grid.spacing()[axis]);
        Field::from_fn(Centering::Cell, shape, |cell| {
            let next = (cell[axis] + 1) % n;
            let prev = (cell[axis] + n - 1) % n;
            (u.get(with_axis(cell, axis, next)) - u.get(with_axis(cell, axis, prev))) * inv
        })
    }

    fn assert_cell_field(&self, u: &Field<T, D>) {
        assert_eq!(
            u.centering(),
            Centering::Cell,
            "expected a cell-centred field"
        );
        assert_eq!(
            u.shape(),
            self.grid.shape(),
            "field shape does not match the grid"
        );
    }
}
