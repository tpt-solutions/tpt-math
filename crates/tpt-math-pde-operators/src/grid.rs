//! Uniform structured grids: shape, spacing and per-axis boundary treatment.

use core::fmt;

use tpt_math_numeric::Scalar;

use crate::Boundary;

/// Errors from constructing a [`Grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridError {
    /// An axis has no cells.
    ZeroExtent {
        /// The offending axis.
        axis: usize,
    },
    /// An axis spacing is not finite and strictly positive.
    BadSpacing {
        /// The offending axis.
        axis: usize,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroExtent { axis } => write!(f, "axis {axis} has zero cells"),
            Self::BadSpacing { axis } => {
                write!(
                    f,
                    "axis {axis} spacing must be finite and greater than zero"
                )
            }
        }
    }
}

impl core::error::Error for GridError {}

/// A uniform `D`-dimensional grid of cells, indexed in row-major order (the
/// last axis varies fastest).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid<T, const D: usize> {
    shape: [usize; D],
    spacing: [T; D],
    boundary: [Boundary; D],
}

impl<T: Scalar, const D: usize> Grid<T, D> {
    /// Creates a grid with `shape` cells of width `spacing` on each axis.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::ZeroExtent`] or [`GridError::BadSpacing`] for the
    /// first invalid axis.
    pub fn new(
        shape: [usize; D],
        spacing: [T; D],
        boundary: [Boundary; D],
    ) -> Result<Self, GridError> {
        for axis in 0..D {
            if shape[axis] == 0 {
                return Err(GridError::ZeroExtent { axis });
            }
            let h = spacing[axis];
            if !(h.is_finite() && h > T::zero()) {
                return Err(GridError::BadSpacing { axis });
            }
        }
        Ok(Self {
            shape,
            spacing,
            boundary,
        })
    }

    /// Number of cells along each axis.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// Cell width along each axis.
    pub fn spacing(&self) -> [T; D] {
        self.spacing
    }

    /// Boundary treatment of each axis.
    pub fn boundary(&self) -> [Boundary; D] {
        self.boundary
    }

    /// Total number of cells.
    pub fn num_cells(&self) -> usize {
        self.shape.iter().product()
    }

    /// Volume of one cell.
    pub fn cell_volume(&self) -> T {
        self.spacing.iter().fold(T::one(), |acc, &h| acc * h)
    }

    /// Shape of the face-centred field normal to `axis`: one more face than
    /// cells on a non-periodic axis, one face per cell on a periodic axis.
    pub fn face_shape(&self, axis: usize) -> [usize; D] {
        let mut shape = self.shape;
        if !self.boundary[axis].is_periodic() {
            shape[axis] += 1;
        }
        shape
    }

    /// Whether face `index` on `axis` lies on a wall.
    pub fn is_wall_face(&self, axis: usize, index: usize) -> bool {
        !self.boundary[axis].is_periodic() && (index == 0 || index == self.shape[axis])
    }

    /// Quadrature weight of a face normal to `axis`: the cross-sectional
    /// volume. Wall faces count half a cell, since the wall is half a cell
    /// outside the boundary cell centre.
    pub fn face_weight(&self, axis: usize, face: [usize; D]) -> T {
        let two = T::one() + T::one();
        let mut weight = T::one();
        for b in 0..D {
            let h = self.spacing[b];
            let width = if b == axis && self.is_wall_face(axis, face[axis]) {
                h / two
            } else {
                h
            };
            weight = weight * width;
        }
        weight
    }
}

/// Row-major offset of `index` within an array of the given `shape`.
pub(crate) fn flat_index<const D: usize>(shape: &[usize; D], index: [usize; D]) -> usize {
    index.iter().zip(shape).fold(0, |acc, (&i, &n)| acc * n + i)
}

/// Inverse of [`flat_index`].
pub(crate) fn unflatten<const D: usize>(shape: &[usize; D], flat: usize) -> [usize; D] {
    let mut index = [0; D];
    let mut rest = flat;
    for (slot, &n) in index.iter_mut().zip(shape).rev() {
        *slot = rest % n;
        rest /= n;
    }
    index
}

/// Copy of `index` with coordinate `axis` replaced by `value`.
pub(crate) fn with_axis<const D: usize>(
    mut index: [usize; D],
    axis: usize,
    value: usize,
) -> [usize; D] {
    index[axis] = value;
    index
}
