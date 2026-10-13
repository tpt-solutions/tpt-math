//! Scalar fields on a structured grid, centred on cells or faces.

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_numeric::Scalar;

use crate::grid::{flat_index, unflatten};

/// Where a field's unknowns live on the staggered grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Centering {
    /// At cell centres.
    Cell,
    /// On the faces normal to the given axis.
    Face(usize),
}

/// A scalar field on a structured grid, stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Field<T, const D: usize> {
    centering: Centering,
    shape: [usize; D],
    data: Vec<T>,
}

impl<T: Scalar, const D: usize> Field<T, D> {
    /// A field of zeros with the given centering and shape.
    pub fn zeros(centering: Centering, shape: [usize; D]) -> Self {
        let len: usize = shape.iter().product();
        Self {
            centering,
            shape,
            data: vec![T::zero(); len],
        }
    }

    /// A field whose value at each multi-index is `f(index)`.
    pub fn from_fn(
        centering: Centering,
        shape: [usize; D],
        mut f: impl FnMut([usize; D]) -> T,
    ) -> Self {
        let len: usize = shape.iter().product();
        let data = (0..len).map(|flat| f(unflatten(&shape, flat))).collect();
        Self {
            centering,
            shape,
            data,
        }
    }

    /// Where the unknowns live.
    pub fn centering(&self) -> Centering {
        self.centering
    }

    /// Number of entries along each axis.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the field has no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Value at `index`.
    ///
    /// # Panics
    ///
    /// If any coordinate of `index` is out of range.
    pub fn get(&self, index: [usize; D]) -> T {
        self.data[flat_index(&self.shape, index)]
    }

    /// Sets the value at `index`.
    ///
    /// # Panics
    ///
    /// If any coordinate of `index` is out of range.
    pub fn set(&mut self, index: [usize; D], value: T) {
        let flat = flat_index(&self.shape, index);
        self.data[flat] = value;
    }

    /// Row-major view of the entries.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable row-major view of the entries.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}
