/// Boundary treatment of one axis.
///
/// A non-periodic axis has walls half a cell outside its first and last cell
/// centres, and the boundary faces sit on those walls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// The axis wraps around: the last cell is the neighbour of the first.
    Periodic,
    /// Homogeneous Dirichlet: the field is zero on the wall, so the wall
    /// gradient is taken half a cell away from the boundary cell.
    Dirichlet0,
    /// Homogeneous Neumann (zero flux): the normal derivative and the normal
    /// flux are zero on the wall. Any flux stored on such a wall is ignored.
    Neumann0,
}

impl Boundary {
    /// Whether the axis wraps around.
    pub fn is_periodic(self) -> bool {
        matches!(self, Self::Periodic)
    }
}
