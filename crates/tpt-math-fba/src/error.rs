//! Error type for the FBA crate.

use std::fmt;

/// Errors returned by model construction, loading and the LP solvers.
#[derive(Debug, Clone, PartialEq)]
pub enum FbaError {
    /// A reaction or metabolite identifier was not found in the model.
    UnknownId(String),
    /// An identifier was added twice.
    DuplicateId(String),
    /// A reaction has `lower_bound > upper_bound` or a NaN bound.
    InvalidBounds {
        /// Reaction identifier.
        reaction: String,
    },
    /// Vector or matrix dimensions of an LP were inconsistent.
    DimensionMismatch,
    /// The basis matrix became singular and could not be repaired.
    SingularBasis,
    /// The model could not be parsed (JSON syntax or schema problem).
    Parse(String),
    /// The LP is infeasible, so no flux distribution satisfies the constraints.
    Infeasible,
    /// The LP objective is unbounded.
    Unbounded,
    /// The solver hit its iteration limit.
    IterationLimit {
        /// Iterations performed.
        iterations: usize,
    },
}

impl fmt::Display for FbaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FbaError::UnknownId(id) => write!(f, "unknown identifier `{id}`"),
            FbaError::DuplicateId(id) => write!(f, "duplicate identifier `{id}`"),
            FbaError::InvalidBounds { reaction } => {
                write!(f, "invalid bounds on reaction `{reaction}`")
            }
            FbaError::DimensionMismatch => write!(f, "LP dimension mismatch"),
            FbaError::SingularBasis => write!(f, "simplex basis is singular"),
            FbaError::Parse(msg) => write!(f, "model parse error: {msg}"),
            FbaError::Infeasible => write!(f, "problem is infeasible"),
            FbaError::Unbounded => write!(f, "problem is unbounded"),
            FbaError::IterationLimit { iterations } => {
                write!(f, "simplex hit the iteration limit ({iterations})")
            }
        }
    }
}

impl std::error::Error for FbaError {}
