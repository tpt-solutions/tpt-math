//! A common summary of an iterative solve, shared by the solver crates.

/// How an iterative solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveStatus {
    /// The residual met the requested tolerance.
    Converged,
    /// The iteration budget ran out before the tolerance was met.
    NotConverged,
    /// The method hit a numerical breakdown and could not continue.
    Breakdown,
}

impl SolveStatus {
    /// `true` only for [`SolveStatus::Converged`].
    #[must_use]
    pub fn is_converged(self) -> bool {
        self == SolveStatus::Converged
    }
}

impl core::fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let text = match self {
            SolveStatus::Converged => "converged",
            SolveStatus::NotConverged => "not converged",
            SolveStatus::Breakdown => "breakdown",
        };
        f.write_str(text)
    }
}

/// The outcome of an iterative solve: how it ended, how many iterations it
/// took, and the residual it reached.
///
/// `residual` is the solver's own measure, normally the absolute residual norm
/// `‖b − A x‖` at the last iterate. Check the solver's documentation for its
/// exact definition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport<T> {
    /// How the solve ended.
    pub status: SolveStatus,
    /// Iterations performed.
    pub iterations: usize,
    /// Residual at the last iterate.
    pub residual: T,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_converged_is_converged() {
        assert!(SolveStatus::Converged.is_converged());
        assert!(!SolveStatus::NotConverged.is_converged());
        assert!(!SolveStatus::Breakdown.is_converged());
    }
}
