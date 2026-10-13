//! Sparse linear-program representation and the pluggable solver trait.

use tpt_math_linalg_sparse::CscMatrix;

use crate::error::FbaError;

/// A sparse LP in computational form:
/// `min cᵀx  s.t.  A x = b,  lb ≤ x ≤ ub`.
///
/// Bounds may be infinite (`f64::INFINITY` / `f64::NEG_INFINITY`); a variable
/// with both infinite is free.
#[derive(Clone, Debug)]
pub struct SparseLp {
    /// Constraint matrix (`m × n`).
    pub a: CscMatrix<f64>,
    /// Right-hand side (length `m`).
    pub b: Vec<f64>,
    /// Objective coefficients to **minimise** (length `n`).
    pub c: Vec<f64>,
    /// Lower bounds (length `n`).
    pub lb: Vec<f64>,
    /// Upper bounds (length `n`).
    pub ub: Vec<f64>,
}

impl SparseLp {
    /// Check that all dimensions agree and no bound pair is inverted or NaN.
    ///
    /// # Errors
    ///
    /// [`FbaError::DimensionMismatch`] for inconsistent lengths, or
    /// [`FbaError::Infeasible`] when some `lb > ub`.
    pub fn validate(&self) -> Result<(), FbaError> {
        let (m, n) = (self.a.nrows(), self.a.ncols());
        if self.b.len() != m
            || [self.c.len(), self.lb.len(), self.ub.len()]
                .iter()
                .any(|&l| l != n)
        {
            return Err(FbaError::DimensionMismatch);
        }
        if self
            .lb
            .iter()
            .zip(&self.ub)
            .any(|(l, u)| l.is_nan() || u.is_nan() || l > u)
        {
            return Err(FbaError::Infeasible);
        }
        Ok(())
    }
}

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LpStatus {
    /// An optimal solution was found.
    Optimal,
    /// No feasible point exists.
    Infeasible,
    /// The objective is unbounded below.
    Unbounded,
    /// The iteration limit was reached.
    IterationLimit,
}

/// Result of an LP solve.
#[derive(Clone, Debug)]
pub struct LpSolution {
    /// Termination status; `x`/`objective` are meaningful only for `Optimal`.
    pub status: LpStatus,
    /// Objective value `cᵀx` of the minimisation problem.
    pub objective: f64,
    /// Primal solution.
    pub x: Vec<f64>,
    /// Dual values (one per row).
    pub duals: Vec<f64>,
    /// Reduced costs (one per column).
    pub reduced_costs: Vec<f64>,
    /// Simplex iterations (pivots and bound flips).
    pub iterations: usize,
}

/// A backend able to solve a [`SparseLp`].
///
/// [`crate::SimplexSolver`] is the default implementation; the trait lets an
/// interior-point backend be added without touching the FBA layer.
pub trait LpSolver {
    /// Solve `lp`.
    ///
    /// # Errors
    ///
    /// Backend-specific numerical failures (e.g. [`FbaError::SingularBasis`]) or
    /// an invalid LP. Infeasible/unbounded outcomes are reported through
    /// [`LpSolution::status`], not as errors.
    fn solve(&self, lp: &SparseLp) -> Result<LpSolution, FbaError>;
}
