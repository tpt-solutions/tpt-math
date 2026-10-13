//! Flux balance analysis: maximise a linear objective over steady-state fluxes.

use crate::error::FbaError;
use crate::lp::{LpSolver, LpStatus, SparseLp};
use crate::model::Model;

/// Result of a flux balance analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct FbaSolution {
    /// Termination status. `objective` and `fluxes` are only meaningful when `Optimal`.
    pub status: LpStatus,
    /// Maximised objective value `Σ cⱼ vⱼ`.
    pub objective: f64,
    /// Flux of every reaction, indexed as in [`Model::reactions`].
    pub fluxes: Vec<f64>,
}

impl FbaSolution {
    /// Flux of the named reaction, or `None` if it does not exist.
    pub fn flux(&self, model: &Model, reaction: &str) -> Option<f64> {
        model.reaction_index(reaction).map(|j| self.fluxes[j])
    }
}

/// Build the computational-form LP `min −cᵀv, S v = 0, lb ≤ v ≤ ub`.
pub fn build_lp(model: &Model) -> SparseLp {
    let rxns = model.reactions();
    SparseLp {
        a: model.stoichiometry(),
        b: vec![0.0; model.n_metabolites()],
        c: rxns.iter().map(|r| -r.objective).collect(),
        lb: rxns.iter().map(|r| r.lower_bound).collect(),
        ub: rxns.iter().map(|r| r.upper_bound).collect(),
    }
}

/// Run flux balance analysis with the given LP backend.
///
/// # Errors
///
/// Propagates solver errors. An infeasible or unbounded model is reported
/// through [`FbaSolution::status`] rather than as an error.
pub fn fba<S: LpSolver>(model: &Model, solver: &S) -> Result<FbaSolution, FbaError> {
    let sol = solver.solve(&build_lp(model))?;
    Ok(FbaSolution {
        status: sol.status,
        objective: -sol.objective,
        fluxes: sol.x,
    })
}
