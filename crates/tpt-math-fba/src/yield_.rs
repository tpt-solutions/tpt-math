//! Maximum theoretical yield of a product on a substrate.

use crate::error::FbaError;
use crate::fba::{fba, FbaSolution};
use crate::lp::{LpSolver, LpStatus};
use crate::model::Model;

/// Maximum theoretical yield of a product on a substrate.
#[derive(Clone, Debug, PartialEq)]
pub struct YieldResult {
    /// Solver status. The remaining fields are only meaningful when `Optimal`.
    pub status: LpStatus,
    /// Substrate uptake used, in the model's flux units.
    pub substrate_uptake: f64,
    /// Maximum secretion flux of the product.
    pub product_flux: f64,
    /// Yield `product_flux / substrate_uptake`, in mol product per mol substrate.
    pub yield_mol_per_mol: f64,
    /// Full flux distribution at the optimum.
    pub solution: FbaSolution,
}

/// Maximise secretion of `product` with the substrate uptake fixed at `uptake`.
///
/// The substrate exchange `substrate` is pinned to `[-uptake, 0]` and the
/// product exchange `product` is made the sole objective. The model is cloned,
/// so the caller's bounds and objective are left untouched.
///
/// The model must already contain a route from substrate to product (a
/// heterologous pathway for a non-native product). Without one, the yield is zero.
///
/// # Errors
///
/// [`FbaError::UnknownId`] if either exchange is missing, [`FbaError::InvalidBounds`]
/// if `uptake` is negative or NaN, or any solver error.
pub fn max_theoretical_yield<S: LpSolver>(
    model: &Model,
    solver: &S,
    substrate: &str,
    product: &str,
    uptake: f64,
) -> Result<YieldResult, FbaError> {
    if uptake.is_nan() || uptake < 0.0 {
        return Err(FbaError::InvalidBounds {
            reaction: substrate.to_owned(),
        });
    }
    let mut m = model.clone();
    m.clear_objectives();
    m.set_bounds(substrate, -uptake, 0.0)?;
    m.set_objective(product, 1.0)?;

    let solution = fba(&m, solver)?;
    let product_flux = solution.flux(&m, product).unwrap_or(0.0);
    let yield_mol_per_mol = if uptake > 0.0 {
        product_flux / uptake
    } else {
        0.0
    };
    Ok(YieldResult {
        status: solution.status,
        substrate_uptake: uptake,
        product_flux,
        yield_mol_per_mol,
        solution,
    })
}
