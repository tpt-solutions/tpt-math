//! Metabolic network model: metabolites, reactions and the stoichiometric matrix.

use std::collections::HashMap;

use tpt_math_linalg_sparse::{CooMatrix, CscMatrix};

use crate::error::FbaError;

/// A metabolite (row of the stoichiometric matrix).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metabolite {
    /// Identifier, e.g. `glc__D_e`.
    pub id: String,
}

/// A reaction (column of the stoichiometric matrix).
#[derive(Clone, Debug, PartialEq)]
pub struct Reaction {
    /// Identifier, e.g. `EX_glc__D_e`.
    pub id: String,
    /// Lower flux bound.
    pub lower_bound: f64,
    /// Upper flux bound.
    pub upper_bound: f64,
    /// Coefficient in the maximised objective (usually biomass or a product exchange).
    pub objective: f64,
    /// Non-zero stoichiometry as `(metabolite index, coefficient)`.
    pub stoichiometry: Vec<(usize, f64)>,
}

/// A genome-scale (or toy) metabolic network.
///
/// Stoichiometry is stored per reaction and assembled into a sparse matrix on
/// demand with [`Model::stoichiometry`]. Sign convention follows COBRA/BiGG:
/// a negative coefficient on an exchange reaction marks an uptake.
#[derive(Clone, Debug, Default)]
pub struct Model {
    metabolites: Vec<Metabolite>,
    reactions: Vec<Reaction>,
    met_index: HashMap<String, usize>,
    rxn_index: HashMap<String, usize>,
}

impl Model {
    /// Empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metabolite and return its index.
    ///
    /// # Errors
    ///
    /// [`FbaError::DuplicateId`] if the identifier already exists.
    pub fn add_metabolite(&mut self, id: &str) -> Result<usize, FbaError> {
        if self.met_index.contains_key(id) {
            return Err(FbaError::DuplicateId(id.to_owned()));
        }
        let idx = self.metabolites.len();
        self.metabolites.push(Metabolite { id: id.to_owned() });
        self.met_index.insert(id.to_owned(), idx);
        Ok(idx)
    }

    /// Add a reaction with the given bounds and stoichiometry, and return its index.
    ///
    /// Stoichiometry entries name existing metabolites; repeated metabolites are summed.
    ///
    /// # Errors
    ///
    /// [`FbaError::DuplicateId`] for a repeated reaction id, [`FbaError::UnknownId`]
    /// for an unknown metabolite, [`FbaError::InvalidBounds`] if `lower > upper` or a bound is NaN.
    pub fn add_reaction(
        &mut self,
        id: &str,
        lower_bound: f64,
        upper_bound: f64,
        stoichiometry: &[(&str, f64)],
    ) -> Result<usize, FbaError> {
        if self.rxn_index.contains_key(id) {
            return Err(FbaError::DuplicateId(id.to_owned()));
        }
        check_bounds(id, lower_bound, upper_bound)?;
        let stoichiometry = stoichiometry
            .iter()
            .map(|&(met, coeff)| {
                self.met_index
                    .get(met)
                    .map(|&i| (i, coeff))
                    .ok_or_else(|| FbaError::UnknownId(met.to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let idx = self.reactions.len();
        self.reactions.push(Reaction {
            id: id.to_owned(),
            lower_bound,
            upper_bound,
            objective: 0.0,
            stoichiometry,
        });
        self.rxn_index.insert(id.to_owned(), idx);
        Ok(idx)
    }

    /// Set the flux bounds of a reaction.
    ///
    /// # Errors
    ///
    /// [`FbaError::UnknownId`] or [`FbaError::InvalidBounds`].
    pub fn set_bounds(
        &mut self,
        id: &str,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Result<(), FbaError> {
        check_bounds(id, lower_bound, upper_bound)?;
        let r = self.reaction_mut(id)?;
        r.lower_bound = lower_bound;
        r.upper_bound = upper_bound;
        Ok(())
    }

    /// Set a reaction's coefficient in the maximised objective.
    ///
    /// # Errors
    ///
    /// [`FbaError::UnknownId`] if the reaction does not exist.
    pub fn set_objective(&mut self, id: &str, coefficient: f64) -> Result<(), FbaError> {
        self.reaction_mut(id)?.objective = coefficient;
        Ok(())
    }

    /// Reset every reaction's objective coefficient to zero.
    pub fn clear_objectives(&mut self) {
        self.reactions.iter_mut().for_each(|r| r.objective = 0.0);
    }

    fn reaction_mut(&mut self, id: &str) -> Result<&mut Reaction, FbaError> {
        let idx = *self
            .rxn_index
            .get(id)
            .ok_or_else(|| FbaError::UnknownId(id.to_owned()))?;
        Ok(&mut self.reactions[idx])
    }

    /// Index of a reaction by id.
    pub fn reaction_index(&self, id: &str) -> Option<usize> {
        self.rxn_index.get(id).copied()
    }

    /// Index of a metabolite by id.
    pub fn metabolite_index(&self, id: &str) -> Option<usize> {
        self.met_index.get(id).copied()
    }

    /// All reactions, in insertion order.
    pub fn reactions(&self) -> &[Reaction] {
        &self.reactions
    }

    /// All metabolites, in insertion order.
    pub fn metabolites(&self) -> &[Metabolite] {
        &self.metabolites
    }

    /// Number of metabolites (rows of `S`).
    pub fn n_metabolites(&self) -> usize {
        self.metabolites.len()
    }

    /// Number of reactions (columns of `S`).
    pub fn n_reactions(&self) -> usize {
        self.reactions.len()
    }

    /// The stoichiometric matrix `S` (metabolites × reactions) in CSC form.
    pub fn stoichiometry(&self) -> CscMatrix<f64> {
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for (j, rxn) in self.reactions.iter().enumerate() {
            for &(i, v) in &rxn.stoichiometry {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        CooMatrix::from_triplets(
            self.metabolites.len(),
            self.reactions.len(),
            rows,
            cols,
            vals,
        )
        .to_csc()
    }
}

fn check_bounds(id: &str, lower: f64, upper: f64) -> Result<(), FbaError> {
    if lower.is_nan() || upper.is_nan() || lower > upper {
        return Err(FbaError::InvalidBounds {
            reaction: id.to_owned(),
        });
    }
    Ok(())
}
