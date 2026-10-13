//! Flux balance analysis on a sparse, in-house bounded revised simplex.
//!
//! The crate loads a metabolic network, assembles its stoichiometric matrix
//! `S` in sparse form, and solves
//! `max cᵀv  subject to  S v = 0,  lb ≤ v ≤ ub`.
//! On top of that it computes the maximum theoretical yield of a product on a
//! substrate, the usual first question for a strain-design target such as
//! xylitol from glucose.
//!
//! The LP layer is pluggable through [`LpSolver`]; [`SimplexSolver`] is the
//! default backend.
//!
//! # Example
//!
//! A two-step chain `EX → A → B → biomass` with uptake capped at 10:
//!
//! ```
//! use tpt_math_fba::{fba, Model, SimplexSolver, LpStatus};
//!
//! let mut m = Model::new();
//! m.add_metabolite("A")?;
//! m.add_metabolite("B")?;
//! m.add_reaction("EX_A", 0.0, 10.0, &[("A", 1.0)])?;
//! m.add_reaction("R1", 0.0, 1000.0, &[("A", -1.0), ("B", 1.0)])?;
//! m.add_reaction("BIOMASS", 0.0, 1000.0, &[("B", -1.0)])?;
//! m.set_objective("BIOMASS", 1.0)?;
//!
//! let sol = fba(&m, &SimplexSolver::new())?;
//! assert_eq!(sol.status, LpStatus::Optimal);
//! assert!((sol.objective - 10.0).abs() < 1e-6);
//! # Ok::<(), tpt_math_fba::FbaError>(())
//! ```

#![warn(missing_docs)]

mod error;
mod fba;
mod lp;
mod model;
mod presolve;
mod simplex;
mod yield_;

#[cfg(feature = "json")]
mod json;

pub use error::FbaError;
pub use fba::{build_lp, fba, FbaSolution};
pub use lp::{LpSolution, LpSolver, LpStatus, SparseLp};
pub use model::{Metabolite, Model, Reaction};
pub use simplex::{SimplexOptions, SimplexSolver};
pub use yield_::{max_theoretical_yield, YieldResult};
