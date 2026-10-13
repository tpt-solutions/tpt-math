//! Flux balance and theoretical-yield tests on hand-checkable networks.

use tpt_math_fba::{fba, max_theoretical_yield, FbaError, LpStatus, Model, SimplexSolver};

/// Mass-balance residual `max_i |Σ_j S_ij v_j|` for a flux vector.
fn residual(model: &Model, v: &[f64]) -> f64 {
    let mut r = vec![0.0; model.n_metabolites()];
    for (j, rxn) in model.reactions().iter().enumerate() {
        for &(i, s) in &rxn.stoichiometry {
            r[i] += s * v[j];
        }
    }
    r.iter().fold(0.0f64, |a, x| a.max(x.abs()))
}

/// Glucose → xylitol with 2 glucose per xylitol (carbon-limited yield 0.5),
/// plus a competing respiration route that does not make product.
fn xylitol_model() -> Model {
    let mut m = Model::new();
    for id in ["glc", "xylt", "co2"] {
        m.add_metabolite(id).unwrap();
    }
    m.add_reaction("EX_glc__D_e", -1000.0, 0.0, &[("glc", -1.0)])
        .unwrap();
    m.add_reaction("PATHWAY", 0.0, 1000.0, &[("glc", -2.0), ("xylt", 1.0)])
        .unwrap();
    m.add_reaction("RESP", 0.0, 1000.0, &[("glc", -1.0), ("co2", 6.0)])
        .unwrap();
    m.add_reaction("EX_xylt", 0.0, 1000.0, &[("xylt", -1.0)])
        .unwrap();
    m.add_reaction("EX_co2", 0.0, 1000.0, &[("co2", -1.0)])
        .unwrap();
    m
}

#[test]
fn fba_maximises_chain_and_respects_mass_balance() {
    let mut m = Model::new();
    m.add_metabolite("A").unwrap();
    m.add_metabolite("B").unwrap();
    m.add_reaction("EX_A", 0.0, 10.0, &[("A", 1.0)]).unwrap();
    m.add_reaction("R1", 0.0, 1000.0, &[("A", -1.0), ("B", 1.0)])
        .unwrap();
    m.add_reaction("BIOMASS", 0.0, 1000.0, &[("B", -1.0)])
        .unwrap();
    m.set_objective("BIOMASS", 1.0).unwrap();

    let sol = fba(&m, &SimplexSolver::new()).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert!((sol.objective - 10.0).abs() < 1e-7);
    assert!(residual(&m, &sol.fluxes) < 1e-7);
    assert_eq!(sol.flux(&m, "BIOMASS"), Some(sol.fluxes[2]));
}

#[test]
fn fba_reports_infeasible_steady_state() {
    // A is produced by nothing but consumed by a required reaction → no steady state with flux.
    let mut m = Model::new();
    m.add_metabolite("A").unwrap();
    m.add_reaction("SINK", 1.0, 1000.0, &[("A", -1.0)]).unwrap();
    let sol = fba(&m, &SimplexSolver::new()).unwrap();
    assert_eq!(sol.status, LpStatus::Infeasible);
}

#[test]
fn xylitol_yield_is_half_mol_per_mol() {
    let m = xylitol_model();
    let r =
        max_theoretical_yield(&m, &SimplexSolver::new(), "EX_glc__D_e", "EX_xylt", 10.0).unwrap();
    assert_eq!(r.status, LpStatus::Optimal);
    assert!(
        (r.product_flux - 5.0).abs() < 1e-7,
        "product {}",
        r.product_flux
    );
    assert!((r.yield_mol_per_mol - 0.5).abs() < 1e-9);
    assert!(residual(&m, &r.solution.fluxes) < 1e-7);
}

#[test]
fn yield_leaves_original_model_untouched() {
    let m = xylitol_model();
    let _ =
        max_theoretical_yield(&m, &SimplexSolver::new(), "EX_glc__D_e", "EX_xylt", 10.0).unwrap();
    assert_eq!(m.reactions()[0].lower_bound, -1000.0);
    assert_eq!(m.reactions()[3].objective, 0.0);
}

#[test]
fn yield_without_pathway_is_zero() {
    let mut m = Model::new();
    m.add_metabolite("glc").unwrap();
    m.add_metabolite("xylt").unwrap();
    m.add_reaction("EX_glc__D_e", -1000.0, 0.0, &[("glc", -1.0)])
        .unwrap();
    m.add_reaction("EX_xylt", 0.0, 1000.0, &[("xylt", -1.0)])
        .unwrap();
    let r =
        max_theoretical_yield(&m, &SimplexSolver::new(), "EX_glc__D_e", "EX_xylt", 10.0).unwrap();
    assert_eq!(r.status, LpStatus::Optimal);
    assert!(r.yield_mol_per_mol.abs() < 1e-9);
}

#[test]
fn unknown_exchange_is_an_error() {
    let m = xylitol_model();
    let err =
        max_theoretical_yield(&m, &SimplexSolver::new(), "EX_nope", "EX_xylt", 10.0).unwrap_err();
    assert_eq!(err, FbaError::UnknownId("EX_nope".to_owned()));
}

#[test]
fn json_model_round_trips_to_same_yield() {
    let json = r#"{
        "metabolites": [{"id":"glc"},{"id":"xylt"}],
        "reactions": [
          {"id":"EX_glc__D_e","metabolites":{"glc":-1.0}},
          {"id":"PATHWAY","lower_bound":0.0,"upper_bound":1000.0,
           "metabolites":{"glc":-2.0,"xylt":1.0}},
          {"id":"EX_xylt","lower_bound":0.0,"metabolites":{"xylt":-1.0}}
        ]
    }"#;
    let m = Model::from_json_str(json).unwrap();
    assert_eq!(m.n_reactions(), 3);
    assert_eq!(m.reactions()[0].lower_bound, -1000.0, "COBRA default bound");
    let r =
        max_theoretical_yield(&m, &SimplexSolver::new(), "EX_glc__D_e", "EX_xylt", 10.0).unwrap();
    assert!((r.yield_mol_per_mol - 0.5).abs() < 1e-9);
}

#[test]
fn json_rejects_malformed_input() {
    assert!(matches!(Model::from_json_str("{"), Err(FbaError::Parse(_))));
    assert!(matches!(
        Model::from_json_str(
            r#"{"metabolites":[],"reactions":[{"id":"R","metabolites":{"x":1.0}}]}"#
        ),
        Err(FbaError::UnknownId(_))
    ));
}
