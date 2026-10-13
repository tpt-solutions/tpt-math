# tpt-math-fba

Flux balance analysis (FBA) for genome-scale metabolic models, built on an
in-house sparse bounded revised simplex.

Given a stoichiometric matrix `S` and flux bounds `lb ≤ v ≤ ub`, FBA solves

    max  cᵀv   subject to   S v = 0,   lb ≤ v ≤ ub

This crate provides:

- `Model`: metabolites, reactions, bounds and objective, with a builder API and
  a BiGG / COBRA JSON loader (`Model::from_json_str`, behind the `json` feature).
- `SimplexSolver`: a bounded-variable revised simplex on a sparse LU basis, with
  geometric-mean scaling, phase-1 artificials and Bland's rule anti-cycling.
- `LpSolver`: the solver trait. Other backends (for example interior point) can
  be plugged in without changing the FBA layer.
- `fba`: flux balance analysis on a `Model`.
- `max_theoretical_yield`: the maximum product secretion per unit substrate
  uptake, for questions such as xylitol from glucose.

## Example: maximum xylitol yield

```rust,ignore
use tpt_math_fba::{max_theoretical_yield, Model, SimplexSolver};

let model = Model::from_json_str(&std::fs::read_to_string("iML1515.json")?)?;
// The model must contain a xylose → xylitol route, for example added with `add_reaction`.
let r = max_theoretical_yield(
    &model,
    &SimplexSolver::new(),
    "EX_glc__D_e",  // substrate exchange
    "EX_xylt_e",    // product exchange
    10.0,           // glucose uptake, mmol/gDW/h
)?;
println!("yield = {:.3} mol/mol", r.yield_mol_per_mol);
```

## Status and limitations

- The basis is refactorised on every iteration (no Forrest–Tomlin update yet),
  so large models are correct but slow. Expect improvement from an
  update-based basis and from an interior-point backend.
- Flux variability analysis is not yet implemented.
- SBML input is not supported; use BiGG / COBRA JSON.

Licensed under MIT OR Apache-2.0.
