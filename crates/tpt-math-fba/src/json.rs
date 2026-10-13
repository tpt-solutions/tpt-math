//! BiGG / COBRA JSON model loader.

use crate::error::FbaError;
use crate::model::Model;

const DEFAULT_LOWER: f64 = -1000.0;
const DEFAULT_UPPER: f64 = 1000.0;

impl Model {
    /// Parse a BiGG / COBRA JSON model.
    ///
    /// Expects `metabolites` (objects with `id`) and `reactions` (objects with
    /// `id`, `metabolites` as a map of id → coefficient, and optionally
    /// `lower_bound`, `upper_bound`, `objective_coefficient`). Missing bounds
    /// default to ±1000, as COBRA does.
    ///
    /// # Errors
    ///
    /// [`FbaError::Parse`] for malformed JSON or a schema violation, and any
    /// error from [`Model::add_reaction`] (duplicate ids, unknown metabolites,
    /// inverted bounds).
    pub fn from_json_str(s: &str) -> Result<Model, FbaError> {
        let root: serde_json::Value =
            serde_json::from_str(s).map_err(|e| FbaError::Parse(e.to_string()))?;
        let mut model = Model::new();

        for met in array(&root, "metabolites")? {
            model.add_metabolite(id_of(met)?)?;
        }
        for rxn in array(&root, "reactions")? {
            let id = id_of(rxn)?;
            let lb = number(rxn, "lower_bound")?.unwrap_or(DEFAULT_LOWER);
            let ub = number(rxn, "upper_bound")?.unwrap_or(DEFAULT_UPPER);
            let obj = number(rxn, "objective_coefficient")?.unwrap_or(0.0);

            let mets = rxn
                .get("metabolites")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| parse_err(id, "missing `metabolites` map"))?;
            let stoich: Vec<(&str, f64)> = mets
                .iter()
                .map(|(met, coeff)| {
                    coeff
                        .as_f64()
                        .map(|c| (met.as_str(), c))
                        .ok_or_else(|| parse_err(id, "non-numeric coefficient"))
                })
                .collect::<Result<_, _>>()?;

            let idx = model.add_reaction(id, lb, ub, &stoich)?;
            if obj != 0.0 {
                model.set_objective(model.reactions()[idx].id.clone().as_str(), obj)?;
            }
        }
        Ok(model)
    }
}

fn parse_err(ctx: &str, msg: &str) -> FbaError {
    FbaError::Parse(format!("{ctx}: {msg}"))
}

fn array<'a>(
    root: &'a serde_json::Value,
    key: &str,
) -> Result<&'a Vec<serde_json::Value>, FbaError> {
    root.get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| FbaError::Parse(format!("missing `{key}` array")))
}

fn id_of(v: &serde_json::Value) -> Result<&str, FbaError> {
    v.get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| FbaError::Parse("entry without a string `id`".to_owned()))
}

fn number(v: &serde_json::Value, key: &str) -> Result<Option<f64>, FbaError> {
    match v.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(x) => x
            .as_f64()
            .map(Some)
            .ok_or_else(|| parse_err(id_of(v).unwrap_or("?"), &format!("`{key}` is not a number"))),
    }
}
