//! Multiple-comparison corrections for a family of p-values.
//!
//! Each function takes the raw p-values and returns adjusted p-values in the
//! same order. Comparing an adjusted value against a level `α` controls the
//! error rate named by the procedure: family-wise error for Bonferroni and
//! Holm, false discovery rate for Benjamini–Hochberg.
//!
//! All adjusted values are capped at `1`.

use crate::error::{check_all_finite, StatsError};

/// Validates that every p-value lies in `[0, 1]`.
fn check_p_values(p_values: &[f64]) -> Result<(), StatsError> {
    check_all_finite("p_values", p_values)?;
    match p_values.iter().position(|p| !(0.0..=1.0).contains(p)) {
        None => Ok(()),
        Some(index) => Err(StatsError::ProbabilityOutOfRange {
            value: p_values[index],
        }),
    }
}

/// Indices of `p_values` sorted by ascending p-value.
fn ascending_order(p_values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..p_values.len()).collect();
    order.sort_by(|&i, &j| p_values[i].total_cmp(&p_values[j]));
    order
}

/// Bonferroni correction: `p_adj = min(1, m · p)` for `m` tests.
///
/// # Errors
///
/// Returns [`StatsError::ProbabilityOutOfRange`] for a p-value outside `[0, 1]`
/// and [`StatsError::NotFinite`] for a non-finite one.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_bonferroni;
/// let adjusted = try_bonferroni(&[0.01, 0.04, 0.5]).unwrap();
/// assert!((adjusted[0] - 0.03).abs() < 1e-12);
/// assert!((adjusted[1] - 0.12).abs() < 1e-12);
/// assert_eq!(adjusted[2], 1.0);
/// ```
pub fn try_bonferroni(p_values: &[f64]) -> Result<Vec<f64>, StatsError> {
    check_p_values(p_values)?;
    let m = p_values.len() as f64;
    Ok(p_values.iter().map(|p| (m * p).min(1.0)).collect())
}

/// Bonferroni correction. See [`try_bonferroni`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_bonferroni`].
pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    try_bonferroni(p_values).unwrap_or_else(|e| panic!("bonferroni: {e}"))
}

/// Holm step-down correction, which controls the family-wise error rate and is
/// uniformly more powerful than Bonferroni.
///
/// The `j`-th smallest p-value (1-based) is multiplied by `m − j + 1`, and a
/// running maximum enforces monotonicity.
///
/// # Errors
///
/// As [`try_bonferroni`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_holm;
/// let adjusted = try_holm(&[0.01, 0.04, 0.03, 0.005]).unwrap();
/// assert!((adjusted[0] - 0.03).abs() < 1e-12);
/// assert!((adjusted[1] - 0.06).abs() < 1e-12);
/// assert!((adjusted[2] - 0.06).abs() < 1e-12);
/// assert!((adjusted[3] - 0.02).abs() < 1e-12);
/// ```
pub fn try_holm(p_values: &[f64]) -> Result<Vec<f64>, StatsError> {
    check_p_values(p_values)?;
    let m = p_values.len();
    let mut adjusted = vec![0.0; m];
    let mut running_max: f64 = 0.0;
    for (rank, &i) in ascending_order(p_values).iter().enumerate() {
        let scaled = ((m - rank) as f64 * p_values[i]).min(1.0);
        running_max = running_max.max(scaled);
        adjusted[i] = running_max;
    }
    Ok(adjusted)
}

/// Holm correction. See [`try_holm`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_holm`].
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    try_holm(p_values).unwrap_or_else(|e| panic!("holm: {e}"))
}

/// Benjamini–Hochberg step-up correction, which controls the false discovery
/// rate.
///
/// The `j`-th smallest p-value (1-based) is multiplied by `m / j`, and a running
/// minimum from the largest p-value enforces monotonicity.
///
/// # Errors
///
/// As [`try_bonferroni`].
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_benjamini_hochberg;
/// let adjusted = try_benjamini_hochberg(&[0.01, 0.04, 0.03, 0.005]).unwrap();
/// assert!((adjusted[0] - 0.02).abs() < 1e-12);
/// assert!((adjusted[1] - 0.04).abs() < 1e-12);
/// assert!((adjusted[2] - 0.04).abs() < 1e-12);
/// assert!((adjusted[3] - 0.02).abs() < 1e-12);
/// ```
pub fn try_benjamini_hochberg(p_values: &[f64]) -> Result<Vec<f64>, StatsError> {
    check_p_values(p_values)?;
    let m = p_values.len();
    let mut adjusted = vec![0.0; m];
    let mut running_min: f64 = 1.0;
    for (rank, &i) in ascending_order(p_values).iter().enumerate().rev() {
        let scaled = (m as f64 * p_values[i] / (rank + 1) as f64).min(1.0);
        running_min = running_min.min(scaled);
        adjusted[i] = running_min;
    }
    Ok(adjusted)
}

/// Benjamini–Hochberg correction. See [`try_benjamini_hochberg`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_benjamini_hochberg`].
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    try_benjamini_hochberg(p_values).unwrap_or_else(|e| panic!("benjamini_hochberg: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn bonferroni_scales_and_caps() {
        assert_all_close(
            &try_bonferroni(&[0.01, 0.04, 0.03, 0.5]).unwrap(),
            &[0.04, 0.16, 0.12, 1.0],
        );
    }

    #[test]
    fn holm_is_monotone_step_down() {
        // Sorted: 0.005·4 = 0.02, 0.01·3 = 0.03, 0.03·2 = 0.06, 0.04·1 = 0.04,
        // and the running maximum lifts the last to 0.06.
        assert_all_close(
            &try_holm(&[0.01, 0.04, 0.03, 0.005]).unwrap(),
            &[0.03, 0.06, 0.06, 0.02],
        );
    }

    #[test]
    fn holm_is_never_less_powerful_than_bonferroni() {
        let p = [0.001, 0.02, 0.2, 0.9];
        let holm = try_holm(&p).unwrap();
        let bonf = try_bonferroni(&p).unwrap();
        for (h, b) in holm.iter().zip(&bonf) {
            assert!(h <= b);
        }
    }

    #[test]
    fn benjamini_hochberg_step_up() {
        // Sorted from the top: 0.04·4/4, then 0.03·4/3 = 0.04, 0.01·4/2 = 0.02,
        // 0.005·4 = 0.02 (the running minimum keeps it at 0.02).
        assert_all_close(
            &try_benjamini_hochberg(&[0.01, 0.04, 0.03, 0.005]).unwrap(),
            &[0.02, 0.04, 0.04, 0.02],
        );
    }

    #[test]
    fn adjustments_reject_invalid_p_values() {
        assert!(matches!(
            try_holm(&[0.5, 1.5]),
            Err(StatsError::ProbabilityOutOfRange { .. })
        ));
        assert!(matches!(
            try_bonferroni(&[f64::NAN]),
            Err(StatsError::NotFinite { .. })
        ));
    }

    #[test]
    fn empty_family_is_empty() {
        assert!(try_holm(&[]).unwrap().is_empty());
        assert!(try_benjamini_hochberg(&[]).unwrap().is_empty());
    }
}
