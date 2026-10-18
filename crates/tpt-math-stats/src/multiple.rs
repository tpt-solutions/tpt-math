//! Multiple linear regression by least squares.

use tpt_math_linalg_dense::{DMatrix, DVector, Qr};

use crate::error::{check_all_finite, check_equal_len, StatsError};

/// The result of an ordinary least-squares fit with an intercept.
#[derive(Clone, Debug, PartialEq)]
pub struct MultipleFit {
    /// Coefficients in order: the intercept, then one per predictor.
    pub coefficients: Vec<f64>,
    /// Residual sum of squares, `Σ (yᵢ − ŷᵢ)²`.
    pub residual_sum_of_squares: f64,
    /// Coefficient of determination `1 − RSS / TSS`. `NaN` when `y` is constant.
    pub r_squared: f64,
}

/// Fit `y ≈ β₀ + β₁ x₁ + … + β_p x_p` by least squares.
///
/// `rows[i]` holds the `p` predictor values of observation `i`. The intercept is
/// added automatically. The fit uses Householder QR, so it is more accurate than
/// the normal equations when predictors are correlated.
///
/// # Errors
///
/// * [`StatsError::LengthMismatch`] if `rows` and `y` differ in length.
/// * [`StatsError::NotEnoughData`] if there are not more observations than
///   coefficients (`p + 1`).
/// * [`StatsError::NotFinite`] for any non-finite value.
/// * [`StatsError::RankDeficientDesign`] if a predictor is a linear combination of
///   the others (or the intercept), so the coefficients are not identified.
///
/// # Examples
///
/// ```
/// use tpt_math_stats::try_multiple_regression;
///
/// // y = 1 + 2 x₁ − x₂ exactly.
/// let rows = [[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = [0.0, 3.0, 2.0, 6.0, 5.0];
/// let fit = try_multiple_regression(&rows.map(|r| r.to_vec()), &y).unwrap();
/// assert!((fit.coefficients[0] - 1.0).abs() < 1e-10);
/// assert!((fit.coefficients[1] - 2.0).abs() < 1e-10);
/// assert!((fit.coefficients[2] + 1.0).abs() < 1e-10);
/// ```
pub fn try_multiple_regression(rows: &[Vec<f64>], y: &[f64]) -> Result<MultipleFit, StatsError> {
    check_equal_len(rows.len(), y.len())?;
    let p = rows.first().map_or(0, Vec::len);
    let k = p + 1;
    if rows.len() < k + 1 {
        return Err(StatsError::NotEnoughData {
            sample: "rows",
            required: k + 1,
            found: rows.len(),
        });
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != p {
            return Err(StatsError::LengthMismatch {
                first: p,
                second: row.len(),
            });
        }
        for &v in row {
            if !v.is_finite() {
                return Err(StatsError::NotFinite {
                    sample: "rows",
                    index: i,
                    value: v,
                });
            }
        }
    }
    check_all_finite("y", y)?;

    let design = DMatrix::from_fn(
        rows.len(),
        k,
        |i, j| if j == 0 { 1.0 } else { rows[i][j - 1] },
    );
    // Any failure from QR here means the design lost full column rank.
    let rhs = DVector::from_vec(y.to_vec());
    let beta = Qr::new(&design)
        .and_then(|qr| qr.solve_least_squares(&rhs))
        .map_err(|_| StatsError::RankDeficientDesign)?;
    let coefficients: Vec<f64> = beta.iter().copied().collect();

    let n = y.len() as f64;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut rss = 0.0;
    let mut tss = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        let fitted = coefficients[0]
            + rows[i]
                .iter()
                .zip(&coefficients[1..])
                .map(|(x, b)| x * b)
                .sum::<f64>();
        rss += (yi - fitted) * (yi - fitted);
        tss += (yi - mean_y) * (yi - mean_y);
    }
    let r_squared = if tss == 0.0 {
        f64::NAN
    } else {
        1.0 - rss / tss
    };
    Ok(MultipleFit {
        coefficients,
        residual_sum_of_squares: rss,
        r_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fit_has_zero_residuals_and_unit_r_squared() {
        let rows: Vec<Vec<f64>> = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![2.0, 3.0],
            vec![3.0, 1.0],
            vec![4.0, 4.0],
        ];
        let y = [0.0, 3.0, 2.0, 6.0, 5.0];
        let fit = try_multiple_regression(&rows, &y).unwrap();
        assert!(fit.residual_sum_of_squares < 1e-18);
        assert!((fit.r_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn noisy_fit_matches_normal_equations_residual_orthogonality() {
        let rows: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![i as f64, (i * i % 7) as f64])
            .collect();
        let y: Vec<f64> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| 0.5 + 1.5 * r[0] - 0.7 * r[1] + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let fit = try_multiple_regression(&rows, &y).unwrap();
        // At the least-squares optimum the residuals are orthogonal to each column.
        let resid: Vec<f64> = rows
            .iter()
            .zip(&y)
            .map(|(r, yi)| {
                yi - (fit.coefficients[0] + r[0] * fit.coefficients[1] + r[1] * fit.coefficients[2])
            })
            .collect();
        let dot_one: f64 = resid.iter().sum();
        let dot_x1: f64 = resid.iter().zip(&rows).map(|(e, r)| e * r[0]).sum();
        let dot_x2: f64 = resid.iter().zip(&rows).map(|(e, r)| e * r[1]).sum();
        assert!(dot_one.abs() < 1e-9 && dot_x1.abs() < 1e-9 && dot_x2.abs() < 1e-9);
        assert!(fit.r_squared > 0.9 && fit.r_squared < 1.0);
    }

    #[test]
    fn collinear_predictors_are_reported() {
        let rows: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 2.0 * i as f64]).collect();
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            try_multiple_regression(&rows, &y).unwrap_err(),
            StatsError::RankDeficientDesign
        );
    }

    #[test]
    fn bad_input_is_rejected() {
        let rows = vec![vec![1.0], vec![2.0]];
        assert!(matches!(
            try_multiple_regression(&rows, &[1.0, 2.0]),
            Err(StatsError::NotEnoughData { .. })
        ));
        let rows: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        assert!(matches!(
            try_multiple_regression(&rows, &[1.0; 4]),
            Err(StatsError::LengthMismatch { .. })
        ));
    }
}

/// The result of a logistic regression fit.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticFit {
    /// Coefficients in order: the intercept, then one per predictor.
    pub coefficients: Vec<f64>,
    /// Newton iterations used.
    pub iterations: usize,
}

/// Iteration budget for [`try_logistic_regression`].
const LOGISTIC_MAX_ITER: usize = 50;

/// Fit `P(y = 1) = σ(β₀ + β·x)` by maximum likelihood, using Newton's method
/// (iteratively reweighted least squares).
///
/// `y` must hold `0` or `1`. The intercept is added automatically.
///
/// # Errors
///
/// * [`StatsError::LengthMismatch`] and [`StatsError::NotEnoughData`] as for
///   [`try_multiple_regression`].
/// * [`StatsError::NotFinite`] for a non-finite predictor, or for a response that
///   is not exactly `0` or `1`.
/// * [`StatsError::RankDeficientDesign`] if the Newton system is singular.
/// * [`StatsError::DidNotConverge`] if the coefficients do not settle within
///   50 iterations. This is the usual symptom of perfectly separated data, where
///   the maximum-likelihood estimate does not exist.
///
/// # Examples
///
/// ```
/// use tpt_math_stats::try_logistic_regression;
///
/// let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 / 10.0 - 1.0]).collect();
/// let y: Vec<f64> = (0..20).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
/// let fit = try_logistic_regression(&x, &y).unwrap();
/// assert_eq!(fit.coefficients.len(), 2);
/// ```
pub fn try_logistic_regression(rows: &[Vec<f64>], y: &[f64]) -> Result<LogisticFit, StatsError> {
    // Reuse the shape and finiteness checks of the linear fit.
    let zero_one_y: Vec<f64> = y
        .iter()
        .map(|&v| if v == 1.0 { 1.0 } else { 0.0 })
        .collect();
    if let Some(i) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
        return Err(StatsError::NotFinite {
            sample: "y",
            index: i,
            value: y[i],
        });
    }
    let p = rows.first().map_or(0, Vec::len);
    let k = p + 1;
    check_equal_len(rows.len(), y.len())?;
    if rows.len() < k + 1 {
        return Err(StatsError::NotEnoughData {
            sample: "rows",
            required: k + 1,
            found: rows.len(),
        });
    }
    for row in rows {
        if row.len() != p {
            return Err(StatsError::LengthMismatch {
                first: p,
                second: row.len(),
            });
        }
        check_all_finite("rows", row)?;
    }

    let mut beta = vec![0.0; k];
    let x_at = |i: usize, j: usize| if j == 0 { 1.0 } else { rows[i][j - 1] };
    for iter in 1..=LOGISTIC_MAX_ITER {
        let mu: Vec<f64> = (0..rows.len())
            .map(|i| {
                let eta: f64 = (0..k).map(|j| x_at(i, j) * beta[j]).sum();
                1.0 / (1.0 + (-eta).exp())
            })
            .collect();
        // Newton step: (XᵀWX) δ = Xᵀ(y − μ), with W = diag(μ(1 − μ)).
        let hessian = DMatrix::from_fn(k, k, |a, b| {
            (0..rows.len())
                .map(|i| x_at(i, a) * x_at(i, b) * mu[i] * (1.0 - mu[i]))
                .sum()
        });
        let gradient = DVector::from_fn(k, |a| {
            (0..rows.len())
                .map(|i| x_at(i, a) * (zero_one_y[i] - mu[i]))
                .sum()
        });
        let delta = hessian
            .solve(&gradient)
            .map_err(|_| StatsError::RankDeficientDesign)?;
        let mut step = 0.0_f64;
        for j in 0..k {
            beta[j] += delta[j];
            step = step.max(delta[j].abs());
        }
        if !step.is_finite() {
            break;
        }
        if step < 1e-10 {
            return Ok(LogisticFit {
                coefficients: beta,
                iterations: iter,
            });
        }
    }
    Err(StatsError::DidNotConverge {
        iterations: LOGISTIC_MAX_ITER,
    })
}

#[cfg(test)]
mod logistic_tests {
    use super::*;

    #[test]
    fn recovers_known_coefficients_from_a_balanced_design() {
        // Deterministic design with both classes at every predictor level, so the
        // maximum-likelihood estimate exists. Probabilities come from β = (−0.5, 1.2).
        let xs: Vec<f64> = (0..40).map(|i| -2.0 + 4.0 * i as f64 / 39.0).collect();
        let rows: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        // Expected-value labels: y = p(x) rounded through a fixed threshold pattern
        // gives a noisy but non-separable sample.
        let y: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let p = 1.0 / (1.0 + (0.5 - 1.2 * x).exp());
                let u = ((i * 37 % 40) as f64 + 0.5) / 40.0;
                if u < p {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let fit = try_logistic_regression(&rows, &y).unwrap();
        assert!(fit.iterations <= 50);
        // The score equations hold at the optimum: Σ(y − μ) = 0 and Σx(y − μ) = 0.
        let mu: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 / (1.0 + (-(fit.coefficients[0] + fit.coefficients[1] * x)).exp()))
            .collect();
        let s0: f64 = y.iter().zip(&mu).map(|(yi, m)| yi - m).sum();
        let s1: f64 = xs
            .iter()
            .zip(y.iter().zip(&mu))
            .map(|(x, (yi, m))| x * (yi - m))
            .sum();
        assert!(s0.abs() < 1e-6 && s1.abs() < 1e-6, "scores {s0}, {s1}");
    }

    #[test]
    fn separated_data_reports_non_convergence_or_rank_loss() {
        let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        assert!(matches!(
            try_logistic_regression(&rows, &y),
            Err(StatsError::DidNotConverge { .. }) | Err(StatsError::RankDeficientDesign)
        ));
    }

    #[test]
    fn response_must_be_zero_or_one() {
        let rows: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
        assert!(matches!(
            try_logistic_regression(&rows, &[0.0, 1.0, 0.5, 1.0, 0.0, 1.0]),
            Err(StatsError::NotFinite {
                sample: "y",
                index: 2,
                ..
            })
        ));
    }
}
