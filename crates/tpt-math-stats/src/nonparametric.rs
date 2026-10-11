//! Rank-based and distribution-free two-sample tests.
//!
//! These tests make no normality assumption. The rank tests use midranks for
//! ties, and their normal approximations include the tie correction. Each
//! returns `(statistic, p_value)`, with the p-value two-sided.
//!
//! The normal approximations drop the continuity correction. They are accurate
//! for moderate sample sizes (roughly 8 or more per group). For very small
//! samples an exact distribution would be more appropriate.

use crate::error::{check_all_finite, check_equal_len, check_min_len, StatsError};
use crate::special::erfc;

/// Midranks of `values`, plus the sizes of every tie group of two or more.
///
/// Tied values share the average of the ranks they occupy, so the ranks always
/// sum to `n(n+1)/2`.
fn midranks(values: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));

    let mut ranks = vec![0.0; values.len()];
    let mut ties = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // Ranks are 1-based: this group occupies ranks start+1 ..= end.
        let midrank = (start + 1 + end) as f64 / 2.0;
        for &k in &order[start..end] {
            ranks[k] = midrank;
        }
        if end - start > 1 {
            ties.push(end - start);
        }
        start = end;
    }
    (ranks, ties)
}

/// `Σ (t³ − t)` over tie groups of size `t`, the variance correction term.
fn tie_term(ties: &[usize]) -> f64 {
    ties.iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum()
}

/// Two-sided p-value of `z` under the standard normal, `2·Φ(-|z|)`.
fn two_sided_normal_p(z: f64) -> f64 {
    erfc(z.abs() / core::f64::consts::SQRT_2).clamp(0.0, 1.0)
}

/// Mann–Whitney U test of two independent samples.
///
/// Returns `(U, p)`, where `U` is the count of pairs `(a_i, b_j)` with
/// `a_i > b_j`, counting ties as one half. The p-value uses the normal
/// approximation with the tie-corrected variance. If every observation is tied
/// the variance is zero and the p-value is `1`.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if either sample is empty, and
/// [`StatsError::NotFinite`] for a non-finite observation.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_mann_whitney_u;
/// // Complete separation: every `a` is below every `b`, so U = 0.
/// let (u, p) = try_mann_whitney_u(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
/// assert_eq!(u, 0.0);
/// assert!(p < 0.1);
/// ```
pub fn try_mann_whitney_u(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    check_min_len("a", a, 1)?;
    check_min_len("b", b, 1)?;
    check_all_finite("a", a)?;
    check_all_finite("b", b)?;

    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let total = n1 + n2;
    let combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let (ranks, ties) = midranks(&combined);

    let rank_sum_a: f64 = ranks[..a.len()].iter().sum();
    let u = rank_sum_a - n1 * (n1 + 1.0) / 2.0;

    let mean = n1 * n2 / 2.0;
    let variance = n1 * n2 / 12.0 * ((total + 1.0) - tie_term(&ties) / (total * (total - 1.0)));
    if variance <= 0.0 {
        return Ok((u, 1.0));
    }
    let z = (u - mean) / variance.sqrt();
    Ok((u, two_sided_normal_p(z)))
}

/// Mann–Whitney U test. See [`try_mann_whitney_u`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_mann_whitney_u`].
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> (f64, f64) {
    try_mann_whitney_u(a, b).unwrap_or_else(|e| panic!("mann_whitney_u: {e}"))
}

/// Wilcoxon signed-rank test for paired samples.
///
/// Pairs with a zero difference are dropped, as is conventional. Returns
/// `(W+, p)`, where `W+` is the sum of the ranks of the positive differences
/// (ranks of `|a_i − b_i|`, with midranks for ties).
///
/// # Errors
///
/// Returns [`StatsError::LengthMismatch`] if the samples differ in length,
/// [`StatsError::NotEnoughData`] if no pair has a non-zero difference, and
/// [`StatsError::NotFinite`] for a non-finite observation.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_wilcoxon_signed_rank;
/// // Every difference is positive, so W+ is the full rank sum 1 + … + 5 = 15.
/// let (w, p) = try_wilcoxon_signed_rank(&[2.0, 4.0, 6.0, 8.0, 10.0], &[1.0, 2.0, 3.0, 4.0, 5.0])
///     .unwrap();
/// assert_eq!(w, 15.0);
/// assert!(p < 0.1);
/// ```
pub fn try_wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    check_equal_len(a.len(), b.len())?;
    check_all_finite("a", a)?;
    check_all_finite("b", b)?;

    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .filter(|d| *d != 0.0)
        .collect();
    if diffs.is_empty() {
        return Err(StatsError::NotEnoughData {
            sample: "non-zero differences",
            required: 1,
            found: 0,
        });
    }

    let n = diffs.len() as f64;
    let magnitudes: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let (ranks, ties) = midranks(&magnitudes);
    let w_plus: f64 = diffs
        .iter()
        .zip(&ranks)
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| r)
        .sum();

    let mean = n * (n + 1.0) / 4.0;
    let variance = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tie_term(&ties) / 48.0;
    if variance <= 0.0 {
        return Ok((w_plus, 1.0));
    }
    let z = (w_plus - mean) / variance.sqrt();
    Ok((w_plus, two_sided_normal_p(z)))
}

/// Wilcoxon signed-rank test. See [`try_wilcoxon_signed_rank`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_wilcoxon_signed_rank`].
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> (f64, f64) {
    try_wilcoxon_signed_rank(a, b).unwrap_or_else(|e| panic!("wilcoxon_signed_rank: {e}"))
}

/// Two-sample Kolmogorov–Smirnov test.
///
/// The statistic `D` is the largest gap between the two empirical CDFs. The
/// p-value uses the asymptotic Kolmogorov distribution with the Stephens
/// correction `λ = (√n_e + 0.12 + 0.11/√n_e)·D`, where `n_e = n₁n₂/(n₁+n₂)`.
///
/// # Errors
///
/// Returns [`StatsError::NotEnoughData`] if either sample is empty, and
/// [`StatsError::NotFinite`] for a non-finite observation.
///
/// # Examples
///
/// ```
/// # use tpt_math_stats::try_kolmogorov_smirnov_two_sample;
/// // The samples do not overlap, so the CDFs differ by 1 at x = 3.
/// let (d, p) = try_kolmogorov_smirnov_two_sample(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
/// assert_eq!(d, 1.0);
/// assert!(p < 0.05);
/// ```
pub fn try_kolmogorov_smirnov_two_sample(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    check_min_len("a", a, 1)?;
    check_min_len("b", b, 1)?;
    check_all_finite("a", a)?;
    check_all_finite("b", b)?;

    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_by(f64::total_cmp);
    sb.sort_by(f64::total_cmp);
    let (n1, n2) = (sa.len(), sb.len());

    // Evaluate both CDFs at every observed point and keep the largest gap.
    let mut d: f64 = 0.0;
    let (mut i, mut j) = (0, 0);
    for &x in sa.iter().chain(sb.iter()) {
        while i < n1 && sa[i] <= x {
            i += 1;
        }
        while j < n2 && sb[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / n1 as f64 - j as f64 / n2 as f64).abs());
    }

    let n_eff = (n1 * n2) as f64 / (n1 + n2) as f64;
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * d;
    Ok((d, kolmogorov_survival(lambda)))
}

/// Kolmogorov survival function `Q(λ) = 2 Σ (−1)^{k−1} e^{−2k²λ²}`.
fn kolmogorov_survival(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut sign = 1.0;
    for k in 1..=200 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * lambda * lambda).exp();
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
        sign = -sign;
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Two-sample Kolmogorov–Smirnov test. See [`try_kolmogorov_smirnov_two_sample`].
///
/// # Panics
///
/// Panics on the conditions listed for [`try_kolmogorov_smirnov_two_sample`].
pub fn kolmogorov_smirnov_two_sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    try_kolmogorov_smirnov_two_sample(a, b)
        .unwrap_or_else(|e| panic!("kolmogorov_smirnov_two_sample: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midranks_average_tied_positions() {
        let (ranks, ties) = midranks(&[10.0, 20.0, 20.0, 5.0]);
        assert_eq!(ranks, vec![2.0, 3.5, 3.5, 1.0]);
        assert_eq!(ties, vec![2]);
    }

    #[test]
    fn mann_whitney_complete_separation() {
        let (u, p) = try_mann_whitney_u(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(u, 0.0);
        // z = (0 − 4.5)/√5.25, p = erfc(|z|/√2).
        let z = 4.5 / 5.25f64.sqrt();
        assert!((p - erfc(z / core::f64::consts::SQRT_2)).abs() < 1e-12);
    }

    #[test]
    fn mann_whitney_counts_ties_as_half() {
        // a = {2, 4}, b = {1, 2, 3}: pairs where a > b are (2,1), (4,1), (4,2),
        // (4,3) = 4 wins, plus the tie (2,2) worth one half.
        let (u, _) = try_mann_whitney_u(&[2.0, 4.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((u - 4.5).abs() < 1e-12);
    }

    #[test]
    fn mann_whitney_all_tied_has_no_evidence() {
        let (_, p) = try_mann_whitney_u(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(p, 1.0);
    }

    #[test]
    fn wilcoxon_full_positive_rank_sum() {
        let (w, p) =
            try_wilcoxon_signed_rank(&[2.0, 4.0, 6.0, 8.0, 10.0], &[1.0, 2.0, 3.0, 4.0, 5.0])
                .unwrap();
        assert_eq!(w, 15.0);
        // mean = 7.5, variance = 5·6·11/24 = 13.75.
        let z = (15.0 - 7.5) / 13.75f64.sqrt();
        assert!((p - erfc(z.abs() / core::f64::consts::SQRT_2)).abs() < 1e-12);
    }

    #[test]
    fn wilcoxon_drops_zero_differences_and_uses_midranks() {
        // Zero difference is dropped, leaving the same test as above.
        let with_zero = try_wilcoxon_signed_rank(
            &[3.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            &[3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(with_zero.0, 15.0);
        // |d| = (1, 1, 2): the two 1s share ranks 1 and 2, so each gets 1.5.
        // W+ = 1.5 (for +1) + 3 (for +2) = 4.5.
        let (w, _) = try_wilcoxon_signed_rank(&[1.0, 0.0, 2.0], &[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(w, 4.5);
    }

    #[test]
    fn wilcoxon_rejects_all_zero_differences() {
        assert!(matches!(
            try_wilcoxon_signed_rank(&[1.0, 2.0], &[1.0, 2.0]),
            Err(StatsError::NotEnoughData { .. })
        ));
    }

    #[test]
    fn ks_statistic_is_exact_for_separated_samples() {
        let (d, p) = try_kolmogorov_smirnov_two_sample(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(d, 1.0);
        // λ = (√1.5 + 0.12 + 0.11/√1.5)·1 ≈ 1.4345, so Q(λ) ≈ 0.0326.
        assert!((p - 0.032_62).abs() < 1e-4, "p = {p}");
    }

    #[test]
    fn ks_identical_samples_have_no_evidence() {
        let x = [0.5, 1.5, 2.5, 3.5];
        let (d, p) = try_kolmogorov_smirnov_two_sample(&x, &x).unwrap();
        assert_eq!(d, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn kolmogorov_survival_reference_points() {
        // Q(1.0) ≈ 0.2700 and Q(1.36) ≈ 0.0495 (the classic 5% critical value).
        assert!((kolmogorov_survival(1.0) - 0.269_999_9).abs() < 1e-4);
        assert!((kolmogorov_survival(1.358) - 0.05).abs() < 2e-3);
    }

    #[test]
    fn rank_tests_reject_empty_and_nonfinite() {
        assert!(matches!(
            try_mann_whitney_u(&[], &[1.0]),
            Err(StatsError::NotEnoughData { .. })
        ));
        assert!(matches!(
            try_kolmogorov_smirnov_two_sample(&[f64::NAN], &[1.0]),
            Err(StatsError::NotFinite { .. })
        ));
        assert!(matches!(
            try_wilcoxon_signed_rank(&[1.0], &[1.0, 2.0]),
            Err(StatsError::LengthMismatch { .. })
        ));
    }
}
