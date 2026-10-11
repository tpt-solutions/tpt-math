//! Convergence diagnostics for MCMC output and self-normalised importance
//! sampling.
//!
//! * [`split_r_hat`] — the split-R̂ potential scale reduction factor across
//!   chains. Values near 1 suggest the chains have mixed.
//! * [`effective_sample_size`] — the number of independent draws a correlated
//!   chain is worth, from Geyer's initial positive sequence estimator.
//! * [`importance_estimate`] — a self-normalised importance-sampling mean with
//!   its effective sample size.
//!
//! Every function returns `None` when the input is too short or degenerate to
//! give a meaningful answer, rather than a `NaN`.

/// Split-R̂ over one or more chains of equal or unequal length.
///
/// Each chain is split in half, so a chain that drifts is detected even when it
/// is the only chain supplied. Returns `None` unless there are at least two
/// draws per split half (four draws per chain), and `None` if any draw is not
/// finite.
///
/// A degenerate case with zero within-chain variance returns `Some(1.0)` when
/// the chains agree exactly and `Some(f64::INFINITY)` when they do not.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::diagnostics::split_r_hat;
///
/// // Two chains with the same distribution mix well.
/// let a: Vec<f64> = (0..200).map(|i| ((i * 37) % 100) as f64).collect();
/// let b: Vec<f64> = (0..200).map(|i| ((i * 53) % 100) as f64).collect();
/// let r = split_r_hat(&[&a, &b]).unwrap();
/// assert!((r - 1.0).abs() < 0.05);
/// ```
pub fn split_r_hat(chains: &[&[f64]]) -> Option<f64> {
    if chains.is_empty() || chains.iter().any(|c| c.iter().any(|x| !x.is_finite())) {
        return None;
    }
    // Split each chain into two halves, dropping a trailing odd draw.
    let half = chains.iter().map(|c| c.len() / 2).min()?;
    if half < 2 {
        return None;
    }
    let mut halves: Vec<&[f64]> = Vec::with_capacity(2 * chains.len());
    for chain in chains {
        halves.push(&chain[..half]);
        halves.push(&chain[chain.len() - half..]);
    }

    let m = halves.len() as f64;
    let n = half as f64;
    let means: Vec<f64> = halves.iter().map(|h| mean(h)).collect();
    let grand = means.iter().sum::<f64>() / m;
    let between = n / (m - 1.0) * means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>();
    let within = halves
        .iter()
        .zip(&means)
        .map(|(h, mu)| h.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>()
        / m;

    if within == 0.0 {
        return Some(if between == 0.0 { 1.0 } else { f64::INFINITY });
    }
    let var_plus = (n - 1.0) / n * within + between / n;
    Some((var_plus / within).sqrt())
}

/// Effective sample size of one chain, `n / τ`, where `τ` is the integrated
/// autocorrelation time.
///
/// `τ` is estimated with Geyer's initial positive sequence: autocorrelations
/// are summed in adjacent pairs until a pair's sum turns non-positive. An
/// anticorrelated chain can have `ESS > n`.
///
/// Returns `None` for fewer than four draws, a non-finite draw, or a constant
/// chain (zero variance), since its autocorrelation is undefined.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::diagnostics::effective_sample_size;
///
/// // A steadily trending chain is almost perfectly autocorrelated.
/// let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
/// assert!(effective_sample_size(&x).unwrap() < 20.0);
/// ```
pub fn effective_sample_size(chain: &[f64]) -> Option<f64> {
    let n = chain.len();
    if n < 4 || chain.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mu = mean(chain);
    let centred: Vec<f64> = chain.iter().map(|x| x - mu).collect();
    let gamma0: f64 = centred.iter().map(|x| x * x).sum::<f64>() / n as f64;
    if gamma0 == 0.0 {
        return None;
    }

    // Biased autocorrelation at lag t.
    let rho = |t: usize| -> f64 {
        let s: f64 = (0..n - t).map(|i| centred[i] * centred[i + t]).sum();
        s / n as f64 / gamma0
    };

    let mut tau = -1.0;
    let mut lag = 0;
    while lag + 1 < n {
        let pair = rho(lag) + rho(lag + 1);
        if pair <= 0.0 {
            break;
        }
        tau += 2.0 * pair;
        lag += 2;
    }
    Some(n as f64 / tau.max(f64::MIN_POSITIVE))
}

/// Self-normalised importance-sampling estimate of `E[f(X)]`.
///
/// `values[i]` is `f(x_i)` and `log_weights[i]` is the log of the unnormalised
/// importance weight `p(x_i)/q(x_i)`. Weights are normalised with the maximum
/// subtracted first, so large log-weights do not overflow. Returns
/// `(estimate, effective_sample_size)`, where the ESS is `(Σw)² / Σw²`.
///
/// Returns `None` for mismatched lengths, empty input, a non-finite value, or a
/// log-weight that is `NaN` or `+∞`. Weights of `−∞` are allowed and contribute
/// nothing. If every weight is `−∞`, returns `None`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::diagnostics::importance_estimate;
///
/// // Equal weights reduce to the plain mean, with ESS = n.
/// let (est, ess) = importance_estimate(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]).unwrap();
/// assert!((est - 2.0).abs() < 1e-12);
/// assert!((ess - 3.0).abs() < 1e-12);
/// ```
pub fn importance_estimate(values: &[f64], log_weights: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() || values.len() != log_weights.len() {
        return None;
    }
    if values.iter().any(|v| !v.is_finite())
        || log_weights
            .iter()
            .any(|w| w.is_nan() || *w == f64::INFINITY)
    {
        return None;
    }
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return None;
    }
    let weights: Vec<f64> = log_weights.iter().map(|lw| (lw - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let weighted: f64 = weights.iter().zip(values).map(|(w, v)| w * v).sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    Some((weighted / total, total * total / sum_sq))
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::{Rng, SplitMix64};

    /// Deterministic uniform draws on `[-0.5, 0.5)`.
    fn uniform_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = SplitMix64::seed_from_u64(seed);
        (0..n).map(|_| rng.next_f64() - 0.5).collect()
    }

    fn ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let noise = uniform_noise(n, seed);
        let mut x = vec![0.0; n];
        for t in 1..n {
            x[t] = phi * x[t - 1] + noise[t];
        }
        x
    }

    #[test]
    fn iid_chain_has_ess_near_n() {
        let x = uniform_noise(4000, 11);
        let ess = effective_sample_size(&x).unwrap();
        assert!(ess > 0.7 * 4000.0 && ess < 1.3 * 4000.0, "ess {ess}");
    }

    #[test]
    fn ar1_chain_matches_theoretical_ess() {
        // τ = (1 + φ)/(1 − φ), so ESS ≈ n(1 − φ)/(1 + φ) = 20000/19 ≈ 1053.
        let x = ar1(20_000, 0.9, 5);
        let ess = effective_sample_size(&x).unwrap();
        assert!(ess > 700.0 && ess < 1500.0, "ess {ess}");
    }

    #[test]
    fn well_mixed_chains_have_r_hat_near_one() {
        let a = uniform_noise(2000, 1);
        let b = uniform_noise(2000, 2);
        let r = split_r_hat(&[&a, &b]).unwrap();
        assert!((r - 1.0).abs() < 0.02, "r_hat {r}");
    }

    #[test]
    fn separated_chains_have_large_r_hat() {
        let a = uniform_noise(1000, 3);
        let b: Vec<f64> = uniform_noise(1000, 4).iter().map(|x| x + 5.0).collect();
        let r = split_r_hat(&[&a, &b]).unwrap();
        assert!(r > 2.0, "r_hat {r}");
    }

    #[test]
    fn split_detects_a_single_drifting_chain() {
        // One chain whose first half and second half have different means.
        let mut x = uniform_noise(1000, 6);
        for v in &mut x[500..] {
            *v += 4.0;
        }
        let r = split_r_hat(&[&x]).unwrap();
        assert!(r > 2.0, "r_hat {r}");
    }

    #[test]
    fn degenerate_inputs_return_none_or_documented_values() {
        assert_eq!(split_r_hat(&[&[1.0, 2.0, 3.0]]), None);
        assert_eq!(split_r_hat(&[&[1.0, f64::NAN, 3.0, 4.0]]), None);
        assert_eq!(split_r_hat(&[&[2.0; 8], &[2.0; 8]]), Some(1.0));
        assert_eq!(split_r_hat(&[&[2.0; 8], &[3.0; 8]]), Some(f64::INFINITY));
        assert_eq!(effective_sample_size(&[1.0; 10]), None);
        assert_eq!(effective_sample_size(&[1.0, 2.0]), None);
    }

    #[test]
    fn importance_equal_weights_is_plain_mean() {
        let (est, ess) = importance_estimate(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4]).unwrap();
        assert!((est - 2.5).abs() < 1e-12);
        assert!((ess - 4.0).abs() < 1e-12);
    }

    #[test]
    fn importance_dominant_weight_collapses_ess() {
        let (est, ess) = importance_estimate(&[10.0, 20.0, 30.0], &[0.0, -50.0, -50.0]).unwrap();
        assert!((est - 10.0).abs() < 1e-9);
        assert!((ess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn importance_handles_large_log_weights_without_overflow() {
        let (est, _) = importance_estimate(&[1.0, 3.0], &[1000.0, 1000.0]).unwrap();
        assert!((est - 2.0).abs() < 1e-12);
    }

    #[test]
    fn importance_rejects_bad_input() {
        assert_eq!(importance_estimate(&[], &[]), None);
        assert_eq!(importance_estimate(&[1.0], &[0.0, 0.0]), None);
        assert_eq!(importance_estimate(&[1.0], &[f64::NAN]), None);
        assert_eq!(importance_estimate(&[1.0], &[f64::NEG_INFINITY]), None);
    }
}
