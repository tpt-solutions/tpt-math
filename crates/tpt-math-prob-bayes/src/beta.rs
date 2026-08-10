//! The Beta distribution: the conjugate prior for a Bernoulli/Binomial rate.

use tpt_math_prob_core::{Distribution, Rng};

use crate::gamma::sample_standard_gamma;
use crate::likelihood::{bernoulli_log_likelihood, binomial_log_likelihood};
use crate::special::{ln_beta, ln_binomial, regularized_incomplete_beta, xlogy};
use crate::ParameterError;

/// The Beta distribution `Beta(α, β)` on the closed unit interval.
///
/// The density is
///
/// ```text
/// p(x) = x^(α-1) (1-x)^(β-1) / B(α, β),   0 ≤ x ≤ 1
/// ```
///
/// `Beta` is the conjugate prior for the success probability of a
/// Bernoulli/Binomial likelihood: observing `s` successes and `f` failures
/// turns `Beta(α, β)` into `Beta(α + s, β + f)` — see [`Beta::update`].
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::Beta;
///
/// let prior = Beta::uniform();              // Beta(1, 1)
/// let posterior = prior.update(7, 3);       // 7 heads, 3 tails
/// assert_eq!((posterior.alpha(), posterior.beta()), (8.0, 4.0));
/// assert!((posterior.mean() - 8.0 / 12.0).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    /// Create `Beta(alpha, beta)`.
    ///
    /// # Panics
    ///
    /// Panics unless both parameters are finite and strictly positive. Use
    /// [`Beta::try_new`] for a non-panicking version.
    #[must_use]
    pub fn new(alpha: f64, beta: f64) -> Self {
        match Self::try_new(alpha, beta) {
            Ok(dist) => dist,
            Err(err) => panic!("invalid Beta parameters: {err}"),
        }
    }

    /// Fallible constructor for `Beta(alpha, beta)`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] if `alpha` or `beta` is not finite and
    /// strictly positive.
    pub fn try_new(alpha: f64, beta: f64) -> Result<Self, ParameterError> {
        crate::check_positive("alpha", alpha)?;
        crate::check_positive("beta", beta)?;
        Ok(Beta { alpha, beta })
    }

    /// The uniform prior `Beta(1, 1)`.
    #[must_use]
    pub fn uniform() -> Self {
        Beta {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Jeffreys' prior `Beta(1/2, 1/2)`.
    #[must_use]
    pub fn jeffreys() -> Self {
        Beta {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Haldane's improper-limit prior approximated by `Beta(ε, ε)`.
    ///
    /// Uses a small but strictly positive `ε` so the distribution stays valid.
    #[must_use]
    pub fn haldane() -> Self {
        Beta {
            alpha: 1e-3,
            beta: 1e-3,
        }
    }

    /// The `α` (pseudo-successes) parameter.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The `β` (pseudo-failures) parameter.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The concentration `α + β`, i.e. the effective sample size.
    #[must_use]
    pub fn concentration(&self) -> f64 {
        self.alpha + self.beta
    }

    /// The mean `α / (α + β)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    ///
    /// assert!((Beta::new(2.0, 2.0).mean() - 0.5).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.alpha / self.concentration()
    }

    /// The variance `αβ / ((α + β)² (α + β + 1))`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let s = self.concentration();
        self.alpha * self.beta / (s * s * (s + 1.0))
    }

    /// The standard deviation.
    #[must_use]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// The mode `(α − 1) / (α + β − 2)`, which exists only for `α, β > 1`.
    #[must_use]
    pub fn mode(&self) -> Option<f64> {
        (self.alpha > 1.0 && self.beta > 1.0)
            .then(|| (self.alpha - 1.0) / (self.concentration() - 2.0))
    }

    /// The log-density at `x`; `-∞` outside `[0, 1]`.
    ///
    /// Boundary terms use the `0 · ln 0 = 0` convention, so `Beta(1, 1)` has a
    /// finite log-density at both endpoints.
    #[must_use]
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        xlogy(self.alpha - 1.0, x) + xlogy(self.beta - 1.0, 1.0 - x)
            - ln_beta(self.alpha, self.beta)
    }

    /// The probability density at `x`; zero outside `[0, 1]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    ///
    /// // Beta(2, 2) has density 6x(1-x).
    /// let d = Beta::new(2.0, 2.0);
    /// assert!((d.pdf(0.5) - 1.5).abs() < 1e-12);
    /// assert_eq!(d.pdf(1.5), 0.0);
    /// ```
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    /// The CDF at `x`, i.e. the regularized incomplete beta `I_x(α, β)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        regularized_incomplete_beta(self.alpha, self.beta, x)
    }

    /// The inverse CDF at probability `p ∈ [0, 1]`, found by bisection.
    ///
    /// Accurate to roughly 1e-12; returns `0.0`/`1.0` at the endpoints and
    /// `NaN` for `p` outside `[0, 1]`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if p == 0.0 {
            return 0.0;
        }
        if p == 1.0 {
            return 1.0;
        }
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-15 {
                break;
            }
        }
        0.5 * (lo + hi)
    }

    /// The equal-tailed credible interval containing probability `mass`.
    ///
    /// # Panics
    ///
    /// Panics unless `mass` lies in `(0, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    ///
    /// let (lo, hi) = Beta::uniform().update(60, 40).credible_interval(0.95);
    /// assert!(lo < 0.6 && 0.6 < hi);
    /// ```
    #[must_use]
    pub fn credible_interval(&self, mass: f64) -> (f64, f64) {
        assert!(
            mass > 0.0 && mass < 1.0 && mass.is_finite(),
            "credible interval mass must lie in (0, 1), got {mass}"
        );
        let tail = 0.5 * (1.0 - mass);
        (self.quantile(tail), self.quantile(1.0 - tail))
    }

    /// Draw one sample in `[0, 1]`.
    ///
    /// Uses two independent Gamma variates: if `G₁ ~ Gamma(α, 1)` and
    /// `G₂ ~ Gamma(β, 1)` then `G₁ / (G₁ + G₂) ~ Beta(α, β)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    /// use tpt_math_prob_core::SplitMix64;
    ///
    /// let mut rng = SplitMix64::seed_from_u64(7);
    /// let x = Beta::new(2.0, 5.0).sample(&mut rng);
    /// assert!((0.0..=1.0).contains(&x));
    /// ```
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.draw(rng)
    }

    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // A handful of retries covers the astronomically unlikely case where
        // both gamma draws underflow to zero for very small α and β.
        for _ in 0..16 {
            let g1 = sample_standard_gamma(self.alpha, rng);
            let g2 = sample_standard_gamma(self.beta, rng);
            let total = g1 + g2;
            if total > 0.0 && total.is_finite() {
                let x = g1 / total;
                if x.is_finite() {
                    return x.clamp(0.0, 1.0);
                }
            }
        }
        // Limiting behaviour as α, β → 0: mass collapses onto {0, 1}.
        if rng.next_f64() < self.mean() {
            1.0
        } else {
            0.0
        }
    }

    /// Conjugate update against a Bernoulli/Binomial likelihood.
    ///
    /// Returns the posterior `Beta(α + successes, β + failures)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    ///
    /// let posterior = Beta::new(2.0, 2.0).update(10, 5);
    /// assert_eq!((posterior.alpha(), posterior.beta()), (12.0, 7.0));
    /// ```
    #[must_use]
    pub fn update(&self, successes: u64, failures: u64) -> Beta {
        Beta {
            alpha: self.alpha + successes as f64,
            beta: self.beta + failures as f64,
        }
    }

    /// Conjugate update from a slice of Bernoulli trials.
    #[must_use]
    pub fn update_bernoulli(&self, observations: &[bool]) -> Beta {
        let successes = observations.iter().filter(|hit| **hit).count() as u64;
        self.update(successes, observations.len() as u64 - successes)
    }

    /// Conjugate update from a single Bernoulli trial.
    #[must_use]
    pub fn update_one(&self, success: bool) -> Beta {
        if success {
            self.update(1, 0)
        } else {
            self.update(0, 1)
        }
    }

    /// Conjugate update from a binomial observation of `successes` out of
    /// `trials`.
    ///
    /// # Panics
    ///
    /// Panics if `successes > trials`.
    #[must_use]
    pub fn update_binomial(&self, successes: u64, trials: u64) -> Beta {
        assert!(
            successes <= trials,
            "successes ({successes}) cannot exceed trials ({trials})"
        );
        self.update(successes, trials - successes)
    }

    /// Probability that the next Bernoulli trial is a success under the
    /// posterior predictive, which equals the mean.
    #[must_use]
    pub fn predictive_success_probability(&self) -> f64 {
        self.mean()
    }

    /// Log-likelihood of Bernoulli data with success probability `p`.
    ///
    /// A convenience alias for [`bernoulli_log_likelihood`]; it does not
    /// depend on the prior, hence the associated-function form.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    ///
    /// // Two successes and one failure at p = 0.5: ln(0.125).
    /// assert!((Beta::log_likelihood(0.5, 2, 1) - 0.125f64.ln()).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn log_likelihood(p: f64, successes: u64, failures: u64) -> f64 {
        bernoulli_log_likelihood(p, successes, failures)
    }

    /// Log-likelihood of a binomial count, including the `C(n, k)` term.
    #[must_use]
    pub fn log_likelihood_binomial(p: f64, successes: u64, trials: u64) -> f64 {
        binomial_log_likelihood(p, successes, trials)
    }

    /// Unnormalised log-posterior `ln p(x) + ln L(data | x)` at `x`.
    ///
    /// Handy as a target for [`Metropolis`](crate::Metropolis) when checking a
    /// sampler against a distribution with a known closed form.
    #[must_use]
    pub fn log_unnormalized_posterior(&self, x: f64, successes: u64, failures: u64) -> f64 {
        self.log_pdf(x) + bernoulli_log_likelihood(x, successes, failures)
    }

    /// Log marginal likelihood (evidence) of a binomial observation.
    ///
    /// `ln p(s successes in s+f trials) = ln C(n, s) + ln B(α+s, β+f) − ln B(α, β)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Beta;
    ///
    /// // Under a uniform prior every count in 0..=n is equally likely.
    /// let evidence = Beta::uniform().log_marginal_likelihood(3, 7);
    /// assert!((evidence - (1.0f64 / 11.0).ln()).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn log_marginal_likelihood(&self, successes: u64, failures: u64) -> f64 {
        let posterior = self.update(successes, failures);
        ln_binomial(successes + failures, successes) + ln_beta(posterior.alpha, posterior.beta)
            - ln_beta(self.alpha, self.beta)
    }
}

impl Default for Beta {
    fn default() -> Self {
        Beta::uniform()
    }
}

impl Distribution<f64> for Beta {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.draw(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::SplitMix64;

    #[test]
    fn beta_2_2_has_mean_one_half() {
        let d = Beta::new(2.0, 2.0);
        assert!((d.mean() - 0.5).abs() < 1e-15);
        assert!((d.variance() - 0.05).abs() < 1e-15);
        assert!((d.mode().unwrap() - 0.5).abs() < 1e-15);
        assert!((d.std() - 0.05f64.sqrt()).abs() < 1e-15);
        assert_eq!(d.concentration(), 4.0);
    }

    #[test]
    fn moments_of_asymmetric_beta() {
        let d = Beta::new(2.0, 5.0);
        assert!((d.mean() - 2.0 / 7.0).abs() < 1e-15);
        assert!((d.variance() - (10.0 / (49.0 * 8.0))).abs() < 1e-15);
        assert!((d.mode().unwrap() - 0.2).abs() < 1e-15);
        assert_eq!(Beta::jeffreys().mode(), None);
        assert_eq!(Beta::uniform().mode(), None);
    }

    #[test]
    fn pdf_matches_closed_form() {
        let d = Beta::new(2.0, 2.0);
        for &x in &[0.0, 0.1, 0.5, 0.9, 1.0] {
            let expected = 6.0 * x * (1.0 - x);
            assert!((d.pdf(x) - expected).abs() < 1e-12, "x = {x}");
        }
        assert_eq!(d.pdf(-0.1), 0.0);
        assert_eq!(d.pdf(1.1), 0.0);
        assert!(d.log_pdf(2.0).is_infinite());
    }

    #[test]
    fn uniform_pdf_is_flat_including_endpoints() {
        let d = Beta::uniform();
        for &x in &[0.0, 0.25, 0.5, 1.0] {
            assert!((d.pdf(x) - 1.0).abs() < 1e-14, "x = {x}");
        }
        // α < 1 blows up at the left endpoint.
        assert!(Beta::jeffreys().pdf(0.0).is_infinite());
    }

    #[test]
    fn pdf_integrates_to_one() {
        let d = Beta::new(3.0, 5.0);
        let steps = 200_000;
        let h = 1.0 / steps as f64;
        let mut total = 0.5 * (d.pdf(0.0) + d.pdf(1.0));
        for i in 1..steps {
            total += d.pdf(i as f64 * h);
        }
        assert!((total * h - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cdf_and_quantile_round_trip() {
        let d = Beta::new(2.0, 3.0);
        assert!((d.cdf(0.5) - 0.6875).abs() < 1e-12);
        assert_eq!(d.cdf(0.0), 0.0);
        assert_eq!(d.cdf(1.0), 1.0);
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let x = d.quantile(p);
            assert!((d.cdf(x) - p).abs() < 1e-10, "p = {p}");
        }
        assert_eq!(Beta::uniform().quantile(0.0), 0.0);
        assert_eq!(Beta::uniform().quantile(1.0), 1.0);
        assert!(Beta::uniform().quantile(1.5).is_nan());
        // Beta(1, 1) is uniform, so its quantile is the identity.
        assert!((Beta::uniform().quantile(0.37) - 0.37).abs() < 1e-12);
    }

    #[test]
    fn credible_interval_brackets_the_mean() {
        let d = Beta::uniform().update(60, 40);
        let (lo, hi) = d.credible_interval(0.95);
        assert!(lo < d.mean() && d.mean() < hi);
        assert!((d.cdf(hi) - d.cdf(lo) - 0.95).abs() < 1e-9);
    }

    #[test]
    fn conjugate_update_adds_counts() {
        let prior = Beta::new(2.0, 3.0);
        let posterior = prior.update(7, 5);
        assert_eq!(posterior, Beta::new(9.0, 8.0));
        assert_eq!(prior.update(0, 0), prior);
        assert_eq!(prior.update_binomial(7, 12), posterior);
        assert_eq!(prior.update_one(true), Beta::new(3.0, 3.0));
        assert_eq!(prior.update_one(false), Beta::new(2.0, 4.0));
        assert_eq!(
            prior.update_bernoulli(&[true, false, true, true]),
            Beta::new(5.0, 4.0)
        );
    }

    #[test]
    fn update_is_order_independent_and_sequential() {
        let prior = Beta::jeffreys();
        let batch = prior.update(3, 2);
        let sequential = prior
            .update_one(true)
            .update_one(false)
            .update_one(true)
            .update_one(true)
            .update_one(false);
        assert_eq!(batch, sequential);
    }

    #[test]
    fn posterior_concentrates_on_the_true_rate() {
        let posterior = Beta::uniform().update(700, 300);
        assert!((posterior.mean() - 0.7).abs() < 1e-3);
        assert!(posterior.std() < 0.02);
        assert!((posterior.predictive_success_probability() - posterior.mean()).abs() < 1e-15);
    }

    #[test]
    fn sampling_recovers_moments() {
        let mut rng = SplitMix64::seed_from_u64(31_337);
        for &(a, b) in &[(2.0, 2.0), (0.7, 3.0), (8.0, 2.0)] {
            let d = Beta::new(a, b);
            let n = 60_000;
            let xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
            assert!(xs.iter().all(|x| (0.0..=1.0).contains(x)));
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0);
            assert!((mean - d.mean()).abs() < 0.01, "Beta({a}, {b}) mean {mean}");
            assert!(
                (var - d.variance()).abs() < 0.01,
                "Beta({a}, {b}) var {var} vs {}",
                d.variance()
            );
        }
    }

    #[test]
    fn samples_track_the_cdf() {
        let d = Beta::new(2.0, 5.0);
        let mut rng = SplitMix64::seed_from_u64(4);
        let n = 40_000;
        let xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let threshold = d.quantile(q);
            let empirical = xs.iter().filter(|x| **x <= threshold).count() as f64 / n as f64;
            assert!(
                (empirical - q).abs() < 0.02,
                "q = {q}, empirical = {empirical}"
            );
        }
    }

    #[test]
    fn log_likelihood_helpers() {
        assert!((Beta::log_likelihood(0.5, 2, 1) - 0.125f64.ln()).abs() < 1e-12);
        assert_eq!(Beta::log_likelihood(0.0, 1, 0), f64::NEG_INFINITY);
        assert_eq!(Beta::log_likelihood(0.0, 0, 3), 0.0);
        // C(3, 2) = 3 extra ways to see two successes.
        let bernoulli = Beta::log_likelihood(0.5, 2, 1);
        let binomial = Beta::log_likelihood_binomial(0.5, 2, 3);
        assert!((binomial - bernoulli - 3f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn unnormalized_posterior_is_proportional_to_the_conjugate_posterior() {
        let prior = Beta::new(2.0, 3.0);
        let (s, f) = (5u64, 4u64);
        let posterior = prior.update(s, f);
        let offset = prior.log_unnormalized_posterior(0.5, s, f) - posterior.log_pdf(0.5);
        for &x in &[0.1, 0.3, 0.7, 0.95] {
            let diff = prior.log_unnormalized_posterior(x, s, f) - posterior.log_pdf(x);
            assert!((diff - offset).abs() < 1e-12, "x = {x}");
        }
        // The offset is exactly the log evidence (minus the binomial coefficient).
        let evidence = prior.log_marginal_likelihood(s, f) - ln_binomial(s + f, s);
        assert!((offset - evidence).abs() < 1e-12);
    }

    #[test]
    fn evidence_under_uniform_prior_is_uniform_over_counts() {
        let prior = Beta::uniform();
        let n = 7u64;
        let total: f64 = (0..=n)
            .map(|k| prior.log_marginal_likelihood(k, n - k).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        for k in 0..=n {
            let p = prior.log_marginal_likelihood(k, n - k).exp();
            assert!((p - 1.0 / (n as f64 + 1.0)).abs() < 1e-12, "k = {k}");
        }
    }

    #[test]
    fn distribution_trait_and_inherent_sample_agree() {
        let d = Beta::new(2.0, 2.0);
        let mut a = SplitMix64::seed_from_u64(77);
        let mut b = SplitMix64::seed_from_u64(77);
        assert_eq!(d.sample(&mut a), Distribution::<f64>::sample(&d, &mut b));
        assert_eq!(Beta::default(), Beta::uniform());
        assert!(Beta::haldane().mean() > 0.0);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(Beta::try_new(0.0, 1.0).is_err());
        assert!(Beta::try_new(1.0, f64::NAN).is_err());
        assert!(Beta::try_new(-1.0, 1.0).is_err());
    }

    #[test]
    #[should_panic(expected = "invalid Beta parameters")]
    fn new_panics_on_bad_alpha() {
        let _ = Beta::new(0.0, 1.0);
    }
}
