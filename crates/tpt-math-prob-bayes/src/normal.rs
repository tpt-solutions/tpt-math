//! The Gaussian (normal) distribution and its Normal–Normal conjugate update.

use tpt_math_prob_core::{Distribution, Rng};

use crate::special::{standard_normal_cdf, standard_normal_quantile, LN_2PI};
use crate::ParameterError;

/// The Gaussian distribution `N(μ, σ²)`.
///
/// Besides the usual density/sampling surface, this type implements the
/// Normal–Normal conjugate model: [`Gaussian::update`] treats `self` as a
/// prior over an unknown mean and returns the posterior after observing data
/// whose variance is known.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::Gaussian;
///
/// let prior = Gaussian::new(0.0, 1.0);
/// let posterior = prior.update(&[1.0, 2.0, 3.0], 1.0);
/// assert!((posterior.mean() - 1.5).abs() < 1e-12);
/// assert!((posterior.variance() - 0.25).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaussian {
    mean: f64,
    std: f64,
}

/// Alias for [`Gaussian`], for callers who prefer the name `Normal`.
pub type Normal = Gaussian;

impl Gaussian {
    /// Create `N(mean, std²)` from a mean and a **standard deviation**.
    ///
    /// # Panics
    ///
    /// Panics unless `mean` is finite and `std` is finite and strictly
    /// positive. Use [`Gaussian::try_new`] for a non-panicking version.
    #[must_use]
    pub fn new(mean: f64, std: f64) -> Self {
        match Self::try_new(mean, std) {
            Ok(dist) => dist,
            Err(err) => panic!("invalid Gaussian parameters: {err}"),
        }
    }

    /// Fallible constructor for `N(mean, std²)`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] if `mean` is not finite or `std` is not
    /// finite and strictly positive.
    pub fn try_new(mean: f64, std: f64) -> Result<Self, ParameterError> {
        crate::check_finite("mean", mean)?;
        crate::check_positive("std", std)?;
        Ok(Gaussian { mean, std })
    }

    /// Create `N(mean, variance)` from a mean and a **variance**.
    ///
    /// # Panics
    ///
    /// Panics unless `mean` is finite and `variance` is finite and strictly
    /// positive.
    #[must_use]
    pub fn from_variance(mean: f64, variance: f64) -> Self {
        match crate::check_positive("variance", variance) {
            Ok(()) => Gaussian::new(mean, variance.sqrt()),
            Err(err) => panic!("invalid Gaussian parameters: {err}"),
        }
    }

    /// The standard normal `N(0, 1)`.
    #[must_use]
    pub fn standard() -> Self {
        Gaussian {
            mean: 0.0,
            std: 1.0,
        }
    }

    /// The mean `μ`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The standard deviation `σ`.
    #[must_use]
    pub fn std(&self) -> f64 {
        self.std
    }

    /// The variance `σ²`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.std * self.std
    }

    /// The precision `1 / σ²`.
    #[must_use]
    pub fn precision(&self) -> f64 {
        1.0 / self.variance()
    }

    /// The number of standard deviations `x` lies above the mean.
    #[must_use]
    pub fn z_score(&self, x: f64) -> f64 {
        (x - self.mean) / self.std
    }

    /// The log-density at `x`.
    #[must_use]
    pub fn log_pdf(&self, x: f64) -> f64 {
        let z = self.z_score(x);
        -0.5 * z * z - self.std.ln() - 0.5 * LN_2PI
    }

    /// The probability density at `x`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    /// The cumulative distribution function at `x`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        standard_normal_cdf(self.z_score(x))
    }

    /// The inverse CDF at probability `p ∈ [0, 1]`.
    ///
    /// Returns `∓∞` at the endpoints.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        self.mean + self.std * standard_normal_quantile(p)
    }

    /// The equal-tailed credible interval containing probability `mass`.
    ///
    /// # Panics
    ///
    /// Panics unless `mass` lies in `(0, 1)`.
    #[must_use]
    pub fn credible_interval(&self, mass: f64) -> (f64, f64) {
        assert!(
            mass > 0.0 && mass < 1.0 && mass.is_finite(),
            "credible interval mass must lie in (0, 1), got {mass}"
        );
        let tail = 0.5 * (1.0 - mass);
        (self.quantile(tail), self.quantile(1.0 - tail))
    }

    /// Draw one sample (Marsaglia polar method).
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.draw(rng)
    }

    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.mean + self.std * sample_standard_normal(rng)
    }

    /// The log-likelihood of independent `data` under this distribution.
    #[must_use]
    pub fn log_likelihood(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let sum_sq: f64 = data
            .iter()
            .map(|x| {
                let z = self.z_score(*x);
                z * z
            })
            .sum();
        -0.5 * sum_sq - n * (self.std.ln() + 0.5 * LN_2PI)
    }

    /// Normal–Normal conjugate update for an unknown mean with known variance.
    ///
    /// Treats `self` as the prior `N(μ₀, σ₀²)` over the unknown mean of a
    /// Gaussian likelihood whose variance `known_variance` is fixed. With
    /// data `x₁ … xₙ` the posterior precision and mean are
    ///
    /// ```text
    /// 1/σ₁² = 1/σ₀² + n/σ²
    /// μ₁    = (μ₀/σ₀² + Σxᵢ/σ²) · σ₁²
    /// ```
    ///
    /// Empty data returns the prior unchanged.
    ///
    /// # Panics
    ///
    /// Panics unless `known_variance` is finite and strictly positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_bayes::Gaussian;
    ///
    /// // A vague prior barely moves the data mean.
    /// let posterior = Gaussian::new(0.0, 100.0).update(&[5.0, 5.0, 5.0, 5.0], 1.0);
    /// assert!((posterior.mean() - 5.0).abs() < 1e-3);
    /// ```
    #[must_use]
    pub fn update(&self, data: &[f64], known_variance: f64) -> Gaussian {
        let n = data.len() as f64;
        let sample_mean = if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f64>() / n
        };
        self.update_summary(data.len() as u64, sample_mean, known_variance)
    }

    /// Normal–Normal conjugate update from sufficient statistics.
    ///
    /// Equivalent to [`Gaussian::update`] but takes the sample size and the
    /// sample mean directly, which is all the data enters through.
    ///
    /// # Panics
    ///
    /// Panics unless `known_variance` is finite and strictly positive, or if
    /// `sample_mean` is not finite for a non-empty sample.
    #[must_use]
    pub fn update_summary(
        &self,
        observations: u64,
        sample_mean: f64,
        known_variance: f64,
    ) -> Gaussian {
        if let Err(err) = crate::check_positive("known_variance", known_variance) {
            panic!("invalid Normal-Normal update: {err}");
        }
        if observations == 0 {
            return *self;
        }
        if let Err(err) = crate::check_finite("sample_mean", sample_mean) {
            panic!("invalid Normal-Normal update: {err}");
        }
        let n = observations as f64;
        let prior_precision = self.precision();
        let data_precision = n / known_variance;
        let posterior_precision = prior_precision + data_precision;
        let posterior_variance = 1.0 / posterior_precision;
        let posterior_mean =
            (self.mean * prior_precision + sample_mean * data_precision) * posterior_variance;
        Gaussian {
            mean: posterior_mean,
            std: posterior_variance.sqrt(),
        }
    }

    /// The posterior predictive distribution for one new observation.
    ///
    /// If `self` is the posterior over the mean and the observation variance
    /// is `known_variance`, a new draw is distributed `N(μ, σ² + σ_obs²)`.
    ///
    /// # Panics
    ///
    /// Panics unless `known_variance` is finite and strictly positive.
    #[must_use]
    pub fn posterior_predictive(&self, known_variance: f64) -> Gaussian {
        if let Err(err) = crate::check_positive("known_variance", known_variance) {
            panic!("invalid posterior predictive: {err}");
        }
        Gaussian {
            mean: self.mean,
            std: (self.variance() + known_variance).sqrt(),
        }
    }

    /// Log marginal likelihood (evidence) of `data` under the Normal–Normal
    /// model with the given known observation variance.
    ///
    /// Uses the identity `p(x) = p(x | μ) p(μ) / p(μ | x)`, evaluated at the
    /// prior mean, which is exact for conjugate Gaussians.
    #[must_use]
    pub fn log_marginal_likelihood(&self, data: &[f64], known_variance: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let posterior = self.update(data, known_variance);
        let mu = self.mean;
        let likelihood: f64 = data
            .iter()
            .map(|x| Gaussian::new(*x, known_variance.sqrt()).log_pdf(mu))
            .sum();
        likelihood + self.log_pdf(mu) - posterior.log_pdf(mu)
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Gaussian::standard()
    }
}

impl Distribution<f64> for Gaussian {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.draw(rng)
    }
}

/// Draw one standard normal variate with the Marsaglia polar method.
pub(crate) fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u = 2.0 * rng.next_f64() - 1.0;
        let v = 2.0 * rng.next_f64() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::SplitMix64;

    fn mean_and_variance(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        (mean, var)
    }

    #[test]
    fn moments_and_accessors() {
        let g = Gaussian::new(2.0, 3.0);
        assert_eq!(g.mean(), 2.0);
        assert_eq!(g.std(), 3.0);
        assert_eq!(g.variance(), 9.0);
        assert!((g.precision() - 1.0 / 9.0).abs() < 1e-15);
        assert_eq!(Gaussian::from_variance(2.0, 9.0), g);
        assert_eq!(Gaussian::default(), Gaussian::standard());
        assert!((g.z_score(5.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn pdf_matches_closed_form() {
        let g = Gaussian::new(1.0, 2.0);
        let peak = 1.0 / (2.0 * (2.0 * core::f64::consts::PI).sqrt());
        assert!((g.pdf(1.0) - peak).abs() < 1e-15);
        // Symmetry about the mean.
        assert!((g.pdf(1.0 - 0.7) - g.pdf(1.0 + 0.7)).abs() < 1e-15);
        // Reference value for the standard normal at x = 1.
        assert!((Gaussian::standard().pdf(1.0) - 0.241_970_724_519_143_37).abs() < 1e-15);
        assert!((Gaussian::standard().log_pdf(0.0) - (-0.918_938_533_204_672_7)).abs() < 1e-14);
    }

    #[test]
    fn pdf_integrates_to_one() {
        let g = Gaussian::new(-1.0, 0.75);
        let (lo, hi, steps) = (-11.0, 9.0, 200_000);
        let h = (hi - lo) / steps as f64;
        let mut total = 0.5 * (g.pdf(lo) + g.pdf(hi));
        for i in 1..steps {
            total += g.pdf(lo + i as f64 * h);
        }
        assert!((total * h - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cdf_and_quantile_round_trip() {
        let g = Gaussian::new(3.0, 2.0);
        assert!((g.cdf(3.0) - 0.5).abs() < 1e-15);
        assert!((g.cdf(5.0) - 0.841_344_746_068_542_9).abs() < 1e-12);
        for &p in &[0.01, 0.25, 0.5, 0.75, 0.99] {
            assert!((g.cdf(g.quantile(p)) - p).abs() < 1e-12, "p = {p}");
        }
        let (lo, hi) = g.credible_interval(0.95);
        assert!((lo - (3.0 - 1.959_963_984_540_054 * 2.0)).abs() < 1e-9);
        assert!((hi - (3.0 + 1.959_963_984_540_054 * 2.0)).abs() < 1e-9);
    }

    #[test]
    fn sampling_recovers_moments() {
        let g = Gaussian::new(-2.5, 1.5);
        let mut rng = SplitMix64::seed_from_u64(2024);
        let xs: Vec<f64> = (0..100_000).map(|_| g.sample(&mut rng)).collect();
        let (mean, var) = mean_and_variance(&xs);
        assert!((mean - g.mean()).abs() < 0.02, "mean = {mean}");
        assert!((var - g.variance()).abs() < 0.05, "var = {var}");
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let g = Gaussian::standard();
        let mut a = SplitMix64::seed_from_u64(5);
        let mut b = SplitMix64::seed_from_u64(5);
        for _ in 0..32 {
            assert_eq!(g.sample(&mut a), Distribution::<f64>::sample(&g, &mut b));
        }
    }

    #[test]
    fn normal_normal_conjugate_update() {
        let prior = Gaussian::new(0.0, 1.0);
        let posterior = prior.update(&[1.0, 2.0, 3.0], 1.0);
        // precision 1 + 3 = 4 -> variance 0.25; mean = (0 + 6) / 4 = 1.5
        assert!((posterior.variance() - 0.25).abs() < 1e-12);
        assert!((posterior.mean() - 1.5).abs() < 1e-12);
        assert_eq!(posterior, prior.update_summary(3, 2.0, 1.0));
    }

    #[test]
    fn update_is_sequentially_consistent() {
        let prior = Gaussian::new(1.0, 2.0);
        let data = [0.4, -1.2, 3.3, 2.9, 0.1];
        let batch = prior.update(&data, 0.5);
        let mut sequential = prior;
        for x in data {
            sequential = sequential.update(&[x], 0.5);
        }
        assert!((batch.mean() - sequential.mean()).abs() < 1e-12);
        assert!((batch.std() - sequential.std()).abs() < 1e-12);
    }

    #[test]
    fn update_shrinks_variance_and_ignores_empty_data() {
        let prior = Gaussian::new(0.0, 1.0);
        assert_eq!(prior.update(&[], 1.0), prior);
        let posterior = prior.update(&[1.0; 10], 1.0);
        assert!(posterior.variance() < prior.variance());
        // Predictive is wider than the posterior over the mean.
        let predictive = posterior.posterior_predictive(1.0);
        assert!(predictive.variance() > posterior.variance());
        assert!((predictive.variance() - (posterior.variance() + 1.0)).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_sums_log_pdfs() {
        let g = Gaussian::new(0.5, 1.25);
        let data = [0.0, 1.0, -2.0, 3.5];
        let expected: f64 = data.iter().map(|x| g.log_pdf(*x)).sum();
        assert!((g.log_likelihood(&data) - expected).abs() < 1e-12);
        assert_eq!(g.log_likelihood(&[]), 0.0);
    }

    #[test]
    fn evidence_matches_numeric_integration() {
        let prior = Gaussian::new(0.0, 1.0);
        let data = [0.8, 1.4];
        let known_variance = 0.5;
        let analytic = prior.log_marginal_likelihood(&data, known_variance);

        let (lo, hi, steps) = (-12.0f64, 12.0f64, 400_000);
        let h = (hi - lo) / steps as f64;
        let integrand = |mu: f64| {
            let lik = Gaussian::new(mu, known_variance.sqrt()).log_likelihood(&data);
            (lik + prior.log_pdf(mu)).exp()
        };
        let mut total = 0.5 * (integrand(lo) + integrand(hi));
        for i in 1..steps {
            total += integrand(lo + i as f64 * h);
        }
        let numeric = (total * h).ln();
        assert!((analytic - numeric).abs() < 1e-8, "{analytic} vs {numeric}");
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(Gaussian::try_new(0.0, 0.0).is_err());
        assert!(Gaussian::try_new(0.0, -1.0).is_err());
        assert!(Gaussian::try_new(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    #[should_panic(expected = "invalid Gaussian parameters")]
    fn new_panics_on_bad_std() {
        let _ = Gaussian::new(0.0, -1.0);
    }
}
