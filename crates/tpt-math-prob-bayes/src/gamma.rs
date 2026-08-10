//! The Gamma distribution and the underlying gamma variate sampler.

use tpt_math_prob_core::{Distribution, Rng};

use crate::normal::sample_standard_normal;
use crate::special::{ln_factorial, ln_gamma, xlogy};
use crate::ParameterError;

/// The Gamma distribution `Gamma(k, β)` in *shape/rate* parameterisation.
///
/// The density is
///
/// ```text
/// p(x) = β^k x^(k-1) e^(-βx) / Γ(k),   x > 0
/// ```
///
/// `Gamma` is the conjugate prior for the rate of a Poisson likelihood (see
/// [`Gamma::update_poisson`]) and is also the building block used to draw
/// [`Beta`](crate::Beta) variates.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::Gamma;
///
/// let prior = Gamma::new(2.0, 1.0);
/// // Observe 3 intervals with 4, 6 and 5 events.
/// let posterior = prior.update_poisson(&[4, 6, 5]);
/// assert_eq!(posterior.shape(), 17.0);
/// assert_eq!(posterior.rate(), 4.0);
/// assert!((posterior.mean() - 4.25).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

impl Gamma {
    /// Create `Gamma(shape, rate)`.
    ///
    /// # Panics
    ///
    /// Panics unless both parameters are finite and strictly positive. Use
    /// [`Gamma::try_new`] for a non-panicking version.
    #[must_use]
    pub fn new(shape: f64, rate: f64) -> Self {
        match Self::try_new(shape, rate) {
            Ok(dist) => dist,
            Err(err) => panic!("invalid Gamma parameters: {err}"),
        }
    }

    /// Fallible constructor for `Gamma(shape, rate)`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] if `shape` or `rate` is not finite and
    /// strictly positive.
    pub fn try_new(shape: f64, rate: f64) -> Result<Self, ParameterError> {
        crate::check_positive("shape", shape)?;
        crate::check_positive("rate", rate)?;
        Ok(Gamma { shape, rate })
    }

    /// The shape parameter `k`.
    #[must_use]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// The rate parameter `β` (the reciprocal of the scale).
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// The scale parameter `θ = 1 / β`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        1.0 / self.rate
    }

    /// The mean `k / β`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// The variance `k / β²`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// The mode `(k − 1) / β`, which exists only for `k ≥ 1`.
    #[must_use]
    pub fn mode(&self) -> Option<f64> {
        (self.shape >= 1.0).then(|| (self.shape - 1.0) / self.rate)
    }

    /// The log-density at `x`; `-∞` outside the support.
    #[must_use]
    pub fn log_pdf(&self, x: f64) -> f64 {
        if x.is_nan() || x < 0.0 {
            return f64::NEG_INFINITY;
        }
        self.shape * self.rate.ln() + xlogy(self.shape - 1.0, x)
            - self.rate * x
            - ln_gamma(self.shape)
    }

    /// The probability density at `x`; zero outside the support.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Draw one sample.
    ///
    /// Marsaglia–Tsang squeeze method, with the `k < 1` boost
    /// `Gamma(k) = Gamma(k + 1) · U^(1/k)`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.draw(rng)
    }

    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_standard_gamma(self.shape, rng) / self.rate
    }

    /// Conjugate update against a Poisson likelihood with unknown rate.
    ///
    /// Given `Gamma(k, β)` as the prior over the Poisson rate `λ` and counts
    /// `x₁ … xₙ`, the posterior is `Gamma(k + Σxᵢ, β + n)`.
    #[must_use]
    pub fn update_poisson(&self, counts: &[u64]) -> Gamma {
        let total: u64 = counts.iter().copied().sum();
        self.update_poisson_summary(total, counts.len() as u64)
    }

    /// Conjugate Poisson update from sufficient statistics.
    ///
    /// `total_count` is `Σxᵢ` and `observations` is `n`.
    #[must_use]
    pub fn update_poisson_summary(&self, total_count: u64, observations: u64) -> Gamma {
        Gamma {
            shape: self.shape + total_count as f64,
            rate: self.rate + observations as f64,
        }
    }

    /// Log marginal likelihood (evidence) of Poisson `counts` under this prior.
    ///
    /// `p(x) = ∫ Poisson(x | λ) Gamma(λ | k, β) dλ`, a negative-binomial
    /// mixture, evaluated in log space.
    #[must_use]
    pub fn log_marginal_likelihood_poisson(&self, counts: &[u64]) -> f64 {
        let n = counts.len() as f64;
        let total: u64 = counts.iter().copied().sum();
        let log_factorials: f64 = counts.iter().map(|&k| ln_factorial(k)).sum();
        let post_shape = self.shape + total as f64;
        let post_rate = self.rate + n;
        self.shape * self.rate.ln() - ln_gamma(self.shape) + ln_gamma(post_shape)
            - post_shape * post_rate.ln()
            - log_factorials
    }
}

impl Distribution<f64> for Gamma {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.draw(rng)
    }
}

/// Draw from `Gamma(shape, 1)` using the Marsaglia–Tsang method.
///
/// Assumes `shape > 0`; callers in this crate validate at construction time.
pub(crate) fn sample_standard_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    // For k < 1 sample Gamma(k + 1) and scale by U^(1/k).
    let (shape, boost) = if shape < 1.0 {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        (shape + 1.0, u.powf(1.0 / shape))
    } else {
        (shape, 1.0)
    };

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.033_1 * x * x * x * x {
            return boost * d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return boost * d * v;
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
    fn moments_are_analytic() {
        let g = Gamma::new(3.0, 2.0);
        assert!((g.mean() - 1.5).abs() < 1e-15);
        assert!((g.variance() - 0.75).abs() < 1e-15);
        assert!((g.mode().unwrap() - 1.0).abs() < 1e-15);
        assert_eq!(Gamma::new(0.5, 1.0).mode(), None);
        assert!((g.scale() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn pdf_matches_closed_form() {
        // Gamma(1, β) is Exponential(β).
        let g = Gamma::new(1.0, 2.0);
        for &x in &[0.0f64, 0.25, 1.0, 3.0] {
            let expected = 2.0 * (-2.0 * x).exp();
            assert!((g.pdf(x) - expected).abs() < 1e-12, "x = {x}");
        }
        assert_eq!(g.pdf(-1.0), 0.0);
        assert!(g.log_pdf(-1.0).is_infinite());
    }

    #[test]
    fn pdf_integrates_to_one() {
        let g = Gamma::new(2.5, 1.5);
        let (lo, hi, steps) = (0.0, 25.0, 250_000);
        let h = (hi - lo) / steps as f64;
        let mut total = 0.5 * (g.pdf(lo) + g.pdf(hi));
        for i in 1..steps {
            total += g.pdf(lo + i as f64 * h);
        }
        assert!((total * h - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sampling_recovers_moments() {
        let mut rng = SplitMix64::seed_from_u64(20_240_617);
        for &(shape, rate) in &[(0.4, 1.0), (1.0, 2.0), (7.5, 3.0)] {
            let g = Gamma::new(shape, rate);
            let xs: Vec<f64> = (0..50_000).map(|_| g.sample(&mut rng)).collect();
            assert!(xs.iter().all(|x| *x > 0.0 && x.is_finite()));
            let (mean, var) = mean_and_variance(&xs);
            assert!(
                (mean - g.mean()).abs() < 0.05 * g.mean().max(1.0),
                "shape {shape}: mean {mean} vs {}",
                g.mean()
            );
            assert!(
                (var - g.variance()).abs() < 0.15 * g.variance().max(1.0),
                "shape {shape}: var {var} vs {}",
                g.variance()
            );
        }
    }

    #[test]
    fn poisson_conjugate_update() {
        let prior = Gamma::new(2.0, 1.0);
        let posterior = prior.update_poisson(&[4, 6, 5]);
        assert_eq!(posterior.shape(), 17.0);
        assert_eq!(posterior.rate(), 4.0);
        assert_eq!(posterior, prior.update_poisson_summary(15, 3));
        // Empty data leaves the prior untouched.
        assert_eq!(prior.update_poisson(&[]), prior);
    }

    #[test]
    fn poisson_evidence_matches_direct_integration() {
        let prior = Gamma::new(2.0, 1.5);
        let counts = [1u64, 3, 2];
        let log_evidence = prior.log_marginal_likelihood_poisson(&counts);

        // Numerically integrate ∫ p(x | λ) p(λ) dλ.
        let (lo, hi, steps) = (0.0f64, 60.0f64, 600_000);
        let h = (hi - lo) / steps as f64;
        let integrand = |lambda: f64| {
            if lambda <= 0.0 {
                return 0.0;
            }
            let log_lik: f64 = counts
                .iter()
                .map(|&k| k as f64 * lambda.ln() - lambda - ln_factorial(k))
                .sum();
            (log_lik + prior.log_pdf(lambda)).exp()
        };
        let mut total = 0.5 * (integrand(lo) + integrand(hi));
        for i in 1..steps {
            total += integrand(lo + i as f64 * h);
        }
        let numeric = (total * h).ln();
        assert!(
            (log_evidence - numeric).abs() < 1e-6,
            "{log_evidence} vs {numeric}"
        );
    }

    #[test]
    fn distribution_trait_and_inherent_sample_agree() {
        let g = Gamma::new(2.0, 3.0);
        let mut a = SplitMix64::seed_from_u64(11);
        let mut b = SplitMix64::seed_from_u64(11);
        let via_inherent = g.sample(&mut a);
        let via_trait = Distribution::<f64>::sample(&g, &mut b);
        assert_eq!(via_inherent, via_trait);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(Gamma::try_new(0.0, 1.0).is_err());
        assert!(Gamma::try_new(1.0, -2.0).is_err());
        assert!(Gamma::try_new(f64::NAN, 1.0).is_err());
    }
}
