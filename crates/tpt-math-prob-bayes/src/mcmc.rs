//! A random-walk Metropolis(–Hastings) sampler for scalar targets.

use tpt_math_prob_core::{Distribution, Rng};

use crate::Gaussian;

/// A random-walk Metropolis sampler over a scalar state.
///
/// `F` is an **unnormalised log-target**: any function proportional (in log
/// space) to the density you want to sample, such as
/// `log prior + log likelihood`. `P` is a **symmetric proposal** that supplies
/// the random *increment* added to the current state — symmetry (`q(a → b) =
/// q(b → a)`) is what lets the Hastings ratio collapse to the plain target
/// ratio, so a zero-mean [`Gaussian`] or any other symmetric perturbation is
/// the right choice.
///
/// One step draws `x' = x + ε`, then accepts with probability
/// `min(1, exp(logπ(x') − logπ(x)))`. Candidates whose log-target is `-∞`
/// (outside the support) or `NaN` are always rejected, which makes bounded
/// targets such as a [`Beta`](crate::Beta) posterior safe to sample directly.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_bayes::{Gaussian, Metropolis};
/// use tpt_math_prob_core::SplitMix64;
///
/// // Target: N(3, 1.5²), given only up to a constant.
/// let target = |x: f64| -0.5 * ((x - 3.0) / 1.5).powi(2);
/// let mut sampler = Metropolis::with_gaussian_proposal(target, 2.0);
///
/// let mut rng = SplitMix64::seed_from_u64(12345);
/// let trace = sampler.run_with_burn_in(&mut rng, 0.0, 2_000, 40_000);
///
/// let mean = trace.iter().sum::<f64>() / trace.len() as f64;
/// assert!((mean - 3.0).abs() < 0.1, "mean = {mean}");
/// assert!(sampler.acceptance_rate() > 0.0);
/// ```
#[derive(Clone, Debug)]
pub struct Metropolis<F, P> {
    log_target: F,
    proposal: P,
    accepted: u64,
    proposals: u64,
}

impl<F, P> Metropolis<F, P>
where
    F: Fn(f64) -> f64,
    P: Distribution<f64>,
{
    /// Build a sampler from an unnormalised log-target and a symmetric
    /// proposal over increments.
    pub fn new(log_target: F, proposal: P) -> Self {
        Metropolis {
            log_target,
            proposal,
            accepted: 0,
            proposals: 0,
        }
    }

    /// Perform one Metropolis step from `current` and return the next state.
    ///
    /// The returned value is either the accepted candidate or `current`
    /// repeated; both are valid trace entries.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R, current: f64) -> f64 {
        let log_current = (self.log_target)(current);
        self.step_cached(rng, current, log_current).0
    }

    /// Run the chain for `n` steps from `init`, returning the trace.
    ///
    /// The trace has exactly `n` entries and does **not** include `init`.
    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R, init: f64, n: usize) -> Vec<f64> {
        self.run_with_burn_in(rng, init, 0, n)
    }

    /// Run `burn_in` discarded steps, then collect `n` trace samples.
    pub fn run_with_burn_in<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        init: f64,
        burn_in: usize,
        n: usize,
    ) -> Vec<f64> {
        let mut state = init;
        let mut log_state = (self.log_target)(init);
        for _ in 0..burn_in {
            let (next, log_next) = self.step_cached(rng, state, log_state);
            state = next;
            log_state = log_next;
        }
        let mut trace = Vec::with_capacity(n);
        for _ in 0..n {
            let (next, log_next) = self.step_cached(rng, state, log_state);
            state = next;
            log_state = log_next;
            trace.push(state);
        }
        trace
    }

    /// Run the chain and keep every `thin`-th sample after `burn_in`.
    ///
    /// Thinning trades samples for lower autocorrelation; the returned trace
    /// has `n` entries drawn `thin` steps apart.
    ///
    /// # Panics
    ///
    /// Panics if `thin` is zero.
    pub fn run_thinned<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        init: f64,
        burn_in: usize,
        n: usize,
        thin: usize,
    ) -> Vec<f64> {
        assert!(thin > 0, "thinning interval must be non-zero");
        let mut state = init;
        let mut log_state = (self.log_target)(init);
        for _ in 0..burn_in {
            let (next, log_next) = self.step_cached(rng, state, log_state);
            state = next;
            log_state = log_next;
        }
        let mut trace = Vec::with_capacity(n);
        for _ in 0..n {
            for _ in 0..thin {
                let (next, log_next) = self.step_cached(rng, state, log_state);
                state = next;
                log_state = log_next;
            }
            trace.push(state);
        }
        trace
    }

    /// One step with the current log-target value already known.
    ///
    /// Returns the next state together with its log-target, so a chain never
    /// evaluates the target more than once per step.
    fn step_cached<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        current: f64,
        log_current: f64,
    ) -> (f64, f64) {
        let candidate = current + self.proposal.sample(rng);
        let log_candidate = (self.log_target)(candidate);
        self.proposals += 1;
        if accept(rng, log_current, log_candidate) {
            self.accepted += 1;
            (candidate, log_candidate)
        } else {
            (current, log_current)
        }
    }

    /// Number of accepted proposals so far.
    #[must_use]
    pub fn accepted(&self) -> u64 {
        self.accepted
    }

    /// Number of proposals made so far.
    #[must_use]
    pub fn proposals(&self) -> u64 {
        self.proposals
    }

    /// The running acceptance rate, or `0.0` before the first proposal.
    ///
    /// For a scalar random walk, rates around 0.4–0.5 are usually efficient;
    /// values near 0 or 1 mean the proposal scale is too large or too small.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposals as f64
        }
    }

    /// Reset the acceptance counters (e.g. after burn-in).
    pub fn reset_statistics(&mut self) {
        self.accepted = 0;
        self.proposals = 0;
    }

    /// Borrow the proposal distribution.
    pub fn proposal(&self) -> &P {
        &self.proposal
    }

    /// Consume the sampler and return its log-target and proposal.
    pub fn into_parts(self) -> (F, P) {
        (self.log_target, self.proposal)
    }
}

impl<F> Metropolis<F, Gaussian>
where
    F: Fn(f64) -> f64,
{
    /// Build a sampler with a zero-mean Gaussian random-walk proposal.
    ///
    /// # Panics
    ///
    /// Panics unless `step_std` is finite and strictly positive.
    pub fn with_gaussian_proposal(log_target: F, step_std: f64) -> Self {
        Metropolis::new(log_target, Gaussian::new(0.0, step_std))
    }
}

/// The Metropolis accept/reject rule for a symmetric proposal.
fn accept<R: Rng + ?Sized>(rng: &mut R, log_current: f64, log_candidate: f64) -> bool {
    if log_candidate.is_nan() || log_candidate == f64::NEG_INFINITY {
        return false;
    }
    if log_candidate >= log_current {
        return true;
    }
    rng.next_f64().ln() < log_candidate - log_current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Beta, Gaussian};
    use tpt_math_prob_core::SplitMix64;

    fn mean_and_std(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    #[test]
    fn recovers_a_gaussian_target() {
        let (mu, sigma) = (3.0, 1.5);
        let target = move |x: f64| -0.5 * ((x - mu) / sigma).powi(2);
        let mut sampler = Metropolis::with_gaussian_proposal(target, 2.5);
        let mut rng = SplitMix64::seed_from_u64(0xBEEF);

        let trace = sampler.run_with_burn_in(&mut rng, -10.0, 5_000, 100_000);
        assert_eq!(trace.len(), 100_000);

        let (mean, std) = mean_and_std(&trace);
        assert!((mean - mu).abs() < 0.05, "mean = {mean}");
        assert!((std - sigma).abs() < 0.08, "std = {std}");

        let rate = sampler.acceptance_rate();
        assert!(rate > 0.2 && rate < 0.9, "acceptance rate = {rate}");
    }

    #[test]
    fn recovers_the_mode_of_a_bimodal_target() {
        // A mixture with a dominant mode at +2 and a small one at -2.
        let target = |x: f64| {
            let big = (-0.5 * (x - 2.0f64).powi(2)).exp();
            let small = 0.1 * (-0.5 * (x + 2.0f64).powi(2)).exp();
            (big + small).ln()
        };
        let mut sampler = Metropolis::with_gaussian_proposal(target, 2.0);
        let mut rng = SplitMix64::seed_from_u64(99);
        let trace = sampler.run_with_burn_in(&mut rng, 0.0, 5_000, 120_000);

        // Histogram the trace and check the dominant mode is where we expect.
        let mut bins = [0usize; 80];
        for x in &trace {
            let idx = (((x + 6.0) / 12.0) * bins.len() as f64) as isize;
            if (0..bins.len() as isize).contains(&idx) {
                bins[idx as usize] += 1;
            }
        }
        let best = bins.iter().enumerate().max_by_key(|(_, c)| **c).unwrap().0;
        let mode = -6.0 + (best as f64 + 0.5) * (12.0 / bins.len() as f64);
        assert!((mode - 2.0).abs() < 0.4, "mode = {mode}");

        // Roughly 10/11 of the mass sits in the dominant component.
        let right = trace.iter().filter(|x| **x > 0.0).count() as f64 / trace.len() as f64;
        assert!(right > 0.85, "right-hand mass = {right}");
    }

    #[test]
    fn recovers_a_beta_posterior() {
        // Beta(1, 1) prior with 30 successes and 70 failures -> Beta(31, 71).
        let prior = Beta::uniform();
        let (s, f) = (30u64, 70u64);
        let posterior = prior.update(s, f);
        let target = move |x: f64| prior.log_unnormalized_posterior(x, s, f);

        let mut sampler = Metropolis::with_gaussian_proposal(target, 0.1);
        let mut rng = SplitMix64::seed_from_u64(7);
        let trace = sampler.run_with_burn_in(&mut rng, 0.5, 2_000, 60_000);

        // The target is -inf outside [0, 1], so the chain must stay inside.
        assert!(trace.iter().all(|x| (0.0..=1.0).contains(x)));
        let (mean, std) = mean_and_std(&trace);
        assert!((mean - posterior.mean()).abs() < 0.01, "mean = {mean}");
        assert!((std - posterior.std()).abs() < 0.005, "std = {std}");
    }

    #[test]
    fn step_returns_current_or_candidate_and_counts_proposals() {
        let target = |x: f64| -0.5 * x * x;
        let mut sampler = Metropolis::with_gaussian_proposal(target, 1.0);
        let mut rng = SplitMix64::seed_from_u64(3);
        let mut state = 0.0;
        for _ in 0..1_000 {
            let next = sampler.step(&mut rng, state);
            assert!(next.is_finite());
            state = next;
        }
        assert_eq!(sampler.proposals(), 1_000);
        assert!(sampler.accepted() > 0 && sampler.accepted() <= 1_000);
        assert!(
            (sampler.acceptance_rate() - sampler.accepted() as f64 / sampler.proposals() as f64)
                .abs()
                < 1e-15
        );
        sampler.reset_statistics();
        assert_eq!(sampler.proposals(), 0);
        assert_eq!(sampler.acceptance_rate(), 0.0);
    }

    #[test]
    fn run_is_deterministic_for_a_seed() {
        let target = |x: f64| -0.5 * x * x;
        let mut a = Metropolis::with_gaussian_proposal(target, 1.0);
        let mut b = Metropolis::with_gaussian_proposal(target, 1.0);
        let mut rng_a = SplitMix64::seed_from_u64(42);
        let mut rng_b = SplitMix64::seed_from_u64(42);
        assert_eq!(a.run(&mut rng_a, 0.0, 500), b.run(&mut rng_b, 0.0, 500));
    }

    #[test]
    fn rejects_impossible_candidates() {
        // Support restricted to [0, 1]; start inside and never leave.
        let target = |x: f64| {
            if (0.0..=1.0).contains(&x) {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        };
        let mut sampler = Metropolis::with_gaussian_proposal(target, 0.5);
        let mut rng = SplitMix64::seed_from_u64(17);
        let trace = sampler.run(&mut rng, 0.5, 20_000);
        assert!(trace.iter().all(|x| (0.0..=1.0).contains(x)));
        // A uniform target over [0, 1] should average about 0.5.
        let (mean, _) = mean_and_std(&trace);
        assert!((mean - 0.5).abs() < 0.02, "mean = {mean}");

        // NaN candidates are rejected too.
        let mut nan_sampler = Metropolis::with_gaussian_proposal(|_x: f64| f64::NAN, 1.0);
        assert_eq!(nan_sampler.step(&mut rng, 1.25), 1.25);
        assert_eq!(nan_sampler.accepted(), 0);
    }

    #[test]
    fn thinning_preserves_the_target() {
        let target = |x: f64| -0.5 * ((x - 1.0) / 0.5f64).powi(2);
        let mut sampler = Metropolis::with_gaussian_proposal(target, 1.0);
        let mut rng = SplitMix64::seed_from_u64(2_024);
        let trace = sampler.run_thinned(&mut rng, 0.0, 1_000, 20_000, 3);
        assert_eq!(trace.len(), 20_000);
        let (mean, std) = mean_and_std(&trace);
        assert!((mean - 1.0).abs() < 0.03, "mean = {mean}");
        assert!((std - 0.5).abs() < 0.03, "std = {std}");
    }

    #[test]
    fn custom_symmetric_proposal_is_accepted() {
        // Any Distribution<f64> works as a proposal; here a wide Gaussian.
        let target = |x: f64| -0.5 * ((x - 5.0) / 2.0f64).powi(2);
        let mut sampler = Metropolis::new(target, Gaussian::new(0.0, 4.0));
        let mut rng = SplitMix64::seed_from_u64(555);
        let trace = sampler.run_with_burn_in(&mut rng, 0.0, 2_000, 50_000);
        let (mean, std) = mean_and_std(&trace);
        assert!((mean - 5.0).abs() < 0.1, "mean = {mean}");
        assert!((std - 2.0).abs() < 0.1, "std = {std}");
        assert_eq!(sampler.proposal().mean(), 0.0);
        let (_target, proposal) = sampler.into_parts();
        assert_eq!(proposal.std(), 4.0);
    }

    #[test]
    fn zero_length_run_is_empty() {
        let mut sampler = Metropolis::with_gaussian_proposal(|x: f64| -x * x, 1.0);
        let mut rng = SplitMix64::seed_from_u64(1);
        assert!(sampler.run(&mut rng, 0.0, 0).is_empty());
        assert_eq!(sampler.proposals(), 0);
    }
}
