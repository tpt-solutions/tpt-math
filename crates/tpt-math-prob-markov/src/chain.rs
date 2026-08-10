//! Discrete-time, finite-state Markov chains.
//!
//! See [`MarkovChain`] for the main type.

use crate::error::MarkovError;
use crate::{sample_categorical, PROBABILITY_TOLERANCE};
use tpt_math_prob_core::Rng;

/// Convergence tolerance used by [`MarkovChain::stationary`].
///
/// Power iteration stops once the L1 distance between two successive iterates
/// drops below this value.
pub const DEFAULT_STATIONARY_TOLERANCE: f64 = 1e-12;

/// Iteration cap used by [`MarkovChain::stationary`].
pub const DEFAULT_STATIONARY_MAX_ITER: usize = 10_000;

/// A time-homogeneous Markov chain over `states` states.
///
/// The chain is described by a row-stochastic transition matrix: entry
/// `transition[i][j]` is `P(X_{t+1} = j | X_t = i)`, and every row sums to `1`.
///
/// # Invariants
///
/// The fields are public so that a model can be built or inspected directly,
/// which means the invariants are *yours* to uphold when you bypass the
/// constructors:
///
/// * `transition.len() == states`, and every row has length `states`;
/// * every entry is finite and non-negative;
/// * every row sums to `1`.
///
/// [`validate`](MarkovChain::validate) checks all three, and
/// [`normalize`](MarkovChain::normalize) restores the last one. Methods never
/// read out of bounds: a missing row or entry is treated as probability `0`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_markov::MarkovChain;
///
/// // A sticky two-state chain: state 0 rarely leaves, state 1 is a coin flip.
/// let mut chain = MarkovChain::new(2);
/// chain.set_transition(0, 0, 0.9);
/// chain.set_transition(0, 1, 0.1);
/// chain.set_transition(1, 0, 0.5);
/// chain.set_transition(1, 1, 0.5);
/// assert!(chain.validate().is_ok());
///
/// // Stationary distribution: pi = pi * P  =>  pi = [5/6, 1/6].
/// let pi = chain.stationary();
/// assert!((pi[0] - 5.0 / 6.0).abs() < 1e-9);
/// assert!((pi[1] - 1.0 / 6.0).abs() < 1e-9);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovChain {
    /// Number of states; state indices are `0..states`.
    pub states: usize,
    /// Row-stochastic transition matrix, `states` x `states`.
    pub transition: Vec<Vec<f64>>,
}

impl MarkovChain {
    /// Create a chain over `states` states with an all-zero transition matrix.
    ///
    /// The result is *not* yet stochastic: fill it with
    /// [`set_transition`](Self::set_transition) and then either supply rows
    /// that already sum to `1` or call [`normalize`](Self::normalize).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::new(3);
    /// assert_eq!(chain.states, 3);
    /// assert_eq!(chain.transition_prob(0, 0), 0.0);
    /// ```
    pub fn new(states: usize) -> Self {
        MarkovChain {
            states,
            transition: vec![vec![0.0; states]; states],
        }
    }

    /// Create a chain that jumps to a uniformly random state at every step.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::uniform(4);
    /// assert_eq!(chain.transition_prob(2, 3), 0.25);
    /// assert!(chain.validate().is_ok());
    /// ```
    pub fn uniform(states: usize) -> Self {
        let p = if states == 0 {
            0.0
        } else {
            1.0 / states as f64
        };
        MarkovChain {
            states,
            transition: vec![vec![p; states]; states],
        }
    }

    /// Create a chain in which every state is absorbing (the identity matrix).
    pub fn identity(states: usize) -> Self {
        let mut chain = MarkovChain::new(states);
        for (i, row) in chain.transition.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        chain
    }

    /// Build a chain from an already-stochastic square matrix.
    ///
    /// # Errors
    ///
    /// Returns a [`MarkovError`] if the matrix is empty, is not square,
    /// contains a negative or non-finite entry, or has a row that does not sum
    /// to `1` (within [`PROBABILITY_TOLERANCE`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::from_rows(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
    /// assert_eq!(chain.states, 2);
    ///
    /// assert!(MarkovChain::from_rows(vec![vec![0.9, 0.2], vec![0.5, 0.5]]).is_err());
    /// ```
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self, MarkovError> {
        let chain = MarkovChain {
            states: rows.len(),
            transition: rows,
        };
        chain.validate()?;
        Ok(chain)
    }

    /// Build a chain from a square matrix of non-negative weights, rescaling
    /// each row to sum to `1`.
    ///
    /// # Errors
    ///
    /// Returns a [`MarkovError`] if the matrix is empty, is not square,
    /// contains a negative or non-finite entry, or has an all-zero row.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// // Raw counts are fine; rows are rescaled for you.
    /// let chain = MarkovChain::from_rows_normalized(vec![vec![9.0, 1.0], vec![1.0, 1.0]]).unwrap();
    /// assert_eq!(chain.transition_prob(0, 0), 0.9);
    /// assert_eq!(chain.transition_prob(1, 1), 0.5);
    /// ```
    pub fn from_rows_normalized(rows: Vec<Vec<f64>>) -> Result<Self, MarkovError> {
        let mut chain = MarkovChain {
            states: rows.len(),
            transition: rows,
        };
        chain.check_shape()?;
        for row in 0..chain.states {
            chain.try_normalize_row(row)?;
        }
        Ok(chain)
    }

    /// The probability of moving from `from` to `to` in one step.
    ///
    /// Out-of-range indices read as `0.0` rather than panicking, so this is
    /// safe to call on a partially built or malformed chain.
    pub fn transition_prob(&self, from: usize, to: usize) -> f64 {
        self.transition
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(0.0)
    }

    /// The outgoing distribution of `from`, or an empty slice if `from` is out
    /// of range.
    pub fn row(&self, from: usize) -> &[f64] {
        self.transition.get(from).map_or(&[], |row| row.as_slice())
    }

    /// Set `P(to | from)`.
    ///
    /// This does not renormalize the row; call
    /// [`normalize_row`](Self::normalize_row) once the row is complete.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is out of range, or if `prob` is negative,
    /// infinite, or `NaN`. Use [`try_set_transition`](Self::try_set_transition)
    /// to get a [`MarkovError`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let mut chain = MarkovChain::new(2);
    /// chain.set_transition(0, 1, 0.25);
    /// assert_eq!(chain.transition_prob(0, 1), 0.25);
    /// ```
    pub fn set_transition(&mut self, from: usize, to: usize, prob: f64) {
        self.try_set_transition(from, to, prob)
            .expect("set_transition: invalid argument");
    }

    /// Checked form of [`set_transition`](Self::set_transition).
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::StateOutOfRange`] if `from` or `to` is not a
    /// valid state, or [`MarkovError::InvalidProbability`] if `prob` is
    /// negative, infinite, or `NaN`.
    pub fn try_set_transition(
        &mut self,
        from: usize,
        to: usize,
        prob: f64,
    ) -> Result<(), MarkovError> {
        if !prob.is_finite() || prob < 0.0 {
            return Err(MarkovError::InvalidProbability { value: prob });
        }
        let states = self.states;
        let slot = self
            .transition
            .get_mut(from)
            .ok_or(MarkovError::StateOutOfRange {
                state: from,
                states,
            })?
            .get_mut(to)
            .ok_or(MarkovError::StateOutOfRange { state: to, states })?;
        *slot = prob;
        Ok(())
    }

    /// Rescale row `from` so that it sums to `1`.
    ///
    /// An all-zero (or non-finite) row is replaced by the uniform
    /// distribution, keeping the chain usable; use
    /// [`try_normalize_row`](Self::try_normalize_row) if you would rather be
    /// told about it.
    ///
    /// # Panics
    ///
    /// Panics if `from` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let mut chain = MarkovChain::new(2);
    /// chain.set_transition(0, 0, 3.0);
    /// chain.set_transition(0, 1, 1.0);
    /// chain.normalize_row(0);
    /// assert_eq!(chain.transition_prob(0, 0), 0.75);
    ///
    /// // An untouched row is degenerate, so it becomes uniform.
    /// chain.normalize_row(1);
    /// assert_eq!(chain.transition_prob(1, 0), 0.5);
    /// ```
    pub fn normalize_row(&mut self, from: usize) {
        let states = self.states;
        let row = self.transition.get_mut(from).unwrap_or_else(|| {
            panic!("normalize_row: state {from} is out of range for {states} states")
        });
        normalize_in_place_or_uniform(row);
    }

    /// Checked form of [`normalize_row`](Self::normalize_row).
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::StateOutOfRange`] if `from` is not a valid
    /// state, or [`MarkovError::DegenerateRow`] if the row sums to zero (or to
    /// a non-finite value) and therefore cannot be rescaled.
    pub fn try_normalize_row(&mut self, from: usize) -> Result<(), MarkovError> {
        let states = self.states;
        let row = self
            .transition
            .get_mut(from)
            .ok_or(MarkovError::StateOutOfRange {
                state: from,
                states,
            })?;
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(MarkovError::DegenerateRow { row: from });
        }
        for p in row.iter_mut() {
            *p /= sum;
        }
        Ok(())
    }

    /// Rescale every row so that it sums to `1`, turning arbitrary
    /// non-negative weights (for example transition counts) into a stochastic
    /// matrix.
    ///
    /// All-zero rows become uniform, as in [`normalize_row`](Self::normalize_row).
    pub fn normalize(&mut self) {
        for row in self.transition.iter_mut() {
            normalize_in_place_or_uniform(row);
        }
    }

    /// Whether the matrix is square, non-negative, finite, and has rows summing
    /// to `1` within `tol`.
    pub fn is_stochastic(&self, tol: f64) -> bool {
        self.states > 0
            && self.transition.len() == self.states
            && self.transition.iter().all(|row| {
                row.len() == self.states
                    && row.iter().all(|p| p.is_finite() && *p >= 0.0)
                    && (row.iter().sum::<f64>() - 1.0).abs() <= tol
            })
    }

    /// Check every documented invariant of the chain.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::ZeroStates`] for an empty chain,
    /// [`MarkovError::DimensionMismatch`] if the matrix is not `states` x
    /// `states`, [`MarkovError::InvalidProbability`] for a negative or
    /// non-finite entry, and [`MarkovError::NotStochastic`] for a row that does
    /// not sum to `1` within [`PROBABILITY_TOLERANCE`].
    pub fn validate(&self) -> Result<(), MarkovError> {
        self.check_shape()?;
        for (i, row) in self.transition.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(MarkovError::NotStochastic { row: i, sum });
            }
        }
        Ok(())
    }

    /// The "lazy" chain `(P + I) / 2`: at each step it stays put with
    /// probability `1/2`, otherwise it follows `P`.
    ///
    /// A lazy chain is aperiodic and has exactly the same stationary
    /// distribution as the original, which makes it the standard fix when
    /// power iteration oscillates on a periodic chain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// // A period-2 chain: power iteration on `P` itself never settles.
    /// let flip = MarkovChain::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    /// let pi = flip.lazy().stationary();
    /// assert!((pi[0] - 0.5).abs() < 1e-9);
    /// ```
    pub fn lazy(&self) -> MarkovChain {
        let states = self.states;
        let transition = (0..states)
            .map(|i| {
                (0..states)
                    .map(|j| {
                        let stay = if i == j { 0.5 } else { 0.0 };
                        stay + 0.5 * self.transition_prob(i, j)
                    })
                    .collect()
            })
            .collect();
        MarkovChain { states, transition }
    }

    /// Advance a distribution over states by one step: `dist * P`.
    ///
    /// Entries beyond `states` are ignored and missing entries read as `0`, so
    /// the result always has length `states`.
    pub fn step_distribution(&self, dist: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.states];
        for (i, &mass) in dist.iter().enumerate().take(self.states) {
            if mass == 0.0 {
                continue;
            }
            if let Some(row) = self.transition.get(i) {
                for (slot, &p) in out.iter_mut().zip(row.iter()) {
                    *slot += mass * p;
                }
            }
        }
        out
    }

    /// The distribution over states after `steps` steps, starting from `initial`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::from_rows(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
    /// let after = chain.distribution_after(&[1.0, 0.0], 1);
    /// assert_eq!(after, vec![0.9, 0.1]);
    ///
    /// // Long-run behaviour matches the stationary distribution.
    /// let far = chain.distribution_after(&[1.0, 0.0], 500);
    /// assert!((far[0] - 5.0 / 6.0).abs() < 1e-9);
    /// ```
    pub fn distribution_after(&self, initial: &[f64], steps: usize) -> Vec<f64> {
        let mut dist = vec![0.0; self.states];
        for (slot, &p) in dist.iter_mut().zip(initial.iter()) {
            *slot = p;
        }
        for _ in 0..steps {
            dist = self.step_distribution(&dist);
        }
        dist
    }

    /// The stationary distribution `pi` solving `pi = pi * P`, computed by
    /// power iteration from the uniform distribution.
    ///
    /// Uses [`DEFAULT_STATIONARY_TOLERANCE`] and
    /// [`DEFAULT_STATIONARY_MAX_ITER`]; see
    /// [`stationary_with`](Self::stationary_with) to choose your own.
    ///
    /// The iterate is returned even if it has not converged. Convergence to
    /// *the* stationary distribution is guaranteed for an irreducible,
    /// aperiodic chain; a periodic chain oscillates forever, so use
    /// [`lazy`](Self::lazy) first, and a reducible chain converges to a
    /// stationary distribution that depends on the (uniform) starting vector.
    /// Use [`try_stationary`](Self::try_stationary) to detect non-convergence.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::from_rows(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
    /// let pi = chain.stationary();
    /// assert!((pi[0] - 5.0 / 6.0).abs() < 1e-9);
    /// assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// ```
    pub fn stationary(&self) -> Vec<f64> {
        self.stationary_with(DEFAULT_STATIONARY_TOLERANCE, DEFAULT_STATIONARY_MAX_ITER)
    }

    /// [`stationary`](Self::stationary) with an explicit tolerance and
    /// iteration cap.
    ///
    /// Iteration stops as soon as the L1 distance between successive iterates
    /// is at most `tol`, or after `max_iter` iterations, whichever comes first.
    pub fn stationary_with(&self, tol: f64, max_iter: usize) -> Vec<f64> {
        self.power_iterate(tol, max_iter).0
    }

    /// Checked form of [`stationary_with`](Self::stationary_with).
    ///
    /// # Errors
    ///
    /// Returns any error from [`validate`](Self::validate) if the chain is not
    /// a valid stochastic matrix, or [`MarkovError::NotConverged`] if the
    /// residual is still above `tol` after `max_iter` iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::{MarkovChain, MarkovError};
    ///
    /// // A period-2 chain: the iterate oscillates forever instead of settling.
    /// let periodic = MarkovChain::from_rows(vec![
    ///     vec![0.0, 0.5, 0.5],
    ///     vec![1.0, 0.0, 0.0],
    ///     vec![1.0, 0.0, 0.0],
    /// ])
    /// .unwrap();
    /// assert!(matches!(
    ///     periodic.try_stationary(1e-12, 1_000),
    ///     Err(MarkovError::NotConverged { .. })
    /// ));
    ///
    /// // The lazy chain has the same stationary distribution and converges.
    /// let pi = periodic.lazy().try_stationary(1e-12, 10_000).unwrap();
    /// assert!((pi[0] - 0.5).abs() < 1e-9);
    /// ```
    pub fn try_stationary(&self, tol: f64, max_iter: usize) -> Result<Vec<f64>, MarkovError> {
        self.validate()?;
        let (pi, iterations, residual) = self.power_iterate(tol, max_iter);
        if residual > tol {
            return Err(MarkovError::NotConverged {
                iterations,
                residual,
            });
        }
        Ok(pi)
    }

    /// Sample the next state given that the chain is currently in `current`.
    ///
    /// The draw uses one [`Standard`](tpt_math_prob_core::Standard) uniform
    /// from `rng` and inverse-CDF sampling over the outgoing row, so it is
    /// reproducible for a seeded generator.
    ///
    /// If `current` is out of range, or its row carries no probability mass,
    /// `current` is returned unchanged (an implicit self-loop).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_core::SplitMix64;
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    /// let mut rng = SplitMix64::seed_from_u64(1);
    /// assert_eq!(chain.step(&mut rng, 0), 1);
    /// assert_eq!(chain.step(&mut rng, 1), 0);
    /// ```
    pub fn step<R: Rng + ?Sized>(&self, rng: &mut R, current: usize) -> usize {
        sample_categorical(rng, self.row(current)).unwrap_or(current)
    }

    /// Simulate a trajectory of exactly `n` states.
    ///
    /// The first element is `init`, so the walk performs `n - 1` transitions;
    /// `n == 0` yields an empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_core::SplitMix64;
    /// use tpt_math_prob_markov::MarkovChain;
    ///
    /// let chain = MarkovChain::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    /// let mut rng = SplitMix64::seed_from_u64(42);
    /// assert_eq!(chain.run(&mut rng, 0, 5), vec![0, 1, 0, 1, 0]);
    /// ```
    pub fn run<R: Rng + ?Sized>(&self, rng: &mut R, init: usize, n: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(n);
        if n == 0 {
            return path;
        }
        let mut current = init;
        path.push(current);
        for _ in 1..n {
            current = self.step(rng, current);
            path.push(current);
        }
        path
    }

    /// How often each state is visited along a simulated trajectory of `n`
    /// states, as a distribution summing to `1`.
    ///
    /// For a long run of an irreducible, aperiodic chain this is a Monte Carlo
    /// estimate of [`stationary`](Self::stationary).
    pub fn empirical_distribution<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        init: usize,
        n: usize,
    ) -> Vec<f64> {
        let mut counts = vec![0.0; self.states];
        for state in self.run(rng, init, n) {
            if let Some(slot) = counts.get_mut(state) {
                *slot += 1.0;
            }
        }
        if n > 0 {
            for c in counts.iter_mut() {
                *c /= n as f64;
            }
        }
        counts
    }

    /// Shape and entry checks shared by [`validate`](Self::validate) and
    /// [`from_rows_normalized`](Self::from_rows_normalized).
    fn check_shape(&self) -> Result<(), MarkovError> {
        if self.states == 0 {
            return Err(MarkovError::ZeroStates);
        }
        if self.transition.len() != self.states {
            return Err(MarkovError::DimensionMismatch {
                expected: self.states,
                found: self.transition.len(),
            });
        }
        for row in &self.transition {
            if row.len() != self.states {
                return Err(MarkovError::DimensionMismatch {
                    expected: self.states,
                    found: row.len(),
                });
            }
            for &p in row {
                if !p.is_finite() || p < 0.0 {
                    return Err(MarkovError::InvalidProbability { value: p });
                }
            }
        }
        Ok(())
    }

    /// Power iteration returning `(iterate, iterations_performed, residual)`.
    fn power_iterate(&self, tol: f64, max_iter: usize) -> (Vec<f64>, usize, f64) {
        let n = self.states;
        if n == 0 {
            return (Vec::new(), 0, 0.0);
        }
        let mut pi = vec![1.0 / n as f64; n];
        let mut residual = f64::INFINITY;
        let mut iterations = 0;
        for _ in 0..max_iter {
            iterations += 1;
            let mut next = self.step_distribution(&pi);
            // Renormalize to contain drift (and to cope with slightly
            // sub-stochastic rows supplied by the caller).
            let sum: f64 = next.iter().sum();
            if sum.is_finite() && sum > 0.0 {
                for p in next.iter_mut() {
                    *p /= sum;
                }
            } else {
                // All mass has vanished; the uniform vector is the best
                // answer we can give.
                return (vec![1.0 / n as f64; n], iterations, f64::INFINITY);
            }
            residual = pi.iter().zip(next.iter()).map(|(a, b)| (a - b).abs()).sum();
            pi = next;
            if residual <= tol {
                break;
            }
        }
        (pi, iterations, residual)
    }
}

/// Rescale `row` to sum to `1`, falling back to the uniform distribution when
/// it carries no usable mass.
fn normalize_in_place_or_uniform(row: &mut [f64]) {
    if row.is_empty() {
        return;
    }
    let sum: f64 = row.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for p in row.iter_mut() {
            *p /= sum;
        }
    } else {
        let p = 1.0 / row.len() as f64;
        for slot in row.iter_mut() {
            *slot = p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::SplitMix64;

    /// The worked two-state example: `pi = [5/6, 1/6]`.
    fn two_state() -> MarkovChain {
        MarkovChain::from_rows(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap()
    }

    #[test]
    fn new_is_zeroed_and_square() {
        let chain = MarkovChain::new(3);
        assert_eq!(chain.states, 3);
        assert_eq!(chain.transition.len(), 3);
        assert!(chain.transition.iter().all(|r| r.len() == 3));
        assert!(chain.transition.iter().flatten().all(|&p| p == 0.0));
    }

    #[test]
    fn set_transition_and_normalize_row_build_a_valid_chain() {
        let mut chain = MarkovChain::new(2);
        chain.set_transition(0, 0, 9.0);
        chain.set_transition(0, 1, 1.0);
        chain.set_transition(1, 0, 1.0);
        chain.set_transition(1, 1, 1.0);
        chain.normalize();

        assert!(chain.validate().is_ok());
        assert!((chain.transition_prob(0, 0) - 0.9).abs() < 1e-15);
        assert!((chain.transition_prob(0, 1) - 0.1).abs() < 1e-15);
        assert_eq!(chain.transition_prob(1, 0), 0.5);
    }

    #[test]
    fn normalize_row_makes_an_empty_row_uniform() {
        let mut chain = MarkovChain::new(4);
        chain.normalize_row(2);
        assert_eq!(chain.row(2), &[0.25, 0.25, 0.25, 0.25]);
        assert_eq!(
            chain.try_normalize_row(0).unwrap_err(),
            MarkovError::DegenerateRow { row: 0 }
        );
    }

    #[test]
    fn try_set_transition_rejects_bad_input() {
        let mut chain = MarkovChain::new(2);
        assert_eq!(
            chain.try_set_transition(2, 0, 0.5).unwrap_err(),
            MarkovError::StateOutOfRange {
                state: 2,
                states: 2
            }
        );
        assert_eq!(
            chain.try_set_transition(0, 2, 0.5).unwrap_err(),
            MarkovError::StateOutOfRange {
                state: 2,
                states: 2
            }
        );
        assert!(matches!(
            chain.try_set_transition(0, 0, -0.5).unwrap_err(),
            MarkovError::InvalidProbability { .. }
        ));
        assert!(matches!(
            chain.try_set_transition(0, 0, f64::NAN).unwrap_err(),
            MarkovError::InvalidProbability { .. }
        ));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn normalize_row_panics_on_bad_index() {
        MarkovChain::new(2).normalize_row(5);
    }

    #[test]
    fn validate_catches_malformed_chains() {
        assert_eq!(
            MarkovChain::new(0).validate().unwrap_err(),
            MarkovError::ZeroStates
        );

        let ragged = MarkovChain {
            states: 2,
            transition: vec![vec![1.0], vec![0.5, 0.5]],
        };
        assert_eq!(
            ragged.validate().unwrap_err(),
            MarkovError::DimensionMismatch {
                expected: 2,
                found: 1
            }
        );

        let unnormalized = MarkovChain {
            states: 1,
            transition: vec![vec![0.5]],
        };
        assert!(matches!(
            unnormalized.validate().unwrap_err(),
            MarkovError::NotStochastic { row: 0, .. }
        ));

        assert!(!unnormalized.is_stochastic(PROBABILITY_TOLERANCE));
        assert!(two_state().is_stochastic(PROBABILITY_TOLERANCE));
    }

    #[test]
    fn from_rows_rejects_non_square_and_non_stochastic_input() {
        assert!(MarkovChain::from_rows(vec![vec![1.0, 0.0]]).is_err());
        assert!(MarkovChain::from_rows(vec![vec![0.6, 0.6], vec![0.5, 0.5]]).is_err());
        assert!(MarkovChain::from_rows_normalized(vec![vec![0.0, 0.0], vec![1.0, 1.0]]).is_err());

        let counts = MarkovChain::from_rows_normalized(vec![vec![3.0, 1.0], vec![2.0, 2.0]])
            .expect("counts normalize to a valid chain");
        assert_eq!(counts.transition_prob(0, 0), 0.75);
        assert!(counts.validate().is_ok());
    }

    #[test]
    fn stationary_matches_the_analytic_two_state_solution() {
        let pi = two_state().stationary();
        assert_eq!(pi.len(), 2);
        assert!((pi[0] - 5.0 / 6.0).abs() < 1e-12, "pi = {pi:?}");
        assert!((pi[1] - 1.0 / 6.0).abs() < 1e-12, "pi = {pi:?}");
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stationary_is_a_fixed_point_of_the_transition_matrix() {
        let chain = MarkovChain::from_rows(vec![
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.1, 0.8],
            vec![0.6, 0.2, 0.2],
        ])
        .unwrap();
        let pi = chain.stationary();
        let stepped = chain.step_distribution(&pi);
        for (a, b) in pi.iter().zip(stepped.iter()) {
            assert!((a - b).abs() < 1e-10, "pi = {pi:?}, pi*P = {stepped:?}");
        }
    }

    #[test]
    fn stationary_of_a_uniform_chain_is_uniform() {
        let pi = MarkovChain::uniform(5).stationary();
        assert!(pi.iter().all(|&p| (p - 0.2).abs() < 1e-12));
    }

    /// A bipartite (period-2) chain with stationary distribution
    /// `[0.5, 0.25, 0.25]`, on which plain power iteration oscillates.
    fn periodic() -> MarkovChain {
        MarkovChain::from_rows(vec![
            vec![0.0, 0.5, 0.5],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ])
        .unwrap()
    }

    #[test]
    fn try_stationary_reports_non_convergence_for_a_periodic_chain() {
        assert!(matches!(
            periodic().try_stationary(1e-12, 500),
            Err(MarkovError::NotConverged { .. })
        ));
    }

    #[test]
    fn try_stationary_rejects_a_non_stochastic_chain() {
        let bogus = MarkovChain {
            states: 2,
            transition: vec![vec![0.9, 0.9], vec![0.5, 0.5]],
        };
        assert!(matches!(
            bogus.try_stationary(1e-12, 10),
            Err(MarkovError::NotStochastic { row: 0, .. })
        ));
    }

    #[test]
    fn the_lazy_form_of_a_periodic_chain_converges_to_its_stationary_law() {
        let pi = periodic().lazy().try_stationary(1e-12, 100_000).unwrap();
        let expected = [0.5, 0.25, 0.25];
        for (a, b) in pi.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9, "pi = {pi:?}");
        }
    }

    #[test]
    fn lazy_chain_preserves_the_stationary_distribution() {
        let chain = two_state();
        let pi = chain.stationary();
        let lazy_pi = chain.lazy().stationary();
        for (a, b) in pi.iter().zip(lazy_pi.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn distribution_after_agrees_with_repeated_steps() {
        let chain = two_state();
        let mut manual = vec![1.0, 0.0];
        for _ in 0..7 {
            manual = chain.step_distribution(&manual);
        }
        let direct = chain.distribution_after(&[1.0, 0.0], 7);
        for (a, b) in manual.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn step_follows_a_deterministic_chain() {
        let flip = MarkovChain::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
        let mut rng = SplitMix64::seed_from_u64(7);
        let mut state = 0;
        for expected in [1, 0, 1, 0, 1, 0] {
            state = flip.step(&mut rng, state);
            assert_eq!(state, expected);
        }
    }

    #[test]
    fn step_on_an_empty_row_stays_put() {
        let chain = MarkovChain::new(2);
        let mut rng = SplitMix64::seed_from_u64(3);
        assert_eq!(chain.step(&mut rng, 1), 1);
        // Out-of-range states are returned unchanged too.
        assert_eq!(chain.step(&mut rng, 99), 99);
    }

    #[test]
    fn run_returns_exactly_n_states_starting_at_init() {
        let chain = two_state();
        let mut rng = SplitMix64::seed_from_u64(11);
        let path = chain.run(&mut rng, 1, 32);
        assert_eq!(path.len(), 32);
        assert_eq!(path[0], 1);
        assert!(path.iter().all(|&s| s < 2));
        assert!(chain.run(&mut rng, 0, 0).is_empty());
    }

    #[test]
    fn run_is_reproducible_for_a_seeded_rng() {
        let chain = two_state();
        let a = chain.run(&mut SplitMix64::seed_from_u64(2024), 0, 64);
        let b = chain.run(&mut SplitMix64::seed_from_u64(2024), 0, 64);
        assert_eq!(a, b);
    }

    #[test]
    fn simulation_frequencies_approach_the_stationary_distribution() {
        let chain = two_state();
        let pi = chain.stationary();
        let mut rng = SplitMix64::seed_from_u64(20_260_810);
        let empirical = chain.empirical_distribution(&mut rng, 0, 400_000);
        for (a, b) in pi.iter().zip(empirical.iter()) {
            assert!(
                (a - b).abs() < 5e-3,
                "stationary = {pi:?}, empirical = {empirical:?}"
            );
        }
    }
}
