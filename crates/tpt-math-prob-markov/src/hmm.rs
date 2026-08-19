//! Hidden Markov models: decoding, likelihood, and generation.
//!
//! See [`Hmm`] for the main type.

use crate::chain::MarkovChain;
use crate::error::MarkovError;
use crate::{sample_categorical, PROBABILITY_TOLERANCE};
use tpt_math_prob_core::Rng;

/// A discrete hidden Markov model.
///
/// A hidden state `X_t` evolves as a Markov chain over `states` states while
/// only an emitted symbol `Y_t`, drawn from an alphabet of `observations`
/// symbols, is visible:
///
/// * `initial[i] = P(X_0 = i)`,
/// * `transition[i][j] = P(X_{t+1} = j | X_t = i)`,
/// * `emission[i][k] = P(Y_t = k | X_t = i)`.
///
/// # Invariants
///
/// As with [`MarkovChain`], the fields are public and the constructors keep
/// them consistent; if you build one by hand you are responsible for:
///
/// * `initial.len() == states` and `initial` sums to `1`;
/// * `transition` is `states` x `states`, row-stochastic;
/// * `emission` is `states` x `observations`, row-stochastic.
///
/// [`validate`](Hmm::validate) checks all of it and
/// [`normalize`](Hmm::normalize) rescales rows in place. Reads never go out of
/// bounds: a missing entry counts as probability `0`.
///
/// # Examples
///
/// The classic health/thermometer model: the patient is `0 = healthy` or
/// `1 = fever`, and you only observe `0 = normal`, `1 = cold`, `2 = dizzy`.
///
/// ```
/// use tpt_math_prob_markov::Hmm;
///
/// let model = Hmm::from_parts(
///     vec![0.6, 0.4],
///     vec![vec![0.7, 0.3], vec![0.4, 0.6]],
///     vec![vec![0.5, 0.4, 0.1], vec![0.1, 0.3, 0.6]],
/// )
/// .unwrap();
///
/// // normal, cold, dizzy  =>  healthy, healthy, fever
/// assert_eq!(model.viterbi(&[0, 1, 2]), vec![0, 0, 1]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Hmm {
    /// Number of hidden states; state indices are `0..states`.
    pub states: usize,
    /// Size of the observation alphabet; symbols are `0..observations`.
    pub observations: usize,
    /// Hidden-state transition matrix, `states` x `states`.
    pub transition: Vec<Vec<f64>>,
    /// Emission matrix, `states` x `observations`.
    pub emission: Vec<Vec<f64>>,
    /// Distribution of the initial hidden state, length `states`.
    pub initial: Vec<f64>,
}

/// A spelled-out alias for [`Hmm`].
pub type HiddenMarkovModel = Hmm;

impl Hmm {
    /// Create a model with all-zero matrices and an all-zero initial
    /// distribution.
    ///
    /// Fill it with [`set_initial`](Self::set_initial),
    /// [`set_transition`](Self::set_transition) and
    /// [`set_emission`](Self::set_emission), then call
    /// [`normalize`](Self::normalize) or supply rows that already sum to `1`.
    pub fn new(states: usize, observations: usize) -> Self {
        Hmm {
            states,
            observations,
            transition: vec![vec![0.0; states]; states],
            emission: vec![vec![0.0; observations]; states],
            initial: vec![0.0; states],
        }
    }

    /// Create a model in which every distribution is uniform.
    pub fn uniform(states: usize, observations: usize) -> Self {
        let ps = if states == 0 {
            0.0
        } else {
            1.0 / states as f64
        };
        let po = if observations == 0 {
            0.0
        } else {
            1.0 / observations as f64
        };
        Hmm {
            states,
            observations,
            transition: vec![vec![ps; states]; states],
            emission: vec![vec![po; observations]; states],
            initial: vec![ps; states],
        }
    }

    /// Build a model from an initial distribution, a transition matrix and an
    /// emission matrix.
    ///
    /// The sizes are inferred: `states` from `initial`, `observations` from the
    /// width of `emission`.
    ///
    /// # Errors
    ///
    /// Returns a [`MarkovError`] if any matrix has the wrong shape, contains a
    /// negative or non-finite entry, or has a row that does not sum to `1`
    /// (within [`PROBABILITY_TOLERANCE`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// let model = Hmm::from_parts(
    ///     vec![1.0, 0.0],
    ///     vec![vec![0.5, 0.5], vec![0.5, 0.5]],
    ///     vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    /// )
    /// .unwrap();
    /// assert_eq!((model.states, model.observations), (2, 2));
    /// ```
    pub fn from_parts(
        initial: Vec<f64>,
        transition: Vec<Vec<f64>>,
        emission: Vec<Vec<f64>>,
    ) -> Result<Self, MarkovError> {
        let states = initial.len();
        let observations = emission.first().map_or(0, Vec::len);
        let model = Hmm {
            states,
            observations,
            transition,
            emission,
            initial,
        };
        model.validate()?;
        Ok(model)
    }

    /// `P(X_0 = state)`; `0.0` for an out-of-range state.
    pub fn initial_prob(&self, state: usize) -> f64 {
        self.initial.get(state).copied().unwrap_or(0.0)
    }

    /// `P(X_{t+1} = to | X_t = from)`; `0.0` for an out-of-range state.
    pub fn transition_prob(&self, from: usize, to: usize) -> f64 {
        self.transition
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(0.0)
    }

    /// `P(Y_t = observation | X_t = state)`; `0.0` for an out-of-range index.
    pub fn emission_prob(&self, state: usize, observation: usize) -> f64 {
        self.emission
            .get(state)
            .and_then(|row| row.get(observation))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set `P(X_0 = state)`.
    ///
    /// # Panics
    ///
    /// Panics if `state` is out of range or `prob` is not a finite,
    /// non-negative number. Use [`try_set_initial`](Self::try_set_initial) to
    /// get a [`MarkovError`] instead.
    pub fn set_initial(&mut self, state: usize, prob: f64) {
        self.try_set_initial(state, prob)
            .expect("set_initial: invalid argument")
    }

    /// Checked form of [`set_initial`](Self::set_initial).
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::StateOutOfRange`] if `state` is not a valid state,
    /// or [`MarkovError::InvalidProbability`] if `prob` is negative, infinite,
    /// or `NaN`.
    pub fn try_set_initial(&mut self, state: usize, prob: f64) -> Result<(), MarkovError> {
        if !prob.is_finite() || prob < 0.0 {
            return Err(MarkovError::InvalidProbability { value: prob });
        }
        let states = self.states;
        let slot = self
            .initial
            .get_mut(state)
            .ok_or(MarkovError::StateOutOfRange { state, states })?;
        *slot = prob;
        Ok(())
    }

    /// Set `P(X_{t+1} = to | X_t = from)`.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is out of range, or `prob` is not a finite,
    /// non-negative number. Use [`try_set_transition`](Self::try_set_transition)
    /// to get a [`MarkovError`] instead.
    pub fn set_transition(&mut self, from: usize, to: usize, prob: f64) {
        self.try_set_transition(from, to, prob)
            .expect("set_transition: invalid argument")
    }

    /// Checked form of [`set_transition`](Self::set_transition).
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::StateOutOfRange`] if `from` or `to` is not a valid
    /// state, or [`MarkovError::InvalidProbability`] if `prob` is negative,
    /// infinite, or `NaN`.
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
        let row = self
            .transition
            .get_mut(from)
            .ok_or(MarkovError::StateOutOfRange {
                state: from,
                states,
            })?;
        row.get_mut(to)
            .ok_or(MarkovError::StateOutOfRange { state: to, states })?;
        self.transition[from][to] = prob;
        Ok(())
    }

    /// Set `P(Y_t = observation | X_t = state)`.
    ///
    /// # Panics
    ///
    /// Panics if `state` or `observation` is out of range, or `prob` is not a
    /// finite, non-negative number. Use [`try_set_emission`](Self::try_set_emission)
    /// to get a [`MarkovError`] instead.
    pub fn set_emission(&mut self, state: usize, observation: usize, prob: f64) {
        self.try_set_emission(state, observation, prob)
            .expect("set_emission: invalid argument")
    }

    /// Checked form of [`set_emission`](Self::set_emission).
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::StateOutOfRange`] if `state` is not a valid state,
    /// [`MarkovError::ObservationOutOfRange`] if `observation` is not a valid
    /// symbol, or [`MarkovError::InvalidProbability`] if `prob` is negative,
    /// infinite, or `NaN`.
    pub fn try_set_emission(
        &mut self,
        state: usize,
        observation: usize,
        prob: f64,
    ) -> Result<(), MarkovError> {
        if !prob.is_finite() || prob < 0.0 {
            return Err(MarkovError::InvalidProbability { value: prob });
        }
        let (states, observations) = (self.states, self.observations);
        let row = self
            .emission
            .get_mut(state)
            .ok_or(MarkovError::StateOutOfRange { state, states })?;
        row.get_mut(observation)
            .ok_or(MarkovError::ObservationOutOfRange {
                observation,
                observations,
            })?;
        self.emission[state][observation] = prob;
        Ok(())
    }

    /// Rescale the initial distribution and every transition and emission row
    /// so that each sums to `1`.
    ///
    /// Rows carrying no mass become uniform, which keeps a partially specified
    /// model usable.
    pub fn normalize(&mut self) {
        normalize_or_uniform(&mut self.initial);
        for row in self.transition.iter_mut().chain(self.emission.iter_mut()) {
            normalize_or_uniform(row);
        }
    }

    /// Check every documented invariant of the model.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::ZeroStates`] or [`MarkovError::ZeroObservations`]
    /// for an empty model, [`MarkovError::DimensionMismatch`] for a
    /// wrongly-shaped matrix or initial vector,
    /// [`MarkovError::InvalidProbability`] for a negative or non-finite entry,
    /// and [`MarkovError::NotStochastic`] for a row that does not sum to `1`.
    pub fn validate(&self) -> Result<(), MarkovError> {
        if self.states == 0 {
            return Err(MarkovError::ZeroStates);
        }
        if self.observations == 0 {
            return Err(MarkovError::ZeroObservations);
        }
        check_dist(&self.initial, self.states, 0)?;
        check_matrix(&self.transition, self.states, self.states)?;
        check_matrix(&self.emission, self.states, self.observations)?;
        Ok(())
    }

    /// The hidden-state process on its own, as a [`MarkovChain`].
    ///
    /// Handy for asking questions about the latent dynamics, such as the
    /// long-run share of time spent in each hidden state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// let model = Hmm::from_parts(
    ///     vec![0.6, 0.4],
    ///     vec![vec![0.7, 0.3], vec![0.4, 0.6]],
    ///     vec![vec![0.5, 0.4, 0.1], vec![0.1, 0.3, 0.6]],
    /// )
    /// .unwrap();
    /// let pi = model.hidden_chain().stationary();
    /// assert!((pi[0] - 4.0 / 7.0).abs() < 1e-9);
    /// ```
    pub fn hidden_chain(&self) -> MarkovChain {
        MarkovChain {
            states: self.states,
            transition: self.transition.clone(),
        }
    }

    /// The most likely hidden-state sequence for `observations` (the Viterbi
    /// path).
    ///
    /// Returns a vector the same length as `observations`; an empty input
    /// yields an empty path. The recursion runs in log space, so long
    /// sequences do not underflow.
    ///
    /// If no path can produce the observations (their joint probability is
    /// zero), the returned path is the one the tie-breaking rule reaches
    /// first — check [`log_likelihood`](Self::log_likelihood) for
    /// `-inf` when that matters.
    ///
    /// # Panics
    ///
    /// Panics if the model has no states or if any symbol is outside
    /// `0..observations`. Use [`try_viterbi`](Self::try_viterbi) to get a
    /// [`MarkovError`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// // Two states that almost always emit their own symbol.
    /// let model = Hmm::from_parts(
    ///     vec![0.5, 0.5],
    ///     vec![vec![0.9, 0.1], vec![0.1, 0.9]],
    ///     vec![vec![0.95, 0.05], vec![0.05, 0.95]],
    /// )
    /// .unwrap();
    /// assert_eq!(model.viterbi(&[0, 0, 1, 1]), vec![0, 0, 1, 1]);
    /// ```
    pub fn viterbi(&self, observations: &[usize]) -> Vec<usize> {
        self.try_viterbi(observations)
            .expect("viterbi: invalid model or observation")
    }

    /// Checked form of [`viterbi`](Self::viterbi).
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::ZeroStates`] if the model has no states, or
    /// [`MarkovError::ObservationOutOfRange`] if a symbol is outside
    /// `0..observations`.
    pub fn try_viterbi(&self, observations: &[usize]) -> Result<Vec<usize>, MarkovError> {
        self.try_viterbi_with_log_prob(observations)
            .map(|(path, _)| path)
    }

    /// The Viterbi path together with its log probability,
    /// `ln P(path, observations)`.
    ///
    /// The log probability is `-inf` when no path can explain the
    /// observations.
    ///
    /// # Errors
    ///
    /// As for [`try_viterbi`](Self::try_viterbi).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// let model = Hmm::from_parts(
    ///     vec![1.0, 0.0],
    ///     vec![vec![0.0, 1.0], vec![1.0, 0.0]],
    ///     vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    /// )
    /// .unwrap();
    ///
    /// // The only possible path has probability 1.
    /// let (path, log_prob) = model.try_viterbi_with_log_prob(&[0, 1, 0]).unwrap();
    /// assert_eq!(path, vec![0, 1, 0]);
    /// assert!(log_prob.abs() < 1e-12);
    ///
    /// // This sequence cannot be emitted at all.
    /// let (_, impossible) = model.try_viterbi_with_log_prob(&[0, 0]).unwrap();
    /// assert_eq!(impossible, f64::NEG_INFINITY);
    /// ```
    pub fn try_viterbi_with_log_prob(
        &self,
        observations: &[usize],
    ) -> Result<(Vec<usize>, f64), MarkovError> {
        if self.states == 0 {
            return Err(MarkovError::ZeroStates);
        }
        self.check_observations(observations)?;
        let steps = observations.len();
        if steps == 0 {
            return Ok((Vec::new(), 0.0));
        }
        let n = self.states;

        // delta[i]: log probability of the best path ending in state `i`.
        let mut delta: Vec<f64> = (0..n)
            .map(|i| ln(self.initial_prob(i)) + ln(self.emission_prob(i, observations[0])))
            .collect();
        // backpointer[t][j]: best predecessor of state `j` at time `t + 1`.
        let mut backpointer: Vec<Vec<usize>> = Vec::with_capacity(steps.saturating_sub(1));

        for &symbol in &observations[1..] {
            let mut next = vec![f64::NEG_INFINITY; n];
            let mut best_prev = vec![0usize; n];
            for (j, (slot, prev)) in next.iter_mut().zip(best_prev.iter_mut()).enumerate() {
                let mut best = f64::NEG_INFINITY;
                let mut arg = 0usize;
                for (i, &score) in delta.iter().enumerate() {
                    let candidate = score + ln(self.transition_prob(i, j));
                    if candidate > best {
                        best = candidate;
                        arg = i;
                    }
                }
                *slot = best + ln(self.emission_prob(j, symbol));
                *prev = arg;
            }
            delta = next;
            backpointer.push(best_prev);
        }

        let (mut state, log_prob) = argmax(&delta);
        let mut path = vec![0usize; steps];
        path[steps - 1] = state;
        for t in (0..steps - 1).rev() {
            state = backpointer[t][state];
            path[t] = state;
        }
        Ok((path, log_prob))
    }

    /// The scaled forward probabilities together with the sequence
    /// log-likelihood.
    ///
    /// Element `t` of the first return value is the filtered distribution
    /// `P(X_t = i | Y_0..Y_t)`; the second is `ln P(Y_0..Y_{T-1})`, which is
    /// `-inf` if the observations are impossible under the model.
    ///
    /// Scaling each time slice keeps the recursion numerically stable for
    /// arbitrarily long sequences.
    ///
    /// # Errors
    ///
    /// As for [`try_viterbi`](Self::try_viterbi).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// let model = Hmm::from_parts(
    ///     vec![0.5, 0.5],
    ///     vec![vec![0.9, 0.1], vec![0.1, 0.9]],
    ///     vec![vec![0.95, 0.05], vec![0.05, 0.95]],
    /// )
    /// .unwrap();
    ///
    /// let (filtered, log_likelihood) = model.forward(&[0, 0, 0]).unwrap();
    /// assert_eq!(filtered.len(), 3);
    /// // Repeated "0" observations make state 0 increasingly likely.
    /// assert!(filtered[2][0] > filtered[0][0]);
    /// assert!(log_likelihood < 0.0);
    /// ```
    pub fn forward(&self, observations: &[usize]) -> Result<(Vec<Vec<f64>>, f64), MarkovError> {
        if self.states == 0 {
            return Err(MarkovError::ZeroStates);
        }
        self.check_observations(observations)?;
        let n = self.states;
        let mut filtered = Vec::with_capacity(observations.len());
        let mut log_likelihood = 0.0;
        let mut alpha = vec![0.0; n];

        for (t, &symbol) in observations.iter().enumerate() {
            let mut next = vec![0.0; n];
            for (j, slot) in next.iter_mut().enumerate() {
                let predicted = if t == 0 {
                    self.initial_prob(j)
                } else {
                    alpha
                        .iter()
                        .enumerate()
                        .map(|(i, &a)| a * self.transition_prob(i, j))
                        .sum()
                };
                *slot = predicted * self.emission_prob(j, symbol);
            }
            let scale: f64 = next.iter().sum();
            if !scale.is_finite() || scale <= 0.0 {
                // The observations are impossible from here on.
                filtered.push(vec![0.0; n]);
                for _ in filtered.len()..observations.len() {
                    filtered.push(vec![0.0; n]);
                }
                return Ok((filtered, f64::NEG_INFINITY));
            }
            for p in next.iter_mut() {
                *p /= scale;
            }
            log_likelihood += scale.ln();
            filtered.push(next.clone());
            alpha = next;
        }

        Ok((filtered, log_likelihood))
    }

    /// `ln P(observations)` under the model, via the forward algorithm.
    ///
    /// An empty sequence has log-likelihood `0` (probability `1`); an
    /// impossible sequence has `-inf`.
    ///
    /// # Panics
    ///
    /// Panics if the model has no states or a symbol is out of range; see
    /// [`forward`](Self::forward) for the checked form.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// let model = Hmm::from_parts(
    ///     vec![1.0, 0.0],
    ///     vec![vec![0.0, 1.0], vec![1.0, 0.0]],
    ///     vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    /// )
    /// .unwrap();
    /// assert!(model.log_likelihood(&[0, 1, 0]).abs() < 1e-12);
    /// assert_eq!(model.log_likelihood(&[1, 1]), f64::NEG_INFINITY);
    /// ```
    pub fn log_likelihood(&self, observations: &[usize]) -> f64 {
        self.forward(observations)
            .expect("log_likelihood: invalid model or observation")
            .1
    }

    /// Generate `n` steps from the model, returning the hidden path and the
    /// observations emitted along it.
    ///
    /// Both vectors have length `n`. Sampling consumes two uniforms per step
    /// (one for the state, one for the symbol), so a seeded generator gives a
    /// reproducible sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_prob_core::SplitMix64;
    /// use tpt_math_prob_markov::Hmm;
    ///
    /// let model = Hmm::from_parts(
    ///     vec![0.5, 0.5],
    ///     vec![vec![0.9, 0.1], vec![0.1, 0.9]],
    ///     vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    /// )
    /// .unwrap();
    ///
    /// let mut rng = SplitMix64::seed_from_u64(5);
    /// let (hidden, observed) = model.sample(&mut rng, 20);
    /// // Emissions are deterministic here, so the states are fully revealed.
    /// assert_eq!(hidden, observed);
    /// ```
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut hidden = Vec::with_capacity(n);
        let mut observed = Vec::with_capacity(n);
        let mut state = 0usize;
        for t in 0..n {
            state = if t == 0 {
                sample_categorical(rng, &self.initial).unwrap_or(0)
            } else {
                sample_categorical(rng, self.transition.get(state).map_or(&[], Vec::as_slice))
                    .unwrap_or(state)
            };
            let symbol =
                sample_categorical(rng, self.emission.get(state).map_or(&[], Vec::as_slice))
                    .unwrap_or(0);
            hidden.push(state);
            observed.push(symbol);
        }
        (hidden, observed)
    }

    fn check_observations(&self, observations: &[usize]) -> Result<(), MarkovError> {
        for &symbol in observations {
            if symbol >= self.observations {
                return Err(MarkovError::ObservationOutOfRange {
                    observation: symbol,
                    observations: self.observations,
                });
            }
        }
        Ok(())
    }
}

/// `ln(p)`, mapping zero, negative and `NaN` inputs to `-inf` so that log-space
/// recursions can carry "impossible" without producing `NaN`.
fn ln(p: f64) -> f64 {
    if p > 0.0 {
        p.ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Index and value of the largest element, ties going to the lowest index.
fn argmax(values: &[f64]) -> (usize, f64) {
    let mut best_index = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best_value = v;
            best_index = i;
        }
    }
    (best_index, best_value)
}

fn normalize_or_uniform(row: &mut [f64]) {
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

fn check_dist(dist: &[f64], expected: usize, row: usize) -> Result<(), MarkovError> {
    if dist.len() != expected {
        return Err(MarkovError::DimensionMismatch {
            expected,
            found: dist.len(),
        });
    }
    for &p in dist {
        if !p.is_finite() || p < 0.0 {
            return Err(MarkovError::InvalidProbability { value: p });
        }
    }
    let sum: f64 = dist.iter().sum();
    if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(MarkovError::NotStochastic { row, sum });
    }
    Ok(())
}

fn check_matrix(rows: &[Vec<f64>], height: usize, width: usize) -> Result<(), MarkovError> {
    if rows.len() != height {
        return Err(MarkovError::DimensionMismatch {
            expected: height,
            found: rows.len(),
        });
    }
    for (i, row) in rows.iter().enumerate() {
        check_dist(row, width, i)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::SplitMix64;

    /// The textbook health/thermometer model.
    ///
    /// Hidden: `0 = healthy`, `1 = fever`. Observed: `0 = normal`,
    /// `1 = cold`, `2 = dizzy`.
    fn weather() -> Hmm {
        Hmm::from_parts(
            vec![0.6, 0.4],
            vec![vec![0.7, 0.3], vec![0.4, 0.6]],
            vec![vec![0.5, 0.4, 0.1], vec![0.1, 0.3, 0.6]],
        )
        .unwrap()
    }

    /// Two near-deterministic states, each emitting "its own" symbol.
    fn sticky() -> Hmm {
        Hmm::from_parts(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
        )
        .unwrap()
    }

    #[test]
    fn constructors_agree_on_shape() {
        let model = Hmm::new(2, 3);
        assert_eq!((model.states, model.observations), (2, 3));
        assert_eq!(model.transition.len(), 2);
        assert_eq!(model.emission[0].len(), 3);
        assert_eq!(model.initial.len(), 2);

        let uniform = Hmm::uniform(2, 4);
        assert!(uniform.validate().is_ok());
        assert_eq!(uniform.emission_prob(1, 3), 0.25);
        assert_eq!(uniform.initial_prob(0), 0.5);
    }

    #[test]
    fn setters_then_normalize_produce_a_valid_model() {
        let mut model = Hmm::new(2, 2);
        model.set_initial(0, 3.0);
        model.set_initial(1, 1.0);
        model.set_transition(0, 0, 1.0);
        model.set_transition(0, 1, 1.0);
        model.set_transition(1, 0, 1.0);
        model.set_transition(1, 1, 3.0);
        model.set_emission(0, 0, 4.0);
        model.set_emission(0, 1, 1.0);
        model.set_emission(1, 0, 1.0);
        model.set_emission(1, 1, 4.0);
        model.normalize();

        assert!(model.validate().is_ok());
        assert_eq!(model.initial_prob(0), 0.75);
        assert_eq!(model.transition_prob(1, 1), 0.75);
        assert_eq!(model.emission_prob(0, 0), 0.8);
    }

    #[test]
    fn try_setters_reject_bad_input() {
        let mut model = Hmm::new(2, 2);
        assert_eq!(
            model.try_set_initial(2, 0.5).unwrap_err(),
            MarkovError::StateOutOfRange {
                state: 2,
                states: 2
            }
        );
        assert!(matches!(
            model.try_set_initial(0, -0.5).unwrap_err(),
            MarkovError::InvalidProbability { .. }
        ));

        assert_eq!(
            model.try_set_transition(2, 0, 0.5).unwrap_err(),
            MarkovError::StateOutOfRange {
                state: 2,
                states: 2
            }
        );
        assert_eq!(
            model.try_set_transition(0, 2, 0.5).unwrap_err(),
            MarkovError::StateOutOfRange {
                state: 2,
                states: 2
            }
        );
        assert!(matches!(
            model.try_set_transition(0, 0, f64::NAN).unwrap_err(),
            MarkovError::InvalidProbability { .. }
        ));

        assert_eq!(
            model.try_set_emission(2, 0, 0.5).unwrap_err(),
            MarkovError::StateOutOfRange {
                state: 2,
                states: 2
            }
        );
        assert_eq!(
            model.try_set_emission(0, 2, 0.5).unwrap_err(),
            MarkovError::ObservationOutOfRange {
                observation: 2,
                observations: 2
            }
        );
    }

    #[test]
    #[should_panic(expected = "invalid argument")]
    fn set_emission_panics_on_bad_symbol() {
        Hmm::new(2, 2).set_emission(0, 7, 0.5);
    }

    #[test]
    #[should_panic(expected = "invalid argument")]
    fn set_initial_panics_on_bad_state() {
        Hmm::new(2, 2).set_initial(5, 0.5);
    }

    #[test]
    #[should_panic(expected = "invalid argument")]
    fn set_transition_panics_on_bad_state() {
        Hmm::new(2, 2).set_transition(5, 0, 0.5);
    }

    #[test]
    fn validate_catches_malformed_models() {
        assert_eq!(
            Hmm::new(0, 2).validate().unwrap_err(),
            MarkovError::ZeroStates
        );
        assert_eq!(
            Hmm::new(2, 0).validate().unwrap_err(),
            MarkovError::ZeroObservations
        );
        // `new` leaves every row at zero, which is not stochastic.
        assert!(matches!(
            Hmm::new(2, 2).validate().unwrap_err(),
            MarkovError::NotStochastic { .. }
        ));

        let mut wrong_width = weather();
        wrong_width.emission[1] = vec![0.5, 0.5];
        assert_eq!(
            wrong_width.validate().unwrap_err(),
            MarkovError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        );

        assert!(Hmm::from_parts(
            vec![0.5, 0.4],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![vec![1.0], vec![1.0]],
        )
        .is_err());
    }

    #[test]
    fn viterbi_recovers_the_textbook_health_sequence() {
        // normal, cold, dizzy => healthy, healthy, fever
        assert_eq!(weather().viterbi(&[0, 1, 2]), vec![0, 0, 1]);
    }

    #[test]
    fn viterbi_matches_brute_force_on_the_weather_model() {
        let model = weather();
        let observations = [0, 1, 2, 2, 1, 0, 0, 2];
        let (path, log_prob) = model.try_viterbi_with_log_prob(&observations).unwrap();

        // Exhaustively score all 2^8 hidden paths.
        let mut best_path = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        for code in 0..(1u32 << observations.len()) {
            let candidate: Vec<usize> = (0..observations.len())
                .map(|t| ((code >> t) & 1) as usize)
                .collect();
            let mut score = ln(model.initial_prob(candidate[0]));
            score += ln(model.emission_prob(candidate[0], observations[0]));
            for t in 1..candidate.len() {
                score += ln(model.transition_prob(candidate[t - 1], candidate[t]));
                score += ln(model.emission_prob(candidate[t], observations[t]));
            }
            if score > best_score {
                best_score = score;
                best_path = candidate;
            }
        }

        assert_eq!(path, best_path);
        assert!((log_prob - best_score).abs() < 1e-12);
    }

    #[test]
    fn viterbi_recovers_a_generated_hidden_sequence() {
        // Emissions are deterministic, so the hidden path is identifiable.
        let model = Hmm::from_parts(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let mut rng = SplitMix64::seed_from_u64(1234);
        let (hidden, observed) = model.sample(&mut rng, 200);
        assert_eq!(model.viterbi(&observed), hidden);
    }

    #[test]
    fn viterbi_handles_edge_cases() {
        let model = sticky();
        assert!(model.viterbi(&[]).is_empty());
        assert_eq!(model.viterbi(&[1]), vec![1]);
        assert_eq!(model.viterbi(&[0, 0, 0, 0, 0, 0]).len(), 6);
        assert_eq!(
            model.try_viterbi(&[0, 5]).unwrap_err(),
            MarkovError::ObservationOutOfRange {
                observation: 5,
                observations: 2
            }
        );
        assert_eq!(
            Hmm::new(0, 1).try_viterbi(&[0]).unwrap_err(),
            MarkovError::ZeroStates
        );
    }

    #[test]
    fn viterbi_is_stable_on_long_sequences() {
        let model = sticky();
        let observations: Vec<usize> = (0..5_000).map(|t| usize::from(t >= 2_500)).collect();
        let (path, log_prob) = model.try_viterbi_with_log_prob(&observations).unwrap();
        assert_eq!(path, observations);
        assert!(log_prob.is_finite() && log_prob < 0.0);
    }

    #[test]
    fn viterbi_reports_impossible_sequences() {
        let model = Hmm::from_parts(
            vec![1.0, 0.0],
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let (_, log_prob) = model.try_viterbi_with_log_prob(&[0, 0]).unwrap();
        assert_eq!(log_prob, f64::NEG_INFINITY);
        assert_eq!(model.log_likelihood(&[0, 0]), f64::NEG_INFINITY);
    }

    #[test]
    fn forward_likelihood_matches_brute_force_enumeration() {
        let model = weather();
        let observations = [2, 0, 1];
        let mut total = 0.0;
        for code in 0..(1u32 << observations.len()) {
            let path: Vec<usize> = (0..observations.len())
                .map(|t| ((code >> t) & 1) as usize)
                .collect();
            let mut p = model.initial_prob(path[0]) * model.emission_prob(path[0], observations[0]);
            for t in 1..path.len() {
                p *= model.transition_prob(path[t - 1], path[t])
                    * model.emission_prob(path[t], observations[t]);
            }
            total += p;
        }
        assert!((model.log_likelihood(&observations) - total.ln()).abs() < 1e-12);
    }

    #[test]
    fn forward_returns_filtered_distributions() {
        let model = sticky();
        let (filtered, log_likelihood) = model.forward(&[0, 0, 0, 0]).unwrap();
        assert_eq!(filtered.len(), 4);
        for slice in &filtered {
            assert!((slice.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert!(filtered[3][0] > filtered[0][0]);
        assert!(log_likelihood < 0.0);
        assert_eq!(model.log_likelihood(&[]), 0.0);
    }

    #[test]
    fn the_viterbi_path_is_never_more_likely_than_the_whole_sequence() {
        let model = weather();
        let observations = [0, 1, 2, 1, 0];
        let (_, path_log_prob) = model.try_viterbi_with_log_prob(&observations).unwrap();
        assert!(path_log_prob <= model.log_likelihood(&observations) + 1e-12);
    }

    #[test]
    fn sampling_is_reproducible_and_in_range() {
        let model = weather();
        let (h1, o1) = model.sample(&mut SplitMix64::seed_from_u64(9), 50);
        let (h2, o2) = model.sample(&mut SplitMix64::seed_from_u64(9), 50);
        assert_eq!((h1.len(), o1.len()), (50, 50));
        assert_eq!((h1, o1), (h2, o2.clone()));
        assert!(o2.iter().all(|&o| o < model.observations));
        assert!(model.log_likelihood(&o2).is_finite());
    }

    #[test]
    fn hidden_chain_exposes_the_latent_dynamics() {
        let chain = weather().hidden_chain();
        assert!(chain.validate().is_ok());
        let pi = chain.stationary();
        // Solve pi = pi P for [[0.7, 0.3], [0.4, 0.6]]: pi = [4/7, 3/7].
        assert!((pi[0] - 4.0 / 7.0).abs() < 1e-9);
        assert!((pi[1] - 3.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn hidden_markov_model_alias_refers_to_hmm() {
        let model: HiddenMarkovModel = weather();
        assert_eq!(model, weather());
    }
}
