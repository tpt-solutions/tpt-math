//! Markov chain and hidden-Markov-model primitives.
//!
//! Two small, dependency-light building blocks sit on top of the [`Rng`] and
//! [`Distribution`] traits from [`tpt_math_prob_core`]:
//!
//! * [`MarkovChain`] — a finite, time-homogeneous chain: build a transition
//!   matrix, normalize it, find its stationary distribution by power
//!   iteration, and simulate trajectories from any [`Rng`].
//! * [`Hmm`] (alias [`HiddenMarkovModel`]) — a discrete hidden Markov model
//!   with Viterbi decoding, a scaled forward pass for filtering and
//!   likelihoods, and generative sampling.
//!
//! Matrices are plain `Vec<Vec<f64>>` — row `i` is the outgoing distribution of
//! state `i` — so models can be written out literally, and nothing here needs a
//! linear-algebra backend. Every sampling entry point is generic over [`Rng`],
//! so seeding with [`SplitMix64`] makes a whole simulation reproducible.
//!
//! # Examples
//!
//! A two-state chain, its stationary distribution, and a simulated walk:
//!
//! ```
//! use tpt_math_prob_core::SplitMix64;
//! use tpt_math_prob_markov::MarkovChain;
//!
//! let mut chain = MarkovChain::new(2);
//! chain.set_transition(0, 0, 0.9);
//! chain.set_transition(0, 1, 0.1);
//! chain.set_transition(1, 0, 0.5);
//! chain.set_transition(1, 1, 0.5);
//!
//! let pi = chain.stationary();
//! assert!((pi[0] - 5.0 / 6.0).abs() < 1e-9);
//!
//! let mut rng = SplitMix64::seed_from_u64(42);
//! let path = chain.run(&mut rng, 0, 1_000);
//! assert_eq!(path.len(), 1_000);
//! ```
//!
//! Decoding a hidden sequence with Viterbi:
//!
//! ```
//! use tpt_math_prob_markov::Hmm;
//!
//! // Hidden: 0 = healthy, 1 = fever. Observed: 0 = normal, 1 = cold, 2 = dizzy.
//! let model = Hmm::from_parts(
//!     vec![0.6, 0.4],
//!     vec![vec![0.7, 0.3], vec![0.4, 0.6]],
//!     vec![vec![0.5, 0.4, 0.1], vec![0.1, 0.3, 0.6]],
//! )
//! .unwrap();
//!
//! assert_eq!(model.viterbi(&[0, 1, 2]), vec![0, 0, 1]);
//! ```
//!
//! # Conventions
//!
//! * States are `usize` indices into `0..states`; observation symbols are
//!   `usize` indices into `0..observations`.
//! * Reading a probability out of range yields `0.0` instead of panicking, so
//!   partially built models stay inspectable.
//! * Every operation that can fail on malformed input has a checked `try_*`
//!   (or `validate`) form returning [`MarkovError`]; the plain form either
//!   falls back to a documented default or panics on a caller mistake such as
//!   an out-of-range index.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod chain;
pub mod error;
pub mod hmm;

pub use chain::{MarkovChain, DEFAULT_STATIONARY_MAX_ITER, DEFAULT_STATIONARY_TOLERANCE};
pub use error::MarkovError;
pub use hmm::{HiddenMarkovModel, Hmm};

pub use tpt_math_prob_core;
pub use tpt_math_prob_core::{Distribution, Rng, Sampler, SplitMix64, Standard};

/// How far a row sum may drift from `1` before `validate` calls it an error.
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Draw an index from `weights`, treated as relative (not necessarily
/// normalized) probabilities.
///
/// This is the inverse-CDF draw used by [`MarkovChain::step`] and
/// [`Hmm::sample`]: it consumes exactly one [`Standard`] uniform from `rng` and
/// scans the weights in order, so a seeded generator yields a reproducible
/// sequence. Negative, infinite, and `NaN` weights are skipped.
///
/// Returns `None` if `weights` is empty or carries no positive, finite mass.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::SplitMix64;
/// use tpt_math_prob_markov::sample_categorical;
///
/// let mut rng = SplitMix64::seed_from_u64(3);
///
/// // Only index 1 can be drawn.
/// assert_eq!(sample_categorical(&mut rng, &[0.0, 2.0, 0.0]), Some(1));
///
/// // Weights need not sum to one.
/// let i = sample_categorical(&mut rng, &[3.0, 1.0]).unwrap();
/// assert!(i < 2);
///
/// assert_eq!(sample_categorical(&mut rng, &[]), None);
/// assert_eq!(sample_categorical(&mut rng, &[0.0, 0.0]), None);
/// ```
pub fn sample_categorical<R: Rng + ?Sized>(rng: &mut R, weights: &[f64]) -> Option<usize> {
    let total: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }

    let uniform: f64 = Standard.sample(rng);
    let target = uniform * total;
    let mut cumulative = 0.0;
    let mut last = None;
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        cumulative += w;
        last = Some(i);
        if target < cumulative {
            return Some(i);
        }
    }
    // Only reachable through floating-point drift in the final comparison.
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_categorical_respects_zero_weights() {
        let mut rng = SplitMix64::seed_from_u64(17);
        for _ in 0..1_000 {
            assert_eq!(
                sample_categorical(&mut rng, &[0.0, 1.0, 0.0, 2.0]).map(|i| i % 2),
                Some(1)
            );
        }
    }

    #[test]
    fn sample_categorical_is_none_without_usable_mass() {
        let mut rng = SplitMix64::seed_from_u64(18);
        assert_eq!(sample_categorical(&mut rng, &[]), None);
        assert_eq!(sample_categorical(&mut rng, &[0.0, 0.0]), None);
        assert_eq!(sample_categorical(&mut rng, &[-1.0, f64::NAN]), None);
        assert_eq!(sample_categorical(&mut rng, &[f64::NAN, 1.0]), Some(1));
    }

    #[test]
    fn sample_categorical_frequencies_track_the_weights() {
        let weights = [0.5, 0.3, 0.2];
        let mut counts = [0.0; 3];
        let mut rng = SplitMix64::seed_from_u64(2026);
        let draws = 200_000;
        for _ in 0..draws {
            counts[sample_categorical(&mut rng, &weights).unwrap()] += 1.0;
        }
        for (count, weight) in counts.iter().zip(weights.iter()) {
            assert!(
                (count / draws as f64 - weight).abs() < 5e-3,
                "counts = {counts:?}"
            );
        }
    }

    #[test]
    fn sample_categorical_accepts_unnormalized_weights() {
        let mut rng = SplitMix64::seed_from_u64(19);
        let mut counts = [0.0_f64; 2];
        for _ in 0..100_000 {
            counts[sample_categorical(&mut rng, &[30.0, 10.0]).unwrap()] += 1.0;
        }
        assert!((counts[0] / 100_000.0 - 0.75).abs() < 5e-3);
    }
}
