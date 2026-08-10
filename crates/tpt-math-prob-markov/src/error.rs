//! The error type shared by the checked entry points of this crate.

use std::error::Error;
use std::fmt;

/// Errors reported by the checked (`try_*`, `validate`, `from_rows`) entry
/// points of this crate.
///
/// The infallible entry points (for example [`MarkovChain::stationary`]) never
/// produce a `MarkovError`; they either fall back to a documented default or
/// panic on a programming error such as an out-of-range state index.
///
/// [`MarkovChain::stationary`]: crate::MarkovChain::stationary
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MarkovError {
    /// The model declares zero states, so no distribution over states exists.
    ZeroStates,

    /// The model declares zero observation symbols.
    ZeroObservations,

    /// A state index was greater than or equal to the number of states.
    StateOutOfRange {
        /// The offending index.
        state: usize,
        /// The number of states in the model.
        states: usize,
    },

    /// An observation symbol was greater than or equal to the alphabet size.
    ObservationOutOfRange {
        /// The offending symbol.
        observation: usize,
        /// The number of distinct observation symbols in the model.
        observations: usize,
    },

    /// A probability was negative, infinite, or `NaN`.
    InvalidProbability {
        /// The offending value.
        value: f64,
    },

    /// A matrix or vector did not have the shape implied by the model size.
    DimensionMismatch {
        /// The length required by the model.
        expected: usize,
        /// The length actually found.
        found: usize,
    },

    /// A row (or the initial distribution) summed to zero, so it cannot be
    /// rescaled into a probability distribution.
    DegenerateRow {
        /// Index of the offending row; `0` for a standalone vector.
        row: usize,
    },

    /// A row (or the initial distribution) did not sum to `1`.
    NotStochastic {
        /// Index of the offending row; `0` for a standalone vector.
        row: usize,
        /// The sum that was found.
        sum: f64,
    },

    /// Power iteration hit its iteration cap before reaching the requested
    /// tolerance. Typically caused by a periodic chain; see
    /// [`MarkovChain::lazy`](crate::MarkovChain::lazy).
    NotConverged {
        /// Number of iterations performed.
        iterations: usize,
        /// The final L1 change between successive iterates.
        residual: f64,
    },
}

impl fmt::Display for MarkovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkovError::ZeroStates => write!(f, "the model has zero states"),
            MarkovError::ZeroObservations => write!(f, "the model has zero observation symbols"),
            MarkovError::StateOutOfRange { state, states } => {
                write!(f, "state index {state} is out of range for {states} states")
            }
            MarkovError::ObservationOutOfRange {
                observation,
                observations,
            } => write!(
                f,
                "observation symbol {observation} is out of range for {observations} symbols"
            ),
            MarkovError::InvalidProbability { value } => {
                write!(
                    f,
                    "{value} is not a valid probability (expected a finite value >= 0)"
                )
            }
            MarkovError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "dimension mismatch: expected length {expected}, found {found}"
                )
            }
            MarkovError::DegenerateRow { row } => {
                write!(f, "row {row} sums to zero and cannot be normalized")
            }
            MarkovError::NotStochastic { row, sum } => {
                write!(f, "row {row} sums to {sum}, expected 1")
            }
            MarkovError::NotConverged {
                iterations,
                residual,
            } => write!(
                f,
                "power iteration did not converge after {iterations} iterations \
                 (residual {residual})"
            ),
        }
    }
}

impl Error for MarkovError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_mentions_the_offending_values() {
        let msg = MarkovError::StateOutOfRange {
            state: 7,
            states: 3,
        }
        .to_string();
        assert!(msg.contains('7') && msg.contains('3'), "{msg}");
    }

    #[test]
    fn is_a_std_error() {
        fn assert_error<E: Error>(_: &E) {}
        assert_error(&MarkovError::ZeroStates);
    }
}
