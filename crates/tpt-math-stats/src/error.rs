//! The error type shared by the checked (`try_*`) entry points of this crate.

use std::error::Error;
use std::fmt;

/// Errors reported by the checked (`try_*`) entry points of this crate.
///
/// Every panicking convenience wrapper (for example [`mean`](crate::mean))
/// simply unwraps its `try_*` twin, so a panic message is always the
/// [`Display`](fmt::Display) text of one of these variants.
///
/// Degenerate — but well defined — results are *not* errors. A sample with
/// zero variance, or a perfectly flat predictor, yields a documented
/// `NaN`/infinite statistic rather than an `Err`; see the individual
/// functions for the exact convention.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum StatsError {
    /// The sample held fewer observations than the statistic requires.
    NotEnoughData {
        /// Name of the offending argument.
        sample: &'static str,
        /// Minimum number of observations the statistic needs.
        required: usize,
        /// Number of observations actually supplied.
        found: usize,
    },

    /// Two paired inputs had different lengths.
    LengthMismatch {
        /// Length of the first argument.
        first: usize,
        /// Length of the second argument.
        second: usize,
    },

    /// An observation was `NaN` or infinite.
    NotFinite {
        /// Name of the offending argument.
        sample: &'static str,
        /// Index of the offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },

    /// A scalar parameter (such as a hypothesised mean) was `NaN` or infinite.
    ParameterNotFinite {
        /// Name of the offending parameter.
        parameter: &'static str,
        /// The offending value.
        value: f64,
    },

    /// An expected cell count was zero or negative, so the chi-squared
    /// statistic would divide by zero.
    NonPositiveExpected {
        /// Index of the offending category.
        index: usize,
        /// The offending value.
        value: f64,
    },

    /// The degrees of freedom implied by the inputs are not a usable
    /// distribution parameter (not finite, or not strictly positive).
    InvalidDegreesOfFreedom {
        /// The rejected degrees-of-freedom value.
        freedom: f64,
    },
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::NotEnoughData {
                sample,
                required,
                found,
            } => write!(
                f,
                "`{sample}` needs at least {required} observation(s), found {found}"
            ),
            StatsError::LengthMismatch { first, second } => write!(
                f,
                "paired samples must have equal lengths, found {first} and {second}"
            ),
            StatsError::NotFinite {
                sample,
                index,
                value,
            } => write!(f, "`{sample}[{index}]` is {value}, expected a finite value"),
            StatsError::ParameterNotFinite { parameter, value } => {
                write!(f, "`{parameter}` is {value}, expected a finite value")
            }
            StatsError::NonPositiveExpected { index, value } => write!(
                f,
                "`expected[{index}]` is {value}, expected a strictly positive count"
            ),
            StatsError::InvalidDegreesOfFreedom { freedom } => write!(
                f,
                "{freedom} is not a valid number of degrees of freedom \
                 (expected a finite value > 0)"
            ),
        }
    }
}

impl Error for StatsError {}

/// Validate that `values` holds at least `required` observations.
pub(crate) fn check_min_len(
    sample: &'static str,
    values: &[f64],
    required: usize,
) -> Result<(), StatsError> {
    if values.len() >= required {
        Ok(())
    } else {
        Err(StatsError::NotEnoughData {
            sample,
            required,
            found: values.len(),
        })
    }
}

/// Validate that two paired inputs have the same length.
pub(crate) fn check_equal_len(first: usize, second: usize) -> Result<(), StatsError> {
    if first == second {
        Ok(())
    } else {
        Err(StatsError::LengthMismatch { first, second })
    }
}

/// Validate that every observation in `values` is finite.
pub(crate) fn check_all_finite(sample: &'static str, values: &[f64]) -> Result<(), StatsError> {
    match values.iter().position(|v| !v.is_finite()) {
        None => Ok(()),
        Some(index) => Err(StatsError::NotFinite {
            sample,
            index,
            value: values[index],
        }),
    }
}

/// Validate that a scalar parameter is finite.
pub(crate) fn check_parameter_finite(
    parameter: &'static str,
    value: f64,
) -> Result<(), StatsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(StatsError::ParameterNotFinite { parameter, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_mentions_the_offending_values() {
        let msg = StatsError::NotEnoughData {
            sample: "samples",
            required: 2,
            found: 1,
        }
        .to_string();
        assert!(
            msg.contains("samples") && msg.contains('2') && msg.contains('1'),
            "{msg}"
        );

        let msg = StatsError::LengthMismatch {
            first: 5,
            second: 4,
        }
        .to_string();
        assert!(msg.contains('5') && msg.contains('4'), "{msg}");

        let msg = StatsError::NonPositiveExpected {
            index: 2,
            value: 0.0,
        }
        .to_string();
        assert!(msg.contains("expected[2]"), "{msg}");

        let msg = StatsError::InvalidDegreesOfFreedom { freedom: 0.0 }.to_string();
        assert!(msg.contains("degrees of freedom"), "{msg}");
    }

    #[test]
    fn validators_accept_good_input_and_reject_bad_input() {
        assert!(check_min_len("x", &[1.0, 2.0], 2).is_ok());
        assert_eq!(
            check_min_len("x", &[1.0], 2),
            Err(StatsError::NotEnoughData {
                sample: "x",
                required: 2,
                found: 1
            })
        );

        assert!(check_equal_len(3, 3).is_ok());
        assert_eq!(
            check_equal_len(3, 2),
            Err(StatsError::LengthMismatch {
                first: 3,
                second: 2
            })
        );

        assert!(check_all_finite("x", &[1.0, -2.0, 3.0]).is_ok());
        assert!(matches!(
            check_all_finite("x", &[1.0, f64::NAN]),
            Err(StatsError::NotFinite {
                sample: "x",
                index: 1,
                ..
            })
        ));
        assert!(matches!(
            check_all_finite("x", &[f64::INFINITY]),
            Err(StatsError::NotFinite { index: 0, .. })
        ));

        assert!(check_parameter_finite("mu0", 1.5).is_ok());
        assert!(matches!(
            check_parameter_finite("mu0", f64::NAN),
            Err(StatsError::ParameterNotFinite {
                parameter: "mu0",
                ..
            })
        ));
    }

    #[test]
    fn is_a_std_error() {
        fn assert_error<E: Error>(_: &E) {}
        assert_error(&StatsError::LengthMismatch {
            first: 1,
            second: 2,
        });
    }
}
