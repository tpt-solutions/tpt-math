//! Errors returned by the runtime unit machinery.

use core::fmt;

use crate::dimension::Dimension;

/// Convenience alias for results produced by this crate.
pub type Result<T, E = UnitError> = core::result::Result<T, E>;

/// Everything that can go wrong when units are resolved at runtime.
///
/// All variants carry enough context to build an actionable message for a
/// configuration file or an API payload.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UnitError {
    /// The unit symbol was not found in the registry that was consulted.
    UnknownUnit {
        /// The symbol as it was supplied by the caller.
        name: String,
    },
    /// Two dimensions were required to be equal but were not.
    DimensionMismatch {
        /// Short description of the operation that failed, e.g. `"add"`.
        context: &'static str,
        /// The dimension that was required.
        expected: Dimension,
        /// The dimension that was actually supplied.
        actual: Dimension,
    },
    /// A dimension exponent left the representable `i8` range.
    ///
    /// This only happens for absurd expressions such as `m^100 * m^100`.
    ExponentOverflow {
        /// Short description of the operation that overflowed.
        context: &'static str,
    },
    /// A root was requested of a dimension whose exponents are not divisible
    /// by the root, e.g. the square root of a `m^3` volume.
    NoIntegerRoot {
        /// The dimension the root was requested of.
        dimension: Dimension,
        /// The requested root (2 for a square root).
        root: i32,
    },
    /// A textual quantity such as `"1.5 km"` could not be parsed.
    Malformed {
        /// The offending input.
        input: String,
        /// Why the input was rejected.
        reason: &'static str,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownUnit { name } => write!(f, "unknown unit `{name}`"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in `{context}`: expected [{expected}], found [{actual}]"
            ),
            Self::ExponentOverflow { context } => {
                write!(f, "dimension exponent overflow in `{context}`")
            }
            Self::NoIntegerRoot { dimension, root } => {
                write!(f, "dimension [{dimension}] has no integer root of order {root}")
            }
            Self::Malformed { input, reason } => {
                write!(f, "malformed quantity `{input}`: {reason}")
            }
        }
    }
}

impl std::error::Error for UnitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_human_readable() {
        let err = UnitError::UnknownUnit {
            name: "furlong".to_owned(),
        };
        assert_eq!(err.to_string(), "unknown unit `furlong`");

        let err = UnitError::DimensionMismatch {
            context: "add",
            expected: Dimension::LENGTH,
            actual: Dimension::MASS,
        };
        assert_eq!(
            err.to_string(),
            "dimension mismatch in `add`: expected [m], found [kg]"
        );
    }

    #[test]
    fn implements_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&UnitError::ExponentOverflow { context: "mul" });
    }
}
