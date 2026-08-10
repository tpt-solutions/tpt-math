//! Runtime dimension-checked units for dynamic/config payloads.
//!
//! [`tpt_math_units`] (a wrapper around [`uom`](tpt_math_units::uom)) checks
//! units *at compile time*, which is the right tool whenever the units are
//! known while the code is written. It cannot help when the unit is only
//! known at runtime — a YAML file that says `speed_limit: "90 km/h"`, a JSON
//! payload with `{"value": 12, "unit": "kWh"}`, a spreadsheet column header,
//! a plugin that reports whatever unit its sensor uses. This crate fills that
//! gap:
//!
//! * [`Dimension`] — the seven SI base-dimension exponents as a `Copy` value
//!   that multiplies, divides and compares like the quantity it describes.
//! * [`UnitDef`] / [`UnitRegistry`] — a symbol table mapping unit names such
//!   as `"km"`, `"min"` or `"kWh"` onto a dimension plus the conversion to
//!   the SI base unit, with a large built-in table and room for custom units.
//! * [`DynQuantity`] — a value stored in SI base units, tagged with its
//!   dimension, whose arithmetic is dimension-checked at runtime.
//! * [`interop`] — conversions to and from the compile-time quantities of
//!   [`tpt_math_units`], so dynamic input can be validated once at the
//!   boundary and be statically typed everywhere else.
//!
//! # Examples
//!
//! ```
//! use tpt_math_units_dyn::prelude::*;
//!
//! // Values that arrived as (number, unit) pairs from a config file.
//! let distance = DynQuantity::parse(1.0, "km")?;
//! let extra = DynQuantity::parse(200.0, "m")?;
//!
//! // Same dimension: addition is allowed, and unit conversion is implicit
//! // because everything is stored in SI base units.
//! let total = distance.try_add(&extra)?;
//! assert!((total.convert_to("km")? - 1.2).abs() < 1e-12);
//!
//! // Different dimensions: multiplication composes them ...
//! let duration = DynQuantity::parse(30.0, "min")?;
//! let speed = total.try_div(&duration)?;
//! assert_eq!(*speed.dim(), Dimension::VELOCITY);
//! assert!((speed.convert_to("km/h")? - 2.4).abs() < 1e-12);
//!
//! // ... but addition is rejected.
//! let mass = DynQuantity::parse(70.0, "kg")?;
//! assert!(total.try_add(&mass).is_err());
//! # Ok::<(), UnitError>(())
//! ```
//!
//! # Design notes
//!
//! * A [`DynQuantity`] always stores its magnitude in SI base units, so no
//!   conversion bookkeeping is needed during arithmetic and comparisons are
//!   exact for values that came from the same unit.
//! * `Add`/`Sub` **panic** on a dimension mismatch (a mismatch is a bug in
//!   the same way indexing out of bounds is); use the `try_*` methods to get
//!   a [`UnitError`] instead. `Mul`/`Div` never mismatch — they compose
//!   dimensions.
//! * Unit symbols are case sensitive, as SI requires: `"m"` is metre, `"T"`
//!   is tesla and `"t"` is tonne.
//! * Degrees Celsius and Fahrenheit are supported through the affine
//!   `offset` field of [`UnitDef`].

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(clippy::all)]

pub mod dimension;
pub mod error;
pub mod interop;
pub mod quantity;
pub mod unit;

pub use crate::dimension::{BaseDimension, Dimension, BASE_COUNT};
pub use crate::error::{Result, UnitError};
pub use crate::quantity::DynQuantity;
pub use crate::unit::{
    builtin_registry, builtin_units, convert, lookup, lookup_unit, UnitDef, UnitRegistry,
    BUILTIN_UNITS,
};

/// The types most users need.
///
/// ```
/// use tpt_math_units_dyn::prelude::*;
///
/// let energy = DynQuantity::parse(2.5, "kWh").unwrap();
/// assert_eq!(*energy.dim(), Dimension::ENERGY);
/// ```
pub mod prelude {
    pub use crate::dimension::{BaseDimension, Dimension};
    pub use crate::error::UnitError;
    pub use crate::quantity::DynQuantity;
    pub use crate::unit::{convert, lookup, lookup_unit, UnitDef, UnitRegistry};
}
