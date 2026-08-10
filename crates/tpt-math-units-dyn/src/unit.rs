//! Unit definitions and the runtime unit registry.
//!
//! A [`UnitDef`] maps a textual symbol (the kind of thing that appears in a
//! YAML/JSON config file) onto a [`Dimension`] plus the affine transform that
//! takes a value expressed in that unit to the corresponding SI base value:
//!
//! ```text
//! base = value * scale + offset
//! ```
//!
//! Almost every unit is purely multiplicative (`offset == 0`); the offset
//! exists so that degrees Celsius and Fahrenheit can be handled correctly.
//!
//! [`lookup`] consults the process-wide [`builtin_registry`]; build your own
//! [`UnitRegistry`] when an application needs domain specific units.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use crate::dimension::Dimension;
use crate::error::{Result, UnitError};
use crate::quantity::DynQuantity;

/// A named unit: its dimension and the affine transform to SI base units.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitDef {
    /// The symbol the unit is looked up by, e.g. `"km"`.
    pub symbol: Cow<'static, str>,
    /// The dimension the unit measures.
    pub dimension: Dimension,
    /// Multiplicative factor to the SI base unit (`km` -> `1000.0`).
    pub scale: f64,
    /// Additive offset applied *after* scaling (`degC` -> `273.15`).
    pub offset: f64,
}

impl UnitDef {
    /// Creates a purely multiplicative unit from a `'static` symbol.
    ///
    /// Usable in `const`/`static` context, which is how the built-in table is
    /// defined.
    #[inline]
    pub const fn new_static(symbol: &'static str, dimension: Dimension, scale: f64) -> Self {
        Self {
            symbol: Cow::Borrowed(symbol),
            dimension,
            scale,
            offset: 0.0,
        }
    }

    /// Creates an affine unit (`base = value * scale + offset`) from a
    /// `'static` symbol.
    #[inline]
    pub const fn new_affine_static(
        symbol: &'static str,
        dimension: Dimension,
        scale: f64,
        offset: f64,
    ) -> Self {
        Self {
            symbol: Cow::Borrowed(symbol),
            dimension,
            scale,
            offset,
        }
    }

    /// Creates a purely multiplicative unit from any owned or borrowed symbol.
    ///
    /// ```
    /// use tpt_math_units_dyn::{Dimension, UnitDef};
    ///
    /// let smoot = UnitDef::new(String::from("smoot"), Dimension::LENGTH, 1.702);
    /// assert_eq!(smoot.to_base(1.0), 1.702);
    /// ```
    pub fn new(
        symbol: impl Into<Cow<'static, str>>,
        dimension: Dimension,
        scale: f64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            dimension,
            scale,
            offset: 0.0,
        }
    }

    /// Returns the same unit with an additive offset, for affine scales.
    #[must_use]
    pub fn with_offset(mut self, offset: f64) -> Self {
        self.offset = offset;
        self
    }

    /// `true` when the unit has a non-zero offset (e.g. `degC`).
    #[inline]
    pub fn is_affine(&self) -> bool {
        self.offset != 0.0
    }

    /// Converts a value expressed in this unit to the SI base unit.
    #[inline]
    pub fn to_base(&self, value: f64) -> f64 {
        value * self.scale + self.offset
    }

    /// Converts a value expressed in the SI base unit to this unit.
    #[inline]
    pub fn from_base(&self, base_value: f64) -> f64 {
        (base_value - self.offset) / self.scale
    }
}

impl fmt::Display for UnitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.symbol)
    }
}

const fn u(symbol: &'static str, dimension: Dimension, scale: f64) -> UnitDef {
    UnitDef::new_static(symbol, dimension, scale)
}

const fn a(symbol: &'static str, dimension: Dimension, scale: f64, offset: f64) -> UnitDef {
    UnitDef::new_affine_static(symbol, dimension, scale, offset)
}

/// Every unit known out of the box, in a `const`-evaluated table.
///
/// Symbols are case sensitive, as SI requires (`m` is metre, `M` is not).
/// Angles (`rad`, `deg`) and ratios (`%`, `ppm`) are dimensionless, matching
/// the SI treatment of plane angle.
pub static BUILTIN_UNITS: &[UnitDef] = &[
    // ----- length -------------------------------------------------------
    u("m", Dimension::LENGTH, 1.0),
    u("metre", Dimension::LENGTH, 1.0),
    u("meter", Dimension::LENGTH, 1.0),
    u("km", Dimension::LENGTH, 1.0e3),
    u("kilometre", Dimension::LENGTH, 1.0e3),
    u("kilometer", Dimension::LENGTH, 1.0e3),
    u("dm", Dimension::LENGTH, 1.0e-1),
    u("cm", Dimension::LENGTH, 1.0e-2),
    u("centimetre", Dimension::LENGTH, 1.0e-2),
    u("centimeter", Dimension::LENGTH, 1.0e-2),
    u("mm", Dimension::LENGTH, 1.0e-3),
    u("millimetre", Dimension::LENGTH, 1.0e-3),
    u("millimeter", Dimension::LENGTH, 1.0e-3),
    u("um", Dimension::LENGTH, 1.0e-6),
    u("µm", Dimension::LENGTH, 1.0e-6),
    u("nm", Dimension::LENGTH, 1.0e-9),
    u("in", Dimension::LENGTH, 0.0254),
    u("ft", Dimension::LENGTH, 0.3048),
    u("yd", Dimension::LENGTH, 0.9144),
    u("mi", Dimension::LENGTH, 1609.344),
    u("nmi", Dimension::LENGTH, 1852.0),
    // ----- mass ---------------------------------------------------------
    u("kg", Dimension::MASS, 1.0),
    u("kilogram", Dimension::MASS, 1.0),
    u("g", Dimension::MASS, 1.0e-3),
    u("gram", Dimension::MASS, 1.0e-3),
    u("mg", Dimension::MASS, 1.0e-6),
    u("ug", Dimension::MASS, 1.0e-9),
    u("µg", Dimension::MASS, 1.0e-9),
    u("t", Dimension::MASS, 1.0e3),
    u("tonne", Dimension::MASS, 1.0e3),
    u("lb", Dimension::MASS, 0.453_592_37),
    u("oz", Dimension::MASS, 0.028_349_523_125),
    // ----- time ---------------------------------------------------------
    u("s", Dimension::TIME, 1.0),
    u("sec", Dimension::TIME, 1.0),
    u("second", Dimension::TIME, 1.0),
    u("ms", Dimension::TIME, 1.0e-3),
    u("us", Dimension::TIME, 1.0e-6),
    u("µs", Dimension::TIME, 1.0e-6),
    u("ns", Dimension::TIME, 1.0e-9),
    u("min", Dimension::TIME, 60.0),
    u("minute", Dimension::TIME, 60.0),
    u("h", Dimension::TIME, 3600.0),
    u("hr", Dimension::TIME, 3600.0),
    u("hour", Dimension::TIME, 3600.0),
    u("d", Dimension::TIME, 86_400.0),
    u("day", Dimension::TIME, 86_400.0),
    // ----- electric current ---------------------------------------------
    u("A", Dimension::ELECTRIC_CURRENT, 1.0),
    u("ampere", Dimension::ELECTRIC_CURRENT, 1.0),
    u("mA", Dimension::ELECTRIC_CURRENT, 1.0e-3),
    u("kA", Dimension::ELECTRIC_CURRENT, 1.0e3),
    // ----- temperature (affine for Celsius/Fahrenheit) -------------------
    u("K", Dimension::TEMPERATURE, 1.0),
    u("kelvin", Dimension::TEMPERATURE, 1.0),
    a("degC", Dimension::TEMPERATURE, 1.0, 273.15),
    a("°C", Dimension::TEMPERATURE, 1.0, 273.15),
    a(
        "degF",
        Dimension::TEMPERATURE,
        5.0 / 9.0,
        273.15 - 32.0 * 5.0 / 9.0,
    ),
    a(
        "°F",
        Dimension::TEMPERATURE,
        5.0 / 9.0,
        273.15 - 32.0 * 5.0 / 9.0,
    ),
    u("degR", Dimension::TEMPERATURE, 5.0 / 9.0),
    // ----- amount of substance / luminous intensity ----------------------
    u("mol", Dimension::AMOUNT_OF_SUBSTANCE, 1.0),
    u("mole", Dimension::AMOUNT_OF_SUBSTANCE, 1.0),
    u("mmol", Dimension::AMOUNT_OF_SUBSTANCE, 1.0e-3),
    u("kmol", Dimension::AMOUNT_OF_SUBSTANCE, 1.0e3),
    u("cd", Dimension::LUMINOUS_INTENSITY, 1.0),
    u("candela", Dimension::LUMINOUS_INTENSITY, 1.0),
    // ----- dimensionless -------------------------------------------------
    u("1", Dimension::DIMENSIONLESS, 1.0),
    u("rad", Dimension::DIMENSIONLESS, 1.0),
    u("radian", Dimension::DIMENSIONLESS, 1.0),
    u("deg", Dimension::DIMENSIONLESS, std::f64::consts::PI / 180.0),
    u("°", Dimension::DIMENSIONLESS, std::f64::consts::PI / 180.0),
    u("%", Dimension::DIMENSIONLESS, 1.0e-2),
    u("ppm", Dimension::DIMENSIONLESS, 1.0e-6),
    // ----- area / volume --------------------------------------------------
    u("m^2", Dimension::AREA, 1.0),
    u("m2", Dimension::AREA, 1.0),
    u("cm^2", Dimension::AREA, 1.0e-4),
    u("cm2", Dimension::AREA, 1.0e-4),
    u("km^2", Dimension::AREA, 1.0e6),
    u("km2", Dimension::AREA, 1.0e6),
    u("ha", Dimension::AREA, 1.0e4),
    u("m^3", Dimension::VOLUME, 1.0),
    u("m3", Dimension::VOLUME, 1.0),
    u("cm^3", Dimension::VOLUME, 1.0e-6),
    u("cm3", Dimension::VOLUME, 1.0e-6),
    u("L", Dimension::VOLUME, 1.0e-3),
    u("litre", Dimension::VOLUME, 1.0e-3),
    u("liter", Dimension::VOLUME, 1.0e-3),
    u("mL", Dimension::VOLUME, 1.0e-6),
    // ----- kinematics ------------------------------------------------------
    u("m/s", Dimension::VELOCITY, 1.0),
    u("km/h", Dimension::VELOCITY, 1.0 / 3.6),
    u("kph", Dimension::VELOCITY, 1.0 / 3.6),
    u("mph", Dimension::VELOCITY, 0.447_04),
    u("kn", Dimension::VELOCITY, 1852.0 / 3600.0),
    u("ft/s", Dimension::VELOCITY, 0.3048),
    u("m/s^2", Dimension::ACCELERATION, 1.0),
    u("m/s2", Dimension::ACCELERATION, 1.0),
    u("g0", Dimension::ACCELERATION, 9.806_65),
    // ----- mechanics --------------------------------------------------------
    u("N", Dimension::FORCE, 1.0),
    u("newton", Dimension::FORCE, 1.0),
    u("kN", Dimension::FORCE, 1.0e3),
    u("MN", Dimension::FORCE, 1.0e6),
    u("lbf", Dimension::FORCE, 4.448_221_615_260_5),
    u("J", Dimension::ENERGY, 1.0),
    u("joule", Dimension::ENERGY, 1.0),
    u("kJ", Dimension::ENERGY, 1.0e3),
    u("MJ", Dimension::ENERGY, 1.0e6),
    u("Wh", Dimension::ENERGY, 3600.0),
    u("kWh", Dimension::ENERGY, 3.6e6),
    u("MWh", Dimension::ENERGY, 3.6e9),
    u("cal", Dimension::ENERGY, 4.184),
    u("kcal", Dimension::ENERGY, 4184.0),
    u("eV", Dimension::ENERGY, 1.602_176_634e-19),
    u("W", Dimension::POWER, 1.0),
    u("watt", Dimension::POWER, 1.0),
    u("mW", Dimension::POWER, 1.0e-3),
    u("kW", Dimension::POWER, 1.0e3),
    u("MW", Dimension::POWER, 1.0e6),
    u("GW", Dimension::POWER, 1.0e9),
    u("hp", Dimension::POWER, 745.699_871_582_270_2),
    u("Pa", Dimension::PRESSURE, 1.0),
    u("pascal", Dimension::PRESSURE, 1.0),
    u("hPa", Dimension::PRESSURE, 1.0e2),
    u("kPa", Dimension::PRESSURE, 1.0e3),
    u("MPa", Dimension::PRESSURE, 1.0e6),
    u("bar", Dimension::PRESSURE, 1.0e5),
    u("mbar", Dimension::PRESSURE, 1.0e2),
    u("atm", Dimension::PRESSURE, 101_325.0),
    u("psi", Dimension::PRESSURE, 6_894.757_293_168_361),
    u("torr", Dimension::PRESSURE, 101_325.0 / 760.0),
    u("Hz", Dimension::FREQUENCY, 1.0),
    u("hertz", Dimension::FREQUENCY, 1.0),
    u("kHz", Dimension::FREQUENCY, 1.0e3),
    u("MHz", Dimension::FREQUENCY, 1.0e6),
    u("GHz", Dimension::FREQUENCY, 1.0e9),
    u("rpm", Dimension::FREQUENCY, 1.0 / 60.0),
    u("kg/m^3", Dimension::DENSITY, 1.0),
    u("g/cm^3", Dimension::DENSITY, 1.0e3),
    // ----- electromagnetism ---------------------------------------------------
    u("C", Dimension::ELECTRIC_CHARGE, 1.0),
    u("coulomb", Dimension::ELECTRIC_CHARGE, 1.0),
    u("Ah", Dimension::ELECTRIC_CHARGE, 3600.0),
    u("mAh", Dimension::ELECTRIC_CHARGE, 3.6),
    u("V", Dimension::ELECTRIC_POTENTIAL, 1.0),
    u("volt", Dimension::ELECTRIC_POTENTIAL, 1.0),
    u("mV", Dimension::ELECTRIC_POTENTIAL, 1.0e-3),
    u("kV", Dimension::ELECTRIC_POTENTIAL, 1.0e3),
    u("ohm", Dimension::ELECTRICAL_RESISTANCE, 1.0),
    u("Ω", Dimension::ELECTRICAL_RESISTANCE, 1.0),
    u("kohm", Dimension::ELECTRICAL_RESISTANCE, 1.0e3),
    u("Mohm", Dimension::ELECTRICAL_RESISTANCE, 1.0e6),
    u("S", Dimension::ELECTRICAL_CONDUCTANCE, 1.0),
    u("siemens", Dimension::ELECTRICAL_CONDUCTANCE, 1.0),
    u("F", Dimension::CAPACITANCE, 1.0),
    u("farad", Dimension::CAPACITANCE, 1.0),
    u("uF", Dimension::CAPACITANCE, 1.0e-6),
    u("nF", Dimension::CAPACITANCE, 1.0e-9),
    u("pF", Dimension::CAPACITANCE, 1.0e-12),
    u("H", Dimension::INDUCTANCE, 1.0),
    u("henry", Dimension::INDUCTANCE, 1.0),
    u("mH", Dimension::INDUCTANCE, 1.0e-3),
    u("Wb", Dimension::MAGNETIC_FLUX, 1.0),
    u("weber", Dimension::MAGNETIC_FLUX, 1.0),
    u("T", Dimension::MAGNETIC_FLUX_DENSITY, 1.0),
    u("tesla", Dimension::MAGNETIC_FLUX_DENSITY, 1.0),
    u("mT", Dimension::MAGNETIC_FLUX_DENSITY, 1.0e-3),
];

/// A runtime map from unit symbol to [`UnitDef`].
///
/// Use this when an application needs units that are not built in, for
/// instance because they come from a configuration file.
///
/// ```
/// use tpt_math_units_dyn::{Dimension, UnitRegistry};
///
/// let mut registry = UnitRegistry::with_builtins();
/// registry.define("furlong", Dimension::LENGTH, 201.168);
///
/// let furlong = registry.quantity(2.0, "furlong").unwrap();
/// assert!((furlong.convert_to_with("m", &registry).unwrap() - 402.336).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UnitRegistry {
    units: HashMap<String, UnitDef>,
}

impl UnitRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
        }
    }

    /// Creates a registry pre-populated with [`BUILTIN_UNITS`].
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        for def in BUILTIN_UNITS {
            registry.insert(def.clone());
        }
        registry
    }

    /// Inserts a unit definition, returning the definition it replaced.
    pub fn insert(&mut self, def: UnitDef) -> Option<UnitDef> {
        self.units.insert(def.symbol.to_string(), def)
    }

    /// Convenience wrapper around [`UnitRegistry::insert`] for a purely
    /// multiplicative unit.
    pub fn define(
        &mut self,
        symbol: impl Into<Cow<'static, str>>,
        dimension: Dimension,
        scale: f64,
    ) -> Option<UnitDef> {
        self.insert(UnitDef::new(symbol, dimension, scale))
    }

    /// Removes a unit, returning its definition.
    pub fn remove(&mut self, symbol: &str) -> Option<UnitDef> {
        self.units.remove(symbol)
    }

    /// Looks up the full definition of a unit.
    #[must_use]
    pub fn get(&self, symbol: &str) -> Option<&UnitDef> {
        self.units.get(symbol)
    }

    /// Looks up the dimension and base scale of a unit.
    ///
    /// For affine units such as `degC` only the multiplicative part is
    /// returned; use [`UnitRegistry::get`] when the offset matters.
    #[must_use]
    pub fn lookup(&self, symbol: &str) -> Option<(Dimension, f64)> {
        self.get(symbol).map(|def| (def.dimension, def.scale))
    }

    /// `true` when the symbol is known.
    #[must_use]
    pub fn contains(&self, symbol: &str) -> bool {
        self.units.contains_key(symbol)
    }

    /// Number of registered symbols (aliases count individually).
    #[must_use]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// `true` when no unit is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Iterates over the registered symbols in unspecified order.
    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.units.keys().map(String::as_str)
    }

    /// Iterates over the registered definitions in unspecified order.
    pub fn definitions(&self) -> impl Iterator<Item = &UnitDef> {
        self.units.values()
    }

    /// Builds a [`DynQuantity`] from a value expressed in `symbol`.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::UnknownUnit`] if the symbol is not registered.
    pub fn quantity(&self, value: f64, symbol: &str) -> Result<DynQuantity> {
        let def = self
            .get(symbol)
            .ok_or_else(|| UnitError::UnknownUnit {
                name: symbol.to_owned(),
            })?;
        Ok(DynQuantity::from_base(def.to_base(value), def.dimension))
    }

    /// Converts `value` from unit `from` to unit `to`.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::UnknownUnit`] for unknown symbols and
    /// [`UnitError::DimensionMismatch`] when the two units measure different
    /// dimensions.
    pub fn convert(&self, value: f64, from: &str, to: &str) -> Result<f64> {
        self.quantity(value, from)?.convert_to_with(to, self)
    }
}

/// The process-wide registry containing [`BUILTIN_UNITS`].
///
/// It is built once, on first use.
#[must_use]
pub fn builtin_registry() -> &'static UnitRegistry {
    static REGISTRY: OnceLock<UnitRegistry> = OnceLock::new();
    REGISTRY.get_or_init(UnitRegistry::with_builtins)
}

/// All built-in unit definitions.
#[must_use]
pub fn builtin_units() -> &'static [UnitDef] {
    BUILTIN_UNITS
}

/// Looks up a built-in unit, returning its dimension and base scale.
///
/// The scale converts the unit to the SI base unit of its dimension, so
/// `lookup("km") == Some((Dimension::LENGTH, 1000.0))`.
///
/// For affine units such as `degC` only the multiplicative part is returned;
/// use [`lookup_unit`] when the offset matters.
///
/// ```
/// use tpt_math_units_dyn::{lookup, Dimension};
///
/// assert_eq!(lookup("km"), Some((Dimension::LENGTH, 1000.0)));
/// assert_eq!(lookup("min"), Some((Dimension::TIME, 60.0)));
/// assert_eq!(lookup("kg"), Some((Dimension::MASS, 1.0)));
/// assert_eq!(lookup("parsec-per-fortnight"), None);
/// ```
#[must_use]
pub fn lookup(name: &str) -> Option<(Dimension, f64)> {
    builtin_registry().lookup(name)
}

/// Looks up the full definition of a built-in unit.
#[must_use]
pub fn lookup_unit(name: &str) -> Option<&'static UnitDef> {
    builtin_registry().get(name)
}

/// Converts `value` between two built-in units.
///
/// # Errors
///
/// Returns [`UnitError::UnknownUnit`] for unknown symbols and
/// [`UnitError::DimensionMismatch`] when the units are incompatible.
///
/// ```
/// use tpt_math_units_dyn::convert;
///
/// assert!((convert(1.0, "h", "s").unwrap() - 3600.0).abs() < 1e-9);
/// assert!(convert(1.0, "kg", "m").is_err());
/// ```
pub fn convert(value: f64, from: &str, to: &str) -> Result<f64> {
    builtin_registry().convert(value, from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_symbols_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for def in BUILTIN_UNITS {
            assert!(
                seen.insert(def.symbol.as_ref()),
                "duplicate builtin unit `{}`",
                def.symbol
            );
        }
        assert_eq!(builtin_registry().len(), BUILTIN_UNITS.len());
    }

    #[test]
    fn lookup_returns_dimension_and_scale() {
        assert_eq!(lookup("m"), Some((Dimension::LENGTH, 1.0)));
        assert_eq!(lookup("km"), Some((Dimension::LENGTH, 1000.0)));
        assert_eq!(lookup("cm"), Some((Dimension::LENGTH, 0.01)));
        assert_eq!(lookup("mm"), Some((Dimension::LENGTH, 0.001)));
        assert_eq!(lookup("s"), Some((Dimension::TIME, 1.0)));
        assert_eq!(lookup("min"), Some((Dimension::TIME, 60.0)));
        assert_eq!(lookup("h"), Some((Dimension::TIME, 3600.0)));
        assert_eq!(lookup("kg"), Some((Dimension::MASS, 1.0)));
        assert_eq!(lookup("g"), Some((Dimension::MASS, 0.001)));
        assert_eq!(lookup("N"), Some((Dimension::FORCE, 1.0)));
        assert_eq!(lookup("J"), Some((Dimension::ENERGY, 1.0)));
        assert_eq!(lookup("W"), Some((Dimension::POWER, 1.0)));
        assert_eq!(lookup("nope"), None);
    }

    #[test]
    fn lookup_is_case_sensitive() {
        assert_eq!(lookup("s").map(|(d, _)| d), Some(Dimension::TIME));
        assert_eq!(
            lookup("S").map(|(d, _)| d),
            Some(Dimension::ELECTRICAL_CONDUCTANCE)
        );
        assert_eq!(lookup("KM"), None);
    }

    #[test]
    fn derived_units_have_the_right_dimensions() {
        let (dim, scale) = lookup("kWh").unwrap();
        assert_eq!(dim, Dimension::ENERGY);
        assert!((scale - 3.6e6).abs() < 1e-6);
        assert_eq!(lookup("N").unwrap().0, Dimension::MASS * Dimension::ACCELERATION);
        assert_eq!(lookup("W").unwrap().0, Dimension::ENERGY / Dimension::TIME);
        assert_eq!(lookup("Pa").unwrap().0, Dimension::FORCE / Dimension::AREA);
        assert_eq!(lookup("V").unwrap().0, Dimension::POWER / Dimension::ELECTRIC_CURRENT);
    }

    #[test]
    fn affine_units_round_trip() {
        let celsius = lookup_unit("degC").unwrap();
        assert!(celsius.is_affine());
        assert!((celsius.to_base(0.0) - 273.15).abs() < 1e-12);
        assert!((celsius.from_base(373.15) - 100.0).abs() < 1e-12);

        let fahrenheit = lookup_unit("degF").unwrap();
        assert!((fahrenheit.to_base(32.0) - 273.15).abs() < 1e-9);
        assert!((fahrenheit.from_base(373.15) - 212.0).abs() < 1e-9);

        assert!(!lookup_unit("K").unwrap().is_affine());
    }

    #[test]
    fn convert_between_builtin_units() {
        assert!((convert(1.0, "h", "s").unwrap() - 3600.0).abs() < 1e-12);
        assert!((convert(1.0, "km/h", "m/s").unwrap() - 1.0 / 3.6).abs() < 1e-12);
        assert!((convert(2.5, "kWh", "MJ").unwrap() - 9.0).abs() < 1e-9);
        assert!(convert(1.0, "kg", "s").is_err());
        assert!(convert(1.0, "kg", "nope").is_err());
    }

    #[test]
    fn custom_registry_supports_new_units() {
        let mut registry = UnitRegistry::new();
        assert!(registry.is_empty());
        registry.define("smoot", Dimension::LENGTH, 1.702);
        registry.define("m", Dimension::LENGTH, 1.0);
        assert_eq!(registry.len(), 2);
        assert!(registry.contains("smoot"));
        assert!((registry.convert(1.0, "smoot", "m").unwrap() - 1.702).abs() < 1e-12);

        // Replacing a definition returns the old one.
        let previous = registry.define("smoot", Dimension::LENGTH, 1.7);
        assert_eq!(previous.map(|d| d.scale), Some(1.702));
        assert_eq!(registry.remove("m").map(|d| d.dimension), Some(Dimension::LENGTH));
        assert!(!registry.contains("m"));
        assert_eq!(registry.symbols().collect::<Vec<_>>(), vec!["smoot"]);
        assert_eq!(registry.definitions().count(), 1);
    }

    #[test]
    fn unit_def_display_and_offset_builder() {
        let def = UnitDef::new("degC2", Dimension::TEMPERATURE, 1.0).with_offset(273.15);
        assert_eq!(def.to_string(), "degC2");
        assert!(def.is_affine());
        assert!((def.to_base(25.0) - 298.15).abs() < 1e-12);
    }
}
