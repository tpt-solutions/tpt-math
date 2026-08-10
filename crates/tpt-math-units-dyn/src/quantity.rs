//! A runtime, dimension-checked quantity.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use core::str::FromStr;
use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::error::{Result, UnitError};
use crate::unit::{builtin_registry, UnitRegistry};

/// A scalar value tagged with a runtime [`Dimension`].
///
/// The value is always stored in **SI base units** (metres, kilograms,
/// seconds, …); the unit a quantity was parsed from is only a lens used on
/// the way in and on the way out. That keeps arithmetic trivially correct: a
/// quantity created from `"km"` and one created from `"mm"` add up without
/// any conversion bookkeeping.
///
/// Dimensions are checked at *runtime*, which is exactly what compile-time
/// crates such as [`uom`](tpt_math_units::uom) cannot do for values whose
/// units are only known when a config file or an API payload is read.
///
/// # Examples
///
/// ```
/// use tpt_math_units_dyn::DynQuantity;
///
/// let distance = DynQuantity::parse(1.0, "km").unwrap();
/// let extra = DynQuantity::parse(200.0, "m").unwrap();
/// let total = distance.try_add(&extra).unwrap();
/// assert_eq!(total.value(), 1200.0); // metres, the SI base unit
/// assert!((total.convert_to("km").unwrap() - 1.2).abs() < 1e-12);
///
/// // Adding a mass to a length is rejected at runtime.
/// let mass = DynQuantity::parse(1.0, "kg").unwrap();
/// assert!(distance.try_add(&mass).is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DynQuantity {
    /// Magnitude expressed in the SI base unit of `dim`.
    value: f64,
    /// Runtime dimension of the quantity.
    dim: Dimension,
    /// Canonical SI base-unit symbol for `dim`, e.g. `"m·kg·s^-2"`.
    ///
    /// Always derived from `dim` when the quantity is built, so the two can
    /// never disagree.
    base_unit: Cow<'static, str>,
}

impl DynQuantity {
    /// Creates a quantity from a value that is already expressed in the SI
    /// base unit of `dim`.
    ///
    /// ```
    /// use tpt_math_units_dyn::{Dimension, DynQuantity};
    ///
    /// let q = DynQuantity::from_base(9.81, Dimension::ACCELERATION);
    /// assert_eq!(q.base_unit(), "m·s^-2");
    /// ```
    #[must_use]
    pub fn from_base(value: f64, dim: Dimension) -> Self {
        Self {
            value,
            dim,
            base_unit: dim.base_unit_symbol(),
        }
    }

    /// Creates a dimensionless quantity.
    #[must_use]
    pub fn dimensionless(value: f64) -> Self {
        Self::from_base(value, Dimension::DIMENSIONLESS)
    }

    /// Creates a quantity from a value expressed in the named built-in unit.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::UnknownUnit`] when `unit_name` is not a built-in
    /// unit.
    ///
    /// ```
    /// use tpt_math_units_dyn::{Dimension, DynQuantity};
    ///
    /// let q = DynQuantity::parse(2.0, "km").unwrap();
    /// assert_eq!(q.value(), 2000.0);
    /// assert_eq!(*q.dim(), Dimension::LENGTH);
    /// assert!(DynQuantity::parse(2.0, "smoot").is_err());
    /// ```
    pub fn parse(value: f64, unit_name: &str) -> Result<Self> {
        Self::parse_with(value, unit_name, builtin_registry())
    }

    /// Like [`DynQuantity::parse`], but resolves the unit in `registry`.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::UnknownUnit`] when `unit_name` is not registered.
    pub fn parse_with(value: f64, unit_name: &str, registry: &UnitRegistry) -> Result<Self> {
        registry.quantity(value, unit_name)
    }

    /// The magnitude in SI base units.
    #[inline]
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The runtime dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> &Dimension {
        &self.dim
    }

    /// The canonical SI base-unit symbol, e.g. `"m·kg·s^-2"` for a force and
    /// `"1"` for a dimensionless quantity.
    #[inline]
    #[must_use]
    pub fn base_unit(&self) -> &str {
        &self.base_unit
    }

    /// `true` when the quantity carries no dimension.
    #[inline]
    #[must_use]
    pub fn is_dimensionless(&self) -> bool {
        self.dim.is_dimensionless()
    }

    /// `true` when the magnitude is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Expresses the quantity in the named built-in unit.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::UnknownUnit`] for an unknown symbol and
    /// [`UnitError::DimensionMismatch`] when the target unit measures a
    /// different dimension.
    ///
    /// ```
    /// use tpt_math_units_dyn::DynQuantity;
    ///
    /// let hour = DynQuantity::parse(1.0, "h").unwrap();
    /// assert!((hour.convert_to("s").unwrap() - 3600.0).abs() < 1e-12);
    /// assert!(hour.convert_to("kg").is_err());
    /// ```
    pub fn convert_to(&self, unit_name: &str) -> Result<f64> {
        self.convert_to_with(unit_name, builtin_registry())
    }

    /// Like [`DynQuantity::convert_to`], but resolves the unit in `registry`.
    ///
    /// # Errors
    ///
    /// Same as [`DynQuantity::convert_to`].
    pub fn convert_to_with(&self, unit_name: &str, registry: &UnitRegistry) -> Result<f64> {
        let def = registry
            .get(unit_name)
            .ok_or_else(|| UnitError::UnknownUnit {
                name: unit_name.to_owned(),
            })?;
        if def.dimension != self.dim {
            return Err(UnitError::DimensionMismatch {
                context: "convert_to",
                expected: self.dim,
                actual: def.dimension,
            });
        }
        Ok(def.from_base(self.value))
    }

    /// Renders the quantity in the named unit, e.g. `"1.2 km"`.
    ///
    /// # Errors
    ///
    /// Same as [`DynQuantity::convert_to`].
    pub fn to_string_in(&self, unit_name: &str) -> Result<String> {
        Ok(format!("{} {unit_name}", self.convert_to(unit_name)?))
    }

    /// The magnitude of a dimensionless quantity.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::DimensionMismatch`] when the quantity has a
    /// dimension.
    pub fn as_ratio(&self) -> Result<f64> {
        if self.is_dimensionless() {
            Ok(self.value)
        } else {
            Err(UnitError::DimensionMismatch {
                context: "as_ratio",
                expected: Dimension::DIMENSIONLESS,
                actual: self.dim,
            })
        }
    }

    /// Adds two quantities, checking that the dimensions agree.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::DimensionMismatch`] when the dimensions differ.
    ///
    /// ```
    /// use tpt_math_units_dyn::DynQuantity;
    ///
    /// let km = DynQuantity::parse(1.0, "km").unwrap();
    /// let m = DynQuantity::parse(200.0, "m").unwrap();
    /// let kg = DynQuantity::parse(1.0, "kg").unwrap();
    ///
    /// assert_eq!(km.try_add(&m).unwrap().value(), 1200.0);
    /// assert!(km.try_add(&kg).is_err());
    /// ```
    pub fn try_add(&self, rhs: &Self) -> Result<Self> {
        self.check_same_dim(rhs, "add")?;
        Ok(self.with_value(self.value + rhs.value))
    }

    /// Subtracts two quantities, checking that the dimensions agree.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::DimensionMismatch`] when the dimensions differ.
    pub fn try_sub(&self, rhs: &Self) -> Result<Self> {
        self.check_same_dim(rhs, "sub")?;
        Ok(self.with_value(self.value - rhs.value))
    }

    /// Multiplies two quantities, combining their dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::ExponentOverflow`] if a dimension exponent leaves
    /// the `i8` range.
    ///
    /// ```
    /// use tpt_math_units_dyn::{Dimension, DynQuantity};
    ///
    /// let length = DynQuantity::parse(3.0, "m").unwrap();
    /// let time = DynQuantity::parse(2.0, "s").unwrap();
    /// let product = length.try_mul(&time).unwrap();
    ///
    /// assert_eq!(product.value(), 6.0);
    /// assert_eq!(*product.dim(), Dimension::LENGTH * Dimension::TIME);
    /// assert_eq!(product.base_unit(), "m·s");
    /// ```
    pub fn try_mul(&self, rhs: &Self) -> Result<Self> {
        let dim = self
            .dim
            .checked_mul(rhs.dim)
            .ok_or(UnitError::ExponentOverflow { context: "mul" })?;
        Ok(Self::from_base(self.value * rhs.value, dim))
    }

    /// Divides two quantities, combining their dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::ExponentOverflow`] if a dimension exponent leaves
    /// the `i8` range.
    pub fn try_div(&self, rhs: &Self) -> Result<Self> {
        let dim = self
            .dim
            .checked_div(rhs.dim)
            .ok_or(UnitError::ExponentOverflow { context: "div" })?;
        Ok(Self::from_base(self.value / rhs.value, dim))
    }

    /// Raises the quantity to an integer power.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::ExponentOverflow`] if a dimension exponent leaves
    /// the `i8` range.
    pub fn try_powi(&self, exp: i32) -> Result<Self> {
        let dim = self
            .dim
            .checked_powi(exp)
            .ok_or(UnitError::ExponentOverflow { context: "powi" })?;
        Ok(Self::from_base(self.value.powi(exp), dim))
    }

    /// Takes the square root of the quantity.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::NoIntegerRoot`] when the dimension exponents are
    /// not all even, e.g. for a volume.
    ///
    /// ```
    /// use tpt_math_units_dyn::{Dimension, DynQuantity};
    ///
    /// let area = DynQuantity::parse(9.0, "m^2").unwrap();
    /// let side = area.try_sqrt().unwrap();
    /// assert_eq!(side.value(), 3.0);
    /// assert_eq!(*side.dim(), Dimension::LENGTH);
    /// ```
    pub fn try_sqrt(&self) -> Result<Self> {
        let dim = self.dim.checked_root(2).ok_or(UnitError::NoIntegerRoot {
            dimension: self.dim,
            root: 2,
        })?;
        Ok(Self::from_base(self.value.sqrt(), dim))
    }

    /// The reciprocal quantity (`1 / self`), inverting the dimension.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::ExponentOverflow`] if a dimension exponent leaves
    /// the `i8` range.
    pub fn try_recip(&self) -> Result<Self> {
        let dim = self
            .dim
            .checked_recip()
            .ok_or(UnitError::ExponentOverflow { context: "recip" })?;
        Ok(Self::from_base(self.value.recip(), dim))
    }

    /// The absolute value, keeping the dimension.
    #[must_use]
    pub fn abs(&self) -> Self {
        self.with_value(self.value.abs())
    }

    /// Compares two quantities within an absolute tolerance on the SI base
    /// value; quantities of different dimensions are never approximately
    /// equal.
    #[must_use]
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.dim == other.dim && (self.value - other.value).abs() <= epsilon
    }

    fn with_value(&self, value: f64) -> Self {
        Self {
            value,
            dim: self.dim,
            base_unit: self.base_unit.clone(),
        }
    }

    fn check_same_dim(&self, rhs: &Self, context: &'static str) -> Result<()> {
        if self.dim == rhs.dim {
            Ok(())
        } else {
            Err(UnitError::DimensionMismatch {
                context,
                expected: self.dim,
                actual: rhs.dim,
            })
        }
    }
}

impl fmt::Display for DynQuantity {
    /// Formats the quantity in SI base units, e.g. `1200 m`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{} {}", self.value, self.base_unit)
        }
    }
}

/// Parses strings such as `"1.5 km"`, `"-3e2mm"` or `"0.5"` (dimensionless).
///
/// ```
/// use tpt_math_units_dyn::DynQuantity;
///
/// let q: DynQuantity = "1.5 km".parse().unwrap();
/// assert_eq!(q.value(), 1500.0);
///
/// let ratio: DynQuantity = "0.25".parse().unwrap();
/// assert!(ratio.is_dimensionless());
/// ```
impl FromStr for DynQuantity {
    type Err = UnitError;

    fn from_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(UnitError::Malformed {
                input: s.to_owned(),
                reason: "empty input",
            });
        }

        // Longest prefix that parses as a float wins, so that the `e` of
        // `1e3` is not mistaken for the start of a unit symbol.
        let mut split = 0usize;
        for (index, _) in trimmed
            .char_indices()
            .skip(1)
            .chain(core::iter::once((trimmed.len(), ' ')))
        {
            if trimmed[..index].trim_end().parse::<f64>().is_ok() {
                split = index;
            }
        }
        if split == 0 {
            return Err(UnitError::Malformed {
                input: s.to_owned(),
                reason: "expected a leading numeric value",
            });
        }

        let value: f64 = trimmed[..split]
            .trim()
            .parse()
            .map_err(|_| UnitError::Malformed {
                input: s.to_owned(),
                reason: "invalid numeric value",
            })?;
        let unit = trimmed[split..].trim();
        if unit.is_empty() {
            Ok(Self::dimensionless(value))
        } else {
            Self::parse(value, unit)
        }
    }
}

impl Neg for DynQuantity {
    type Output = Self;

    fn neg(self) -> Self {
        self.with_value(-self.value)
    }
}

impl Neg for &DynQuantity {
    type Output = DynQuantity;

    fn neg(self) -> DynQuantity {
        self.with_value(-self.value)
    }
}

impl Add<&DynQuantity> for &DynQuantity {
    type Output = DynQuantity;

    /// # Panics
    ///
    /// Panics when the dimensions differ; use [`DynQuantity::try_add`] for a
    /// fallible addition.
    fn add(self, rhs: &DynQuantity) -> DynQuantity {
        self.try_add(rhs).unwrap_or_else(|err| panic!("{err}"))
    }
}

impl Sub<&DynQuantity> for &DynQuantity {
    type Output = DynQuantity;

    /// # Panics
    ///
    /// Panics when the dimensions differ; use [`DynQuantity::try_sub`] for a
    /// fallible subtraction.
    fn sub(self, rhs: &DynQuantity) -> DynQuantity {
        self.try_sub(rhs).unwrap_or_else(|err| panic!("{err}"))
    }
}

impl Mul<&DynQuantity> for &DynQuantity {
    type Output = DynQuantity;

    /// # Panics
    ///
    /// Panics only if a dimension exponent overflows `i8`; use
    /// [`DynQuantity::try_mul`] for a fallible multiplication.
    fn mul(self, rhs: &DynQuantity) -> DynQuantity {
        self.try_mul(rhs).unwrap_or_else(|err| panic!("{err}"))
    }
}

impl Div<&DynQuantity> for &DynQuantity {
    type Output = DynQuantity;

    /// # Panics
    ///
    /// Panics only if a dimension exponent overflows `i8`; use
    /// [`DynQuantity::try_div`] for a fallible division.
    fn div(self, rhs: &DynQuantity) -> DynQuantity {
        self.try_div(rhs).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Forwards the owned/borrowed operand combinations onto the `&` / `&` impl.
macro_rules! forward_binop {
    ($trait:ident, $method:ident) => {
        impl $trait<DynQuantity> for DynQuantity {
            type Output = DynQuantity;

            fn $method(self, rhs: DynQuantity) -> DynQuantity {
                $trait::$method(&self, &rhs)
            }
        }

        impl $trait<&DynQuantity> for DynQuantity {
            type Output = DynQuantity;

            fn $method(self, rhs: &DynQuantity) -> DynQuantity {
                $trait::$method(&self, rhs)
            }
        }

        impl $trait<DynQuantity> for &DynQuantity {
            type Output = DynQuantity;

            fn $method(self, rhs: DynQuantity) -> DynQuantity {
                $trait::$method(self, &rhs)
            }
        }
    };
}

forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);
forward_binop!(Div, div);

/// Scaling by a plain number keeps the dimension.
macro_rules! scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<f64> for DynQuantity {
            type Output = DynQuantity;

            fn $method(self, rhs: f64) -> DynQuantity {
                self.with_value(self.value $op rhs)
            }
        }

        impl $trait<f64> for &DynQuantity {
            type Output = DynQuantity;

            fn $method(self, rhs: f64) -> DynQuantity {
                self.with_value(self.value $op rhs)
            }
        }
    };
}

scalar_binop!(Mul, mul, *);
scalar_binop!(Div, div, /);

impl Mul<DynQuantity> for f64 {
    type Output = DynQuantity;

    fn mul(self, rhs: DynQuantity) -> DynQuantity {
        rhs.with_value(self * rhs.value)
    }
}

impl Mul<&DynQuantity> for f64 {
    type Output = DynQuantity;

    fn mul(self, rhs: &DynQuantity) -> DynQuantity {
        rhs.with_value(self * rhs.value)
    }
}

impl Div<DynQuantity> for f64 {
    type Output = DynQuantity;

    /// # Panics
    ///
    /// Panics only if the inverted dimension overflows `i8`.
    fn div(self, rhs: DynQuantity) -> DynQuantity {
        Div::div(self, &rhs)
    }
}

impl Div<&DynQuantity> for f64 {
    type Output = DynQuantity;

    /// # Panics
    ///
    /// Panics only if the inverted dimension overflows `i8`.
    fn div(self, rhs: &DynQuantity) -> DynQuantity {
        rhs.try_recip()
            .map(|q| q.with_value(self / rhs.value))
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(value: f64, unit: &str) -> DynQuantity {
        DynQuantity::parse(value, unit).expect("built-in unit")
    }

    #[test]
    fn parsing_scales_to_base_units() {
        assert_eq!(q(1.0, "km").value(), 1000.0);
        assert_eq!(q(1.0, "g").value(), 0.001);
        assert_eq!(q(2.0, "min").value(), 120.0);
        assert!((q(25.0, "degC").value() - 298.15).abs() < 1e-12);
        assert!(matches!(
            DynQuantity::parse(1.0, "furlong"),
            Err(UnitError::UnknownUnit { .. })
        ));
    }

    #[test]
    fn accessors_expose_value_dimension_and_base_unit() {
        let force = DynQuantity::from_base(12.0, Dimension::FORCE);
        assert_eq!(force.value(), 12.0);
        assert_eq!(*force.dim(), Dimension::FORCE);
        assert_eq!(force.base_unit(), "m·kg·s^-2");
        assert!(force.is_finite());
        assert!(!force.is_dimensionless());
        assert!(DynQuantity::dimensionless(0.5).is_dimensionless());
        assert_eq!(DynQuantity::dimensionless(0.5).as_ratio().unwrap(), 0.5);
        assert!(force.as_ratio().is_err());
    }

    #[test]
    fn base_unit_always_matches_the_dimension() {
        let composed = q(3.0, "m").try_mul(&q(2.0, "s")).unwrap();
        assert_eq!(composed.base_unit(), composed.dim().base_unit_symbol());
        let negated = -composed.clone();
        assert_eq!(negated.base_unit(), composed.base_unit());
    }

    #[test]
    fn addition_requires_equal_dimensions() {
        let total = q(1.0, "km").try_add(&q(200.0, "m")).unwrap();
        assert_eq!(total.value(), 1200.0);
        assert!((total.convert_to("km").unwrap() - 1.2).abs() < 1e-12);

        let err = q(1.0, "m").try_add(&q(1.0, "kg")).unwrap_err();
        assert_eq!(
            err,
            UnitError::DimensionMismatch {
                context: "add",
                expected: Dimension::LENGTH,
                actual: Dimension::MASS,
            }
        );
    }

    #[test]
    fn subtraction_requires_equal_dimensions() {
        assert_eq!(q(1.0, "km").try_sub(&q(200.0, "m")).unwrap().value(), 800.0);
        assert!(q(1.0, "s").try_sub(&q(1.0, "A")).is_err());
    }

    #[test]
    #[should_panic(expected = "dimension mismatch in `add`")]
    fn add_operator_panics_on_mismatch() {
        let _ = q(1.0, "m") + q(1.0, "kg");
    }

    #[test]
    #[should_panic(expected = "dimension mismatch in `sub`")]
    fn sub_operator_panics_on_mismatch() {
        let _ = q(1.0, "m") - q(1.0, "kg");
    }

    #[test]
    fn multiplication_and_division_combine_dimensions() {
        let velocity = q(100.0, "km") / q(1.0, "h");
        assert_eq!(*velocity.dim(), Dimension::VELOCITY);
        assert!((velocity.convert_to("km/h").unwrap() - 100.0).abs() < 1e-9);

        let work = q(10.0, "N") * q(3.0, "m");
        assert_eq!(*work.dim(), Dimension::ENERGY);
        assert!((work.convert_to("J").unwrap() - 30.0).abs() < 1e-12);
    }

    #[test]
    fn operator_reference_combinations_agree() {
        let a = q(1.0, "m");
        let b = q(2.0, "m");
        let expected = 3.0;
        assert_eq!((&a + &b).value(), expected);
        assert_eq!((a.clone() + &b).value(), expected);
        assert_eq!((&a + b.clone()).value(), expected);
        assert_eq!((a.clone() + b.clone()).value(), expected);
    }

    #[test]
    fn scalar_operations_keep_the_dimension() {
        let length = q(2.0, "m");
        assert_eq!((&length * 3.0).value(), 6.0);
        assert_eq!((3.0 * &length).value(), 6.0);
        assert_eq!((&length / 4.0).value(), 0.5);
        assert_eq!(*(&length * 3.0).dim(), Dimension::LENGTH);

        let inverse = 1.0 / &length;
        assert_eq!(inverse.value(), 0.5);
        assert_eq!(*inverse.dim(), Dimension::LENGTH.checked_recip().unwrap());
    }

    #[test]
    fn powers_roots_and_absolute_value() {
        let side = q(3.0, "m");
        let area = side.try_powi(2).unwrap();
        assert_eq!(*area.dim(), Dimension::AREA);
        assert_eq!(area.value(), 9.0);
        assert_eq!(area.try_sqrt().unwrap().value(), 3.0);
        assert!(side.try_powi(3).unwrap().try_sqrt().is_err());
        assert_eq!((-&side).abs().value(), 3.0);
        assert_eq!(side.try_recip().unwrap().value(), 1.0 / 3.0);
    }

    #[test]
    fn conversion_checks_the_target_dimension() {
        assert!((q(1.0, "h").convert_to("s").unwrap() - 3600.0).abs() < 1e-12);
        assert!((q(1.0, "km").convert_to("mm").unwrap() - 1.0e6).abs() < 1e-6);
        assert!((q(100.0, "degC").convert_to("degF").unwrap() - 212.0).abs() < 1e-9);

        let err = q(1.0, "m").convert_to("kg").unwrap_err();
        assert!(matches!(err, UnitError::DimensionMismatch { .. }));
        assert!(matches!(
            q(1.0, "m").convert_to("furlong"),
            Err(UnitError::UnknownUnit { .. })
        ));
    }

    #[test]
    fn display_and_rendering() {
        assert_eq!(q(1.2, "km").to_string(), "1200 m");
        assert_eq!(DynQuantity::dimensionless(0.5).to_string(), "0.5");
        assert_eq!(q(1200.0, "m").to_string_in("km").unwrap(), "1.2 km");
        assert!(q(1200.0, "m").to_string_in("s").is_err());
    }

    #[test]
    fn from_str_accepts_common_spellings() {
        assert_eq!("1.5 km".parse::<DynQuantity>().unwrap().value(), 1500.0);
        assert_eq!("1.5km".parse::<DynQuantity>().unwrap().value(), 1500.0);
        assert_eq!("  -3e2 mm ".parse::<DynQuantity>().unwrap().value(), -0.3);
        assert_eq!("2e3".parse::<DynQuantity>().unwrap().value(), 2000.0);
        assert!("2e3".parse::<DynQuantity>().unwrap().is_dimensionless());
        assert_eq!(
            "60 km/h".parse::<DynQuantity>().unwrap().dim(),
            &Dimension::VELOCITY
        );

        assert!(matches!(
            "".parse::<DynQuantity>(),
            Err(UnitError::Malformed { .. })
        ));
        assert!(matches!(
            "km".parse::<DynQuantity>(),
            Err(UnitError::Malformed { .. })
        ));
        assert!(matches!(
            "1 furlong".parse::<DynQuantity>(),
            Err(UnitError::UnknownUnit { .. })
        ));
    }

    #[test]
    fn approximate_equality_is_dimension_aware() {
        let a = q(1.0, "km");
        let b = q(1_000.000_001, "m");
        assert!(a.approx_eq(&b, 1e-3));
        assert!(!a.approx_eq(&b, 1e-9));
        assert!(!a.approx_eq(&q(1000.0, "kg"), 1e-3));
    }
}
