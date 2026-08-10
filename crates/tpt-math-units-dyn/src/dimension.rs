//! Runtime representation of an SI dimension.
//!
//! A [`Dimension`] is the vector of the seven SI base-dimension exponents. It
//! is `Copy`, cheap to compare, and can be combined with `*` and `/` exactly
//! like the quantities it describes.

use core::fmt;
use core::ops::{Div, Mul};
use std::borrow::Cow;

/// Number of SI base dimensions (length, mass, time, current, temperature,
/// amount of substance, luminous intensity).
pub const BASE_COUNT: usize = 7;

/// One of the seven SI base dimensions.
///
/// The discriminants double as the index into [`Dimension::exponents`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum BaseDimension {
    /// Length, base unit metre (`m`).
    Length = 0,
    /// Mass, base unit kilogram (`kg`).
    Mass = 1,
    /// Time, base unit second (`s`).
    Time = 2,
    /// Electric current, base unit ampere (`A`).
    ElectricCurrent = 3,
    /// Thermodynamic temperature, base unit kelvin (`K`).
    Temperature = 4,
    /// Amount of substance, base unit mole (`mol`).
    AmountOfSubstance = 5,
    /// Luminous intensity, base unit candela (`cd`).
    LuminousIntensity = 6,
}

impl BaseDimension {
    /// All base dimensions, in canonical (exponent-array) order.
    pub const ALL: [BaseDimension; BASE_COUNT] = [
        BaseDimension::Length,
        BaseDimension::Mass,
        BaseDimension::Time,
        BaseDimension::ElectricCurrent,
        BaseDimension::Temperature,
        BaseDimension::AmountOfSubstance,
        BaseDimension::LuminousIntensity,
    ];

    /// Index of this base dimension inside an exponent array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// SI symbol of the corresponding base unit, e.g. `"kg"` for mass.
    #[inline]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Length => "m",
            Self::Mass => "kg",
            Self::Time => "s",
            Self::ElectricCurrent => "A",
            Self::Temperature => "K",
            Self::AmountOfSubstance => "mol",
            Self::LuminousIntensity => "cd",
        }
    }

    /// Human readable name, e.g. `"amount of substance"`.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Mass => "mass",
            Self::Time => "time",
            Self::ElectricCurrent => "electric current",
            Self::Temperature => "temperature",
            Self::AmountOfSubstance => "amount of substance",
            Self::LuminousIntensity => "luminous intensity",
        }
    }

    /// The [`Dimension`] with exponent `1` for this base dimension.
    #[inline]
    pub const fn dimension(self) -> Dimension {
        let mut exponents = [0i8; BASE_COUNT];
        exponents[self as usize] = 1;
        Dimension { exponents }
    }
}

impl fmt::Display for BaseDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The dimension of a physical quantity, stored as the seven SI base
/// exponents `[length, mass, time, current, temperature, amount, luminous]`.
///
/// # Examples
///
/// ```
/// use tpt_math_units_dyn::Dimension;
///
/// let velocity = Dimension::LENGTH / Dimension::TIME;
/// assert_eq!(velocity, Dimension::VELOCITY);
/// assert_eq!(velocity.to_string(), "m·s^-1");
/// assert!((velocity * Dimension::TIME / Dimension::LENGTH).is_dimensionless());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dimension {
    exponents: [i8; BASE_COUNT],
}

impl Dimension {
    /// The dimensionless dimension (all exponents zero).
    pub const DIMENSIONLESS: Self = Self::new(0, 0, 0, 0, 0, 0, 0);
    /// Length, `m`.
    pub const LENGTH: Self = Self::new(1, 0, 0, 0, 0, 0, 0);
    /// Mass, `kg`.
    pub const MASS: Self = Self::new(0, 1, 0, 0, 0, 0, 0);
    /// Time, `s`.
    pub const TIME: Self = Self::new(0, 0, 1, 0, 0, 0, 0);
    /// Electric current, `A`.
    pub const ELECTRIC_CURRENT: Self = Self::new(0, 0, 0, 1, 0, 0, 0);
    /// Thermodynamic temperature, `K`.
    pub const TEMPERATURE: Self = Self::new(0, 0, 0, 0, 1, 0, 0);
    /// Amount of substance, `mol`.
    pub const AMOUNT_OF_SUBSTANCE: Self = Self::new(0, 0, 0, 0, 0, 1, 0);
    /// Luminous intensity, `cd`.
    pub const LUMINOUS_INTENSITY: Self = Self::new(0, 0, 0, 0, 0, 0, 1);

    /// Area, `m^2`.
    pub const AREA: Self = Self::new(2, 0, 0, 0, 0, 0, 0);
    /// Volume, `m^3`.
    pub const VOLUME: Self = Self::new(3, 0, 0, 0, 0, 0, 0);
    /// Velocity, `m·s^-1`.
    pub const VELOCITY: Self = Self::new(1, 0, -1, 0, 0, 0, 0);
    /// Acceleration, `m·s^-2`.
    pub const ACCELERATION: Self = Self::new(1, 0, -2, 0, 0, 0, 0);
    /// Frequency, `s^-1`.
    pub const FREQUENCY: Self = Self::new(0, 0, -1, 0, 0, 0, 0);
    /// Mass density, `m^-3·kg`.
    pub const DENSITY: Self = Self::new(-3, 1, 0, 0, 0, 0, 0);
    /// Momentum, `m·kg·s^-1`.
    pub const MOMENTUM: Self = Self::new(1, 1, -1, 0, 0, 0, 0);
    /// Force, `m·kg·s^-2` (newton).
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);
    /// Energy, `m^2·kg·s^-2` (joule).
    pub const ENERGY: Self = Self::new(2, 1, -2, 0, 0, 0, 0);
    /// Power, `m^2·kg·s^-3` (watt).
    pub const POWER: Self = Self::new(2, 1, -3, 0, 0, 0, 0);
    /// Pressure, `m^-1·kg·s^-2` (pascal).
    pub const PRESSURE: Self = Self::new(-1, 1, -2, 0, 0, 0, 0);
    /// Electric charge, `s·A` (coulomb).
    pub const ELECTRIC_CHARGE: Self = Self::new(0, 0, 1, 1, 0, 0, 0);
    /// Electric potential, `m^2·kg·s^-3·A^-1` (volt).
    pub const ELECTRIC_POTENTIAL: Self = Self::new(2, 1, -3, -1, 0, 0, 0);
    /// Electrical resistance, `m^2·kg·s^-3·A^-2` (ohm).
    pub const ELECTRICAL_RESISTANCE: Self = Self::new(2, 1, -3, -2, 0, 0, 0);
    /// Electrical conductance, `m^-2·kg^-1·s^3·A^2` (siemens).
    pub const ELECTRICAL_CONDUCTANCE: Self = Self::new(-2, -1, 3, 2, 0, 0, 0);
    /// Capacitance, `m^-2·kg^-1·s^4·A^2` (farad).
    pub const CAPACITANCE: Self = Self::new(-2, -1, 4, 2, 0, 0, 0);
    /// Inductance, `m^2·kg·s^-2·A^-2` (henry).
    pub const INDUCTANCE: Self = Self::new(2, 1, -2, -2, 0, 0, 0);
    /// Magnetic flux, `m^2·kg·s^-2·A^-1` (weber).
    pub const MAGNETIC_FLUX: Self = Self::new(2, 1, -2, -1, 0, 0, 0);
    /// Magnetic flux density, `kg·s^-2·A^-1` (tesla).
    pub const MAGNETIC_FLUX_DENSITY: Self = Self::new(0, 1, -2, -1, 0, 0, 0);

    /// Builds a dimension from the seven base exponents.
    ///
    /// ```
    /// use tpt_math_units_dyn::Dimension;
    ///
    /// // joule = m^2 · kg · s^-2
    /// let energy = Dimension::new(2, 1, -2, 0, 0, 0, 0);
    /// assert_eq!(energy, Dimension::ENERGY);
    /// ```
    #[inline]
    pub const fn new(
        length: i8,
        mass: i8,
        time: i8,
        electric_current: i8,
        temperature: i8,
        amount_of_substance: i8,
        luminous_intensity: i8,
    ) -> Self {
        Self {
            exponents: [
                length,
                mass,
                time,
                electric_current,
                temperature,
                amount_of_substance,
                luminous_intensity,
            ],
        }
    }

    /// Builds a dimension directly from an exponent array in canonical order.
    #[inline]
    pub const fn from_exponents(exponents: [i8; BASE_COUNT]) -> Self {
        Self { exponents }
    }

    /// The raw exponent array in canonical order.
    #[inline]
    pub const fn exponents(&self) -> [i8; BASE_COUNT] {
        self.exponents
    }

    /// The exponent of a single base dimension.
    #[inline]
    pub const fn exponent(&self, base: BaseDimension) -> i8 {
        self.exponents[base.index()]
    }

    /// Exponent of length.
    #[inline]
    pub const fn length(&self) -> i8 {
        self.exponents[BaseDimension::Length.index()]
    }

    /// Exponent of mass.
    #[inline]
    pub const fn mass(&self) -> i8 {
        self.exponents[BaseDimension::Mass.index()]
    }

    /// Exponent of time.
    #[inline]
    pub const fn time(&self) -> i8 {
        self.exponents[BaseDimension::Time.index()]
    }

    /// Exponent of electric current.
    #[inline]
    pub const fn electric_current(&self) -> i8 {
        self.exponents[BaseDimension::ElectricCurrent.index()]
    }

    /// Exponent of thermodynamic temperature.
    #[inline]
    pub const fn temperature(&self) -> i8 {
        self.exponents[BaseDimension::Temperature.index()]
    }

    /// Exponent of amount of substance.
    #[inline]
    pub const fn amount_of_substance(&self) -> i8 {
        self.exponents[BaseDimension::AmountOfSubstance.index()]
    }

    /// Exponent of luminous intensity.
    #[inline]
    pub const fn luminous_intensity(&self) -> i8 {
        self.exponents[BaseDimension::LuminousIntensity.index()]
    }

    /// Returns `true` when every base exponent is zero.
    ///
    /// ```
    /// use tpt_math_units_dyn::Dimension;
    ///
    /// assert!(Dimension::DIMENSIONLESS.is_dimensionless());
    /// assert!((Dimension::LENGTH / Dimension::LENGTH).is_dimensionless());
    /// assert!(!Dimension::MASS.is_dimensionless());
    /// ```
    #[inline]
    pub const fn is_dimensionless(&self) -> bool {
        let mut i = 0;
        while i < BASE_COUNT {
            if self.exponents[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Multiplies two dimensions (adds exponents), returning [`None`] on
    /// exponent overflow.
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let mut out = [0i8; BASE_COUNT];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.exponents[index].checked_add(rhs.exponents[index])?;
        }
        Some(Self { exponents: out })
    }

    /// Divides two dimensions (subtracts exponents), returning [`None`] on
    /// exponent overflow.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        let mut out = [0i8; BASE_COUNT];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.exponents[index].checked_sub(rhs.exponents[index])?;
        }
        Some(Self { exponents: out })
    }

    /// The reciprocal dimension (negated exponents), [`None`] on overflow.
    #[inline]
    pub fn checked_recip(self) -> Option<Self> {
        Self::DIMENSIONLESS.checked_div(self)
    }

    /// Raises the dimension to an integer power, [`None`] on overflow.
    ///
    /// ```
    /// use tpt_math_units_dyn::Dimension;
    ///
    /// assert_eq!(Dimension::LENGTH.checked_powi(3), Some(Dimension::VOLUME));
    /// ```
    #[inline]
    pub fn checked_powi(self, exp: i32) -> Option<Self> {
        let mut out = [0i8; BASE_COUNT];
        for (index, slot) in out.iter_mut().enumerate() {
            let scaled = i32::from(self.exponents[index]).checked_mul(exp)?;
            *slot = i8::try_from(scaled).ok()?;
        }
        Some(Self { exponents: out })
    }

    /// Takes the `root`-th root of the dimension.
    ///
    /// Returns [`None`] when `root` is zero or when any exponent is not
    /// divisible by `root` (`sqrt(m^3)` has no dimension).
    ///
    /// ```
    /// use tpt_math_units_dyn::Dimension;
    ///
    /// assert_eq!(Dimension::AREA.checked_root(2), Some(Dimension::LENGTH));
    /// assert_eq!(Dimension::VOLUME.checked_root(2), None);
    /// ```
    #[inline]
    pub fn checked_root(self, root: i32) -> Option<Self> {
        if root == 0 {
            return None;
        }
        let mut out = [0i8; BASE_COUNT];
        for (index, slot) in out.iter_mut().enumerate() {
            let exponent = i32::from(self.exponents[index]);
            if exponent % root != 0 {
                return None;
            }
            *slot = i8::try_from(exponent / root).ok()?;
        }
        Some(Self { exponents: out })
    }

    /// The canonical SI base-unit symbol for this dimension, e.g. `"m·kg·s^-2"`
    /// for a force and `"1"` for a dimensionless quantity.
    ///
    /// Common dimensions resolve to `'static` strings, so no allocation takes
    /// place for them.
    pub fn base_unit_symbol(&self) -> Cow<'static, str> {
        for (dim, symbol) in COMMON_BASE_SYMBOLS {
            if dim == self {
                return Cow::Borrowed(*symbol);
            }
        }
        Cow::Owned(self.to_string())
    }
}

/// Cached canonical symbols; each entry must equal `Display` of its dimension
/// (enforced by a unit test in this module).
static COMMON_BASE_SYMBOLS: &[(Dimension, &str)] = &[
    (Dimension::DIMENSIONLESS, "1"),
    (Dimension::LENGTH, "m"),
    (Dimension::MASS, "kg"),
    (Dimension::TIME, "s"),
    (Dimension::ELECTRIC_CURRENT, "A"),
    (Dimension::TEMPERATURE, "K"),
    (Dimension::AMOUNT_OF_SUBSTANCE, "mol"),
    (Dimension::LUMINOUS_INTENSITY, "cd"),
    (Dimension::AREA, "m^2"),
    (Dimension::VOLUME, "m^3"),
    (Dimension::VELOCITY, "m·s^-1"),
    (Dimension::ACCELERATION, "m·s^-2"),
    (Dimension::FREQUENCY, "s^-1"),
    (Dimension::FORCE, "m·kg·s^-2"),
    (Dimension::ENERGY, "m^2·kg·s^-2"),
    (Dimension::POWER, "m^2·kg·s^-3"),
    (Dimension::PRESSURE, "m^-1·kg·s^-2"),
];

impl Mul for Dimension {
    type Output = Self;

    /// # Panics
    ///
    /// Panics if an exponent leaves the `i8` range; use
    /// [`Dimension::checked_mul`] for a fallible variant.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.checked_mul(rhs)
            .expect("dimension exponent overflow while multiplying dimensions")
    }
}

impl Div for Dimension {
    type Output = Self;

    /// # Panics
    ///
    /// Panics if an exponent leaves the `i8` range; use
    /// [`Dimension::checked_div`] for a fallible variant.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.checked_div(rhs)
            .expect("dimension exponent overflow while dividing dimensions")
    }
}

impl fmt::Display for Dimension {
    /// Formats the dimension in SI base units, e.g. `m^2·kg·s^-2`.
    ///
    /// Factors are emitted in canonical base order and a dimensionless
    /// dimension is rendered as `1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return f.write_str("1");
        }
        let mut first = true;
        for base in BaseDimension::ALL {
            let exp = self.exponents[base.index()];
            if exp == 0 {
                continue;
            }
            if !first {
                f.write_str("·")?;
            }
            first = false;
            if exp == 1 {
                f.write_str(base.symbol())?;
            } else {
                write!(f, "{}^{}", base.symbol(), exp)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_dimension_metadata_is_consistent() {
        for (i, base) in BaseDimension::ALL.into_iter().enumerate() {
            assert_eq!(base.index(), i);
            assert_eq!(base.dimension().exponent(base), 1);
            assert!(!base.symbol().is_empty());
            assert!(!base.name().is_empty());
        }
    }

    #[test]
    fn multiplication_and_division_combine_exponents() {
        let velocity = Dimension::LENGTH / Dimension::TIME;
        assert_eq!(velocity, Dimension::VELOCITY);
        assert_eq!(velocity.length(), 1);
        assert_eq!(velocity.time(), -1);

        let force = Dimension::MASS * Dimension::ACCELERATION;
        assert_eq!(force, Dimension::FORCE);
        assert_eq!(force * Dimension::LENGTH, Dimension::ENERGY);
        assert_eq!(Dimension::ENERGY / Dimension::TIME, Dimension::POWER);
    }

    #[test]
    fn equality_is_exponentwise() {
        assert_eq!(Dimension::default(), Dimension::DIMENSIONLESS);
        assert_ne!(Dimension::LENGTH, Dimension::MASS);
        assert_eq!(
            Dimension::from_exponents([2, 1, -2, 0, 0, 0, 0]),
            Dimension::ENERGY
        );
    }

    #[test]
    fn dimensionless_detection() {
        assert!(Dimension::DIMENSIONLESS.is_dimensionless());
        assert!((Dimension::ENERGY / Dimension::ENERGY).is_dimensionless());
        assert!(!Dimension::TEMPERATURE.is_dimensionless());
    }

    #[test]
    fn powers_and_roots() {
        assert_eq!(Dimension::LENGTH.checked_powi(2), Some(Dimension::AREA));
        assert_eq!(Dimension::AREA.checked_root(2), Some(Dimension::LENGTH));
        assert_eq!(Dimension::VOLUME.checked_root(2), None);
        assert_eq!(Dimension::LENGTH.checked_root(0), None);
        assert_eq!(
            Dimension::VELOCITY.checked_recip(),
            Some(Dimension::new(-1, 0, 1, 0, 0, 0, 0))
        );
    }

    #[test]
    fn exponent_overflow_is_detected() {
        let huge = Dimension::new(100, 0, 0, 0, 0, 0, 0);
        assert_eq!(huge.checked_mul(huge), None);
        assert_eq!(huge.checked_powi(2), None);
        assert!(huge.checked_mul(Dimension::MASS).is_some());
    }

    #[test]
    #[should_panic(expected = "dimension exponent overflow")]
    fn overflowing_multiplication_panics() {
        let huge = Dimension::new(100, 0, 0, 0, 0, 0, 0);
        let _ = huge * huge;
    }

    #[test]
    fn display_uses_si_base_symbols() {
        assert_eq!(Dimension::DIMENSIONLESS.to_string(), "1");
        assert_eq!(Dimension::LENGTH.to_string(), "m");
        assert_eq!(Dimension::ENERGY.to_string(), "m^2·kg·s^-2");
        assert_eq!(Dimension::ELECTRIC_POTENTIAL.to_string(), "m^2·kg·s^-3·A^-1");
    }

    #[test]
    fn cached_base_symbols_match_display() {
        for (dim, symbol) in COMMON_BASE_SYMBOLS {
            assert_eq!(dim.to_string(), *symbol);
            assert_eq!(dim.base_unit_symbol(), *symbol);
        }
        assert_eq!(Dimension::DENSITY.base_unit_symbol(), "m^-3·kg");
    }
}
