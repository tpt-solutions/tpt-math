//! Bridge between the compile-time quantities of [`tpt_math_units`] (i.e.
//! [`uom`](tpt_math_units::uom)) and the runtime [`DynQuantity`].
//!
//! The intended flow is:
//!
//! 1. read a value plus a unit symbol from a config file or API payload,
//! 2. resolve it into a [`DynQuantity`], where the dimension is checked at
//!    runtime,
//! 3. convert it *once* into the statically typed quantity the rest of the
//!    program uses, where the compiler takes over again.
//!
//! # Examples
//!
//! ```
//! use tpt_math_units::prelude::*;
//! use tpt_math_units::si::velocity::kilometer_per_hour;
//! use tpt_math_units_dyn::DynQuantity;
//!
//! // "90 km/h" arrived as text in a configuration file.
//! let dynamic: DynQuantity = "90 km/h".parse().unwrap();
//! let typed = Velocity::try_from(dynamic).unwrap();
//! assert!((typed.get::<kilometer_per_hour>() - 90.0).abs() < 1e-9);
//!
//! // ... and back again.
//! let round_trip = DynQuantity::from(typed);
//! assert!((round_trip.convert_to("m/s").unwrap() - 25.0).abs() < 1e-9);
//! ```

use tpt_math_units::prelude::*;

use crate::dimension::Dimension;
use crate::error::UnitError;
use crate::quantity::DynQuantity;

/// Generates `From<Quantity> for DynQuantity` and the checked reverse
/// conversion for each compile-time quantity type.
macro_rules! uom_bridge {
    ($($quantity:ident, $base_unit:path, $dim:expr, $context:literal;)*) => {
        $(
            impl From<$quantity> for DynQuantity {
                #[doc = concat!("Converts a compile-time `", stringify!($quantity), "` into a runtime quantity.")]
                fn from(value: $quantity) -> Self {
                    DynQuantity::from_base(value.get::<$base_unit>(), $dim)
                }
            }

            impl TryFrom<DynQuantity> for $quantity {
                type Error = UnitError;

                #[doc = concat!("Converts a runtime quantity into a compile-time `", stringify!($quantity), "`.")]
                ///
                /// # Errors
                ///
                /// Returns [`UnitError::DimensionMismatch`] when the runtime
                /// dimension is not the one this quantity type represents.
                fn try_from(value: DynQuantity) -> Result<Self, Self::Error> {
                    if *value.dim() != $dim {
                        return Err(UnitError::DimensionMismatch {
                            context: $context,
                            expected: $dim,
                            actual: *value.dim(),
                        });
                    }
                    Ok(<$quantity>::new::<$base_unit>(value.value()))
                }
            }
        )*
    };
}

uom_bridge! {
    Length, tpt_math_units::si::length::meter, Dimension::LENGTH, "Length::try_from";
    Mass, tpt_math_units::si::mass::kilogram, Dimension::MASS, "Mass::try_from";
    Time, tpt_math_units::si::time::second, Dimension::TIME, "Time::try_from";
    ElectricCurrent, tpt_math_units::si::electric_current::ampere, Dimension::ELECTRIC_CURRENT, "ElectricCurrent::try_from";
    ThermodynamicTemperature, tpt_math_units::si::thermodynamic_temperature::kelvin, Dimension::TEMPERATURE, "ThermodynamicTemperature::try_from";
    AmountOfSubstance, tpt_math_units::si::amount_of_substance::mole, Dimension::AMOUNT_OF_SUBSTANCE, "AmountOfSubstance::try_from";
    LuminousIntensity, tpt_math_units::si::luminous_intensity::candela, Dimension::LUMINOUS_INTENSITY, "LuminousIntensity::try_from";
    Area, tpt_math_units::si::area::square_meter, Dimension::AREA, "Area::try_from";
    Volume, tpt_math_units::si::volume::cubic_meter, Dimension::VOLUME, "Volume::try_from";
    Velocity, tpt_math_units::si::velocity::meter_per_second, Dimension::VELOCITY, "Velocity::try_from";
    Acceleration, tpt_math_units::si::acceleration::meter_per_second_squared, Dimension::ACCELERATION, "Acceleration::try_from";
    Force, tpt_math_units::si::force::newton, Dimension::FORCE, "Force::try_from";
    Energy, tpt_math_units::si::energy::joule, Dimension::ENERGY, "Energy::try_from";
    Power, tpt_math_units::si::power::watt, Dimension::POWER, "Power::try_from";
    Pressure, tpt_math_units::si::pressure::pascal, Dimension::PRESSURE, "Pressure::try_from";
    Frequency, tpt_math_units::si::frequency::hertz, Dimension::FREQUENCY, "Frequency::try_from";
    ElectricCharge, tpt_math_units::si::electric_charge::coulomb, Dimension::ELECTRIC_CHARGE, "ElectricCharge::try_from";
    ElectricPotential, tpt_math_units::si::electric_potential::volt, Dimension::ELECTRIC_POTENTIAL, "ElectricPotential::try_from";
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_units::si::length::kilometer;
    use tpt_math_units::si::time::hour;

    #[test]
    fn static_to_dynamic() {
        let length = Length::new::<kilometer>(1.5);
        let dynamic = DynQuantity::from(length);
        assert_eq!(*dynamic.dim(), Dimension::LENGTH);
        assert!((dynamic.value() - 1500.0).abs() < 1e-9);
        assert!((dynamic.convert_to("km").unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn dynamic_to_static() {
        let dynamic = DynQuantity::parse(1.0, "h").unwrap();
        let time = Time::try_from(dynamic).unwrap();
        assert!((time.get::<hour>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dynamic_to_static_rejects_wrong_dimension() {
        let mass = DynQuantity::parse(1.0, "kg").unwrap();
        let err = Length::try_from(mass).unwrap_err();
        assert_eq!(
            err,
            UnitError::DimensionMismatch {
                context: "Length::try_from",
                expected: Dimension::LENGTH,
                actual: Dimension::MASS,
            }
        );
    }

    #[test]
    fn derived_quantities_round_trip() {
        let dynamic: DynQuantity = "2.5 kWh".parse().unwrap();
        let energy = Energy::try_from(dynamic.clone()).unwrap();
        assert!((energy.get::<tpt_math_units::si::energy::joule>() - 9.0e6).abs() < 1e-3);
        assert!(DynQuantity::from(energy).approx_eq(&dynamic, 1e-6));

        let power: DynQuantity = "1.5 kW".parse().unwrap();
        assert!((Power::try_from(power).unwrap().get::<tpt_math_units::si::power::watt>()
            - 1500.0)
            .abs()
            < 1e-9);
    }

    #[test]
    fn compile_time_and_runtime_agree_on_derived_dimensions() {
        // Velocity computed with uom ...
        let typed: Velocity = Length::new::<kilometer>(100.0) / Time::new::<hour>(1.0);
        // ... and with runtime dimensions.
        let dynamic = DynQuantity::parse(100.0, "km")
            .unwrap()
            .try_div(&DynQuantity::parse(1.0, "h").unwrap())
            .unwrap();
        assert!(DynQuantity::from(typed).approx_eq(&dynamic, 1e-12));
    }
}
