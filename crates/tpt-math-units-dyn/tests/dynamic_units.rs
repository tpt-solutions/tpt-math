//! End-to-end tests of the public API, driven the way a config-loading
//! application would drive it.

use tpt_math_units_dyn::prelude::*;
use tpt_math_units_dyn::{convert, Dimension, DynQuantity, UnitError};

#[test]
fn one_kilometre_plus_two_hundred_metres_converts_correctly() {
    let km = DynQuantity::parse(1.0, "km").unwrap();
    let m = DynQuantity::parse(200.0, "m").unwrap();

    let total = km.try_add(&m).unwrap();

    // Stored in SI base units ...
    assert_eq!(total.value(), 1200.0);
    assert_eq!(total.base_unit(), "m");
    // ... and readable in whatever unit the caller wants.
    assert!((total.convert_to("km").unwrap() - 1.2).abs() < 1e-12);
    assert!((total.convert_to("m").unwrap() - 1200.0).abs() < 1e-12);
    assert!((total.convert_to("cm").unwrap() - 120_000.0).abs() < 1e-9);
    // The operator form agrees with the checked form.
    assert_eq!((&km + &m), total);
}

#[test]
fn length_times_time_gives_a_compound_dimension() {
    let length = DynQuantity::parse(3.0, "m").unwrap();
    let time = DynQuantity::parse(2.0, "s").unwrap();

    let product = &length * &time;

    assert_eq!(product.value(), 6.0);
    assert_eq!(*product.dim(), Dimension::LENGTH * Dimension::TIME);
    assert_eq!(product.dim().exponents(), [1, 0, 1, 0, 0, 0, 0]);
    assert_eq!(product.base_unit(), "m·s");
    assert!(!product.dim().is_dimensionless());
    // The compound dimension is not a length any more.
    assert!(product.convert_to("m").is_err());
    // Dividing the time back out recovers the length.
    assert_eq!(product.try_div(&time).unwrap(), length);
}

#[test]
fn adding_a_length_to_a_mass_is_rejected() {
    let length = DynQuantity::parse(1.0, "m").unwrap();
    let mass = DynQuantity::parse(1.0, "kg").unwrap();

    let err = length.try_add(&mass).unwrap_err();

    assert_eq!(
        err,
        UnitError::DimensionMismatch {
            context: "add",
            expected: Dimension::LENGTH,
            actual: Dimension::MASS,
        }
    );
    assert_eq!(
        err.to_string(),
        "dimension mismatch in `add`: expected [m], found [kg]"
    );
    assert!(length.try_sub(&mass).is_err());
    // Multiplication, in contrast, is always well defined.
    assert_eq!(
        *length.try_mul(&mass).unwrap().dim(),
        Dimension::LENGTH * Dimension::MASS
    );
}

#[test]
#[should_panic(expected = "dimension mismatch in `add`")]
fn adding_a_length_to_a_mass_panics_with_the_operator() {
    let length = DynQuantity::parse(1.0, "m").unwrap();
    let mass = DynQuantity::parse(1.0, "kg").unwrap();
    let _ = length + mass;
}

#[test]
fn one_hour_is_three_thousand_six_hundred_seconds() {
    let hour = DynQuantity::parse(1.0, "h").unwrap();

    assert_eq!(hour.convert_to("s").unwrap(), 3600.0);
    assert_eq!(hour.convert_to("min").unwrap(), 60.0);
    assert_eq!(convert(1.0, "h", "s").unwrap(), 3600.0);
    assert_eq!(*hour.dim(), Dimension::TIME);
}

#[test]
fn lookup_exposes_dimension_and_base_scale() {
    assert_eq!(lookup("km"), Some((Dimension::LENGTH, 1000.0)));
    assert_eq!(lookup("min"), Some((Dimension::TIME, 60.0)));
    assert_eq!(lookup("kg"), Some((Dimension::MASS, 1.0)));
    assert_eq!(lookup("N"), Some((Dimension::FORCE, 1.0)));
    assert_eq!(lookup("does-not-exist"), None);
    assert_eq!(lookup_unit("h").map(|def| def.scale), Some(3600.0));
}

#[test]
fn a_small_config_payload_is_validated_and_combined() {
    // (value, unit) pairs, as they might come out of a deserialiser.
    let payload = [(150.0, "kW"), (2.0, "h")];

    let quantities: Vec<DynQuantity> = payload
        .iter()
        .map(|(value, unit)| DynQuantity::parse(*value, unit).expect("known unit"))
        .collect();

    let energy = quantities[0].try_mul(&quantities[1]).unwrap();
    assert_eq!(*energy.dim(), Dimension::ENERGY);
    assert!((energy.convert_to("kWh").unwrap() - 300.0).abs() < 1e-9);
    assert!((energy.convert_to("MJ").unwrap() - 1080.0).abs() < 1e-9);

    // An unknown unit is reported, not silently accepted.
    let err = DynQuantity::parse(1.0, "kWH").unwrap_err();
    assert_eq!(err.to_string(), "unknown unit `kWH`");
}

#[test]
fn custom_units_extend_the_builtin_table() {
    let mut registry = UnitRegistry::with_builtins();
    registry.define("furlong", Dimension::LENGTH, 201.168);
    registry.insert(UnitDef::new("fortnight", Dimension::TIME, 1_209_600.0));

    let speed = DynQuantity::parse_with(1.0, "furlong", &registry)
        .unwrap()
        .try_div(&DynQuantity::parse_with(1.0, "fortnight", &registry).unwrap())
        .unwrap();

    assert_eq!(*speed.dim(), Dimension::VELOCITY);
    let expected = 201.168 / 1_209_600.0;
    assert!((speed.convert_to_with("m/s", &registry).unwrap() - expected).abs() < 1e-18);
    // The built-in registry is untouched by the custom one.
    assert_eq!(lookup("furlong"), None);
}

#[test]
fn textual_quantities_round_trip() {
    let speed: DynQuantity = "90 km/h".parse().unwrap();
    assert_eq!(*speed.dim(), Dimension::VELOCITY);
    assert!((speed.value() - 25.0).abs() < 1e-12);
    assert_eq!(speed.to_string_in("km/h").unwrap(), "90 km/h");
    assert_eq!(speed.to_string(), "25 m·s^-1");
}

#[test]
fn temperatures_use_affine_conversions() {
    let boiling = DynQuantity::parse(100.0, "degC").unwrap();
    assert!((boiling.value() - 373.15).abs() < 1e-12);
    assert!((boiling.convert_to("degF").unwrap() - 212.0).abs() < 1e-9);
    assert!((boiling.convert_to("K").unwrap() - 373.15).abs() < 1e-12);
    assert!(boiling.convert_to("J").is_err());
}
