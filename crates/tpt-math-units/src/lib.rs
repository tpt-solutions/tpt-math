#![no_std]
//! Compile-time typed units, a thin wrap over [`uom`].
//!
//! `uom` provides quantities (length, mass, time, …) whose units are checked
//! at compile time: `Length<f64>` multiplied by `Time<f64>` is a
//! `Velocity<f64>`, and you cannot accidentally add a `Mass` to a `Length`.
//!
//! This crate re-exports the whole [`uom`] API under [`uom`], plus a set of
//! `f64`-backed quantity type aliases and a [`prelude`] that brings the common
//! SI unit markers into scope.
//!
//! # Examples
//!
//! ```
//! use tpt_math_units::prelude::*;
//! use tpt_math_units::si::length::kilometer;
//! use tpt_math_units::si::time::second;
//!
//! let distance = Length::new::<kilometer>(3.0);
//! let duration = Time::new::<second>(90.0);
//! let speed = distance / duration;
//! assert!((speed.get::<tpt_math_units::si::velocity::kilometer_per_hour>() - 120.0).abs() < 1e-9);
//! ```
//!
//! # no_std
//!
//! `uom` is built with its `std` feature disabled; the `f32`/`f64`
//! transcendental math is provided through [`uom`]'s `libm` backend. The
//! crate therefore compiles for `no_std` targets.

pub use uom;

/// Re-export of [`uom`]'s `si` module (systems, quantities, units).
pub mod si {
    pub use uom::si::*;
}

/// `f64`-backed aliases for the most common SI quantities.
pub mod q {
    pub use uom::si::area::Area;
    pub use uom::si::length::Length;
    pub use uom::si::mass::Mass;
    pub use uom::si::ratio::Ratio;
    pub use uom::si::thermodynamic_temperature::ThermodynamicTemperature;
    pub use uom::si::time::Time;
    pub use uom::si::velocity::Velocity;
    pub use uom::si::volume::Volume;
}

/// Brings the `f64` SI unit markers and base quantities into scope.
pub mod prelude {
    pub use uom::si::f64::*;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use super::si::length::meter;
    use super::si::mass::kilogram;
    use super::si::time::second;

    #[test]
    fn dimensions_are_checked_at_compile_time() {
        let m = Mass::new::<kilogram>(2.0);
        let a = Length::new::<meter>(10.0);
        // velocity = length / time, not mass + length
        let _v = a / Time::new::<second>(2.0);
        assert_eq!(m.get::<kilogram>(), 2.0);
    }

    #[test]
    fn unit_conversion() {
        let km = Length::new::<super::si::length::kilometer>(1.0);
        assert!((km.get::<meter>() - 1000.0).abs() < 1e-9);
    }
}
