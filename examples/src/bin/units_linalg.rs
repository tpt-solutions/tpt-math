//! Cross-crate example: `tpt-math-units` + `tpt-math-linalg`.
//!
//! Demonstrates dimensionally-tagged vectors/matrices. Mixing incompatible
//! units (e.g. adding a `Length` vector to a `Time` vector) is a compile
//! error, while unit-correct operations compile and run.

use tpt_math_linalg::{Mat, Vec};
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_units::prelude::{Length, Time, Velocity};
use tpt_math_units::si::length::meter;
use tpt_math_units::si::time::second;
use tpt_math_units::si::velocity::meter_per_second;

fn main() {
    // Two same-unit vectors add normally; the unit tag is preserved.
    let pos = Vec::<Length>::from_raw(DVector::from_row_slice(&[3.0_f64, 4.0]));
    let offset = Vec::<Length>::from_raw(DVector::from_row_slice(&[1.0_f64, 0.0]));
    let total = pos + offset;
    assert_eq!(total.raw(), &DVector::from_row_slice(&[4.0, 4.0]));

    // `tpt-math-units` computes the dimensionally-correct scalar quantity, which
    // we then wrap in a unit-tagged `tpt-math-linalg` vector.
    let distance = Length::new::<meter>(10.0);
    let duration = Time::new::<second>(2.0);
    let velocity = distance / duration;
    assert!((velocity.get::<meter_per_second>() - 5.0).abs() < 1e-9);
    let v_vec = Vec::<Velocity>::from_raw(DVector::from_row_slice(&[
        velocity.get::<meter_per_second>()
    ]));

    // Matrix multiplication propagates the row/column unit tags:
    // `Mat<Length, Time> * Mat<Time, Velocity> = Mat<Length, Velocity>`.
    let m = Mat::<Length, Time>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]));
    let n =
        Mat::<Time, Velocity>::from_raw(DMatrix::from_row_slice(2, 2, &[1.0_f64, 0.0, 0.0, 1.0]));
    let p: Mat<Length, Velocity> = m * n;
    assert_eq!(p.nrows(), 2);
    assert_eq!(p.ncols(), 2);

    println!(
        "units+linalg: velocity = {} m/s, tagged velocity vector = {:?}",
        velocity.get::<meter_per_second>(),
        v_vec.raw()
    );
}
