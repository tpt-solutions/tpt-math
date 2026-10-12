//! Cross-crate example: `tpt-math-geometry` + `tpt-math-spatial`.
//!
//! Builds a rigid motion (a rotation about `z` followed by a translation),
//! applies it to a point, and checks that the same motion written as a dual
//! quaternion moves the point the same way.

use tpt_math_geometry::{Isometry3, Point3, Quaternion, Rotation3, Translation};
use tpt_math_linalg_fixed::Vector;
use tpt_math_spatial::DualQuaternion;

fn main() {
    let angle = std::f64::consts::FRAC_PI_2;
    let axis = Vector::new([0.0_f64, 0.0, 1.0]);
    let motion = Isometry3::new(
        Translation::new(Vector::new([1.0_f64, 0.0, 0.0])),
        Rotation3::from_axis_angle(&axis, angle),
    );

    let p = Point3::new(Vector::new([1.0_f64, 0.0, 0.0]));
    let moved = motion.transform_point(&p);
    println!(
        "isometry: (1, 0, 0) -> ({:.3}, {:.3}, {:.3})",
        moved.coords[0], moved.coords[1], moved.coords[2]
    );

    // The same rotation as a unit quaternion, with the translation in the dual part.
    let half = angle / 2.0;
    let rotation = Quaternion::new(0.0_f64, 0.0, half.sin(), half.cos());
    let translation = Quaternion::new(0.0_f64, 1.0, 0.0, 0.0);
    let dual = (translation * rotation) * 0.5;
    let dq = DualQuaternion::new(rotation, dual);
    let composed = dq.multiply(&DualQuaternion::new(
        Quaternion::new(0.0, 0.0, 0.0, 1.0),
        Quaternion::new(0.0, 0.0, 0.0, 0.0),
    ));
    // Components are stored as [x, y, z, w], with the scalar part last.
    println!(
        "dual quaternion real scalar part w = {:.3}",
        composed.real.coords[3]
    );
}
