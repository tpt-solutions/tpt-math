//! One dependency for the `tpt-math` numeric substrate.
//!
//! Each area of the workspace is an optional dependency with a short feature
//! name, re-exported under the same name as a module:
//!
//! | Feature       | Module            | Area                                   |
//! |---------------|-------------------|----------------------------------------|
//! | `numeric`     | [`numeric`]       | scalar traits, quadrature, ODE, roots  |
//! | `units`       | [`units`]         | compile-time typed units               |
//! | `linalg`      | [`linalg`]        | dense, sparse, complex and fixed linear algebra |
//! | `exact`       | [`exact`]         | rationals and interval arithmetic      |
//! | `prob`        | [`prob`]          | distributions, sampling, Bayes, Markov |
//! | `stats`       | [`stats`]         | hypothesis tests and regression        |
//! | `autodiff`    | [`autodiff`]      | forward and reverse mode               |
//! | `optimize`    | [`optimize`]      | general and convex optimisation        |
//! | `signal`      | [`signal`]        | FFT and filters                        |
//! | `symbolic`    | [`symbolic`]      | symbolic expressions                   |
//! | `geometry`    | [`geometry`]      | points, rotations, isometries          |
//! | `spatial`     | [`spatial`]       | spatial vectors and screw theory       |
//! | `graph`       | [`graph`]         | graphs, shortest paths, max-flow       |
//! | `interpolate` | [`interpolate`]   | RBF, kriging, PCHIP, B-splines         |
//!
//! `full` enables all of them. Nothing is enabled by default.
//!
//! The [`prelude`] brings in the most common scalar traits and types. Each item
//! is gated on the feature that provides it.
//!
//! # Examples
//!
//! ```
//! # #[cfg(all(feature = "exact", feature = "symbolic"))]
//! # {
//! use tpt_math::exact::BigRational;
//! use tpt_math::prelude::*;
//!
//! // Exact rational arithmetic.
//! let half = BigRational::new(1.into(), 2.into());
//! assert_eq!(&half + &half, BigRational::from_integer(1.into()));
//!
//! // Symbolic differentiation, checked numerically.
//! let x = Expr64::var("x");
//! let d = Expr64::func("sin", x).derivative("x").simplify();
//! assert!((d.eval_str("x", 0.0).unwrap() - 1.0).abs() < 1e-12);
//! # }
//! ```

#[cfg(feature = "numeric")]
pub use tpt_math_numeric as numeric;

#[cfg(feature = "units")]
pub use tpt_math_units as units;

#[cfg(feature = "linalg")]
pub use tpt_math_linalg as linalg;

#[cfg(feature = "exact")]
pub use tpt_math_exact as exact;

#[cfg(feature = "prob")]
pub use tpt_math_prob as prob;

#[cfg(feature = "stats")]
pub use tpt_math_stats as stats;

#[cfg(feature = "autodiff")]
pub use tpt_math_autodiff as autodiff;

#[cfg(feature = "optimize")]
pub use tpt_math_optimize as optimize;

#[cfg(feature = "signal")]
pub use tpt_math_signal as signal;

#[cfg(feature = "symbolic")]
pub use tpt_math_symbolic as symbolic;

#[cfg(feature = "geometry")]
pub use tpt_math_geometry as geometry;

#[cfg(feature = "spatial")]
pub use tpt_math_spatial as spatial;

#[cfg(feature = "graph")]
pub use tpt_math_graph as graph;

#[cfg(feature = "interpolate")]
pub use tpt_math_interpolate as interpolate;

#[cfg(feature = "units-dyn")]
pub use tpt_math_units_dyn as units_dyn;

#[cfg(feature = "linalg-sparse")]
pub use tpt_math_linalg_sparse as linalg_sparse;

#[cfg(feature = "linalg-complex")]
pub use tpt_math_linalg_complex as linalg_complex;

#[cfg(feature = "linalg-fixed")]
pub use tpt_math_linalg_fixed as linalg_fixed;

#[cfg(feature = "pde-operators")]
pub use tpt_math_pde_operators as pde_operators;

/// The most common names, each gated on the feature that provides it.
pub mod prelude {
    #[cfg(feature = "numeric")]
    pub use tpt_math_numeric::prelude::*;

    #[cfg(feature = "linalg")]
    pub use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

    #[cfg(feature = "exact")]
    pub use tpt_math_exact::{Interval, Rational};

    #[cfg(feature = "symbolic")]
    pub use tpt_math_symbolic::{Expr64, Symbolic};

    #[cfg(feature = "linalg")]
    pub use tpt_math_linalg::tpt_math_linalg_dense::{Lu, Qr};

    #[cfg(feature = "signal")]
    pub use tpt_math_signal::fft::Fft;

    #[cfg(feature = "autodiff")]
    pub use tpt_math_autodiff::fwd::Dual;

    #[cfg(feature = "autodiff")]
    pub use tpt_math_autodiff::rev::GradientTape;

    #[cfg(feature = "units-dyn")]
    pub use tpt_math_units_dyn::DynQuantity;
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "linalg")]
    #[test]
    fn linalg_area_is_reachable_through_the_prelude() {
        use crate::prelude::*;
        let v = DVector::from_row_slice(&[3.0_f64, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-12);
    }

    #[cfg(feature = "full")]
    #[test]
    fn every_area_is_reexported() {
        let _ = crate::numeric::report::SolveStatus::Converged;
        let _: Option<crate::units::prelude::Length> = None;
        let _ = crate::prob::dist::density::normal_cdf;
        let _ = crate::signal::fft::fft;
        let _: Option<crate::units_dyn::DynQuantity> = None;
        let _: Option<crate::linalg_sparse::CsrMatrix<f64>> = None;
        let _: Option<crate::pde_operators::Grid<f64, 2>> = None;
    }
}
