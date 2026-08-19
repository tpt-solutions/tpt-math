#![no_std]
#![forbid(unsafe_code)]
// Dense numeric kernels are clearest with explicit indexing; the indexed-loop
// lint does not fit this code.
#![allow(clippy::needless_range_loop)]
//! In-house interpolation & approximation for the `tpt-math` substrate.
//!
//! This crate provides scattered-data and surrogate-modelling routines built
//! on top of the [`tpt_math_linalg_dense`] storage and the
//! [`tpt_math_numeric`] `Scalar` trait, with **no external interpolation
//! backend**:
//!
//! * [`RbfInterpolator`] — radial-basis-function interpolation with the
//!   thin-plate, Gaussian and multiquadric kernels, weights solved via the
//!   dense linear solver.
//! * [`Kriging`] — ordinary Kriging with a configurable variogram model
//!   (spherical, exponential, Gaussian or linear), returning the prediction
//!   and the Kriging variance.
//! * [`Pchip`] — shape-preserving piecewise-cubic-Hermite interpolation that
//!   provably preserves monotonicity of the data.
//! * [`bspline_basis`] and [`BsplineCurve`] — Cox–de Boor B-spline basis
//!   evaluation and a weighted B-spline curve.
//!
//! # Why in-house?
//!
//! The commonly used `scirs2-interpolate` crate is Apache-2.0-only, which is
//! disqualified by this substrate's dual MIT/Apache-2.0 license policy
//! (see ADR-0007). All routines here are implemented from scratch so there is
//! no license exposure and nothing to vendor.
//!
//! # Features
//!
//! * `std` (default) — enable the allocator and the `std` support of deps.
//! * `alloc` — signal allocator availability (dynamic vectors need it).
//!
//! # Examples
//!
//! ```
//! use tpt_math_interpolate::{RbfInterpolator, RbfKernel};
//! use tpt_math_linalg_dense::DVector;
//!
//! let xs = DVector::from_vec(vec![0.0_f64, 1.0, 2.0]);
//! let ys = DVector::from_vec(vec![0.0_f64, 1.0, 4.0]);
//! let rbf = RbfInterpolator::new(xs, ys, RbfKernel::Gaussian { epsilon: 1.5 }).unwrap();
//! // RBF interpolation is exact at the sample nodes.
//! assert!((rbf.eval(1.0) - 1.0).abs() < 1e-6);
//! ```

extern crate alloc;

use tpt_math_linalg_dense::{DMatrix, DVector, DenseError};
use tpt_math_numeric::{Float, Scalar};

#[cfg(feature = "alloc")]
use alloc::{format, vec, vec::Vec};

// ===========================================================================
// Radial basis function interpolation
// ===========================================================================

/// Radial kernels available for [`RbfInterpolator`].
///
/// Every kernel is a function of the (non-negative) distance `r` between two
/// sample points; `epsilon` is a shape parameter that controls the kernel's
/// breadth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RbfKernel {
    /// Thin-plate spline: `φ(r) = r²·ln(r)` (with `φ(0) = 0`).
    ThinPlate {
        /// Shape parameter (currently unused for thin-plate, kept for API
        /// symmetry).
        epsilon: f64,
    },
    /// Gaussian: `φ(r) = exp(-(ε·r)²)`.
    Gaussian {
        /// Shape parameter controlling the width of the bump.
        epsilon: f64,
    },
    /// Multiquadric: `φ(r) = sqrt(1 + (ε·r)²)`.
    Multiquadric {
        /// Shape parameter controlling the curvature.
        epsilon: f64,
    },
}

impl RbfKernel {
    /// Evaluate the kernel at distance `r >= 0`.
    pub fn eval(&self, r: f64) -> f64 {
        match self {
            RbfKernel::ThinPlate { .. } => {
                let rr = r * r;
                if rr == 0.0 {
                    0.0
                } else {
                    rr * Float::ln(rr)
                }
            }
            RbfKernel::Gaussian { epsilon } => {
                let e = *epsilon;
                Float::exp(-(e * r) * (e * r))
            }
            RbfKernel::Multiquadric { epsilon } => {
                let e = *epsilon;
                Float::sqrt(1.0 + (e * r) * (e * r))
            }
        }
    }
}

/// A radial-basis-function interpolant of scattered 1-D data.
///
/// Given samples `(x_i, y_i)`, the interpolant is
/// `s(x) = Σ_j w_j · φ(|x - x_j|)`, where the weight vector `w` solves the
/// symmetric system `A w = y` with `A_ij = φ(|x_i - x_j|)`. The system is
/// solved with the dense LU solver from [`tpt_math_linalg_dense`].
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct RbfInterpolator {
    xs: DVector<f64>,
    weights: DVector<f64>,
    kernel: RbfKernel,
}

#[cfg(feature = "alloc")]
impl RbfInterpolator {
    /// Build the interpolant from samples `xs`/`ys` and a [`RbfKernel`].
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `xs` and `ys` differ in
    /// length, and [`DenseError::Singular`] if the kernel matrix is too
    /// ill-conditioned to solve.
    pub fn new(xs: DVector<f64>, ys: DVector<f64>, kernel: RbfKernel) -> Result<Self, DenseError> {
        let n = xs.len();
        if ys.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("xs has length {n}, ys has length {}", ys.len()),
            });
        }
        let a = DMatrix::from_fn(n, n, |i, j| kernel.eval((xs[i] - xs[j]).abs()));
        let weights = a.solve(&ys)?;
        Ok(RbfInterpolator {
            xs,
            weights,
            kernel,
        })
    }

    /// Evaluate the interpolant at a single point `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let mut s = 0.0_f64;
        for j in 0..self.xs.len() {
            s += self.weights[j] * self.kernel.eval((x - self.xs[j]).abs());
        }
        s
    }

    /// Evaluate the interpolant at each point of `points`.
    pub fn interpolate(&self, points: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(points.len(), |i| self.eval(points[i]))
    }
}

// ===========================================================================
// Ordinary Kriging
// ===========================================================================

/// A variogram model `γ(h)` describing the spatial correlation of residuals.
///
/// All models reduce to the nugget at `h = 0`, i.e. `γ(0) = nugget`, which is
/// what makes ordinary Kriging an exact interpolator at sample points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variogram {
    /// Spherical: `γ(h) = nugget + sill·(1.5·a − 0.5·a³)` for `h ≤ range`, else
    /// `nugget + sill`, where `a = h / range`.
    Spherical {
        /// Partial sill (the variance contributed by the structured component).
        sill: f64,
        /// Distance at which the variogram reaches the sill.
        range: f64,
        /// Jump at the origin (measurement/short-range noise).
        nugget: f64,
    },
    /// Exponential: `γ(h) = nugget + sill·(1 − exp(−h / range))`.
    Exponential {
        /// Partial sill.
        sill: f64,
        /// Practical range (≈ 3·range is where the sill is reached).
        range: f64,
        /// Jump at the origin.
        nugget: f64,
    },
    /// Gaussian: `γ(h) = nugget + sill·(1 − exp(−(h / range)²))`.
    Gaussian {
        /// Partial sill.
        sill: f64,
        /// Range parameter.
        range: f64,
        /// Jump at the origin.
        nugget: f64,
    },
    /// Linear model of regionalized variables: `γ(h) = nugget + slope·h`.
    ///
    /// This is a valid (conditionally negative-definite) variogram in 1-D and
    /// has the useful property that ordinary Kriging reproduces a linear trend
    /// exactly (with `nugget = 0`).
    Linear {
        /// Slope of the variogram (must be non-negative).
        slope: f64,
        /// Jump at the origin.
        nugget: f64,
    },
}

impl Variogram {
    /// Evaluate the variogram at lag `h >= 0`.
    pub fn gamma(&self, h: f64) -> f64 {
        match self {
            Variogram::Spherical {
                sill,
                range,
                nugget,
            } => {
                let r = *range;
                if h >= r {
                    nugget + sill
                } else {
                    let a = h / r;
                    nugget + sill * (1.5 * a - 0.5 * a * a * a)
                }
            }
            Variogram::Exponential {
                sill,
                range,
                nugget,
            } => nugget + sill * (1.0 - Float::exp(-h / *range)),
            Variogram::Gaussian {
                sill,
                range,
                nugget,
            } => {
                let a = h / *range;
                nugget + sill * (1.0 - Float::exp(-a * a))
            }
            Variogram::Linear { slope, nugget } => nugget + slope * h,
        }
    }
}

/// An ordinary-Kriging surrogate of scattered 1-D data.
///
/// The prediction at a point `x*` solves the `(n+1)×(n+1)` Kriging system
/// built from the variogram, with the final row/column enforcing the
/// unbiasedness constraint `Σ_i w_i = 1`. Both the predicted mean and the
/// Kriging variance are returned.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Kriging {
    xs: DVector<f64>,
    ys: DVector<f64>,
    variogram: Variogram,
}

#[cfg(feature = "alloc")]
impl Kriging {
    /// Build a Kriging surrogate from samples `xs`/`ys` and a [`Variogram`].
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `xs` and `ys` differ in
    /// length.
    pub fn new(
        xs: DVector<f64>,
        ys: DVector<f64>,
        variogram: Variogram,
    ) -> Result<Self, DenseError> {
        let n = xs.len();
        if ys.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("xs has length {n}, ys has length {}", ys.len()),
            });
        }
        Ok(Kriging { xs, ys, variogram })
    }

    /// Predict the value (and Kriging variance) at `x`.
    ///
    /// With a `nugget = 0` variogram this is an exact interpolator, so
    /// `predict(x_i) == y_i`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::Singular`] if the Kriging system is singular.
    pub fn predict(&self, x: f64) -> Result<(f64, f64), DenseError> {
        let n = self.xs.len();
        let a = DMatrix::from_fn(n + 1, n + 1, |i, j| {
            if i < n && j < n {
                self.variogram.gamma((self.xs[i] - self.xs[j]).abs())
            } else if (i < n && j == n) || (i == n && j < n) {
                1.0
            } else {
                0.0
            }
        });

        let rhs = DVector::from_fn(n + 1, |i| {
            if i < n {
                self.variogram.gamma((self.xs[i] - x).abs())
            } else {
                1.0
            }
        });

        let w = a.solve(&rhs)?;
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += w[i] * self.ys[i];
        }
        let mut variance = w[n];
        for i in 0..n {
            variance += w[i] * rhs[i];
        }
        Ok((mean, variance))
    }
}

// ===========================================================================
// PCHIP (shape-preserving piecewise cubic Hermite)
// ===========================================================================

/// Shape-preserving piecewise-cubic-Hermite interpolant (PCHIP).
///
/// Given sorted node abscissae `x` and values `y`, the derivative at each node
/// is chosen by the Fritsch–Carlson harmonic-mean limiting rule, which
/// guarantees that a monotone data set yields a monotone interpolant.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct Pchip<T: Scalar> {
    x: DVector<T>,
    y: DVector<T>,
    d: DVector<T>,
}

#[cfg(feature = "alloc")]
impl<T: Scalar> Pchip<T> {
    /// Build a PCHIP interpolant.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `x` and `y` differ in
    /// length (they must each have at least two entries) or if the abscissae
    /// are not strictly increasing.
    pub fn new(x: DVector<T>, y: DVector<T>) -> Result<Self, DenseError> {
        let n = x.len();
        if y.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("x has length {n}, y has length {}", y.len()),
            });
        }
        if n < 2 {
            return Err(DenseError::DimensionMismatch {
                what: format!("need at least 2 nodes, got {n}"),
            });
        }
        let one = T::one();
        let two = one + one;
        let three = two + one;

        let mut h: Vec<T> = vec![T::zero(); n - 1];
        let mut delta: Vec<T> = vec![T::zero(); n - 1];
        for i in 0..(n - 1) {
            h[i] = x[i + 1] - x[i];
            if h[i] <= T::zero() {
                return Err(DenseError::DimensionMismatch {
                    what: format!("abscissae must be strictly increasing at index {i}"),
                });
            }
            delta[i] = (y[i + 1] - y[i]) / h[i];
        }

        let mut d: Vec<T> = vec![T::zero(); n];
        // Interior derivatives (Fritsch–Carlson harmonic-mean rule).
        for i in 1..(n - 1) {
            if delta[i - 1] * delta[i] <= T::zero() {
                d[i] = T::zero();
            } else {
                let w1 = two * h[i - 1] + h[i];
                let w2 = h[i - 1] + two * h[i];
                d[i] = (three * (h[i - 1] + h[i])) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
        // Endpoint derivatives (one-sided Hermite / quadratic estimate).
        d[0] = ((two * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
        let nm = n - 1;
        d[nm] = ((two * h[nm - 1] + h[nm - 2]) * delta[nm - 1] - h[nm - 1] * delta[nm - 2])
            / (h[nm - 1] + h[nm - 2]);

        // Fritsch–Carlson monotonicity enforcement: on each interval, limit the
        // derivatives so the Hermite cubic cannot overshoot. This guarantees a
        // monotone data set yields a monotone interpolant.
        for i in 0..(n - 1) {
            if delta[i] == T::zero() {
                d[i] = T::zero();
                d[i + 1] = T::zero();
                continue;
            }
            let a = d[i] / delta[i];
            let b = d[i + 1] / delta[i];
            if a >= T::zero() && b >= T::zero() && (a * a + b * b) > three * three {
                let tau = three / (a * a + b * b).sqrt();
                d[i] = tau * a * delta[i];
                d[i + 1] = tau * b * delta[i];
            }
            if d[i] * delta[i] < T::zero() {
                d[i] = T::zero();
            }
            if d[i + 1] * delta[i] < T::zero() {
                d[i + 1] = T::zero();
            }
        }

        Ok(Pchip {
            x,
            y,
            d: DVector::from_vec(d),
        })
    }

    /// Evaluate the interpolant at `x` (linear extrapolation outside the
    /// node range).
    pub fn eval(&self, x: T) -> T {
        let one = T::one();
        let two = one + one;
        let three = two + one;
        let n = self.x.len();

        // Locate the bracketing interval by binary search.
        let k = if x <= self.x[0] {
            0
        } else if x >= self.x[n - 1] {
            n - 2
        } else {
            let mut lo = 0_usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.x[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };

        let h = self.x[k + 1] - self.x[k];
        let t = (x - self.x[k]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = two * t3 - three * t2 + one;
        let h10 = t3 - two * t2 + t;
        let h01 = -two * t3 + three * t2;
        let h11 = t3 - t2;

        self.y[k] * h00 + self.d[k] * (h10 * h) + self.y[k + 1] * h01 + self.d[k + 1] * (h11 * h)
    }
}

// ===========================================================================
// B-spline basis (Cox–de Boor) and curve
// ===========================================================================

/// Evaluate the `i`-th B-spline basis function of the given `degree` at `x`.
///
/// The knot vector `knots` must contain the `p + 2` knots spanning basis `i`
/// (so `knots.len()` must be at least `degree + i + 2`). The recursion is the
/// Cox–de Boor formula; basis functions satisfy the partition-of-unity
/// property `Σ_i B_{i,p}(x) = 1` for `x` in the interior of the knot span.
///
/// Returns `0.0` for any out-of-range `i`.
pub fn bspline_basis(degree: usize, knots: &[f64], i: usize, x: f64) -> f64 {
    let n = knots.len();
    if i + degree + 1 >= n {
        return 0.0;
    }
    if degree == 0 {
        let a = knots[i];
        let b = knots[i + 1];
        if x >= a && x < b {
            1.0
        } else if i + 1 == n - 1 && x == b {
            // Include the right endpoint on the last span so that the
            // partition-of-unity holds across the full knot domain.
            1.0
        } else {
            0.0
        }
    } else {
        let denom1 = knots[i + degree] - knots[i];
        let denom2 = knots[i + degree + 1] - knots[i + 1];
        let t1 = if denom1 != 0.0 {
            (x - knots[i]) / denom1
        } else {
            0.0
        };
        let t2 = if denom2 != 0.0 {
            (knots[i + degree + 1] - x) / denom2
        } else {
            0.0
        };
        t1 * bspline_basis(degree - 1, knots, i, x)
            + t2 * bspline_basis(degree - 1, knots, i + 1, x)
    }
}

/// A weighted B-spline curve `C(x) = Σ_i w_i · B_{i,p}(x)`.
///
/// This is a thin convenience wrapper over [`bspline_basis`]: it stores the
/// knot vector and control-point weights and evaluates their weighted sum.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct BsplineCurve {
    degree: usize,
    knots: Vec<f64>,
    weights: Vec<f64>,
}

#[cfg(feature = "alloc")]
impl BsplineCurve {
    /// Build a B-spline curve of `degree` with the given `knots` and control
    /// weights.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != knots.len() - degree - 1`, i.e. the number
    /// of control points is not consistent with the knot vector.
    pub fn new(degree: usize, knots: &[f64], weights: &[f64]) -> Self {
        assert_eq!(
            weights.len(),
            knots.len() - degree - 1,
            "weights length must equal knots.len() - degree - 1"
        );
        BsplineCurve {
            degree,
            knots: knots.to_vec(),
            weights: weights.to_vec(),
        }
    }

    /// Evaluate the weighted B-spline curve at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let knots = self.knots.as_slice();
        let mut s = 0.0_f64;
        for (i, w) in self.weights.iter().enumerate() {
            s += *w * bspline_basis(self.degree, knots, i, x);
        }
        s
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use tpt_math_linalg_dense::DVector;

    // -------------------------------------------------------------------
    // RBF: exact at the sample nodes.
    // -------------------------------------------------------------------
    #[test]
    fn rbf_exact_at_nodes() {
        // The Gaussian RBF matrix is positive definite, so the node values are
        // reproduced exactly.
        let xs = DVector::from_vec(vec![-1.0_f64, 0.0, 1.0]);
        let ys = DVector::from_vec(vec![1.0_f64, 3.0, 2.0]);
        let rbf = RbfInterpolator::new(xs, ys, RbfKernel::Gaussian { epsilon: 1.5 }).unwrap();
        // At the nodes the interpolant reproduces the data exactly.
        assert_relative_eq!(rbf.eval(0.0), 3.0, epsilon = 1e-6);
        assert_relative_eq!(rbf.eval(1.0), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn rbf_thinplate_recovers_sin() {
        // Non-degenerate node spacing so the thin-plate system is solvable.
        let xs = DVector::from_vec(vec![-0.8_f64, -0.3, 0.2, 0.7]);
        let ys = DVector::from_vec(
            xs.iter()
                .map(|&x| (core::f64::consts::PI * x).sin())
                .collect(),
        );
        let rbf = RbfInterpolator::new(xs, ys, RbfKernel::ThinPlate { epsilon: 1.0 }).unwrap();
        // Exact at the nodes.
        assert_relative_eq!(
            rbf.eval(0.2),
            (core::f64::consts::PI * 0.2).sin(),
            epsilon = 1e-6
        );
        // Close to the truth between nodes.
        let val = rbf.eval(0.0);
        let truth = (0.0_f64).sin();
        assert!((val - truth).abs() < 0.2, "val={val}, truth={truth}");
    }

    // -------------------------------------------------------------------
    // Kriging: recovers a linear trend exactly (Linear variogram, nugget 0).
    // -------------------------------------------------------------------
    #[test]
    fn kriging_recovers_line() {
        // y = x sampled at two colinear points; ordinary Kriging with the
        // linear variogram reproduces the line everywhere inside the span.
        let xs = DVector::from_vec(vec![-1.0_f64, 1.0]);
        let ys = DVector::from_vec(vec![-1.0_f64, 1.0]);
        let krig = Kriging::new(
            xs,
            ys,
            Variogram::Linear {
                slope: 1.0,
                nugget: 0.0,
            },
        )
        .unwrap();
        for &x in &[-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let (mean, _var) = krig.predict(x).unwrap();
            assert_relative_eq!(mean, x, epsilon = 1e-9);
        }
    }

    #[test]
    fn kriging_exact_at_nodes() {
        // Spherical variogram with no nugget is an exact interpolator.
        let xs = DVector::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]);
        let ys = DVector::from_vec(vec![0.5_f64, 2.0, 1.0, 4.0]);
        let v = Variogram::Spherical {
            sill: 1.0,
            range: 5.0,
            nugget: 0.0,
        };
        let krig = Kriging::new(xs, ys.clone(), v).unwrap();
        for i in 0..4 {
            let (mean, _) = krig.predict(i as f64).unwrap();
            assert_relative_eq!(mean, ys[i], epsilon = 1e-6);
        }
    }

    // -------------------------------------------------------------------
    // PCHIP: monotonicity preservation.
    // -------------------------------------------------------------------
    #[test]
    fn pchip_preserves_monotonicity() {
        let x = DVector::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]);
        let y = DVector::from_vec(vec![0.0_f64, 1.0, 8.0, 27.0]); // strictly increasing
        let pchip = Pchip::new(x, y).unwrap();
        let mut prev = pchip.eval(0.0);
        for k in 1..=100 {
            let xv = (k as f64) / 100.0 * 3.0;
            let cur = pchip.eval(xv);
            assert!(cur >= prev - 1e-12, "non-monotone at x={xv}");
            prev = cur;
        }
    }

    #[test]
    fn pchip_matches_nodes() {
        let x = DVector::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]);
        let y = DVector::from_vec(vec![0.0_f64, 1.0, 8.0, 27.0]);
        let pchip = Pchip::new(x, y.clone()).unwrap();
        for i in 0..4 {
            assert_relative_eq!(pchip.eval(i as f64), y[i], epsilon = 1e-9);
        }
    }

    // -------------------------------------------------------------------
    // B-spline: partition of unity.
    // -------------------------------------------------------------------
    #[test]
    fn bspline_partition_of_unity() {
        // Open-uniform knot vector; the basis functions form a partition of
        // unity on the interior span (t_p, t_{m-p-1}) = (2, 6).
        let knots: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = 2;
        let n = knots.len() - p - 1;
        // Sample a deterministic grid strictly inside the interior span.
        for k in 3..6 {
            for m in 1..=9 {
                let x = k as f64 + m as f64 / 10.0;
                let mut sum = 0.0_f64;
                for i in 0..n {
                    sum += bspline_basis(p, &knots, i, x);
                }
                assert_relative_eq!(sum, 1.0, epsilon = 1e-9, max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn bspline_curve_uniform_weights_is_partition() {
        let knots: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = 2;
        let n = knots.len() - p - 1;
        let weights: Vec<f64> = vec![1.0; n];
        let curve = BsplineCurve::new(p, &knots, &weights);
        for k in 3..6 {
            let x = k as f64 + 0.5;
            assert_relative_eq!(curve.eval(x), 1.0, epsilon = 1e-9, max_relative = 1e-9);
        }
    }
}
