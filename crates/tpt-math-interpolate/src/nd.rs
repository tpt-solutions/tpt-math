//! Scattered-data interpolation in any number of dimensions.
//!
//! [`RbfInterpolatorNd`] and [`KrigingNd`] are the N-D counterparts of the
//! 1-D [`RbfInterpolator`](crate::RbfInterpolator) and
//! [`Kriging`](crate::Kriging). Sites are the rows of a matrix, so the
//! dimension is the number of columns. Distances are Euclidean.

use alloc::{format, vec::Vec};

use tpt_math_linalg_dense::{DMatrix, DVector, DenseError, Lu};
use tpt_math_numeric::Float;

use crate::{RbfKernel, Variogram};

/// Check that `points` (sites by rows) and `values` agree, and return the
/// number of sites and dimensions.
fn check_sites(points: &DMatrix<f64>, values: &DVector<f64>) -> Result<(usize, usize), DenseError> {
    let (n, d) = (points.nrows(), points.ncols());
    if n == 0 || d == 0 {
        return Err(DenseError::DimensionMismatch {
            what: format!("need at least one site and one dimension, got {n} x {d}"),
        });
    }
    if values.len() != n {
        return Err(DenseError::DimensionMismatch {
            what: format!("points has {n} rows, values has length {}", values.len()),
        });
    }
    Ok((n, d))
}

/// Euclidean distance between row `i` of `points` and the point `x`.
fn distance_to(points: &DMatrix<f64>, i: usize, x: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for (k, &xk) in x.iter().enumerate() {
        let diff = points[(i, k)] - xk;
        s += diff * diff;
    }
    Float::sqrt(s)
}

/// Euclidean distance between rows `i` and `j` of `points`.
fn distance_between(points: &DMatrix<f64>, i: usize, j: usize) -> f64 {
    let mut s = 0.0_f64;
    for k in 0..points.ncols() {
        let diff = points[(i, k)] - points[(j, k)];
        s += diff * diff;
    }
    Float::sqrt(s)
}

/// Value of the affine basis function `index` at site `row`: `1` for index
/// `0`, otherwise the coordinate `index − 1`.
fn affine_basis_value(points: &DMatrix<f64>, row: usize, index: usize) -> f64 {
    if index == 0 {
        1.0
    } else {
        points[(row, index - 1)]
    }
}

/// A radial-basis-function interpolant of scattered data in `d` dimensions.
///
/// The interpolant is `s(x) = Σ_j w_j · φ(‖x − p_j‖) + p(x)`, where `p` is an
/// affine polynomial. For the thin-plate kernel the polynomial tail is
/// required: `r² ln r` is only conditionally positive definite, so the
/// interpolant is unique only once the constant and linear terms are included
/// with the side condition `Σ_j w_j q(p_j) = 0` for every affine `q`. The
/// Gaussian and multiquadric kernels are positive definite and get no tail.
///
/// The coefficients solve the augmented system
/// `[Φ P; Pᵀ 0] [w; c] = [y; 0]`, with `Φ_ij = φ(‖p_i − p_j‖)` and `P` the
/// matrix of affine basis values `(1, x_1, …, x_d)` at the sites.
#[derive(Clone, Debug)]
pub struct RbfInterpolatorNd {
    points: DMatrix<f64>,
    weights: DVector<f64>,
    /// Affine coefficients `(c_0, c_1, …, c_d)`; empty when there is no tail.
    tail: Vec<f64>,
    kernel: RbfKernel,
}

impl RbfInterpolatorNd {
    /// Build the interpolant from sites `points` (one site per row) and their
    /// `values`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the row count and the
    /// length of `values` differ, or there are no sites or no dimensions.
    /// Returns [`DenseError::Singular`] if the system cannot be solved, for
    /// example when the sites are collinear and a thin-plate tail is needed.
    pub fn new(
        points: DMatrix<f64>,
        values: DVector<f64>,
        kernel: RbfKernel,
    ) -> Result<Self, DenseError> {
        let (n, d) = check_sites(&points, &values)?;
        let m = if matches!(kernel, RbfKernel::ThinPlate { .. }) {
            d + 1
        } else {
            0
        };

        let size = n + m;
        let system = DMatrix::from_fn(size, size, |i, j| {
            if i < n && j < n {
                kernel.eval(distance_between(&points, i, j))
            } else if i < n {
                affine_basis_value(&points, i, j - n)
            } else if j < n {
                affine_basis_value(&points, j, i - n)
            } else {
                0.0
            }
        });
        let rhs = DVector::from_fn(size, |i| if i < n { values[i] } else { 0.0 });
        let solution = Lu::new(&system)?.solve(&rhs)?;

        let weights = DVector::from_fn(n, |i| solution[i]);
        let tail = (n..size).map(|i| solution[i]).collect();
        Ok(RbfInterpolatorNd {
            points,
            weights,
            tail,
            kernel,
        })
    }

    /// The number of dimensions of the sites.
    pub fn dimension(&self) -> usize {
        self.points.ncols()
    }

    /// Evaluate the interpolant at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` differs from [`dimension`](Self::dimension).
    pub fn eval(&self, x: &[f64]) -> f64 {
        assert_eq!(
            x.len(),
            self.dimension(),
            "query point has the wrong dimension"
        );
        let mut s = 0.0_f64;
        for j in 0..self.points.nrows() {
            s += self.weights[j] * self.kernel.eval(distance_to(&self.points, j, x));
        }
        if let Some((&constant, linear)) = self.tail.split_first() {
            s += constant;
            for (c, &xk) in linear.iter().zip(x) {
                s += c * xk;
            }
        }
        s
    }
}

/// An ordinary-Kriging surrogate of scattered data in `d` dimensions.
///
/// The N-D counterpart of [`Kriging`](crate::Kriging): the variogram is
/// evaluated at the Euclidean distance between sites, and the unbiasedness
/// constraint `Σ_i w_i = 1` is enforced by a bordered system.
#[derive(Clone, Debug)]
pub struct KrigingNd {
    points: DMatrix<f64>,
    values: DVector<f64>,
    variogram: Variogram,
    factor: Lu,
}

impl KrigingNd {
    /// Build a Kriging surrogate from sites `points` (one per row) and `values`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] for inconsistent sizes and
    /// [`DenseError::Singular`] if the Kriging matrix is singular, for example
    /// duplicate sites with a zero nugget.
    pub fn new(
        points: DMatrix<f64>,
        values: DVector<f64>,
        variogram: Variogram,
    ) -> Result<Self, DenseError> {
        let (n, _) = check_sites(&points, &values)?;
        let matrix = DMatrix::from_fn(n + 1, n + 1, |i, j| {
            if i < n && j < n {
                variogram.gamma(distance_between(&points, i, j))
            } else if i < n || j < n {
                1.0
            } else {
                0.0
            }
        });
        let factor = Lu::new(&matrix)?;
        Ok(KrigingNd {
            points,
            values,
            variogram,
            factor,
        })
    }

    /// The number of dimensions of the sites.
    pub fn dimension(&self) -> usize {
        self.points.ncols()
    }

    /// Predict the value and the Kriging variance at `x`.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if `x` has the wrong length.
    pub fn predict(&self, x: &[f64]) -> Result<(f64, f64), DenseError> {
        if x.len() != self.dimension() {
            return Err(DenseError::DimensionMismatch {
                what: format!(
                    "query has {} coordinates, sites have {}",
                    x.len(),
                    self.dimension()
                ),
            });
        }
        let n = self.points.nrows();
        let rhs = DVector::from_fn(n + 1, |i| {
            if i < n {
                self.variogram.gamma(distance_to(&self.points, i, x))
            } else {
                1.0
            }
        });
        let w = self.factor.solve(&rhs)?;
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += w[i] * self.values[i];
        }
        let mut variance = w[n];
        for i in 0..n {
            variance += w[i] * rhs[i];
        }
        Ok((mean, variance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;

    /// Six sites in the unit square; the data is the plane `x + 2y`.
    fn plane_sites() -> (DMatrix<f64>, DVector<f64>) {
        let pts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.25, 0.75),
        ];
        let data: Vec<f64> = pts.iter().flat_map(|&(x, y)| [x, y]).collect();
        let values: Vec<f64> = pts.iter().map(|&(x, y)| x + 2.0 * y).collect();
        (
            DMatrix::from_row_slice(pts.len(), 2, &data),
            DVector::from_vec(values),
        )
    }

    fn row(pts: &DMatrix<f64>, i: usize) -> Vec<f64> {
        (0..pts.ncols()).map(|k| pts[(i, k)]).collect()
    }

    #[test]
    fn rbf_gaussian_is_exact_at_nodes_in_2d() {
        let (pts, vals) = plane_sites();
        let rbf = RbfInterpolatorNd::new(
            pts.clone(),
            vals.clone(),
            RbfKernel::Gaussian { epsilon: 2.0 },
        )
        .unwrap();
        for i in 0..pts.nrows() {
            assert_relative_eq!(rbf.eval(&row(&pts, i)), vals[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn rbf_thin_plate_reproduces_an_affine_function_everywhere() {
        // The affine tail makes the thin-plate interpolant reproduce any affine
        // function exactly, between the nodes and beyond them.
        let (pts, vals) = plane_sites();
        let rbf = RbfInterpolatorNd::new(pts, vals, RbfKernel::ThinPlate { epsilon: 1.0 }).unwrap();
        for &(x, y) in &[(0.1, 0.9), (0.6, 0.2), (2.0, -1.0), (-0.5, 0.3)] {
            assert_relative_eq!(rbf.eval(&[x, y]), x + 2.0 * y, epsilon = 1e-8);
        }
    }

    #[test]
    fn rbf_multiquadric_is_exact_at_nodes_in_3d() {
        let pts = DMatrix::from_row_slice(
            4,
            3,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let vals = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let rbf = RbfInterpolatorNd::new(
            pts.clone(),
            vals.clone(),
            RbfKernel::Multiquadric { epsilon: 1.0 },
        )
        .unwrap();
        for i in 0..4 {
            assert_relative_eq!(rbf.eval(&row(&pts, i)), vals[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn rbf_rejects_mismatched_sizes() {
        let pts = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let vals = DVector::from_vec(vec![1.0]);
        assert!(matches!(
            RbfInterpolatorNd::new(pts, vals, RbfKernel::ThinPlate { epsilon: 1.0 }),
            Err(DenseError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn kriging_nd_is_exact_at_nodes_with_zero_nugget() {
        let (pts, vals) = plane_sites();
        let kriging = KrigingNd::new(
            pts.clone(),
            vals.clone(),
            Variogram::Linear {
                slope: 1.0,
                nugget: 0.0,
            },
        )
        .unwrap();
        for i in 0..pts.nrows() {
            let (mean, var) = kriging.predict(&row(&pts, i)).unwrap();
            assert_relative_eq!(mean, vals[i], epsilon = 1e-8);
            assert!(var.abs() < 1e-8);
        }
    }

    #[test]
    fn kriging_nd_is_unbiased_for_a_constant_field() {
        // Ordinary Kriging's weights sum to one, so a constant field is
        // reproduced everywhere. (A linear trend is not, in general: that needs
        // universal Kriging.)
        let pts =
            DMatrix::from_row_slice(5, 2, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.4, 0.9]);
        let vals = DVector::from_vec(vec![7.0; 5]);
        let kriging = KrigingNd::new(
            pts,
            vals,
            Variogram::Linear {
                slope: 1.0,
                nugget: 0.0,
            },
        )
        .unwrap();
        for &(x, y) in &[(0.3, 0.6), (-2.0, 5.0), (0.5, 0.5)] {
            let (mean, _) = kriging.predict(&[x, y]).unwrap();
            assert_relative_eq!(mean, 7.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn kriging_nd_variance_grows_away_from_samples() {
        let pts = DMatrix::from_row_slice(3, 2, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let vals = DVector::from_vec(vec![0.0, 1.0, 1.0]);
        let kriging = KrigingNd::new(
            pts,
            vals,
            Variogram::Gaussian {
                sill: 1.0,
                range: 1.0,
                nugget: 0.0,
            },
        )
        .unwrap();
        let (_, near) = kriging.predict(&[0.05, 0.05]).unwrap();
        let (_, far) = kriging.predict(&[3.0, 3.0]).unwrap();
        assert!(far > near);
    }

    #[test]
    fn kriging_nd_rejects_wrong_query_dimension() {
        let (pts, vals) = plane_sites();
        let kriging = KrigingNd::new(
            pts,
            vals,
            Variogram::Linear {
                slope: 1.0,
                nugget: 0.0,
            },
        )
        .unwrap();
        assert!(matches!(
            kriging.predict(&[0.0]),
            Err(DenseError::DimensionMismatch { .. })
        ));
    }
}
