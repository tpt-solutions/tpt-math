//! Natural cubic splines and multilinear interpolation on rectilinear grids.

use alloc::{format, vec, vec::Vec};

use tpt_math_linalg_dense::{solve_tridiagonal, DVector, DenseError};

/// A natural cubic spline through `(x_i, y_i)`.
///
/// The spline is `C²` on the nodes and has zero second derivative at the two
/// end nodes. Between the nodes it is the usual cubic. Beyond the nodes it is
/// extended linearly with the slope at the nearest end, so the extension
/// does not curve away unboundedly.
///
/// The second derivatives at the interior nodes solve a symmetric positive
/// definite tridiagonal system, which is solved in `O(n)`.
#[derive(Clone, Debug, PartialEq)]
pub struct NaturalCubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Second derivative at each node.
    m: Vec<f64>,
}

impl NaturalCubicSpline {
    /// Fit a natural cubic spline through the nodes `(x, y)`.
    ///
    /// # Errors
    ///
    /// * [`DenseError::DimensionMismatch`] if `x` and `y` differ in length, or
    ///   there are fewer than two nodes.
    /// * [`DenseError::DimensionMismatch`] if `x` is not strictly increasing or
    ///   contains a non-finite value, or if `y` contains a non-finite value.
    pub fn new(x: &DVector<f64>, y: &DVector<f64>) -> Result<Self, DenseError> {
        let n = x.len();
        if y.len() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!("x has length {n}, y has length {}", y.len()),
            });
        }
        if n < 2 {
            return Err(DenseError::DimensionMismatch {
                what: format!("a cubic spline needs at least 2 nodes, got {n}"),
            });
        }
        let x: Vec<f64> = (0..n).map(|i| x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| y[i]).collect();
        if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
            return Err(DenseError::DimensionMismatch {
                what: "spline nodes must be finite".into(),
            });
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(DenseError::DimensionMismatch {
                what: "spline abscissae must be strictly increasing".into(),
            });
        }

        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let mut m = vec![0.0; n];
        if n > 2 {
            // Unknowns are M_1 .. M_{n-2}; M_0 = M_{n-1} = 0 (natural ends).
            let k = n - 2;
            let diag: Vec<f64> = (1..n - 1).map(|i| 2.0 * (h[i - 1] + h[i])).collect();
            let off: Vec<f64> = (1..k).map(|j| h[j]).collect();
            let rhs: Vec<f64> = (1..n - 1)
                .map(|i| 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]))
                .collect();
            let interior = solve_tridiagonal(&off, &diag, &off, &rhs)?;
            m[1..n - 1].copy_from_slice(&interior);
        }
        Ok(NaturalCubicSpline { x, y, m })
    }

    /// The `(first, last)` node abscissae.
    pub fn domain(&self) -> (f64, f64) {
        (self.x[0], self.x[self.x.len() - 1])
    }

    /// The spline value at `t`.
    pub fn eval(&self, t: f64) -> f64 {
        let last = self.x.len() - 1;
        if t < self.x[0] {
            return self.y[0] + self.derivative(self.x[0]) * (t - self.x[0]);
        }
        if t > self.x[last] {
            return self.y[last] + self.derivative(self.x[last]) * (t - self.x[last]);
        }
        let i = self.segment(t);
        let h = self.x[i + 1] - self.x[i];
        let a = (self.x[i + 1] - t) / h;
        let b = (t - self.x[i]) / h;
        a * self.y[i]
            + b * self.y[i + 1]
            + ((a * a * a - a) * self.m[i] + (b * b * b - b) * self.m[i + 1]) * h * h / 6.0
    }

    /// The first derivative of the spline at `t`.
    pub fn derivative(&self, t: f64) -> f64 {
        let i = self.segment(t.clamp(self.x[0], self.x[self.x.len() - 1]));
        let h = self.x[i + 1] - self.x[i];
        let a = (self.x[i + 1] - t) / h;
        let b = (t - self.x[i]) / h;
        (self.y[i + 1] - self.y[i]) / h - (3.0 * a * a - 1.0) * h * self.m[i] / 6.0
            + (3.0 * b * b - 1.0) * h * self.m[i + 1] / 6.0
    }

    /// Index `i` of the segment `[x_i, x_{i+1}]` that contains `t`, with `t`
    /// clamped to the domain.
    fn segment(&self, t: f64) -> usize {
        let n = self.x.len();
        let count = self.x.partition_point(|&xi| xi <= t);
        count.saturating_sub(1).min(n - 2)
    }
}

/// Multilinear (bilinear in 2-D, trilinear in 3-D, …) interpolation on a
/// rectilinear grid.
///
/// Each axis is a strictly increasing list of coordinates. The grid values are
/// stored with the last axis varying fastest, so for two axes `(xs, ys)` the
/// value at `(xs[i], ys[j])` is at position `i * ys.len() + j`.
///
/// Queries outside the grid return `None`. The grid does not extrapolate.
#[derive(Clone, Debug, PartialEq)]
pub struct MultilinearGrid {
    axes: Vec<Vec<f64>>,
    values: Vec<f64>,
    /// Stride of each axis in `values`.
    strides: Vec<usize>,
}

impl MultilinearGrid {
    /// Build a grid from its axes and values.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if there are no axes, an axis
    /// has fewer than two coordinates or is not strictly increasing or finite,
    /// or `values` does not have one entry per grid point.
    pub fn new(axes: &[DVector<f64>], values: &DVector<f64>) -> Result<Self, DenseError> {
        if axes.is_empty() {
            return Err(DenseError::DimensionMismatch {
                what: "a grid needs at least one axis".into(),
            });
        }
        let axes: Vec<Vec<f64>> = axes
            .iter()
            .enumerate()
            .map(|(k, ax)| {
                let v: Vec<f64> = (0..ax.len()).map(|i| ax[i]).collect();
                if v.len() < 2 {
                    return Err(DenseError::DimensionMismatch {
                        what: format!("axis {k} needs at least 2 coordinates"),
                    });
                }
                if v.iter().any(|c| !c.is_finite()) || v.windows(2).any(|w| w[1] <= w[0]) {
                    return Err(DenseError::DimensionMismatch {
                        what: format!("axis {k} must be finite and strictly increasing"),
                    });
                }
                Ok(v)
            })
            .collect::<Result<_, _>>()?;

        let mut strides = vec![1usize; axes.len()];
        for k in (0..axes.len() - 1).rev() {
            strides[k] = strides[k + 1] * axes[k + 1].len();
        }
        let total: usize = axes.iter().map(Vec::len).product();
        if values.len() != total {
            return Err(DenseError::DimensionMismatch {
                what: format!(
                    "the grid has {total} points but {} values were given",
                    values.len()
                ),
            });
        }
        let values: Vec<f64> = (0..total).map(|i| values[i]).collect();
        Ok(MultilinearGrid {
            axes,
            values,
            strides,
        })
    }

    /// The number of axes.
    pub fn dimension(&self) -> usize {
        self.axes.len()
    }

    /// Interpolate at `point`, one coordinate per axis.
    ///
    /// Returns `None` if `point` has the wrong number of coordinates, or any
    /// coordinate lies outside its axis (or is `NaN`).
    pub fn eval(&self, point: &[f64]) -> Option<f64> {
        let d = self.axes.len();
        if point.len() != d {
            return None;
        }
        // Lower cell index and fractional position along each axis.
        let mut lower = Vec::with_capacity(d);
        let mut frac = Vec::with_capacity(d);
        for (axis, &x) in self.axes.iter().zip(point) {
            let (first, last) = (axis[0], axis[axis.len() - 1]);
            if !(x >= first && x <= last) {
                return None;
            }
            let i = axis
                .partition_point(|&a| a <= x)
                .saturating_sub(1)
                .min(axis.len() - 2);
            lower.push(i);
            frac.push((x - axis[i]) / (axis[i + 1] - axis[i]));
        }

        // Blend the 2^d corners of the containing cell.
        let mut sum = 0.0_f64;
        for corner in 0..(1usize << d) {
            let mut weight = 1.0_f64;
            let mut index = 0usize;
            for k in 0..d {
                let high = (corner >> k) & 1;
                weight *= if high == 1 { frac[k] } else { 1.0 - frac[k] };
                index += (lower[k] + high) * self.strides[k];
            }
            sum += weight * self.values[index];
        }
        Some(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_vec(v.to_vec())
    }

    #[test]
    fn spline_passes_through_the_nodes() {
        let x = dv(&[0.0, 1.0, 2.5, 4.0, 5.0]);
        let y = dv(&[1.0, -0.5, 2.0, 0.3, 1.7]);
        let s = NaturalCubicSpline::new(&x, &y).unwrap();
        for i in 0..x.len() {
            assert_relative_eq!(s.eval(x[i]), y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn two_nodes_give_the_straight_line() {
        let s = NaturalCubicSpline::new(&dv(&[0.0, 2.0]), &dv(&[1.0, 5.0])).unwrap();
        assert_relative_eq!(s.eval(1.0), 3.0, epsilon = 1e-14);
        assert_relative_eq!(s.derivative(0.5), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn spline_is_c2_across_interior_nodes() {
        let x = dv(&[0.0, 0.7, 1.9, 3.0, 4.2]);
        let y = dv(&[0.0, 1.0, 0.2, -1.0, 0.5]);
        let s = NaturalCubicSpline::new(&x, &y).unwrap();
        let eps = 1e-7;
        for i in 1..x.len() - 1 {
            let left = s.derivative(x[i] - eps);
            let right = s.derivative(x[i] + eps);
            assert!((left - right).abs() < 1e-5, "first derivative jumps at {i}");
            // Second derivative from central differences of the first.
            let d2_left = (s.derivative(x[i] - 2.0 * eps) - s.derivative(x[i] - eps)) / eps;
            let d2_right = (s.derivative(x[i] + eps) - s.derivative(x[i] + 2.0 * eps)) / eps;
            assert!(
                (d2_left - d2_right).abs() < 1e-3,
                "second derivative jumps at {i}"
            );
        }
    }

    #[test]
    fn natural_ends_have_zero_curvature() {
        let x = dv(&[0.0, 1.0, 2.0, 3.0]);
        let y = dv(&[0.0, 2.0, 1.0, 4.0]);
        let s = NaturalCubicSpline::new(&x, &y).unwrap();
        // S''(t) is linear on the first segment with S''(0) = 0, so it is about
        // M_1 · t / h near the end. Sample very close to the end to see that.
        let eps = 1e-7;
        let d2 = |t: f64| (s.derivative(t + eps) - s.derivative(t - eps)) / (2.0 * eps);
        assert!(d2(1e-6).abs() < 1e-4, "S''(0+) = {}", d2(1e-6));
        assert!(d2(3.0 - 1e-6).abs() < 1e-4, "S''(3-) = {}", d2(3.0 - 1e-6));
    }

    #[test]
    fn spline_reproduces_a_straight_line_exactly() {
        let x = dv(&[0.0, 1.0, 3.0, 4.0]);
        let y: DVector<f64> = DVector::from_fn(4, |i| 2.0 * x[i] - 1.0);
        let s = NaturalCubicSpline::new(&x, &y).unwrap();
        for &t in &[0.5, 2.0, 3.7] {
            assert_relative_eq!(s.eval(t), 2.0 * t - 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn spline_extrapolates_linearly_with_the_end_slope() {
        let x = dv(&[0.0, 1.0, 2.0]);
        let y = dv(&[0.0, 1.0, 0.0]);
        let s = NaturalCubicSpline::new(&x, &y).unwrap();
        let slope = s.derivative(2.0);
        assert_relative_eq!(s.eval(3.0), 0.0 + slope * 1.0, epsilon = 1e-12);
    }

    #[test]
    fn spline_rejects_bad_nodes() {
        assert!(NaturalCubicSpline::new(&dv(&[0.0]), &dv(&[1.0])).is_err());
        assert!(NaturalCubicSpline::new(&dv(&[1.0, 0.0]), &dv(&[1.0, 2.0])).is_err());
        assert!(NaturalCubicSpline::new(&dv(&[0.0, 1.0]), &dv(&[1.0])).is_err());
        assert!(NaturalCubicSpline::new(&dv(&[0.0, f64::NAN]), &dv(&[1.0, 2.0])).is_err());
    }

    #[test]
    fn bilinear_interpolates_a_plane_exactly() {
        // z = 3x - 2y + 1 on a 3 x 4 grid.
        let xs = dv(&[0.0, 1.0, 2.5]);
        let ys = dv(&[-1.0, 0.0, 1.0, 3.0]);
        let mut vals = Vec::new();
        for &x in &[0.0, 1.0, 2.5] {
            for &y in &[-1.0, 0.0, 1.0, 3.0] {
                vals.push(3.0 * x - 2.0 * y + 1.0);
            }
        }
        let grid = MultilinearGrid::new(&[xs, ys], &dv(&vals)).unwrap();
        for &(x, y) in &[(0.5, 0.5), (1.7, -0.3), (2.5, 3.0), (0.0, -1.0)] {
            assert_relative_eq!(
                grid.eval(&[x, y]).unwrap(),
                3.0 * x - 2.0 * y + 1.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn trilinear_matches_a_known_corner_blend() {
        // Unit cube, value = corner index in binary: x + 2y + 4z.
        let unit = dv(&[0.0, 1.0]);
        let mut vals = Vec::new();
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    vals.push((x + 2 * y + 4 * z) as f64);
                }
            }
        }
        let grid = MultilinearGrid::new(&[unit.clone(), unit.clone(), unit], &dv(&vals)).unwrap();
        // Multilinear reproduces the affine function x + 2y + 4z everywhere.
        assert_relative_eq!(
            grid.eval(&[0.25, 0.5, 0.75]).unwrap(),
            0.25 + 1.0 + 3.0,
            epsilon = 1e-12
        );
        assert_eq!(grid.dimension(), 3);
    }

    #[test]
    fn grid_is_exact_at_grid_points_and_none_outside() {
        let xs = dv(&[0.0, 2.0, 5.0]);
        let vals = dv(&[1.0, 4.0, 9.0]);
        let grid = MultilinearGrid::new(&[xs], &vals).unwrap();
        assert_eq!(grid.eval(&[2.0]), Some(4.0));
        assert_eq!(grid.eval(&[-0.1]), None);
        assert_eq!(grid.eval(&[5.1]), None);
        assert_eq!(grid.eval(&[f64::NAN]), None);
        assert_eq!(grid.eval(&[1.0, 1.0]), None);
    }

    #[test]
    fn grid_rejects_bad_shapes() {
        let ax = dv(&[0.0, 1.0]);
        assert!(MultilinearGrid::new(&[], &dv(&[1.0])).is_err());
        assert!(MultilinearGrid::new(&[ax.clone(), ax.clone()], &dv(&[1.0, 2.0, 3.0])).is_err());
        assert!(MultilinearGrid::new(&[dv(&[1.0])], &dv(&[1.0])).is_err());
        assert!(MultilinearGrid::new(&[dv(&[1.0, 0.5])], &dv(&[1.0, 2.0])).is_err());
    }
}
