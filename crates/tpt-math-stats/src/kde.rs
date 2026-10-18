//! Gaussian kernel density estimation.
//!
//! A kernel density estimate places a Gaussian bump of width `h` on every
//! observation and averages them:
//!
//! ```text
//! f̂(x) = (1 / (n·h·√(2π))) · Σᵢ exp(−½ ((x − xᵢ)/h)²)
//! ```
//!
//! The default bandwidth is Silverman's rule of thumb,
//! `h = 0.9 · min(s, IQR/1.34) · n^(−1/5)`, which is exact for normal data and
//! a reasonable starting point otherwise.

use core::f64::consts::PI;

use crate::descriptive::{try_quantile, try_std_dev};
use crate::error::{check_all_finite, check_min_len, check_parameter_finite, StatsError};

/// A fitted Gaussian kernel density estimate.
///
/// Build one with [`try_kernel_density`] or [`kernel_density`]. The estimate is
/// a smooth, non-negative function that integrates to one.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDensity {
    samples: Vec<f64>,
    bandwidth: f64,
}

impl KernelDensity {
    /// The bandwidth `h` the estimate was built with.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// The number of observations the estimate is built from.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Always `false`: an estimate needs at least two observations.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The density estimate at `x`.
    pub fn density(&self, x: f64) -> f64 {
        let h = self.bandwidth;
        let sum: f64 = self
            .samples
            .iter()
            .map(|&xi| {
                let u = (x - xi) / h;
                (-0.5 * u * u).exp()
            })
            .sum();
        sum / (self.samples.len() as f64 * h * (2.0 * PI).sqrt())
    }

    /// The density estimate at each point of `grid`, in the same order.
    pub fn evaluate(&self, grid: &[f64]) -> Vec<f64> {
        grid.iter().map(|&x| self.density(x)).collect()
    }
}

/// Fit a Gaussian kernel density estimate.
///
/// `bandwidth` is the kernel width `h`. Pass `None` for Silverman's rule of
/// thumb.
///
/// # Errors
///
/// * [`StatsError::NotEnoughData`] for fewer than two samples.
/// * [`StatsError::NotFinite`] for a `NaN` or infinite sample.
/// * [`StatsError::ParameterNotFinite`] for a `NaN` or infinite bandwidth.
/// * [`StatsError::NonPositiveParameter`] for a bandwidth that is not strictly
///   positive. This includes the default rule when every sample is equal,
///   because the spread is then zero and no bandwidth can be inferred.
///
/// # Examples
///
/// ```
/// use tpt_math_stats::try_kernel_density;
///
/// let data = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let kde = try_kernel_density(&data, Some(0.5)).unwrap();
/// // The estimate peaks near the middle of the data.
/// assert!(kde.density(2.0) > kde.density(-2.0));
/// ```
pub fn try_kernel_density(
    samples: &[f64],
    bandwidth: Option<f64>,
) -> Result<KernelDensity, StatsError> {
    check_min_len("samples", samples, 2)?;
    check_all_finite("samples", samples)?;
    let h = match bandwidth {
        Some(h) => {
            check_parameter_finite("bandwidth", h)?;
            if h <= 0.0 {
                return Err(StatsError::NonPositiveParameter {
                    parameter: "bandwidth",
                    value: h,
                });
            }
            h
        }
        None => silverman_bandwidth(samples)?,
    };
    Ok(KernelDensity {
        samples: samples.to_vec(),
        bandwidth: h,
    })
}

/// Panicking form of [`try_kernel_density`].
///
/// # Panics
///
/// Panics wherever [`try_kernel_density`] returns an error.
pub fn kernel_density(samples: &[f64], bandwidth: Option<f64>) -> KernelDensity {
    try_kernel_density(samples, bandwidth).unwrap_or_else(|e| panic!("kernel_density: {e}"))
}

/// Silverman's rule of thumb, `0.9 · min(s, IQR/1.34) · n^(−1/5)`.
fn silverman_bandwidth(samples: &[f64]) -> Result<f64, StatsError> {
    let n = samples.len() as f64;
    let sd = try_std_dev(samples)?;
    let iqr = try_quantile(samples, 0.75)? - try_quantile(samples, 0.25)?;
    let spread = sd.min(iqr / 1.34);
    let h = 0.9 * spread * n.powf(-0.2);
    if h > 0.0 && h.is_finite() {
        Ok(h)
    } else {
        Err(StatsError::NonPositiveParameter {
            parameter: "bandwidth",
            value: h,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, evenly spread sample on `[-2, 2]`. The tests only need
    /// reproducible data, not a particular distribution.
    fn grid_sample(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let p = (i as f64 + 0.5) / n as f64;
                (p - 0.5) * 4.0
            })
            .collect()
    }

    #[test]
    fn density_integrates_to_one() {
        let data = grid_sample(40);
        let kde = try_kernel_density(&data, None).unwrap();
        let (lo, hi, steps) = (-12.0, 12.0, 24_000);
        let dx = (hi - lo) / steps as f64;
        let area: f64 = (0..=steps)
            .map(|k| {
                let w = if k == 0 || k == steps { 0.5 } else { 1.0 };
                w * kde.density(lo + k as f64 * dx)
            })
            .sum::<f64>()
            * dx;
        assert!((area - 1.0).abs() < 1e-4, "area = {area}");
    }

    #[test]
    fn single_gaussian_kernel_matches_closed_form() {
        // Two points 0 and 0 with h = 1: density is the N(0, 1) pdf.
        let kde = try_kernel_density(&[0.0, 0.0], Some(1.0)).unwrap();
        let want = (-0.5f64).exp() / (2.0 * PI).sqrt();
        assert!((kde.density(1.0) - want).abs() < 1e-14);
        assert!((kde.density(0.0) - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn silverman_bandwidth_for_normal_like_data() {
        // With sd = IQR/1.34 the rule reduces to 0.9 · s · n^(-1/5).
        let data = grid_sample(200);
        let kde = try_kernel_density(&data, None).unwrap();
        let sd = try_std_dev(&data).unwrap();
        let iqr = try_quantile(&data, 0.75).unwrap() - try_quantile(&data, 0.25).unwrap();
        let want = 0.9 * sd.min(iqr / 1.34) * 200f64.powf(-0.2);
        assert!((kde.bandwidth() - want).abs() < 1e-12);
        assert_eq!(kde.len(), 200);
    }

    #[test]
    fn evaluate_matches_pointwise_density() {
        let kde = try_kernel_density(&[1.0, 2.0, 4.0], Some(0.7)).unwrap();
        let grid = [-1.0, 0.0, 2.5, 5.0];
        let batch = kde.evaluate(&grid);
        for (g, b) in grid.iter().zip(&batch) {
            assert_eq!(kde.density(*g), *b);
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            try_kernel_density(&[1.0], None),
            Err(StatsError::NotEnoughData { .. })
        ));
        assert!(matches!(
            try_kernel_density(&[1.0, f64::NAN], None),
            Err(StatsError::NotFinite { .. })
        ));
        assert!(matches!(
            try_kernel_density(&[1.0, 2.0], Some(0.0)),
            Err(StatsError::NonPositiveParameter { .. })
        ));
        assert!(matches!(
            try_kernel_density(&[1.0, 2.0], Some(f64::INFINITY)),
            Err(StatsError::ParameterNotFinite { .. })
        ));
    }

    #[test]
    fn constant_data_needs_an_explicit_bandwidth() {
        assert!(matches!(
            try_kernel_density(&[3.0, 3.0, 3.0], None),
            Err(StatsError::NonPositiveParameter { .. })
        ));
        assert!(try_kernel_density(&[3.0, 3.0, 3.0], Some(0.5)).is_ok());
    }
}
