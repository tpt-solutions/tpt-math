//! Certified linear solves by the Krawczyk operator.
//!
//! [`certified_solve`] returns, for a square system `A x = b` with `f64` data, an
//! interval with exact rational endpoints that is guaranteed to contain the
//! solution. The approximate inverse is computed in `f64` and only serves as a
//! preconditioner. The contraction test and the enclosure are exact rational
//! arithmetic, so rounding in the preconditioner cannot make the enclosure wrong.
//! It can only make it wide, or make the test fail.

use alloc::{vec, vec::Vec};
use core::fmt;

use num_traits::{One, ToPrimitive, Zero};

use crate::{BigRational, Interval, Rational};

/// Why a certified solve could not produce an enclosure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CertifyError {
    /// `a` is not `n × n` for `n = b.len()`.
    DimensionMismatch,
    /// An input entry is `NaN` or infinite.
    NotFinite,
    /// The approximate inverse could not be formed in `f64`, so the matrix is
    /// singular to working precision.
    Singular,
    /// The Krawczyk operator does not contract: the preconditioned matrix is too
    /// far from the identity for this method to certify the solution. `row_sum`
    /// is the bound `‖I − R A‖∞` (at least 1). Tighten the problem or use a
    /// different method.
    NotContracting {
        /// The computed bound on `‖I − R A‖∞`.
        row_sum: f64,
    },
}

impl fmt::Display for CertifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CertifyError::DimensionMismatch => write!(f, "matrix is not square of the rhs size"),
            CertifyError::NotFinite => write!(f, "input contains a NaN or infinite entry"),
            CertifyError::Singular => write!(f, "matrix is singular to working precision"),
            CertifyError::NotContracting { row_sum } => write!(
                f,
                "Krawczyk operator does not contract (‖I − RA‖∞ = {row_sum})"
            ),
        }
    }
}

impl core::error::Error for CertifyError {}

fn rat(x: f64) -> Option<Rational> {
    BigRational::from_float(x)
}

/// Approximate inverse of the row-major `n × n` matrix `a` by Gauss–Jordan with
/// partial pivoting, in `f64`.
fn approximate_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let w = 2 * n;
    let mut m = vec![0.0_f64; n * w];
    for i in 0..n {
        for j in 0..n {
            m[i * w + j] = a[i * n + j];
        }
        m[i * w + n + i] = 1.0;
    }
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&p, &q| m[p * w + col].abs().total_cmp(&m[q * w + col].abs()))?;
        if m[pivot * w + col] == 0.0 || !m[pivot * w + col].is_finite() {
            return None;
        }
        for k in 0..w {
            m.swap(col * w + k, pivot * w + k);
        }
        let p = m[col * w + col];
        for k in 0..w {
            m[col * w + k] /= p;
        }
        for row in 0..n {
            if row != col {
                let factor = m[row * w + col];
                if factor != 0.0 {
                    for k in 0..w {
                        m[row * w + k] -= factor * m[col * w + k];
                    }
                }
            }
        }
    }
    Some(
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| m[i * w + n + j])
            .collect(),
    )
}

/// Certified enclosure of the solution of `A x = b`.
///
/// `a` is the row-major `n × n` matrix and `b` has length `n`. On success, the
/// `i`th interval contains the `i`th component of the exact solution of the
/// system with the given `f64` data.
///
/// The method is the Krawczyk operator with the `f64` inverse of the midpoint
/// matrix as preconditioner. For a candidate box of radius `r` about the
/// approximate solution `m`, it checks that the box maps strictly into itself,
/// which proves a unique solution lies inside it.
///
/// # Errors
///
/// See [`CertifyError`]. A `NotContracting` error means the method cannot decide
/// the question. It does not mean the system has no solution.
///
/// # Examples
///
/// ```
/// use tpt_math_exact::{certified_solve, BigInt, BigRational};
///
/// // [4 1; 2 3] x = [1, 2] has the exact solution (1/10, 3/5).
/// let enc = certified_solve(&[4.0, 1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap();
/// let tenth = BigRational::new(BigInt::from(1), BigInt::from(10));
/// assert!(enc[0].contains(&tenth));
/// ```
pub fn certified_solve(a: &[f64], b: &[f64]) -> Result<Vec<Interval<Rational>>, CertifyError> {
    let n = b.len();
    if a.len() != n * n {
        return Err(CertifyError::DimensionMismatch);
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(CertifyError::NotFinite);
    }
    let inv = approximate_inverse(a, n).ok_or(CertifyError::Singular)?;

    // Midpoint: the f64 approximate solution, converted exactly.
    let m: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| inv[i * n + j] * b[j]).sum())
        .collect();

    // Everything from here on is exact rational arithmetic on the f64 data.
    let to_rat = |x: f64| rat(x).ok_or(CertifyError::NotFinite);
    let ar: Vec<Rational> = a.iter().map(|&x| to_rat(x)).collect::<Result<_, _>>()?;
    let br: Vec<Rational> = b.iter().map(|&x| to_rat(x)).collect::<Result<_, _>>()?;
    let mr: Vec<Rational> = m.iter().map(|&x| to_rat(x)).collect::<Result<_, _>>()?;
    let rr: Vec<Rational> = inv.iter().map(|&x| to_rat(x)).collect::<Result<_, _>>()?;

    // Residual of the midpoint: b − A m.
    let residual: Vec<Rational> = (0..n)
        .map(|i| {
            let mut acc = br[i].clone();
            for j in 0..n {
                acc -= &ar[i * n + j] * &mr[j];
            }
            acc
        })
        .collect();
    // Newton correction d = R (b − A m).
    let d: Vec<Rational> = (0..n)
        .map(|i| {
            let mut acc = Rational::zero();
            for j in 0..n {
                acc += &rr[i * n + j] * &residual[j];
            }
            acc
        })
        .collect();

    // C = I − R A, and its exact infinity norm ρ.
    let c: Vec<Rational> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let mut ra = Rational::zero();
            for k in 0..n {
                ra += &rr[i * n + k] * &ar[k * n + j];
            }
            let identity = if i == j {
                Rational::one()
            } else {
                Rational::zero()
            };
            identity - ra
        })
        .collect();
    let rho = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| c[i * n + j].abs())
                .fold(Rational::zero(), |acc, x| acc + x)
        })
        .fold(Rational::zero(), |acc, x| if x > acc { x } else { acc });
    let one = Rational::one();
    if rho >= one {
        let row_sum = rat_to_f64(&rho);
        return Err(CertifyError::NotContracting { row_sum });
    }

    // Radius: the box m ± r must satisfy max|d| + ρ r ≤ r, i.e. r ≥ max|d| / (1 − ρ).
    let max_d = d
        .iter()
        .map(|x| x.abs())
        .fold(Rational::zero(), |acc, x| if x > acc { x } else { acc });
    let radius = if max_d.is_zero() {
        // m is already the exact solution: any positive radius contains it.
        BigRational::new(
            num_bigint::BigInt::from(1),
            num_bigint::BigInt::from(1) << 60,
        )
    } else {
        let two = Rational::from_integer(num_bigint::BigInt::from(2));
        // Strictly larger than the bound, so the containment is strict.
        two * max_d / (one - rho)
    };

    Ok((0..n)
        .map(|i| {
            let centre = &mr[i];
            Interval::new(centre - &radius, centre + &radius)
        })
        .collect())
}

fn rat_to_f64(x: &Rational) -> f64 {
    x.to_f64().unwrap_or(f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn q(n: i64, d: i64) -> Rational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    #[test]
    fn two_by_two_contains_the_exact_solution() {
        // [4 1; 2 3] x = [1, 2]: x = (1/10, 3/5).
        let enc = certified_solve(&[4.0, 1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap();
        assert!(enc[0].contains(&q(1, 10)));
        assert!(enc[1].contains(&q(3, 5)));
        // The enclosure is tight for a well-conditioned system.
        assert!(enc[0].width() < q(1, 1_000_000));
    }

    #[test]
    fn four_by_four_integer_system_contains_the_rational_solution() {
        // Exact solution computed by hand: x = (1, 2, 3, 4) for the chosen matrix.
        let a = [
            [5.0, 1.0, 0.0, 0.0],
            [1.0, 6.0, 2.0, 0.0],
            [0.0, 2.0, 7.0, 3.0],
            [0.0, 0.0, 3.0, 8.0],
        ];
        let x = [1.0, 2.0, 3.0, 4.0];
        let flat: Vec<f64> = a.iter().flatten().copied().collect();
        let b: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|j| a[i][j] * x[j]).sum())
            .collect();
        let enc = certified_solve(&flat, &b).unwrap();
        for (i, interval) in enc.iter().enumerate() {
            assert!(interval.contains(&q(i as i64 + 1, 1)), "component {i}");
        }
    }

    #[test]
    fn nearly_singular_matrix_that_stays_nonsingular_is_enclosed_tightly() {
        // 1 + 1e-10 is exact in the matrix data, so the system is nonsingular and
        // the true solution (2, 0) must lie inside the enclosure.
        let enc = certified_solve(&[1.0, 1.0, 1.0, 1.0 + 1e-10], &[2.0, 2.0]).unwrap();
        assert!(enc[0].contains(&q(2, 1)));
        assert!(enc[1].contains(&q(0, 1)));
    }

    /// Exact rational solution of `a x = b` by Gauss elimination (test oracle).
    #[allow(clippy::needless_range_loop)]
    fn exact_solve(a: &[f64], b: &[f64]) -> Vec<Rational> {
        let n = b.len();
        let mut m: Vec<Vec<Rational>> = (0..n)
            .map(|i| {
                let mut row: Vec<Rational> = (0..n).map(|j| rat(a[i * n + j]).unwrap()).collect();
                row.push(rat(b[i]).unwrap());
                row
            })
            .collect();
        for col in 0..n {
            let pivot = (col..n).find(|&r| !m[r][col].is_zero()).unwrap();
            m.swap(col, pivot);
            for r in 0..n {
                if r != col && !m[r][col].is_zero() {
                    let f = &m[r][col] / &m[col][col];
                    for k in col..=n {
                        let v = &m[col][k] * &f;
                        m[r][k] -= v;
                    }
                }
            }
        }
        (0..n).map(|i| &m[i][n] / &m[i][i]).collect()
    }

    #[test]
    fn hilbert_enclosure_contains_the_exact_rational_solution() {
        // Order-8 Hilbert matrix from f64 data: the enclosure must contain the
        // exact solution of that f64 system, however ill-conditioned it is.
        let n = 8;
        let a: Vec<f64> = (0..n * n)
            .map(|k| 1.0 / ((k / n + k % n + 1) as f64))
            .collect();
        let b = vec![1.0; n];
        let enc = certified_solve(&a, &b).unwrap();
        let exact = exact_solve(&a, &b);
        for i in 0..n {
            assert!(enc[i].contains(&exact[i]), "component {i}");
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            certified_solve(&[1.0, 2.0], &[1.0, 2.0]),
            Err(CertifyError::DimensionMismatch)
        );
        assert_eq!(
            certified_solve(&[f64::NAN, 0.0, 0.0, 1.0], &[1.0, 1.0]),
            Err(CertifyError::NotFinite)
        );
        assert_eq!(
            certified_solve(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0]),
            Err(CertifyError::Singular)
        );
    }
}
