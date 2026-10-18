//! Eigenvalues of a general (not necessarily symmetric) real matrix.
//!
//! The matrix is first reduced to upper Hessenberg form by Householder
//! reflections, which are orthogonal similarities and so keep the eigenvalues.
//! Francis's double-shift QR iteration then finds the eigenvalues of the
//! Hessenberg matrix, one or two at a time, deflating as subdiagonal entries
//! become negligible. Complex eigenvalues come in conjugate pairs.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{sqrt_f64, DMatrix, DenseError};

/// An eigenvalue `re + i·im` of a real matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eigenvalue {
    /// Real part.
    pub re: f64,
    /// Imaginary part. Zero for a real eigenvalue.
    pub im: f64,
}

impl Eigenvalue {
    /// `true` if the imaginary part is exactly zero.
    pub fn is_real(&self) -> bool {
        self.im == 0.0
    }
}

/// Reduce `a` (square, `n × n`, row-major) to upper Hessenberg form in place.
fn hessenberg(a: &mut [Vec<f64>]) {
    let n = a.len();
    for k in 0..n.saturating_sub(2) {
        // Householder vector annihilating a[k+2.., k].
        let mut v: Vec<f64> = (k + 1..n).map(|i| a[i][k]).collect();
        let norm = sqrt_f64(v.iter().map(|x| x * x).sum::<f64>());
        if norm == 0.0 {
            continue;
        }
        let alpha = if v[0] >= 0.0 { -norm } else { norm };
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        // A ← H A, rows k+1..n.
        for j in 0..n {
            let dot: f64 = (k + 1..n).map(|i| v[i - k - 1] * a[i][j]).sum();
            let f = 2.0 * dot / vnorm2;
            for i in k + 1..n {
                a[i][j] -= f * v[i - k - 1];
            }
        }
        // A ← A H, columns k+1..n.
        for row in a.iter_mut() {
            let dot: f64 = (k + 1..n).map(|j| row[j] * v[j - k - 1]).sum();
            let f = 2.0 * dot / vnorm2;
            for j in k + 1..n {
                row[j] -= f * v[j - k - 1];
            }
        }
    }
}

/// `|a|` with the sign of `b`, as in the classical `SIGN` function.
fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// Francis double-shift QR on an upper Hessenberg matrix. Writes the eigenvalues
/// into `wr`/`wi`. This follows the classical `hqr` algorithm.
fn hqr(a: &mut [Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>), DenseError> {
    let n = a.len();
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    let anorm: f64 = (0..n)
        .map(|i| (i.saturating_sub(1)..n).map(|j| a[i][j].abs()).sum::<f64>())
        .sum();

    let mut nn = n as isize - 1;
    let mut t = 0.0_f64;
    while nn >= 0 {
        let mut its = 0usize;
        loop {
            // Look for a single small subdiagonal element.
            let mut l = nn;
            while l >= 1 {
                let (li, ln) = (l as usize, l as usize);
                let mut s = a[li - 1][li - 1].abs() + a[ln][ln].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if a[ln][li - 1].abs() + s == s {
                    a[ln][li - 1] = 0.0;
                    break;
                }
                l -= 1;
            }
            let nu = nn as usize;
            let x = a[nu][nu];
            if l == nn {
                // One root found.
                wr[nu] = x + t;
                wi[nu] = 0.0;
                nn -= 1;
            } else {
                let y = a[nu - 1][nu - 1];
                let w = a[nu][nu - 1] * a[nu - 1][nu];
                if l == nn - 1 {
                    // Two roots found.
                    let p = 0.5 * (y - x);
                    let q = p * p + w;
                    let z = sqrt_f64(q.abs());
                    let x = x + t;
                    if q >= 0.0 {
                        let z = p + sign(z, p);
                        wr[nu - 1] = x + z;
                        wr[nu] = if z != 0.0 { x - w / z } else { x + z };
                        wi[nu - 1] = 0.0;
                        wi[nu] = 0.0;
                    } else {
                        wr[nu - 1] = x + p;
                        wr[nu] = x + p;
                        wi[nu - 1] = -z;
                        wi[nu] = z;
                    }
                    nn -= 2;
                } else {
                    if its == 60 {
                        return Err(DenseError::NoConvergence {
                            what: "eigenvalues (Hessenberg QR)",
                        });
                    }
                    let (mut x, mut y, mut w) = (x, y, w);
                    if its == 10 || its == 20 {
                        // Exceptional shift to break cycles.
                        t += x;
                        for i in 0..=nu {
                            a[i][i] -= x;
                        }
                        let s = a[nu][nu - 1].abs() + a[nu - 1][nu - 2].abs();
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;

                    // Look for two consecutive small subdiagonal elements.
                    let mut m = nn - 2;
                    let (mut p, mut q, mut r);
                    loop {
                        let mu = m as usize;
                        let z = a[mu][mu];
                        r = x - z;
                        let s = y - z;
                        p = (r * s - w) / a[mu + 1][mu] + a[mu][mu + 1];
                        q = a[mu + 1][mu + 1] - z - r - s;
                        r = a[mu + 2][mu + 1];
                        let s = p.abs() + q.abs() + r.abs();
                        p /= s;
                        q /= s;
                        r /= s;
                        if m == l {
                            break;
                        }
                        let u = a[mu][mu - 1].abs() * (q.abs() + r.abs());
                        let v =
                            p.abs() * (a[mu - 1][mu - 1].abs() + z.abs() + a[mu + 1][mu + 1].abs());
                        if u + v == v {
                            break;
                        }
                        m -= 1;
                    }
                    let mu = m as usize;
                    for i in mu + 2..=nu {
                        a[i][i - 2] = 0.0;
                        if i != mu + 2 {
                            a[i][i - 3] = 0.0;
                        }
                    }

                    // Double QR step on rows l..=nn and columns m..=nn.
                    let mut k = mu;
                    while k < nu {
                        if k != mu {
                            p = a[k][k - 1];
                            q = a[k + 1][k - 1];
                            r = if k != nu - 1 { a[k + 2][k - 1] } else { 0.0 };
                            x = p.abs() + q.abs() + r.abs();
                            if x != 0.0 {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        let s = sign(sqrt_f64(p * p + q * q + r * r), p);
                        if s != 0.0 {
                            if k == mu {
                                if l as usize != mu {
                                    a[k][k - 1] = -a[k][k - 1];
                                }
                            } else {
                                a[k][k - 1] = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            let z = r / s;
                            q /= p;
                            r /= p;
                            for j in k..=nu {
                                let mut pp = a[k][j] + q * a[k + 1][j];
                                if k != nu - 1 {
                                    pp += r * a[k + 2][j];
                                    a[k + 2][j] -= pp * z;
                                }
                                a[k + 1][j] -= pp * y;
                                a[k][j] -= pp * x;
                            }
                            let mmin = if nu < k + 3 { nu } else { k + 3 };
                            for i in l as usize..=mmin {
                                let mut pp = x * a[i][k] + y * a[i][k + 1];
                                if k != nu - 1 {
                                    pp += z * a[i][k + 2];
                                    a[i][k + 2] -= pp * r;
                                }
                                a[i][k + 1] -= pp * q;
                                a[i][k] -= pp;
                            }
                        }
                        k += 1;
                    }
                }
            }
            // Continue the iteration on the active block only while it has more
            // than two rows; otherwise deflate.
            if !(l < nn - 1) {
                break;
            }
        }
    }
    Ok((wr, wi))
}

impl DMatrix<f64> {
    /// All eigenvalues of a square real matrix, real and complex.
    ///
    /// The matrix need not be symmetric. Complex eigenvalues appear in
    /// conjugate pairs. The order is the order the QR iteration deflates them,
    /// which is not sorted. For a symmetric matrix prefer
    /// [`symmetric_eigen`](Self::symmetric_eigen), which returns sorted real values
    /// and eigenvectors.
    ///
    /// For a real eigenvalue, [`eigenvector_for_eigenvalue`](Self::eigenvector_for_eigenvalue)
    /// gives the eigenvector.
    ///
    /// # Errors
    ///
    /// Returns [`DenseError::DimensionMismatch`] if the matrix is not square, and
    /// [`DenseError::NoConvergence`] if the QR iteration does not deflate within its
    /// budget (for example, with a NaN entry).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_linalg_dense::DMatrix;
    ///
    /// // Rotation by 90°: eigenvalues ±i.
    /// let r = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
    /// let mut ev = r.eigenvalues().unwrap();
    /// ev.sort_by(|a, b| a.im.total_cmp(&b.im));
    /// assert!((ev[0].im + 1.0).abs() < 1e-12 && (ev[1].im - 1.0).abs() < 1e-12);
    /// ```
    pub fn eigenvalues(&self) -> Result<Vec<Eigenvalue>, DenseError> {
        let n = self.nrows();
        if self.ncols() != n {
            return Err(DenseError::DimensionMismatch {
                what: format!(
                    "eigenvalues needs a square matrix, got {n}x{}",
                    self.ncols()
                ),
            });
        }
        if self.iter().any(|v| !v.is_finite()) {
            return Err(DenseError::NoConvergence {
                what: "eigenvalues (non-finite entry)",
            });
        }
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| self[(i, j)]).collect())
            .collect();
        hessenberg(&mut a);
        let (wr, wi) = hqr(&mut a)?;
        Ok(wr
            .into_iter()
            .zip(wi)
            .map(|(re, im)| Eigenvalue { re, im })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<Eigenvalue>) -> Vec<Eigenvalue> {
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
        v
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn triangular_matrix_gives_its_diagonal() {
        let a = DMatrix::from_row_slice(3, 3, &[2.0, 5.0, -1.0, 0.0, -3.0, 4.0, 0.0, 0.0, 7.0]);
        let ev = sorted(a.eigenvalues().unwrap());
        let want = [-3.0, 2.0, 7.0];
        for (e, w) in ev.iter().zip(want) {
            assert!(e.im.abs() < 1e-9 && close(e.re, w), "{e:?}");
        }
    }

    #[test]
    fn companion_matrix_of_known_polynomial() {
        // (x − 1)(x − 2)(x − 3)(x + 4) = x⁴ − 2x³ − 13x² + 38x − 24.
        // Companion matrix with first row (c3, c2, c1, c0) = (2, 13, −38, 24).
        let a = DMatrix::from_row_slice(
            4,
            4,
            &[
                2.0, 13.0, -38.0, 24.0, //
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
        );
        let ev = sorted(a.eigenvalues().unwrap());
        let want = [-4.0, 1.0, 2.0, 3.0];
        for (e, w) in ev.iter().zip(want) {
            assert!(e.im.abs() < 1e-8 && close(e.re, w), "{e:?} vs {w}");
        }
    }

    #[test]
    fn complex_pair_in_a_nonsymmetric_block() {
        // Eigenvalues 1 ± 2i.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, -2.0, 2.0, 1.0]);
        let ev = sorted(a.eigenvalues().unwrap());
        assert!(close(ev[0].re, 1.0) && close(ev[0].im, -2.0), "{ev:?}");
        assert!(close(ev[1].re, 1.0) && close(ev[1].im, 2.0), "{ev:?}");
    }

    #[test]
    fn trace_and_determinant_match_for_a_random_looking_matrix() {
        let n = 7;
        let a = DMatrix::from_fn(n, n, |i, j| {
            ((i * 13 + j * 7 + 3) % 11) as f64 - 5.0 + if i == j { 0.5 } else { 0.0 }
        });
        let ev = a.eigenvalues().unwrap();
        assert_eq!(ev.len(), n);
        let trace: f64 = (0..n).map(|i| a[(i, i)]).sum();
        let sum_re: f64 = ev.iter().map(|e| e.re).sum();
        let sum_im: f64 = ev.iter().map(|e| e.im).sum();
        assert!(close(sum_re, trace), "sum {sum_re} vs trace {trace}");
        assert!(sum_im.abs() < 1e-8);
        // The product of the eigenvalues is the determinant.
        let (mut pr, mut pi) = (1.0, 0.0);
        for e in &ev {
            let (r, i) = (pr * e.re - pi * e.im, pr * e.im + pi * e.re);
            pr = r;
            pi = i;
        }
        let det = a.det().unwrap();
        assert!(
            (pr - det).abs() <= 1e-7 * (1.0 + det.abs()),
            "product {pr} vs det {det}"
        );
        assert!(pi.abs() <= 1e-7 * (1.0 + det.abs()));
    }

    #[test]
    fn agrees_with_symmetric_eigen_on_a_symmetric_matrix() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 1.0]);
        let general = sorted(a.eigenvalues().unwrap());
        let (values, _) = a.symmetric_eigen().unwrap();
        for (g, s) in general.iter().zip(values.iter()) {
            assert!(g.im.abs() < 1e-9 && close(g.re, *s), "{g:?} vs {s}");
        }
    }

    #[test]
    fn cyclic_permutation_gives_the_roots_of_unity() {
        // Eigenvalues are the 6th roots of unity: exercises the exceptional shifts.
        let n = 6;
        let p = DMatrix::from_fn(n, n, |i, j| if j == (i + 1) % n { 1.0 } else { 0.0 });
        let ev = p.eigenvalues().unwrap();
        for e in &ev {
            let modulus = sqrt_f64(e.re * e.re + e.im * e.im);
            assert!((modulus - 1.0).abs() < 1e-9, "{e:?}");
            // λⁿ = 1
            let (mut r, mut i) = (1.0f64, 0.0f64);
            for _ in 0..n {
                let (nr, ni) = (r * e.re - i * e.im, r * e.im + i * e.re);
                r = nr;
                i = ni;
            }
            assert!((r - 1.0).abs() < 1e-9 && i.abs() < 1e-9, "{e:?}");
        }
    }

    #[test]
    fn larger_matrix_keeps_trace_and_determinant() {
        let n = 30;
        let a = DMatrix::from_fn(n, n, |i, j| {
            (((i * 31 + j * 17 + 5) % 23) as f64 - 11.0) / 7.0
        });
        let ev = a.eigenvalues().unwrap();
        assert_eq!(ev.len(), n);
        let trace: f64 = (0..n).map(|i| a[(i, i)]).sum();
        let sum_re: f64 = ev.iter().map(|e| e.re).sum();
        assert!(
            (sum_re - trace).abs() < 1e-7 * (1.0 + trace.abs()),
            "{sum_re} vs {trace}"
        );
    }

    #[test]
    fn non_square_and_non_finite_inputs_are_rejected() {
        assert!(DMatrix::zeros(2, 3).eigenvalues().is_err());
        let a = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
        assert!(a.eigenvalues().is_err());
    }
}
