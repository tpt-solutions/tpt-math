//! Gaussian covariance construction, 4D time-conditioning and EWA projection.

use crate::sym::{identity, matmul, Mat, Sym3, Sym4};
use tpt_math_numeric::Scalar;

/// Normalises a `[w, x, y, z]` quaternion; `None` if zero or non-finite.
fn normalize_quat<T: Scalar>(q: [T; 4]) -> Option<[T; 4]> {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if !(n2 > T::zero()) || !n2.is_finite() {
        return None;
    }
    let inv = T::one() / n2.sqrt();
    Some([q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv])
}

/// 3×3 rotation matrix of the (normalised) quaternion `[w, x, y, z]`.
pub fn quat_to_rot3<T: Scalar>(q: [T; 4]) -> Option<Mat<T, 3>> {
    let [w, x, y, z] = normalize_quat(q)?;
    let two = T::one() + T::one();
    let one = T::one();
    Some([
        [
            one - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
        ],
        [
            two * (x * y + w * z),
            one - two * (x * x + z * z),
            two * (y * z - w * x),
        ],
        [
            two * (x * z - w * y),
            two * (y * z + w * x),
            one - two * (x * x + y * y),
        ],
    ])
}

/// 4×4 rotation `L(q_l) · R(q_r)` from a pair of quaternions `[w, x, y, z]`
/// (each normalised). `L` and `R` are left/right quaternion multiplication, so
/// the product is orthogonal with determinant `+1` (a general 4D rotation).
pub fn rot4_from_isoclinic<T: Scalar>(ql: [T; 4], qr: [T; 4]) -> Option<Mat<T, 4>> {
    let [a, b, c, d] = normalize_quat(ql)?;
    let l = [[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]];
    let [a, b, c, d] = normalize_quat(qr)?;
    let r = [[a, -b, -c, -d], [b, a, d, -c], [c, -d, a, b], [d, c, -b, a]];
    Some(matmul(&l, &r))
}

/// `Σ = R S Sᵀ Rᵀ` for a rotation quaternion and per-axis scales. Positive
/// semi-definite by construction (any real `s`, including signs, is valid).
pub fn cov3_from_rot_scale<T: Scalar>(q: [T; 4], s: [T; 3]) -> Option<Sym3<T>> {
    let r = quat_to_rot3(q)?;
    let mut m = [[T::zero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = r[i][j] * s[j];
        }
    }
    Some(Sym3::from_full(&gram(&m)))
}

/// 4D analogue of [`cov3_from_rot_scale`]: `Σ = R S Sᵀ Rᵀ` with
/// `R = L(q_l) R(q_r)`.
pub fn cov4_from_rot_scale<T: Scalar>(ql: [T; 4], qr: [T; 4], s: [T; 4]) -> Option<Sym4<T>> {
    let r = rot4_from_isoclinic(ql, qr)?;
    let mut m = [[T::zero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = r[i][j] * s[j];
        }
    }
    Some(Sym4::from_full(&gram(&m)))
}

/// `M Mᵀ` (always symmetric positive semi-definite).
fn gram<T: Scalar, const N: usize>(m: &Mat<T, N>) -> Mat<T, N> {
    let mut out = identity::<T, N>();
    for i in 0..N {
        for j in 0..N {
            let mut s = T::zero();
            for k in 0..N {
                s = s + m[i][k] * m[j][k];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Spatial marginal of a 4D Gaussian: the top-left 3×3 block of `Σ`.
pub fn marginal_xyz<T: Scalar>(cov: &Sym4<T>) -> Sym3<T> {
    Sym3::new(
        cov.get(0, 0),
        cov.get(0, 1),
        cov.get(0, 2),
        cov.get(1, 1),
        cov.get(1, 2),
        cov.get(2, 2),
    )
}

/// A 4D Gaussian sliced at a fixed time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionedGaussian<T> {
    /// Conditional spatial mean `μ_xyz + Σ_xt Σ_tt⁻¹ (t − μ_t)`.
    pub mean: [T; 3],
    /// Conditional spatial covariance (Schur complement
    /// `Σ_xx − Σ_xt Σ_tt⁻¹ Σ_tx`).
    pub cov: Sym3<T>,
    /// Temporal marginal weight `exp(−½ (t − μ_t)² / Σ_tt)` in `(0, 1]`.
    pub weight: T,
}

/// Conditions the 4D Gaussian `N(mean, cov)` (axes `x, y, z, t`) on time `t`.
///
/// Because the conditioning variable is a scalar, no matrix inverse is needed.
/// Returns `None` if `Σ_tt` is not strictly positive/finite or `t` is
/// non-finite.
pub fn condition_on_time<T: Scalar>(
    mean: [T; 4],
    cov: &Sym4<T>,
    t: T,
) -> Option<ConditionedGaussian<T>> {
    let stt = cov.get(3, 3);
    if !(stt > T::zero()) || !stt.is_finite() || !t.is_finite() {
        return None;
    }
    let dt = t - mean[3];
    let half = T::one() / (T::one() + T::one());
    let weight = (-half * dt * dt / stt).exp();
    let sxt = [cov.get(0, 3), cov.get(1, 3), cov.get(2, 3)];
    let mut m = [T::zero(); 3];
    for i in 0..3 {
        m[i] = mean[i] + sxt[i] / stt * dt;
    }
    let c = |i: usize, j: usize| cov.get(i, j) - sxt[i] * sxt[j] / stt;
    Some(ConditionedGaussian {
        mean: m,
        cov: Sym3::new(c(0, 0), c(0, 1), c(0, 2), c(1, 1), c(1, 2), c(2, 2)),
        weight,
    })
}

/// 2×2 symmetric matrix `[[xx, xy], [xy, yy]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sym2<T> {
    /// `xx` entry.
    pub xx: T,
    /// `xy` entry.
    pub xy: T,
    /// `yy` entry.
    pub yy: T,
}

impl<T: Scalar> Sym2<T> {
    /// Determinant.
    pub fn det(&self) -> T {
        self.xx * self.yy - self.xy * self.xy
    }

    /// Inverse (the screen-space *conic*); `None` if the determinant is not
    /// strictly positive/finite (degenerate splat).
    pub fn inverse(&self) -> Option<Self> {
        let det = self.det();
        if !(det > T::zero()) || !det.is_finite() {
            return None;
        }
        let inv = T::one() / det;
        Some(Self {
            xx: self.yy * inv,
            xy: -self.xy * inv,
            yy: self.xx * inv,
        })
    }

    /// Eigenvalues `(larger, smaller)`; the larger gives the splat's 3σ
    /// screen radius `3 √λ₁`.
    pub fn eigenvalues(&self) -> (T, T) {
        let half = T::one() / (T::one() + T::one());
        let mid = (self.xx + self.yy) * half;
        let disc = (mid * mid - self.det()).max(T::zero()).sqrt();
        (mid + disc, mid - disc)
    }
}

/// Perspective Jacobian `∂(u, v)/∂(x, y, z)` at camera-space point `p`
/// (`u = fx·x/z`, `v = fy·y/z`). `None` for `z <= 0` (behind the camera).
pub fn perspective_jacobian<T: Scalar>(p: [T; 3], fx: T, fy: T) -> Option<[[T; 3]; 2]> {
    let z = p[2];
    if !(z > T::zero()) || !z.is_finite() {
        return None;
    }
    let inv_z = T::one() / z;
    let inv_z2 = inv_z * inv_z;
    Some([
        [fx * inv_z, T::zero(), -fx * p[0] * inv_z2],
        [T::zero(), fy * inv_z, -fy * p[1] * inv_z2],
    ])
}

/// EWA splatting projection `Σ' = (J W) Σ (J W)ᵀ + low_pass · I`.
///
/// `w` is the world→camera rotation and `j` the perspective Jacobian at the
/// splat centre. `low_pass` is the screen-space anti-aliasing term (3DGS uses
/// `0.3`); pass zero for the exact projected covariance.
pub fn project_cov_ewa<T: Scalar>(
    cov: &Sym3<T>,
    w: &Mat<T, 3>,
    j: &[[T; 3]; 2],
    low_pass: T,
) -> Sym2<T> {
    let mut t = [[T::zero(); 3]; 2];
    for a in 0..2 {
        for c in 0..3 {
            let mut s = T::zero();
            for k in 0..3 {
                s = s + j[a][k] * w[k][c];
            }
            t[a][c] = s;
        }
    }
    let quad = |a: usize, b: usize| {
        let mut s = T::zero();
        for i in 0..3 {
            for k in 0..3 {
                s = s + t[a][i] * cov.get(i, k) * t[b][k];
            }
        }
        s
    };
    Sym2 {
        xx: quad(0, 0) + low_pass,
        xy: quad(0, 1),
        yy: quad(1, 1) + low_pass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sym::{spd_inverse, transpose};

    const Q: [f64; 4] = [0.9, 0.1, -0.3, 0.2];
    const P: [f64; 4] = [0.5, 0.5, -0.5, 0.5];

    #[test]
    fn rot3_is_orthogonal() {
        let r = quat_to_rot3(Q).unwrap();
        let rtr = matmul(&transpose(&r), &r);
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - e).abs() < 1e-12);
            }
        }
        assert!(quat_to_rot3([0.0_f64; 4]).is_none());
        assert!(quat_to_rot3([f64::NAN, 0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn rot4_is_orthogonal() {
        let r = rot4_from_isoclinic(Q, P).unwrap();
        let rtr = matmul(&transpose(&r), &r);
        for i in 0..4 {
            for j in 0..4 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - e).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn built_covariances_are_spd() {
        let c3 = cov3_from_rot_scale(Q, [0.5, 1.0, 2.0]).unwrap();
        assert!(c3.is_spd());
        let c4 = cov4_from_rot_scale(Q, P, [0.5, 1.0, 2.0, 0.25]).unwrap();
        assert!(c4.is_spd());
    }

    #[test]
    fn conditioning_matches_precision_matrix_formulation() {
        let c4 = cov4_from_rot_scale(Q, P, [0.5, 1.0, 2.0, 0.25]).unwrap();
        let mean = [1.0, -2.0, 0.5, 3.0];
        let t = 3.4;
        let g = condition_on_time(mean, &c4, t).unwrap();
        // Conditional covariance = inverse of the top-left block of Σ⁻¹.
        let prec = spd_inverse(&c4.to_full()).unwrap();
        let block = Sym3::from_full(&[
            [prec[0][0], prec[0][1], prec[0][2]],
            [prec[1][0], prec[1][1], prec[1][2]],
            [prec[2][0], prec[2][1], prec[2][2]],
        ]);
        let expect = block.spd_inverse().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((g.cov.get(i, j) - expect.get(i, j)).abs() < 1e-10);
            }
        }
        // Conditional mean: μ − Λxx⁻¹ Λxt (t − μt).
        let lxt = [prec[0][3], prec[1][3], prec[2][3]];
        let adj = expect.mul_vec(&lxt);
        for i in 0..3 {
            assert!((g.mean[i] - (mean[i] - adj[i] * (t - mean[3]))).abs() < 1e-10);
        }
        assert!(g.cov.is_spd());
        assert!(g.weight > 0.0 && g.weight <= 1.0);
    }

    #[test]
    fn conditioning_rejects_degenerate_time_variance() {
        let c = Sym4::new([1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(condition_on_time([0.0; 4], &c, 0.0).is_none());
    }

    #[test]
    fn weight_peaks_at_mean_time() {
        let c =
            cov4_from_rot_scale([1.0_f64, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0; 4]).unwrap();
        let at = condition_on_time([0.0, 0.0, 0.0, 2.0], &c, 2.0).unwrap();
        let off = condition_on_time([0.0, 0.0, 0.0, 2.0], &c, 3.0).unwrap();
        assert!((at.weight - 1.0).abs() < 1e-15);
        assert!((off.weight - (-0.5_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn ewa_identity_view_matches_hand_computation() {
        // Isotropic unit covariance, point at (0,0,2), f=1, identity view.
        let cov = Sym3::<f64>::identity();
        let j = perspective_jacobian([0.0, 0.0, 2.0], 1.0, 1.0).unwrap();
        let s = project_cov_ewa(&cov, &identity::<f64, 3>(), &j, 0.0);
        assert!((s.xx - 0.25).abs() < 1e-15);
        assert!((s.yy - 0.25).abs() < 1e-15);
        assert!(s.xy.abs() < 1e-15);
        let conic = s.inverse().unwrap();
        assert!((conic.xx - 4.0).abs() < 1e-12);
        let (l1, l2) = s.eigenvalues();
        assert!((l1 - 0.25).abs() < 1e-12 && (l2 - 0.25).abs() < 1e-12);
        assert!(perspective_jacobian([0.0, 0.0, -1.0], 1.0, 1.0).is_none());
    }
}
