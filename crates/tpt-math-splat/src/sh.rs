//! Real spherical harmonics (degrees 0–3) for view-dependent colour.
//!
//! The basis and sign convention follow the reference 3D Gaussian Splatting
//! implementation, so trained coefficients are interchangeable. Coefficients
//! are ordered by `(l, m)`: index `l² + l + m`.

use tpt_math_numeric::Scalar;

/// Highest supported degree.
pub const MAX_DEGREE: usize = 3;
/// Number of basis functions for a degree: `(degree + 1)²`.
pub const fn coeff_count(degree: usize) -> usize {
    (degree + 1) * (degree + 1)
}

const C0: f64 = 0.282_094_791_773_878_14;
const C1: f64 = 0.488_602_511_902_919_9;
const C2: [f64; 5] = [
    1.092_548_430_592_079_2,
    -1.092_548_430_592_079_2,
    0.315_391_565_252_520_05,
    -1.092_548_430_592_079_2,
    0.546_274_215_296_039_6,
];
const C3: [f64; 7] = [
    -0.590_043_589_926_643_5,
    2.890_611_442_640_554,
    -0.457_045_799_464_465_8,
    0.373_176_332_590_115_4,
    -0.457_045_799_464_465_8,
    1.445_305_721_320_277,
    -0.590_043_589_926_643_5,
];

fn c<T: Scalar>(x: f64) -> T {
    <T as tpt_math_numeric::NumCast>::from(x).unwrap_or_else(T::zero)
}

/// Evaluates the SH basis up to `degree` at direction `dir`.
///
/// `dir` is normalised internally. Returns `None` if `degree > 3`, or `dir` is
/// zero or non-finite. Entries past `coeff_count(degree)` are zero.
pub fn eval_basis<T: Scalar>(degree: usize, dir: [T; 3]) -> Option<[T; 16]> {
    if degree > MAX_DEGREE {
        return None;
    }
    let n2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    if !(n2 > T::zero()) || !n2.is_finite() {
        return None;
    }
    let inv = T::one() / n2.sqrt();
    let (x, y, z) = (dir[0] * inv, dir[1] * inv, dir[2] * inv);
    let mut b = [T::zero(); 16];
    b[0] = c(C0);
    if degree >= 1 {
        let c1 = c::<T>(C1);
        b[1] = -c1 * y;
        b[2] = c1 * z;
        b[3] = -c1 * x;
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, yz, xz) = (x * y, y * z, x * z);
        let two = T::one() + T::one();
        b[4] = c::<T>(C2[0]) * xy;
        b[5] = c::<T>(C2[1]) * yz;
        b[6] = c::<T>(C2[2]) * (two * zz - xx - yy);
        b[7] = c::<T>(C2[3]) * xz;
        b[8] = c::<T>(C2[4]) * (xx - yy);
        if degree >= 3 {
            let three = two + T::one();
            let four = two + two;
            b[9] = c::<T>(C3[0]) * y * (three * xx - yy);
            b[10] = c::<T>(C3[1]) * xy * z;
            b[11] = c::<T>(C3[2]) * y * (four * zz - xx - yy);
            b[12] = c::<T>(C3[3]) * z * (two * zz - three * xx - three * yy);
            b[13] = c::<T>(C3[4]) * x * (four * zz - xx - yy);
            b[14] = c::<T>(C3[5]) * z * (xx - yy);
            b[15] = c::<T>(C3[6]) * x * (xx - three * yy);
        }
    }
    Some(b)
}

/// Raw SH radiance `Σ_i basis_i(dir) · coeffs[i]` per colour channel (no
/// offset, no clamp).
///
/// Returns `None` if `coeffs` has fewer than `coeff_count(degree)` entries or
/// [`eval_basis`] fails.
pub fn eval_radiance<T: Scalar>(degree: usize, coeffs: &[[T; 3]], dir: [T; 3]) -> Option<[T; 3]> {
    let n = coeff_count(degree.min(MAX_DEGREE + 1));
    if degree > MAX_DEGREE || coeffs.len() < n {
        return None;
    }
    let basis = eval_basis(degree, dir)?;
    let mut out = [T::zero(); 3];
    for i in 0..n {
        for ch in 0..3 {
            out[ch] = out[ch] + basis[i] * coeffs[i][ch];
        }
    }
    Some(out)
}

/// 3DGS display colour: [`eval_radiance`] `+ 0.5`, clamped below at zero.
pub fn eval_color<T: Scalar>(degree: usize, coeffs: &[[T; 3]], dir: [T; 3]) -> Option<[T; 3]> {
    let half = T::one() / (T::one() + T::one());
    let r = eval_radiance(degree, coeffs, dir)?;
    Some([
        (r[0] + half).max(T::zero()),
        (r[1] + half).max(T::zero()),
        (r[2] + half).max(T::zero()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Midpoint-in-z × uniform-in-φ quadrature of `∫ Y_i Y_j dΩ`.
    fn gram(degree: usize) -> [[f64; 16]; 16] {
        let n = coeff_count(degree);
        let (nz, nphi) = (400usize, 64usize);
        let mut g = [[0.0; 16]; 16];
        for izi in 0..nz {
            let z = -1.0 + 2.0 * (izi as f64 + 0.5) / nz as f64;
            let r = (1.0 - z * z).sqrt();
            for ip in 0..nphi {
                let phi = 2.0 * core::f64::consts::PI * (ip as f64 + 0.5) / nphi as f64;
                let b = eval_basis(degree, [r * phi.cos(), r * phi.sin(), z]).unwrap();
                let w = (2.0 / nz as f64) * (2.0 * core::f64::consts::PI / nphi as f64);
                for i in 0..n {
                    for j in 0..n {
                        g[i][j] += w * b[i] * b[j];
                    }
                }
            }
        }
        g
    }

    #[test]
    fn basis_is_orthonormal() {
        let g = gram(3);
        for i in 0..16 {
            for j in 0..16 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((g[i][j] - e).abs() < 1e-3, "({i},{j}) = {}", g[i][j]);
            }
        }
    }

    #[test]
    fn degree_zero_is_constant() {
        let a = eval_basis(0, [1.0_f64, 0.0, 0.0]).unwrap();
        let b = eval_basis(0, [0.3, -0.2, 0.9]).unwrap();
        assert_eq!(a[0], b[0]);
        assert!((a[0] - 0.282_094_791_773_878_14).abs() < 1e-15);
        assert!(a[1..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn known_values_on_axes() {
        let bz = eval_basis(3, [0.0_f64, 0.0, 1.0]).unwrap();
        assert!((bz[2] - C1).abs() < 1e-15); // Y_1^0 ∝ +z
        assert!((bz[6] - 2.0 * C2[2]).abs() < 1e-15);
        assert!((bz[12] - 2.0 * C3[3]).abs() < 1e-15);
        let bx = eval_basis(1, [2.0_f64, 0.0, 0.0]).unwrap(); // normalised internally
        assert!((bx[3] + C1).abs() < 1e-15);
    }

    #[test]
    fn invalid_inputs() {
        assert!(eval_basis(4, [0.0_f64, 0.0, 1.0]).is_none());
        assert!(eval_basis(1, [0.0_f64; 3]).is_none());
        assert!(eval_basis(1, [f64::NAN, 0.0, 1.0]).is_none());
        let short = [[0.0_f64; 3]; 3];
        assert!(eval_radiance(2, &short, [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn color_offset_and_clamp() {
        let mut coeffs = [[0.0_f64; 3]; 1];
        let dc = 0.25 / C0;
        coeffs[0] = [dc, 0.0, -10.0];
        let col = eval_color(0, &coeffs, [0.0, 0.0, 1.0]).unwrap();
        assert!((col[0] - 0.75).abs() < 1e-12);
        assert!((col[1] - 0.5).abs() < 1e-12);
        assert_eq!(col[2], 0.0);
    }

    #[test]
    fn f32_matches_f64() {
        let d64 = eval_basis(3, [0.3_f64, -0.5, 0.8]).unwrap();
        let d32 = eval_basis(3, [0.3_f32, -0.5, 0.8]).unwrap();
        for i in 0..16 {
            assert!((d64[i] - f64::from(d32[i])).abs() < 1e-5);
        }
    }
}
