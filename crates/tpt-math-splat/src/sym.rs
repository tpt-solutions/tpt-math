//! Symmetric matrices: packed storage, factorisations and a 3×3 eigensolver.

use tpt_math_numeric::Scalar;

/// Row-major square matrix.
pub type Mat<T, const N: usize> = [[T; N]; N];

/// Index of `(i, j)` (with `i <= j`) in row-major packed upper storage.
const fn packed_index(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    i * n - i * i.saturating_sub(1) / 2 + (j - i)
}

/// The identity matrix.
pub fn identity<T: Scalar, const N: usize>() -> Mat<T, N> {
    let mut m = [[T::zero(); N]; N];
    for i in 0..N {
        m[i][i] = T::one();
    }
    m
}

/// Matrix product `a * b`.
pub fn matmul<T: Scalar, const N: usize>(a: &Mat<T, N>, b: &Mat<T, N>) -> Mat<T, N> {
    let mut out = [[T::zero(); N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut s = T::zero();
            for k in 0..N {
                s = s + a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Transpose.
pub fn transpose<T: Scalar, const N: usize>(a: &Mat<T, N>) -> Mat<T, N> {
    let mut out = *a;
    for i in 0..N {
        for j in 0..N {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// Cholesky factorisation `A = L Lᵀ` of a symmetric positive-definite matrix.
///
/// Only the lower triangle of `a` is read. Returns the lower-triangular `L`,
/// or `None` if a pivot is not strictly positive (matrix not numerically
/// positive definite) or any value is non-finite.
pub fn cholesky<T: Scalar, const N: usize>(a: &Mat<T, N>) -> Option<Mat<T, N>> {
    let mut l = [[T::zero(); N]; N];
    for j in 0..N {
        let mut d = a[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        // `!(d > 0)` also rejects NaN.
        if !(d > T::zero()) || !d.is_finite() {
            return None;
        }
        let ljj = d.sqrt();
        l[j][j] = ljj;
        for i in (j + 1)..N {
            let mut s = a[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            l[i][j] = s / ljj;
        }
    }
    Some(l)
}

/// Square-root-free `A = L D Lᵀ` factorisation (unit lower-triangular `L`,
/// diagonal `D`) of a symmetric positive-definite matrix.
///
/// Returns `None` if any pivot is not strictly positive.
pub fn ldl<T: Scalar, const N: usize>(a: &Mat<T, N>) -> Option<(Mat<T, N>, [T; N])> {
    let mut l = identity::<T, N>();
    let mut d = [T::zero(); N];
    for j in 0..N {
        let mut dj = a[j][j];
        for k in 0..j {
            dj = dj - l[j][k] * l[j][k] * d[k];
        }
        if !(dj > T::zero()) || !dj.is_finite() {
            return None;
        }
        d[j] = dj;
        for i in (j + 1)..N {
            let mut s = a[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k] * d[k];
            }
            l[i][j] = s / dj;
        }
    }
    Some((l, d))
}

/// Solves `L Lᵀ x = b` given the Cholesky factor `l`.
pub fn cholesky_solve<T: Scalar, const N: usize>(l: &Mat<T, N>, b: &[T; N]) -> [T; N] {
    let mut y = *b;
    for i in 0..N {
        let mut s = y[i];
        for k in 0..i {
            s = s - l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    for i in (0..N).rev() {
        let mut s = y[i];
        for k in (i + 1)..N {
            s = s - l[k][i] * y[k];
        }
        y[i] = s / l[i][i];
    }
    y
}

/// Inverse of a symmetric positive-definite matrix (via Cholesky), or `None`
/// if the matrix is not numerically positive definite.
pub fn spd_inverse<T: Scalar, const N: usize>(a: &Mat<T, N>) -> Option<Mat<T, N>> {
    let l = cholesky(a)?;
    let mut inv = [[T::zero(); N]; N];
    for j in 0..N {
        let mut e = [T::zero(); N];
        e[j] = T::one();
        let col = cholesky_solve(&l, &e);
        for i in 0..N {
            inv[i][j] = col[i];
        }
    }
    Some(inv)
}

/// Determinant of a symmetric positive-definite matrix (product of squared
/// Cholesky diagonal), or `None` if not numerically positive definite.
pub fn spd_det<T: Scalar, const N: usize>(a: &Mat<T, N>) -> Option<T> {
    let l = cholesky(a)?;
    let mut det = T::one();
    for i in 0..N {
        det = det * l[i][i] * l[i][i];
    }
    Some(det)
}

macro_rules! sym_common {
    ($name:ident, $n:literal, $len:literal) => {
        impl<T: Scalar> $name<T> {
            /// Builds from a full matrix, symmetrising as `(a_ij + a_ji) / 2`.
            pub fn from_full(a: &Mat<T, $n>) -> Self {
                let half = T::one() / (T::one() + T::one());
                let mut m = [T::zero(); $len];
                for i in 0..$n {
                    for j in i..$n {
                        m[packed_index($n, i, j)] = if i == j {
                            a[i][i]
                        } else {
                            (a[i][j] + a[j][i]) * half
                        };
                    }
                }
                Self { m }
            }

            /// Expands to a full row-major matrix.
            pub fn to_full(&self) -> Mat<T, $n> {
                let mut a = [[T::zero(); $n]; $n];
                for i in 0..$n {
                    for j in 0..$n {
                        a[i][j] = self.m[packed_index($n, i, j)];
                    }
                }
                a
            }

            /// The identity matrix.
            pub fn identity() -> Self {
                Self::from_full(&identity::<T, $n>())
            }

            /// Element `(i, j)`. Panics if an index is out of range.
            pub fn get(&self, i: usize, j: usize) -> T {
                assert!(i < $n && j < $n, "index out of range");
                self.m[packed_index($n, i, j)]
            }

            /// Trace.
            pub fn trace(&self) -> T {
                let mut t = T::zero();
                for i in 0..$n {
                    t = t + self.get(i, i);
                }
                t
            }

            /// Matrix–vector product.
            pub fn mul_vec(&self, v: &[T; $n]) -> [T; $n] {
                let mut out = [T::zero(); $n];
                for i in 0..$n {
                    let mut s = T::zero();
                    for j in 0..$n {
                        s = s + self.get(i, j) * v[j];
                    }
                    out[i] = s;
                }
                out
            }

            /// Quadratic form `vᵀ Σ v` (the squared Mahalanobis length when
            /// `self` is a *precision* matrix).
            pub fn quad_form(&self, v: &[T; $n]) -> T {
                let sv = self.mul_vec(v);
                let mut s = T::zero();
                for i in 0..$n {
                    s = s + v[i] * sv[i];
                }
                s
            }

            /// Cholesky factor `L` (`Σ = L Lᵀ`), or `None` if not numerically
            /// positive definite.
            pub fn cholesky(&self) -> Option<Mat<T, $n>> {
                cholesky(&self.to_full())
            }

            /// `LDLᵀ` factorisation; see [`ldl`].
            pub fn ldl(&self) -> Option<(Mat<T, $n>, [T; $n])> {
                ldl(&self.to_full())
            }

            /// `true` if the matrix is numerically positive definite.
            pub fn is_spd(&self) -> bool {
                self.cholesky().is_some()
            }

            /// Inverse, assuming positive definiteness (`None` otherwise).
            pub fn spd_inverse(&self) -> Option<Self> {
                spd_inverse(&self.to_full()).map(|m| Self::from_full(&m))
            }

            /// Determinant, assuming positive definiteness (`None` otherwise).
            pub fn spd_det(&self) -> Option<T> {
                spd_det(&self.to_full())
            }
        }
    };
}

/// Packed 3×3 symmetric matrix, storage order `xx, xy, xz, yy, yz, zz`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sym3<T> {
    /// Packed upper triangle, row-major.
    pub m: [T; 6],
}

/// Packed 4×4 symmetric matrix, storage order
/// `00, 01, 02, 03, 11, 12, 13, 22, 23, 33`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sym4<T> {
    /// Packed upper triangle, row-major.
    pub m: [T; 10],
}

sym_common!(Sym3, 3, 6);
sym_common!(Sym4, 4, 10);

impl<T: Scalar> Sym3<T> {
    /// Builds from the six unique entries.
    pub fn new(xx: T, xy: T, xz: T, yy: T, yz: T, zz: T) -> Self {
        Self {
            m: [xx, xy, xz, yy, yz, zz],
        }
    }

    /// Closed-form determinant (valid for any symmetric matrix).
    pub fn det(&self) -> T {
        let [xx, xy, xz, yy, yz, zz] = self.m;
        xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
    }

    /// Closed-form adjugate inverse; `None` if the determinant is zero or
    /// non-finite. Prefer [`Sym3::spd_inverse`] when positive definiteness is
    /// expected and a failure signal is wanted.
    pub fn inverse(&self) -> Option<Self> {
        let [xx, xy, xz, yy, yz, zz] = self.m;
        let det = self.det();
        if det == T::zero() || !det.is_finite() {
            return None;
        }
        let inv = T::one() / det;
        Some(Self::new(
            (yy * zz - yz * yz) * inv,
            (xz * yz - xy * zz) * inv,
            (xy * yz - xz * yy) * inv,
            (xx * zz - xz * xz) * inv,
            (xy * xz - xx * yz) * inv,
            (xx * yy - xy * xy) * inv,
        ))
    }

    /// Eigendecomposition by cyclic Jacobi rotations.
    ///
    /// Eigenvalues are ascending; the eigenvector of `values[k]` is the
    /// *column* `k` of `vectors` (`vectors[row][k]`). Non-finite input yields
    /// non-finite output rather than looping forever.
    pub fn eig(&self) -> Eig3<T> {
        let mut a = self.to_full();
        let mut v = identity::<T, 3>();
        let two = T::one() + T::one();
        let scale = {
            let mut s = T::zero();
            for r in &a {
                for x in r {
                    s = s + *x * *x;
                }
            }
            s
        };
        let tol = T::epsilon() * T::epsilon() * scale;
        for _ in 0..50 {
            let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
            if off <= tol || off == T::zero() {
                break;
            }
            for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
                if a[p][q] == T::zero() {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (two * a[p][q]);
                let sign = if theta < T::zero() {
                    -T::one()
                } else {
                    T::one()
                };
                let t = sign / (theta.abs() + (theta * theta + T::one()).sqrt());
                let c = T::one() / (t * t + T::one()).sqrt();
                let s = t * c;
                for k in 0..3 {
                    let (kp, kq) = (a[k][p], a[k][q]);
                    a[k][p] = c * kp - s * kq;
                    a[k][q] = s * kp + c * kq;
                }
                for k in 0..3 {
                    let (pk, qk) = (a[p][k], a[q][k]);
                    a[p][k] = c * pk - s * qk;
                    a[q][k] = s * pk + c * qk;
                }
                for k in 0..3 {
                    let (kp, kq) = (v[k][p], v[k][q]);
                    v[k][p] = c * kp - s * kq;
                    v[k][q] = s * kp + c * kq;
                }
            }
        }
        // Sort ascending (3-element insertion sort, NaN-tolerant).
        let mut idx = [0usize, 1, 2];
        for i in 1..3 {
            let mut j = i;
            while j > 0 && a[idx[j]][idx[j]] < a[idx[j - 1]][idx[j - 1]] {
                idx.swap(j, j - 1);
                j -= 1;
            }
        }
        let values = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
        let mut vectors = [[T::zero(); 3]; 3];
        for (col, &src) in idx.iter().enumerate() {
            for row in 0..3 {
                vectors[row][col] = v[row][src];
            }
        }
        Eig3 { values, vectors }
    }
}

/// Result of [`Sym3::eig`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eig3<T> {
    /// Eigenvalues, ascending.
    pub values: [T; 3],
    /// Orthonormal eigenvectors as columns.
    pub vectors: Mat<T, 3>,
}

impl<T: Scalar> Sym4<T> {
    /// Builds from the ten unique entries in packed order.
    pub fn new(m: [T; 10]) -> Self {
        Self { m }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_index_layout() {
        let mut seen3 = [false; 6];
        for i in 0..3 {
            for j in i..3 {
                seen3[packed_index(3, i, j)] = true;
            }
        }
        assert!(seen3.iter().all(|&b| b));
        let order4: [(usize, usize); 10] = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3),
        ];
        for (k, (i, j)) in order4.iter().enumerate() {
            assert_eq!(packed_index(4, *i, *j), k);
            assert_eq!(packed_index(4, *j, *i), k);
        }
        let order3: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        for (k, (i, j)) in order3.iter().enumerate() {
            assert_eq!(packed_index(3, *i, *j), k);
        }
    }

    #[test]
    fn cholesky_reconstructs() {
        let s = Sym3::new(4.0_f64, 1.0, 0.5, 3.0, 0.2, 2.0);
        let l = s.cholesky().unwrap();
        let rec = matmul(&l, &transpose(&l));
        let full = s.to_full();
        for i in 0..3 {
            for j in 0..3 {
                assert!((rec[i][j] - full[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn indefinite_is_rejected() {
        let s = Sym3::new(1.0_f64, 2.0, 0.0, 1.0, 0.0, 1.0);
        assert!(s.cholesky().is_none());
        assert!(!s.is_spd());
        assert!(s.ldl().is_none());
        let nan = Sym3::new(f64::NAN, 0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(nan.cholesky().is_none());
    }

    #[test]
    fn ldl_reconstructs() {
        let s = Sym4::new([5.0_f64, 1.0, 0.0, 0.3, 4.0, 0.2, 0.1, 3.0, 0.4, 2.0]);
        let (l, d) = s.ldl().unwrap();
        let full = s.to_full();
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += l[i][k] * d[k] * l[j][k];
                }
                assert!((acc - full[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn inverse_and_det() {
        let s = Sym3::new(4.0_f64, 1.0, 0.5, 3.0, 0.2, 2.0);
        let inv = s.inverse().unwrap();
        let spd = s.spd_inverse().unwrap();
        let prod = matmul(&s.to_full(), &inv.to_full());
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i][j] - e).abs() < 1e-12);
                assert!((inv.get(i, j) - spd.get(i, j)).abs() < 1e-12);
            }
        }
        assert!((s.det() - s.spd_det().unwrap()).abs() < 1e-12);
        let singular = Sym3::new(1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(singular.inverse().is_none());
    }

    #[test]
    fn eig_diagonalises() {
        let s = Sym3::new(4.0_f64, 1.0, 0.5, 3.0, 0.2, 2.0);
        let e = s.eig();
        assert!(e.values[0] <= e.values[1] && e.values[1] <= e.values[2]);
        let a = s.to_full();
        for k in 0..3 {
            let v = [e.vectors[0][k], e.vectors[1][k], e.vectors[2][k]];
            let av = s.mul_vec(&v);
            for i in 0..3 {
                assert!((av[i] - e.values[k] * v[i]).abs() < 1e-12);
            }
        }
        let trace: f64 = a[0][0] + a[1][1] + a[2][2];
        assert!((e.values.iter().sum::<f64>() - trace).abs() < 1e-12);
    }

    #[test]
    fn eig_of_diagonal_and_nan() {
        let s = Sym3::new(3.0_f64, 0.0, 0.0, 1.0, 0.0, 2.0);
        assert_eq!(s.eig().values, [1.0, 2.0, 3.0]);
        let nan = Sym3::new(f64::NAN, 1.0, 0.0, 1.0, 0.0, 1.0);
        let _ = nan.eig(); // must terminate
    }
}
