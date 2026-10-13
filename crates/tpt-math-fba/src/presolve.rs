//! Geometric-mean row/column scaling.
//!
//! Genome-scale stoichiometric matrices mix coefficients from 1e-3 (biomass
//! precursors) to 1e2 (ATP maintenance, polymers); scaling keeps the simplex
//! tolerances meaningful.

use tpt_math_linalg_sparse::CooMatrix;

use crate::lp::SparseLp;

/// A scaled copy of an LP plus the factors needed to unscale a solution.
pub(crate) struct Scaled {
    pub lp: SparseLp,
    /// Row scale `R`: `A' = R A C`, `b' = R b`.
    pub row: Vec<f64>,
    /// Column scale `C`: `x = C x'`, `c' = C c`, bounds divided by `C`.
    pub col: Vec<f64>,
}

fn pow2(x: f64) -> f64 {
    2f64.powi(x.log2().round() as i32)
}

pub(crate) fn scale(lp: &SparseLp, passes: usize) -> Scaled {
    let (m, n) = (lp.a.nrows(), lp.a.ncols());
    let mut row = vec![1.0; m];
    let mut col = vec![1.0; n];
    let entries: Vec<(usize, usize, f64)> = lp.a.iter().map(|(r, c, &v)| (r, c, v)).collect();
    for _ in 0..passes {
        // Row pass: R_i = 1 / sqrt(min * max) of |R_i a_ij C_j|.
        let (mut lo, mut hi) = (vec![f64::INFINITY; m], vec![0.0f64; m]);
        for &(r, c, v) in &entries {
            let s = (v * row[r] * col[c]).abs();
            if s > 0.0 {
                lo[r] = lo[r].min(s);
                hi[r] = hi[r].max(s);
            }
        }
        for i in 0..m {
            if hi[i] > 0.0 {
                row[i] *= 1.0 / (lo[i] * hi[i]).sqrt();
            }
        }
        let (mut lo, mut hi) = (vec![f64::INFINITY; n], vec![0.0f64; n]);
        for &(r, c, v) in &entries {
            let s = (v * row[r] * col[c]).abs();
            if s > 0.0 {
                lo[c] = lo[c].min(s);
                hi[c] = hi[c].max(s);
            }
        }
        for j in 0..n {
            if hi[j] > 0.0 {
                col[j] *= 1.0 / (lo[j] * hi[j]).sqrt();
            }
        }
    }
    // Round to powers of two so scaling introduces no rounding error.
    row.iter_mut().for_each(|v| *v = pow2(*v));
    col.iter_mut().for_each(|v| *v = pow2(*v));

    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for &(r, c, v) in &entries {
        rr.push(r);
        cc.push(c);
        vv.push(v * row[r] * col[c]);
    }
    let a = CooMatrix::from_triplets(m, n, rr, cc, vv).to_csc();
    let scaled = SparseLp {
        a,
        b: lp.b.iter().zip(&row).map(|(b, r)| b * r).collect(),
        c: lp.c.iter().zip(&col).map(|(c, s)| c * s).collect(),
        lb: lp.lb.iter().zip(&col).map(|(l, s)| l / s).collect(),
        ub: lp.ub.iter().zip(&col).map(|(u, s)| u / s).collect(),
    };
    Scaled {
        lp: scaled,
        row,
        col,
    }
}
