//! Property tests for covariance construction, conditioning, SH and curves.

use proptest::prelude::*;
use tpt_math_splat::gaussian::{condition_on_time, cov3_from_rot_scale, cov4_from_rot_scale};
use tpt_math_splat::sfc::{
    hilbert_decode, hilbert_encode, morton3_decode, morton3_encode, morton4_decode, morton4_encode,
};
use tpt_math_splat::sh::eval_basis;
use tpt_math_splat::sym::{cholesky, matmul, transpose};

fn quat() -> impl Strategy<Value = [f64; 4]> {
    prop::array::uniform4(-1.0f64..1.0).prop_filter("non-degenerate", |q| {
        q.iter().map(|x| x * x).sum::<f64>() > 1e-3
    })
}

fn scale3() -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3(0.05f64..3.0)
}

fn scale4() -> impl Strategy<Value = [f64; 4]> {
    prop::array::uniform4(0.05f64..3.0)
}

proptest! {
    #[test]
    fn cov3_is_spd_and_cholesky_reconstructs(q in quat(), s in scale3()) {
        let c = cov3_from_rot_scale(q, s).unwrap();
        let l = c.cholesky().expect("SPD by construction");
        let rec = matmul(&l, &transpose(&l));
        let full = c.to_full();
        for i in 0..3 { for j in 0..3 {
            prop_assert!((rec[i][j] - full[i][j]).abs() < 1e-9);
        }}
    }

    #[test]
    fn cov4_is_spd(ql in quat(), qr in quat(), s in scale4()) {
        let c = cov4_from_rot_scale(ql, qr, s).unwrap();
        prop_assert!(cholesky(&c.to_full()).is_some());
    }

    #[test]
    fn conditioned_cov_stays_spd_and_weight_in_unit_interval(
        ql in quat(), qr in quat(), s in scale4(),
        mean in prop::array::uniform4(-5.0f64..5.0), t in -5.0f64..5.0,
    ) {
        let c = cov4_from_rot_scale(ql, qr, s).unwrap();
        let g = condition_on_time(mean, &c, t).unwrap();
        prop_assert!(g.weight >= 0.0 && g.weight <= 1.0);
        prop_assert!(g.cov.is_spd());
        // Conditioning never increases variance along any axis.
        for i in 0..3 {
            prop_assert!(g.cov.get(i, i) <= c.get(i, i) + 1e-12);
        }
    }

    #[test]
    fn sh_basis_is_scale_invariant(
        d in prop::array::uniform3(-1.0f64..1.0), k in 0.1f64..10.0,
    ) {
        prop_assume!(d.iter().map(|x| x * x).sum::<f64>() > 1e-3);
        let a = eval_basis(3, d).unwrap();
        let b = eval_basis(3, [d[0] * k, d[1] * k, d[2] * k]).unwrap();
        for i in 0..16 { prop_assert!((a[i] - b[i]).abs() < 1e-12); }
    }

    #[test]
    fn morton_roundtrip(a in 0u32..(1 << 21), b in 0u32..(1 << 21), c in 0u32..(1 << 21), d in 0u32..(1 << 16)) {
        prop_assert_eq!(morton3_decode(morton3_encode([a, b, c])), [a, b, c]);
        let w = [a & 0xffff, b & 0xffff, c & 0xffff, d];
        prop_assert_eq!(morton4_decode(morton4_encode(w)), w);
    }

    #[test]
    fn morton_is_monotone_per_axis(a in 0u32..(1 << 21) - 1, b in 0u32..(1 << 21), c in 0u32..(1 << 21)) {
        prop_assert!(morton3_encode([a, b, c]) < morton3_encode([a + 1, b, c]));
    }

    #[test]
    fn hilbert_roundtrip_3d_and_4d(a in 0u32..(1 << 21), b in 0u32..(1 << 21), c in 0u32..(1 << 21), d in 0u32..(1 << 16)) {
        let k = hilbert_encode([a, b, c], 21).unwrap();
        prop_assert_eq!(hilbert_decode::<3>(k, 21), Some([a, b, c]));
        let w = [a & 0xffff, b & 0xffff, c & 0xffff, d];
        let k = hilbert_encode(w, 16).unwrap();
        prop_assert_eq!(hilbert_decode::<4>(k, 16), Some(w));
    }
}
