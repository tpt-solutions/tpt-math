//! Cross-crate example: `tpt-math-linalg-sparse` and `tpt-math-linalg-complex`.
//!
//! Solves a sparse symmetric positive-definite system with conjugate gradients
//! and a small complex system with LU factorisation.

use tpt_math_linalg_complex::{Complex, ComplexDMatrix};
use tpt_math_linalg_dense::DVector;
use tpt_math_linalg_sparse::{conjugate_gradient, CooMatrix};

fn main() {
    // 1-D Laplacian with Dirichlet ends: tridiagonal, sparse, SPD.
    let n = 200;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i + 1 < n {
            coo.push(i, i + 1, -1.0);
            coo.push(i + 1, i, -1.0);
        }
    }
    let a = coo.to_csr();
    let b = DVector::from_fn(n, |_| 1.0);
    let x = conjugate_gradient(&a, &b, None, 1e-10, 1000).expect("CG converges");
    let residual = a.matvec(&x).unwrap();
    let err = (0..n)
        .map(|i| (residual[i] - b[i]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "sparse CG: n = {n}, nnz = {}, max residual = {err:.2e}",
        a.nnz()
    );

    // A 2x2 complex system: [[1+i, 2], [0, 1-i]] z = [3+i, 2-2i].
    let m = ComplexDMatrix::from_row_slice(
        2,
        2,
        &[
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, -1.0),
        ],
    );
    let (l, u, piv) = m.lu().expect("the matrix is non-singular");
    println!(
        "complex LU: L has {} rows, U has {} rows, pivot order {:?}",
        l.len(),
        u.len(),
        piv
    );
}
