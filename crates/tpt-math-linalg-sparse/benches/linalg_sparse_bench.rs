use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_linalg_dense::DVector;
use tpt_math_linalg_sparse::{conjugate_gradient, ic0_preconditioned_cg, CooMatrix, CsrMatrix};

/// 2D five-point Laplacian on an `m × m` grid: SPD, sparse, and a standard test.
fn laplacian_2d(m: usize) -> CsrMatrix<f64> {
    let n = m * m;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for r in 0..m {
        for c in 0..m {
            let i = r * m + c;
            coo.push(i, i, 4.0);
            if c + 1 < m {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
            if r + 1 < m {
                coo.push(i, i + m, -1.0);
                coo.push(i + m, i, -1.0);
            }
        }
    }
    coo.to_csr()
}

fn bench_solvers(c: &mut Criterion) {
    let a = laplacian_2d(40);
    let n = a.nrows();
    let b = DVector::from_fn(n, |i| if i % 3 == 0 { 1.0 } else { -0.5 });

    c.bench_function("cg laplacian 40x40", |bch| {
        bch.iter(|| {
            let x = conjugate_gradient(black_box(&a), black_box(&b), None, 1e-8, 2000);
            black_box(x.is_ok())
        })
    });
    c.bench_function("ic0-cg laplacian 40x40", |bch| {
        bch.iter(|| {
            let x = ic0_preconditioned_cg(black_box(&a), black_box(&b), None, 1e-8, 2000);
            black_box(x.is_ok())
        })
    });
}

criterion_group!(benches, bench_solvers);
criterion_main!(benches);
