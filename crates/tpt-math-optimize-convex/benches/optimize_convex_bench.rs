use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_optimize_convex::solve_qp;

fn bench_qp(c: &mut Criterion) {
    let p = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
    let q = DVector::from_vec(vec![0.0, 0.0]);
    let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
    let b_eq = DVector::from_vec(vec![1.0]);

    c.bench_function("qp 2x2", |b| {
        b.iter(|| {
            let x = solve_qp(
                black_box(&p),
                black_box(&q),
                black_box(&a_eq),
                black_box(&b_eq),
                black_box(&[]),
            )
            .unwrap();
            black_box(x.len())
        })
    });
}

fn bench_qp_20(c: &mut Criterion) {
    let n = 20;
    let p = DMatrix::from_fn(n, n, |i, j| if i == j { 2.0 } else { 0.0 });
    let q = DVector::from_fn(n, |i| i as f64 * 0.1);
    let a_eq = DMatrix::from_fn(1, n, |_, _| 1.0);
    let b_eq = DVector::from_vec(vec![1.0]);
    c.bench_function("qp 20 variables", |b| {
        b.iter(|| {
            black_box(
                solve_qp(
                    black_box(&p),
                    black_box(&q),
                    black_box(&a_eq),
                    black_box(&b_eq),
                    black_box(&[]),
                )
                .unwrap(),
            )
        })
    });
}

criterion_group!(benches, bench_qp, bench_qp_20);
criterion_main!(benches);
