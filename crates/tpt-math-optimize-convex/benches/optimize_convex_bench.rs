use criterion::{black_box, criterion_group, criterion_main, Criterion};
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

criterion_group!(benches, bench_qp);
criterion_main!(benches);
