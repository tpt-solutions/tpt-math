use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_linalg::{Mat, Vec};
use tpt_math_linalg_dense::{DMatrix, DVector};

fn bench_matvec(c: &mut Criterion) {
    let m = Mat::<f64, f64>::from_raw(DMatrix::<f64>::from_fn(64, 64, |i, j| (i + j) as f64));
    let v = Vec::<f64>::from_raw(DVector::<f64>::from_fn(64, |i| i as f64));

    c.bench_function("matvec 64x64", |b| {
        b.iter(|| {
            let r = black_box(m.clone()) * black_box(v.clone());
            black_box(r.raw().len())
        })
    });
}

fn bench_solve_inverse(c: &mut Criterion) {
    // A diagonally dominant 32 × 32 system, so both calls succeed.
    let n = 32;
    let m = Mat::<f64, f64>::from_raw(DMatrix::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            n as f64 + 1.0
        } else {
            ((i + 2 * j) as f64).sin()
        }
    }));
    let rhs = Vec::<f64>::from_raw(DVector::<f64>::from_fn(n, |i| i as f64));
    c.bench_function("Mat solve 32", |b| {
        b.iter(|| black_box(m.solve(black_box(&rhs)).unwrap()))
    });
    c.bench_function("Mat inverse 32", |b| {
        b.iter(|| black_box(m.inverse().unwrap()))
    });
}

criterion_group!(benches, bench_matvec, bench_solve_inverse);
criterion_main!(benches);
