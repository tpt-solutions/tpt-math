use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_linalg_dense::{DMatrix, DVector};

fn bench_matrix_vector(c: &mut Criterion) {
    let m = DMatrix::from_fn(64, 64, |i, j| (i * 7 + j * 3) as f64);
    let v = DVector::from_fn(64, |i| i as f64);
    c.bench_function("dmatrix_dvector_mul_64", |b| {
        b.iter(|| black_box(m.clone()) * black_box(v.clone()))
    });
}

fn bench_matrix_matrix(c: &mut Criterion) {
    let m = DMatrix::from_fn(64, 64, |i, j| (i * 7 + j * 3) as f64);
    let n = DMatrix::from_fn(64, 64, |i, j| (i * 5 + j * 11) as f64);
    c.bench_function("dmatrix_dmatrix_mul_64", |b| {
        b.iter(|| black_box(m.clone()) * black_box(n.clone()))
    });
}

fn bench_inverse(c: &mut Criterion) {
    let m = DMatrix::from_fn(32, 32, |i, j| {
        if i == j {
            2.0
        } else {
            0.5 * ((i * 3 + j * 5) as f64).sin()
        }
    });
    c.bench_function("dmatrix_inverse_32", |b| {
        b.iter(|| black_box(&m).inverse().unwrap())
    });
}

criterion_group!(
    benches,
    bench_matrix_vector,
    bench_matrix_matrix,
    bench_inverse
);
criterion_main!(benches);
