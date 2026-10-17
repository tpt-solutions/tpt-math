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

fn well_conditioned(n: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else {
            0.5 * ((i * 3 + j * 5) as f64).sin()
        }
    })
}

fn bench_inverse(c: &mut Criterion) {
    for n in [32, 64, 256] {
        let m = well_conditioned(n);
        c.bench_function(&format!("dmatrix_inverse_{n}"), |b| {
            b.iter(|| black_box(&m).inverse().unwrap())
        });
    }
}

fn bench_solve(c: &mut Criterion) {
    for n in [64, 256] {
        let m = well_conditioned(n);
        let rhs = DVector::from_fn(n, |i| i as f64);
        c.bench_function(&format!("dmatrix_solve_{n}"), |b| {
            b.iter(|| black_box(&m).solve(black_box(&rhs)).unwrap())
        });
    }
}

criterion_group!(
    benches,
    bench_matrix_vector,
    bench_matrix_matrix,
    bench_inverse,
    bench_solve
);
criterion_main!(benches);
