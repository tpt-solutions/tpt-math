use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tpt_math_linalg_dense::{DMatrix, DVector};

/// Deterministic, well-conditioned test data (diagonally dominant for LU).
fn well_conditioned(n: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            n as f64
        } else {
            ((i * 3 + j * 5) as f64).sin()
        }
    })
}

fn bench_matrix_vector(c: &mut Criterion) {
    let m = DMatrix::from_fn(64, 64, |i, j| (i * 7 + j * 3) as f64);
    let v = DVector::from_fn(64, |i| i as f64);
    c.bench_function("dmatrix_dvector_mul_64", |b| {
        b.iter(|| black_box(m.clone()) * black_box(v.clone()))
    });
}

fn bench_matrix_matrix(c: &mut Criterion) {
    for &n in &[64_usize, 256, 512] {
        let m = DMatrix::from_fn(n, n, |i, j| (i * 7 + j * 3) as f64);
        let k = DMatrix::from_fn(n, n, |i, j| (i * 5 + j * 11) as f64);
        c.bench_function(&format!("dmatrix_dmatrix_mul_{n}"), |b| {
            b.iter(|| black_box(m.clone()) * black_box(k.clone()))
        });
    }
}

/// `solve` (which factors on every call) and `Lu::new` at the sizes in the
/// Phase I plan.
fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu");
    for n in [32, 64, 128, 512] {
        let m = dominant(n);
        let rhs = DVector::from_fn(n, |i| i as f64);
        group.bench_with_input(BenchmarkId::new("solve", n), &n, |b, _| {
            b.iter(|| black_box(&m).solve(black_box(&rhs)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("factor", n), &n, |b, _| {
            b.iter(|| tpt_math_linalg_dense::Lu::new(black_box(&m)).unwrap())
        });
    }
    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let m = well_conditioned(32);
    c.bench_function("dmatrix_inverse_32", |b| {
        b.iter(|| black_box(&m).inverse().unwrap())
    });
    for &n in &[64_usize, 256] {
        let m = well_conditioned(n);
        c.bench_function(&format!("dmatrix_inverse_{n}"), |b| {
            b.iter(|| black_box(&m).inverse().unwrap())
        });
    }
}

fn bench_solve(c: &mut Criterion) {
    for &n in &[64_usize, 256] {
        let m = well_conditioned(n);
        let rhs = DVector::from_fn(n, |i| i as f64);
        c.bench_function(&format!("dmatrix_solve_{n}"), |b| {
            b.iter(|| black_box(&m).solve(black_box(&rhs)).unwrap())
        });
    }
}

fn bench_det(c: &mut Criterion) {
    let m = dominant(128);
    c.bench_function("dmatrix_det_128", |b| {
        b.iter(|| black_box(&m).det().unwrap())
    });
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    for n in [128, 512] {
        let a = spd(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(&a).cholesky().unwrap())
        });
    }
    group.finish();
}

fn bench_svd(c: &mut Criterion) {
    let m = DMatrix::from_fn(64, 64, |i, j| ((i * 13 + j * 7) as f64).sin());
    c.bench_function("dmatrix_svd_64", |b| {
        b.iter(|| black_box(&m).svd().unwrap())
    });
}

fn bench_matrix_exp(c: &mut Criterion) {
    let m = DMatrix::from_fn(32, 32, |i, j| 0.1 * ((i * 3 + j * 5) as f64).sin());
    c.bench_function("dmatrix_matrix_exp_32", |b| {
        b.iter(|| black_box(&m).matrix_exp().unwrap())
    });
}

criterion_group!(
    benches,
    bench_matrix_vector,
    bench_matrix_matrix,
    bench_inverse,
    bench_solve
);
criterion_main!(benches);
