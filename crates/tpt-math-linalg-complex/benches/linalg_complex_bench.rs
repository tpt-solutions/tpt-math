use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_linalg_complex::{Complex, ComplexDMatrix, ComplexDVector};

/// A diagonally dominant complex matrix of order `n`, so it is well conditioned.
fn dominant(n: usize) -> ComplexDMatrix<f64> {
    ComplexDMatrix::from_fn(n, n, |i, j| {
        if i == j {
            Complex::new(n as f64 + 1.0, 0.5)
        } else {
            Complex::new(((i * 3 + j) as f64).sin(), ((i + 2 * j) as f64).cos() * 0.5)
        }
    })
}

fn bench_complex(c: &mut Criterion) {
    for n in [64usize, 128, 256] {
        let a = dominant(n);
        let rhs = ComplexDVector::from_fn(n, |i| Complex::new(i as f64, 1.0));
        c.bench_function(&format!("complex_lu_{n}"), |b| {
            b.iter(|| black_box(a.lu().unwrap()))
        });
        c.bench_function(&format!("complex_solve_{n}"), |b| {
            b.iter(|| black_box(a.solve(black_box(&rhs)).unwrap()))
        });
        c.bench_function(&format!("complex_inverse_{n}"), |b| {
            b.iter(|| black_box(a.inverse().unwrap()))
        });
        c.bench_function(&format!("complex_product_{n}"), |b| {
            b.iter(|| black_box(a.product(black_box(&a))))
        });
    }
}

criterion_group!(benches, bench_complex);
criterion_main!(benches);
