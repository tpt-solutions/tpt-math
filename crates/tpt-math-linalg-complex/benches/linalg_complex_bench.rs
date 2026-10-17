use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_linalg_complex::{Complex, ComplexDMatrix, ComplexDVector};

/// Diagonally dominant complex matrix, so the benches measure the kernels and
/// not pivoting noise.
fn well_conditioned(n: usize) -> ComplexDMatrix<f64> {
    ComplexDMatrix::from_fn(n, n, |i, j| {
        if i == j {
            Complex::new(2.0 + n as f64, 0.0)
        } else {
            let re = 0.5 * ((i * 3 + j * 5) as f64).sin();
            let im = 0.5 * ((i * 7 + j) as f64).cos();
            Complex::new(re, im)
        }
    })
}

fn bench_complex_linalg(c: &mut Criterion) {
    for n in [64usize, 256] {
        let m = well_conditioned(n);
        let rhs = ComplexDVector::from_fn(n, |i| Complex::new(i as f64, 1.0));
        c.bench_function(&format!("complex_lu_{n}"), |b| {
            b.iter(|| black_box(&m).lu().unwrap())
        });
        c.bench_function(&format!("complex_solve_{n}"), |b| {
            b.iter(|| black_box(&m).solve(black_box(&rhs)).unwrap())
        });
        c.bench_function(&format!("complex_inverse_{n}"), |b| {
            b.iter(|| black_box(&m).inverse().unwrap())
        });
    }
}

criterion_group!(benches, bench_complex_linalg);
criterion_main!(benches);
