use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_optimize_general::{minimize_lbfgs_with, tpt_math_linalg_dense::DVector, Options};

fn rosenbrock(x: &DVector<f64>) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

fn rosenbrock_grad(x: &DVector<f64>) -> DVector<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
    }
    DVector::from_vec(g)
}

fn bench_optimize(c: &mut Criterion) {
    c.bench_function("L-BFGS Rosenbrock 4-D", |b| {
        b.iter(|| {
            black_box(minimize_lbfgs_with(
                rosenbrock,
                rosenbrock_grad,
                DVector::from_vec(vec![-1.2, 1.0, -1.2, 1.0]),
                5,
                &Options::new(1000),
            ))
        })
    });
}

criterion_group!(benches, bench_optimize);
criterion_main!(benches);
