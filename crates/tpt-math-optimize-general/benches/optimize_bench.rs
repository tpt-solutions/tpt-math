use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_optimize_general::{
    minimize_lbfgs_with, minimize_newton,
    tpt_math_linalg_dense::{DMatrix, DVector},
    Options,
};

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

fn bench_newton(c: &mut Criterion) {
    // Newton on the same 4-D Rosenbrock, with the analytic Hessian.
    // Hessian of Σ 100(x_{i+1} − x_i²)² + (1 − x_i)²: tridiagonal in 4-D.
    let hess = |x: &DVector<f64>| {
        DMatrix::from_fn(4, 4, |r, c| {
            if r == c {
                let mut v = 0.0;
                if r < 3 {
                    v += 1200.0 * x[r] * x[r] - 400.0 * x[r + 1] + 2.0;
                }
                if r >= 1 {
                    v += 200.0;
                }
                v
            } else if r + 1 == c {
                -400.0 * x[r]
            } else if c + 1 == r {
                -400.0 * x[c]
            } else {
                0.0
            }
        })
    };
    c.bench_function("Newton Rosenbrock 4-D", |b| {
        b.iter(|| {
            black_box(minimize_newton(
                rosenbrock,
                rosenbrock_grad,
                hess,
                DVector::from_vec(vec![-1.2, 1.0, -1.2, 1.0]),
                200,
            ))
        })
    });
}

criterion_group!(benches, bench_optimize, bench_newton);
criterion_main!(benches);
