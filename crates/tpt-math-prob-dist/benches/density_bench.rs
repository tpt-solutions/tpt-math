use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_prob_dist::density::{gamma_cdf, gamma_quantile, normal_cdf, student_t_cdf};

fn bench_density(c: &mut Criterion) {
    c.bench_function("normal_cdf x 1k", |b| {
        b.iter(|| {
            (0..1000)
                .map(|i| normal_cdf(0.0, 1.0, i as f64 * 0.01 - 5.0))
                .sum::<f64>()
        })
    });
    c.bench_function("gamma_cdf x 1k", |b| {
        b.iter(|| {
            (0..1000)
                .map(|i| gamma_cdf(3.2, 1.5, i as f64 * 0.02))
                .sum::<f64>()
        })
    });
    c.bench_function("student_t_cdf x 1k", |b| {
        b.iter(|| {
            (0..1000)
                .map(|i| student_t_cdf(7.0, i as f64 * 0.01 - 5.0))
                .sum::<f64>()
        })
    });
    c.bench_function("gamma_quantile bisection", |b| {
        b.iter(|| black_box(gamma_quantile(3.2, 1.5, 0.9)))
    });
}

criterion_group!(benches, bench_density);
criterion_main!(benches);
