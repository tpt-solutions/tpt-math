use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_stats::{
    kolmogorov_smirnov_two_sample, try_median, try_quantile, two_sample_t_test, variance,
};

fn bench_stats(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.37).sin()).collect();
    let b: Vec<f64> = (0..10_000)
        .map(|i| (i as f64 * 0.41).cos() + 0.05)
        .collect();
    c.bench_function("two-sample t-test, 10k each", |bch| {
        bch.iter(|| black_box(two_sample_t_test(&a, &b)))
    });
    c.bench_function("two-sample KS, 10k each", |bch| {
        bch.iter(|| black_box(kolmogorov_smirnov_two_sample(&a, &b)))
    });
}

/// Order statistics and variance at 1e6 samples, the size in the Phase I plan.
fn bench_large_sample(c: &mut Criterion) {
    let x: Vec<f64> = (0..1_000_000)
        .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 1_000_003) as f64 / 7.0)
        .collect();
    c.bench_function("median_1e6", |bch| {
        bch.iter(|| black_box(try_median(black_box(&x)).unwrap()))
    });
    c.bench_function("quantile_0.9_1e6", |bch| {
        bch.iter(|| black_box(try_quantile(black_box(&x), 0.9).unwrap()))
    });
    c.bench_function("variance_1e6", |bch| {
        bch.iter(|| black_box(variance(black_box(&x))))
    });
}

criterion_group!(benches, bench_stats, bench_large_sample);
criterion_main!(benches);
