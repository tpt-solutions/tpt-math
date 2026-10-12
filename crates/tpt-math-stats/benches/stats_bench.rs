use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_stats::{kolmogorov_smirnov_two_sample, two_sample_t_test};

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

criterion_group!(benches, bench_stats);
criterion_main!(benches);
