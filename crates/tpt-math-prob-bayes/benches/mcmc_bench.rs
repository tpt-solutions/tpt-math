use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_prob_bayes::hmc::run_hmc;

fn bench_hmc(c: &mut Criterion) {
    c.bench_function("hmc 2-D standard normal, 1k samples", |b| {
        b.iter(|| {
            black_box(run_hmc(
                |x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>(),
                |x: &[f64]| x.iter().map(|v| -v).collect::<Vec<f64>>(),
                &[0.0, 0.0],
                0.2,
                10,
                1_000,
                200,
                9,
            ))
        })
    });
}

criterion_group!(benches, bench_hmc);
criterion_main!(benches);
