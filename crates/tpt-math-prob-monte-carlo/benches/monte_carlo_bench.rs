use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_prob_core::SplitMix64;
use tpt_math_prob_monte_carlo::integrate;

fn bench_mc(c: &mut Criterion) {
    c.bench_function("crude MC integrate x^2, 100k", |b| {
        b.iter(|| {
            let mut rng = SplitMix64::seed_from_u64(3);
            black_box(integrate(&mut rng, |x: f64| x * x, 0.0, 1.0, 100_000))
        })
    });
}

criterion_group!(benches, bench_mc);
criterion_main!(benches);
