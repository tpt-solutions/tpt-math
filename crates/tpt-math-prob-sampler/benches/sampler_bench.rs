use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_prob_core::SplitMix64;
use tpt_math_prob_sampler::sample_categorical;

fn bench_sampler(c: &mut Criterion) {
    let weights: Vec<f64> = (1..=64).map(|i| 1.0 / f64::from(i)).collect();
    c.bench_function("categorical 64 outcomes x 10k", |b| {
        let mut rng = SplitMix64::seed_from_u64(5);
        b.iter(|| {
            let mut acc = 0usize;
            for _ in 0..10_000 {
                acc += sample_categorical(&weights, &mut rng).unwrap_or(0);
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_sampler);
criterion_main!(benches);
