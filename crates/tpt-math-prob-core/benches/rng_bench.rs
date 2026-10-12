use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_prob_core::{Distribution, SplitMix64, Standard};

fn bench_rng(c: &mut Criterion) {
    c.bench_function("splitmix64 uniform f64 x 10k", |b| {
        let mut rng = SplitMix64::seed_from_u64(1);
        b.iter(|| {
            let mut acc = 0.0;
            for _ in 0..10_000 {
                let u: f64 = Standard.sample(&mut rng);
                acc += u;
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_rng);
criterion_main!(benches);
