use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_prob_markov::MarkovChain;

fn bench_markov(c: &mut Criterion) {
    let n = 50;
    let mut chain = MarkovChain::new(n);
    for i in 0..n {
        chain.set_transition(i, (i + 1) % n, 0.5);
        chain.set_transition(i, (i + n - 1) % n, 0.5);
    }
    c.bench_function("stationary distribution, 50-state ring", |b| {
        b.iter(|| black_box(chain.stationary()))
    });
}

criterion_group!(benches, bench_markov);
criterion_main!(benches);
