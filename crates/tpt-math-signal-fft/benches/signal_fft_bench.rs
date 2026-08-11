use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_signal_fft::Fft;

fn bench_fft(c: &mut Criterion) {
    let signal: Vec<f64> = (0..1024).map(|i| (i as f64).sin()).collect();
    let mut engine = Fft::new();

    c.bench_function("fft 1024", |b| {
        b.iter(|| {
            let spec = engine.forward(black_box(&signal));
            black_box(spec.len())
        })
    });
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);
