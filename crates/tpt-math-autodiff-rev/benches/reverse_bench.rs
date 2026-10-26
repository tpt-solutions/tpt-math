use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_autodiff_rev::value_and_gradient;

fn bench_reverse(c: &mut Criterion) {
    let x: Vec<f64> = (0..20).map(|i| 0.1 * f64::from(i)).collect();
    c.bench_function("reverse-mode gradient, 20 inputs", |b| {
        b.iter(|| {
            black_box(value_and_gradient(&x, |_tape, v| {
                v.iter().enumerate().fold(v[0] * 0.0, |acc, (i, &vi)| {
                    acc + (vi * vi).sin() * f64::from(i as u32 + 1)
                })
            }))
        })
    });
}

criterion_group!(benches, bench_reverse);
criterion_main!(benches);
