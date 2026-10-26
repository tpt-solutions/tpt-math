use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_linalg_fixed::Matrix;

fn bench_fixed(c: &mut Criterion) {
    let m = Matrix::<f64, 4, 4>::from_fn(|r, c| if r == c { 4.0 } else { 0.1 * (r + c) as f64 });
    c.bench_function("4x4 matrix product x 1k", |b| {
        b.iter(|| {
            let mut acc = m;
            for _ in 0..1000 {
                acc = acc * m;
            }
            black_box(acc)
        })
    });
    c.bench_function("4x4 closed-form inverse x 1k", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for _ in 0..1000 {
                total += m.inverse().map_or(0.0, |inv| inv.data[0][0]);
            }
            black_box(total)
        })
    });
}

criterion_group!(benches, bench_fixed);
criterion_main!(benches);
