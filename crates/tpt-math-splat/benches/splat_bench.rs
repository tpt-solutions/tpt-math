use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_splat::gaussian::{condition_on_time, cov4_from_rot_scale};
use tpt_math_splat::sfc::{hilbert_encode, morton3_encode};
use tpt_math_splat::sh::eval_color;

fn bench_splat(c: &mut Criterion) {
    let cov = cov4_from_rot_scale(
        [0.9_f32, 0.1, -0.3, 0.2],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 1.0, 2.0, 0.25],
    )
    .unwrap();
    c.bench_function("condition_on_time f32", |b| {
        b.iter(|| condition_on_time(black_box([0.0_f32; 4]), black_box(&cov), black_box(0.3)))
    });
    let coeffs = [[0.1_f32, 0.2, 0.3]; 16];
    c.bench_function("sh degree 3 color f32", |b| {
        b.iter(|| eval_color(3, black_box(&coeffs), black_box([0.3, -0.5, 0.8])))
    });
    c.bench_function("morton3 x 1k", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for i in 0..1000u32 {
                acc ^= morton3_encode(black_box([i, i * 3, i * 7]));
            }
            acc
        })
    });
    c.bench_function("hilbert3 x 1k", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for i in 0..1000u32 {
                acc ^= hilbert_encode(black_box([i, i * 3, i * 7]), 21).unwrap();
            }
            acc
        })
    });
}

criterion_group!(benches, bench_splat);
criterion_main!(benches);
