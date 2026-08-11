use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_exact::BigRational;

fn bench_rational(c: &mut Criterion) {
    let a = BigRational::new(1u8.into(), 3u8.into());
    let b = BigRational::new(2u8.into(), 7u8.into());

    c.bench_function("rational add", |bencher| {
        bencher.iter(|| {
            let s = black_box(&a) + black_box(&b);
            black_box(s)
        })
    });
}

criterion_group!(benches, bench_rational);
criterion_main!(benches);
