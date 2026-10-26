use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_units_dyn::{convert, Dimension, DynQuantity};

fn bench_units(c: &mut Criterion) {
    c.bench_function("convert km/h to m/s x 1k", |b| {
        b.iter(|| {
            (0..1000)
                .map(|i| convert(f64::from(i), "km/h", "m/s").unwrap())
                .sum::<f64>()
        })
    });
    c.bench_function("dimension-checked multiply x 1k", |b| {
        let a = DynQuantity::from_base(2.0, Dimension::LENGTH);
        let t = DynQuantity::from_base(3.0, Dimension::TIME);
        b.iter(|| {
            let mut q = a.clone();
            for _ in 0..1000 {
                q = q.try_mul(&t).unwrap().try_div(&t).unwrap();
            }
            black_box(q)
        })
    });
}

criterion_group!(benches, bench_units);
criterion_main!(benches);
