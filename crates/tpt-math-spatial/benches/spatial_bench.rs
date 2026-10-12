use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_geometry::Quaternion;
use tpt_math_spatial::DualQuaternion;

fn bench_spatial(c: &mut Criterion) {
    let a = DualQuaternion::new(
        Quaternion::new(0.1_f64, 0.2, 0.3, 0.9),
        Quaternion::new(0.4_f64, 0.0, -0.1, 0.2),
    );
    let b = a;
    c.bench_function("dual quaternion product x 1k", |bch| {
        bch.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc = acc.multiply(&b);
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_spatial);
criterion_main!(benches);
