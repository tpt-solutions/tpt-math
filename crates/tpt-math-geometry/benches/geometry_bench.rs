use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_geometry::{Isometry3, Point3, Rotation3, Translation};

fn bench_geometry(c: &mut Criterion) {
    let iso = Isometry3::new(
        Translation::new(tpt_math_linalg_fixed::Vector::new([1.0_f64, 2.0, 3.0])),
        Rotation3::from_axis_angle(
            &tpt_math_linalg_fixed::Vector::new([0.0_f64, 0.0, 1.0]),
            0.7,
        ),
    );
    let p = Point3::new(tpt_math_linalg_fixed::Vector::new([0.5_f64, -0.25, 2.0]));
    c.bench_function("isometry transform_point x 1k", |b| {
        b.iter(|| {
            let mut q = p;
            for _ in 0..1000 {
                q = iso.transform_point(&q);
            }
            black_box(q)
        })
    });
}

criterion_group!(benches, bench_geometry);
criterion_main!(benches);
