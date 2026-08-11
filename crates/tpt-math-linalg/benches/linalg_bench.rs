use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_linalg::{nalgebra, Mat, Vec};

fn bench_matvec(c: &mut Criterion) {
    let m = Mat::<f64, f64>::from_raw(nalgebra::DMatrix::<f64>::from_fn(64, 64, |i, j| {
        (i + j) as f64
    }));
    let v = Vec::<f64>::from_raw(nalgebra::DVector::<f64>::from_fn(64, |i, _| i as f64));

    c.bench_function("matvec 64x64", |b| {
        b.iter(|| {
            let r = black_box(m.clone()) * black_box(v.clone());
            black_box(r.raw().len())
        })
    });
}

criterion_group!(benches, bench_matvec);
criterion_main!(benches);
