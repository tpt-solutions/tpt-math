use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_interpolate::{Pchip, RbfInterpolator, RbfKernel};
use tpt_math_linalg_dense::DVector;

fn bench_interpolate(c: &mut Criterion) {
    let n = 100;
    let xs = DVector::from_fn(n, |i| i as f64 / n as f64);
    let ys = DVector::from_fn(n, |i| (i as f64 / n as f64 * 6.0).sin());
    c.bench_function("RBF thin-plate fit, 100 points", |b| {
        b.iter(|| {
            black_box(RbfInterpolator::new(
                xs.clone(),
                ys.clone(),
                RbfKernel::ThinPlate { epsilon: 1.0 },
            ))
        })
    });
    let pchip = Pchip::new(xs.clone(), ys.clone()).unwrap();
    c.bench_function("PCHIP eval x 1k", |b| {
        b.iter(|| {
            (0..1000)
                .map(|i| pchip.eval(i as f64 / 1000.0))
                .sum::<f64>()
        })
    });
}

criterion_group!(benches, bench_interpolate);
criterion_main!(benches);
