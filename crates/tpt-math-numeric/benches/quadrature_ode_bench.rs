use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_numeric::ode::dopri5_integrate;
use tpt_math_numeric::quadrature::{adaptive_gauss_kronrod, gauss_kronrod_15};
use tpt_math_numeric::roots::brent;

fn bench_numeric(c: &mut Criterion) {
    c.bench_function("gauss-kronrod 15 on sin", |b| {
        b.iter(|| {
            black_box(gauss_kronrod_15(
                |x: f64| x.sin(),
                0.0,
                core::f64::consts::PI,
            ))
        })
    });
    c.bench_function("adaptive gauss-kronrod 1/sqrt(x)", |b| {
        b.iter(|| {
            black_box(adaptive_gauss_kronrod(
                |x: f64| 1.0 / x.sqrt(),
                0.0,
                1.0,
                1e-8,
                60,
            ))
        })
    });
    c.bench_function("dopri5 harmonic oscillator", |b| {
        b.iter(|| {
            black_box(dopri5_integrate(
                |_t: f64, y: &[f64; 2]| [y[1], -y[0]],
                0.0,
                10.0,
                [1.0, 0.0],
                1e-9,
                1e-12,
                100_000,
            ))
        })
    });
    c.bench_function("brent cos root", |b| {
        b.iter(|| black_box(brent(|x: f64| x.cos() - x, 0.0, 1.0, 1e-12, 200)))
    });
}

criterion_group!(benches, bench_numeric);
criterion_main!(benches);
