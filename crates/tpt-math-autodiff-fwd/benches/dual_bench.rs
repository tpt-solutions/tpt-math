use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_autodiff_fwd::Dual;

fn bench_dual(c: &mut Criterion) {
    c.bench_function("forward-mode sin(x)*x+exp(x), 1k points", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for i in 0..1000 {
                let x = Dual::<f64, 1>::new(i as f64 * 0.001, [1.0]);
                let y = x.sin() * x + x.exp();
                acc += y.deriv()[0];
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_dual);
criterion_main!(benches);
