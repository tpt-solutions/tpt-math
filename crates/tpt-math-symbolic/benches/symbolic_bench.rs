use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_symbolic::{Expr64, Symbolic};

fn bench_symbolic(c: &mut Criterion) {
    let x = Expr64::var("x");
    let mut e = x.clone();
    for _ in 0..6 {
        e = e.clone() * x.clone() + e.clone();
    }
    c.bench_function("simplify + derivative of a degree-7 polynomial", |b| {
        b.iter(|| black_box(e.clone().derivative("x").simplify()))
    });
}

criterion_group!(benches, bench_symbolic);
criterion_main!(benches);
