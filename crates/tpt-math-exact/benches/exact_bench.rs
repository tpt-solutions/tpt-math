use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tpt_math_exact::roots::poly_eval;
use tpt_math_exact::{BigInt, BigRational};

fn frac(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn bench_rational(c: &mut Criterion) {
    let a = BigRational::new(1u8.into(), 3u8.into());
    let b = BigRational::new(2u8.into(), 7u8.into());

    c.bench_function("rational add", |bencher| {
        bencher.iter(|| {
            let s = black_box(&a) + black_box(&b);
            black_box(s)
        })
    });

    c.bench_function("rational mul", |bencher| {
        bencher.iter(|| {
            let s = black_box(&a) * black_box(&b);
            black_box(s)
        })
    });

    let i = BigRational::from_integer(BigInt::from(12345));
    let j = BigRational::from_integer(BigInt::from(678));
    c.bench_function("integer add", |bencher| {
        bencher.iter(|| {
            let s = black_box(&i) + black_box(&j);
            black_box(s)
        })
    });

    // Denominators grow: exercises large gcds and big intermediates.
    c.bench_function("harmonic sum 1..=300", |bencher| {
        bencher.iter(|| {
            let mut acc = BigRational::from_integer(BigInt::from(0));
            for k in 1..=300 {
                acc += frac(1, k);
            }
            black_box(acc)
        })
    });

    // Horner evaluation of a degree-20 polynomial at a non-integer point.
    let coeffs: Vec<BigRational> = (1..=21).map(|k| frac(k, k + 1)).collect();
    let x = frac(3, 5);
    c.bench_function("poly_eval degree 20", |bencher| {
        bencher.iter(|| black_box(poly_eval(black_box(&coeffs), black_box(&x))))
    });
}

criterion_group!(benches, bench_rational);
criterion_main!(benches);
