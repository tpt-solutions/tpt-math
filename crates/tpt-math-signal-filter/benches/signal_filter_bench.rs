use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_signal_filter::FirFilter;

/// Direct convolution against FFT overlap-add, for a short and a long kernel.
///
/// Measured on the development machine (median): 9 taps over 16k samples, direct
/// about 78 µs and overlap-add about 340 µs. 255 taps, direct about 2.3 ms and
/// overlap-add about 0.5 ms. The crossover lies between those two lengths.
fn bench_fir(c: &mut Criterion) {
    let signal: Vec<f64> = (0..16_384).map(|i| (i as f64 * 0.01).sin()).collect();
    let short = FirFilter::lowpass(9, 0.1);
    let fir = FirFilter::lowpass(255, 0.1);

    c.bench_function("fir direct 9 taps x 16k", |b| {
        b.iter(|| black_box(short.filter(black_box(&signal))).len())
    });
    c.bench_function("fir overlap-add 9 taps x 16k", |b| {
        b.iter(|| black_box(short.filter_overlap_add(black_box(&signal))).len())
    });

    c.bench_function("fir direct 255 taps x 16k", |b| {
        b.iter(|| black_box(fir.filter(black_box(&signal))).len())
    });
    c.bench_function("fir overlap-add 255 taps x 16k", |b| {
        b.iter(|| black_box(fir.filter_overlap_add(black_box(&signal))).len())
    });
}

criterion_group!(benches, bench_fir);
criterion_main!(benches);
