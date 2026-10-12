//! Cross-crate example: `tpt-math-signal` (FFT and FIR filters).
//!
//! Separates a slow tone from a fast one: a low-pass FIR filter removes the fast
//! component, and the real FFT shows which bins carry the energy before and after.

use tpt_math_signal::fft::Fft;
use tpt_math_signal::filter::FirFilter;

fn main() {
    let n = 1024;
    let slow = |t: usize| (2.0 * std::f64::consts::PI * 8.0 * t as f64 / n as f64).sin();
    let fast = |t: usize| 0.5 * (2.0 * std::f64::consts::PI * 200.0 * t as f64 / n as f64).sin();
    let signal: Vec<f64> = (0..n).map(|t| slow(t) + fast(t)).collect();

    let mut engine = Fft::new();
    let before = engine.rfft(&signal);
    let peak_before = peak_bin(&before);

    let filter = FirFilter::lowpass(129, 0.05);
    let filtered = filter.filter_overlap_add(&signal);
    let after = engine.rfft(&filtered);
    let peak_after = peak_bin(&after);

    println!("dominant bin before filtering: {peak_before}");
    println!("dominant bin after filtering:  {peak_after}");
    println!(
        "energy above bin 100 before: {:.2}",
        high_energy(&before, 100)
    );
    println!(
        "energy above bin 100 after:  {:.2}",
        high_energy(&after, 100)
    );
}

fn peak_bin(spectrum: &[tpt_math_signal::fft::Complex<f64>]) -> usize {
    spectrum
        .iter()
        .enumerate()
        .skip(1)
        .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
        .map_or(0, |(i, _)| i)
}

fn high_energy(spectrum: &[tpt_math_signal::fft::Complex<f64>], from: usize) -> f64 {
    spectrum.iter().skip(from).map(|z| z.norm_sqr()).sum()
}
