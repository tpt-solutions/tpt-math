use tpt_math_linalg_complex::{Complex, ComplexDMatrix};

fn main() {
    let a = ComplexDMatrix::from_real_row_slice(
        4,
        4,
        &[
            4.0, 1.0, 0.0, 0.0, //
            1.0, 3.0, 1.0, 0.0, //
            0.0, 1.0, 2.0, 1.0, //
            0.0, 0.0, 1.0, 1.0, //
        ],
    );
    let ev = a.eigenvalues();
    let trace: f64 = ev.iter().map(|z| z.re).sum();
    assert!((trace - 10.0).abs() < 1e-9);
    let _ = Complex::new(1.0_f64, 0.0);
}
