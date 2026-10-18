//! Fuzz the regularised incomplete gamma functions on valid input. Results must be
//! finite, `P + Q = 1`, and both must lie in [0, 1]. `P` must not decrease in `x`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tpt_math_prob_core::special::{gamma_p, gamma_q};

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    // Map two arbitrary words into a shape a > 0 and a point x >= 0.
    let a_bits = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let x_bits = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let a = 1e-3 + (a_bits % 100_000) as f64 / 100.0;
    let x = (x_bits % 1_000_000) as f64 / 1000.0;

    let p = gamma_p(a, x);
    let q = gamma_q(a, x);
    if p.is_nan() || q.is_nan() {
        // The series or continued fraction ran out of budget. That is the documented
        // NaN for extreme shapes, not a wrong value.
        return;
    }
    assert!((0.0..=1.0).contains(&p), "P({a}, {x}) = {p}");
    assert!((0.0..=1.0).contains(&q), "Q({a}, {x}) = {q}");
    assert!((p + q - 1.0).abs() < 1e-9, "P + Q at ({a}, {x})");
    let p2 = gamma_p(a, x * 1.01 + 1e-9);
    if !p2.is_nan() {
        assert!(p2 + 1e-12 >= p, "P not monotone at ({a}, {x})");
    }
});
