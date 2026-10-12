//! Fuzz `simplify`: the simplified expression must have the same value as the
//! original wherever both evaluate to finite numbers.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tpt_math_symbolic::{Expr64, Symbolic};

/// Build an expression from the byte stream. Each byte picks a node; running out
/// of bytes produces a leaf, so every input yields some expression.
fn build(data: &[u8], pos: &mut usize, depth: u32) -> Expr64 {
    let byte = data.get(*pos).copied().unwrap_or(0);
    *pos += 1;
    if depth == 0 || byte % 4 == 0 {
        return if byte % 2 == 0 {
            Expr64::var("x")
        } else {
            Expr64::constant(f64::from(i32::from(byte % 7) - 3))
        };
    }
    match byte % 6 {
        1 => build(data, pos, depth - 1) + build(data, pos, depth - 1),
        2 => build(data, pos, depth - 1) - build(data, pos, depth - 1),
        3 => build(data, pos, depth - 1) * build(data, pos, depth - 1),
        4 => -build(data, pos, depth - 1),
        _ => build(data, pos, depth - 1).pow(usize::from(byte % 3)),
    }
}

fuzz_target!(|data: &[u8]| {
    let mut pos = 0;
    let expr = build(data, &mut pos, 5);
    let simplified = expr.clone().simplify();
    for &x in &[-1.5, 0.5, 2.0] {
        if let (Some(before), Some(after)) = (expr.eval_str("x", x), simplified.eval_str("x", x)) {
            if before.is_finite() && after.is_finite() {
                let tol = 1e-9 * (1.0 + before.abs().max(after.abs()));
                assert!(
                    (before - after).abs() <= tol,
                    "simplify changed the value at x = {x}: {before} vs {after}"
                );
            }
        }
    }
});
