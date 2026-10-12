//! Property tests: simplification must preserve the value of an expression.

use proptest::prelude::*;
use tpt_math_symbolic::{Expr64, Symbolic};

/// Random polynomial-style expressions in the single variable `x`.
fn expr_strategy() -> impl Strategy<Value = Expr64> {
    let leaf = prop_oneof![
        Just(Expr64::var("x")),
        (-3i32..=3).prop_map(|c| Expr64::constant(f64::from(c))),
    ];
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
            inner.clone().prop_map(|a| -a),
            (inner, 0usize..=3).prop_map(|(a, n)| a.pow(n)),
        ]
    })
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

proptest! {
    #[test]
    fn simplify_preserves_value(e in expr_strategy()) {
        let simplified = e.clone().simplify();
        for &x in &[-1.5, -0.3, 0.7, 2.0] {
            if let (Some(before), Some(after)) = (e.eval_str("x", x), simplified.eval_str("x", x)) {
                prop_assert!(
                    close(before, after),
                    "x = {x}: {before} before simplify, {after} after"
                );
            }
        }
    }

    #[test]
    fn simplify_is_idempotent(e in expr_strategy()) {
        let once = e.simplify();
        let twice = once.clone().simplify();
        prop_assert_eq!(once, twice);
    }
}
