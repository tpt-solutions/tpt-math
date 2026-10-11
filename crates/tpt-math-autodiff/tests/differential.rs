//! Cross-crate differential tests.
//!
//! Each test computes the same quantity by independent routes and requires them
//! to agree. A bug in one implementation shows up as a disagreement with the
//! others, so the checks need no hand-computed reference values:
//!
//! * **Derivatives:** forward-mode dual numbers, reverse-mode tape gradients,
//!   symbolic differentiation and central finite differences.
//! * **Values:** the plain `f64` function, each autodiff tape, and symbolic
//!   evaluation.
//! * **Linear solves:** exact rational Gauss elimination against the dense
//!   `f64` solver.

use std::collections::HashMap;

use num_traits::ToPrimitive;
use tpt_math_autodiff::fwd::Dual;
use tpt_math_autodiff::rev::value_and_gradient;
use tpt_math_exact::{BigInt, BigRational, Rational};
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_math_symbolic::{Expr64, Symbolic};

/// The test function `f(x, y) = x·y + sin x − ½·e^{y²}`, in plain `f64`.
fn f_plain(x: f64, y: f64) -> f64 {
    x * y + x.sin() - 0.5 * (y * y).exp()
}

fn f_forward(x: Dual<f64, 2>, y: Dual<f64, 2>) -> Dual<f64, 2> {
    x * y + x.sin() - Dual::constant(0.5) * (y * y).exp()
}

fn f_symbolic() -> Expr64 {
    let x = Expr64::var("x");
    let y = Expr64::var("y");
    x.clone() * y.clone() + Expr64::func("sin", x)
        - Expr64::constant(0.5) * Expr64::func("exp", y.clone() * y)
}

/// Grid of evaluation points, avoiding the origin to keep it away from
/// anything special.
fn grid() -> Vec<(f64, f64)> {
    let axis = [-1.5, -0.5, 0.25, 1.0, 1.75];
    axis.iter()
        .flat_map(|&x| axis.iter().map(move |&y| (x, y)))
        .collect()
}

fn env(x: f64, y: f64) -> HashMap<String, f64> {
    HashMap::from([("x".to_string(), x), ("y".to_string(), y)])
}

#[test]
fn values_agree_across_representations() {
    let f_sym = f_symbolic();
    for (x, y) in grid() {
        let plain = f_plain(x, y);
        let fwd = f_forward(Dual::variable(x, 0), Dual::variable(y, 1)).re();
        let (rev, _) = value_and_gradient(&[x, y], |tape, v| {
            v[0] * v[1] + v[0].sin() - tape.constant(0.5) * (v[1] * v[1]).exp()
        });
        let sym = f_sym.eval(&env(x, y)).expect("symbolic evaluation");
        for (name, value) in [("forward", fwd), ("reverse", rev), ("symbolic", sym)] {
            assert!(
                (value - plain).abs() < 1e-12,
                "{name} value {value} vs plain {plain} at ({x}, {y})"
            );
        }
    }
}

#[test]
fn derivatives_agree_across_representations() {
    let f_sym = f_symbolic();
    let dx_sym = f_sym.derivative("x");
    let dy_sym = f_sym.derivative("y");
    let h = 1e-6;
    for (x, y) in grid() {
        // Forward mode: one seeded direction per partial.
        let fwd = f_forward(Dual::variable(x, 0), Dual::variable(y, 1));
        let fwd_grad = [fwd.du(0), fwd.du(1)];

        // Reverse mode: a single backward pass gives both partials.
        let (_, rev_grad) = value_and_gradient(&[x, y], |tape, v| {
            v[0] * v[1] + v[0].sin() - tape.constant(0.5) * (v[1] * v[1]).exp()
        });

        let sym_grad = [
            dx_sym.eval(&env(x, y)).expect("symbolic ∂x"),
            dy_sym.eval(&env(x, y)).expect("symbolic ∂y"),
        ];

        // Central differences: O(h²) truncation error, so a loose tolerance.
        let fd_grad = [
            (f_plain(x + h, y) - f_plain(x - h, y)) / (2.0 * h),
            (f_plain(x, y + h) - f_plain(x, y - h)) / (2.0 * h),
        ];

        for i in 0..2 {
            let reference = fd_grad[i];
            let tol = 1e-6 * (1.0 + reference.abs());
            for (name, value) in [
                ("forward", fwd_grad[i]),
                ("reverse", rev_grad[i]),
                ("symbolic", sym_grad[i]),
            ] {
                assert!(
                    (value - reference).abs() < tol,
                    "∂{} {name} {value} vs finite difference {reference} at ({x}, {y})",
                    if i == 0 { "x" } else { "y" }
                );
            }
            // The two exact-derivative routes must agree much more tightly.
            assert!((fwd_grad[i] - rev_grad[i]).abs() < 1e-12);
            assert!((fwd_grad[i] - sym_grad[i]).abs() < 1e-12);
        }
    }
}

/// Solves `A x = b` exactly over the rationals by Gauss elimination with
/// partial pivoting. Panics on a singular system, which these fixtures avoid.
fn exact_solve(a: [[i64; 3]; 3], b: [i64; 3]) -> [BigRational; 3] {
    let mut m: Vec<Vec<BigRational>> = (0..3)
        .map(|i| {
            let mut row: Vec<BigRational> = a[i]
                .iter()
                .map(|&v| BigRational::from_integer(BigInt::from(v)))
                .collect();
            row.push(BigRational::from_integer(BigInt::from(b[i])));
            row
        })
        .collect();
    for col in 0..3 {
        let pivot = (col..3)
            .find(|&r| m[r][col] != Rational::from_integer(BigInt::from(0)))
            .expect("singular fixture");
        m.swap(col, pivot);
        for r in 0..3 {
            if r != col {
                let factor = m[r][col].clone() / m[col][col].clone();
                let pivot_row = m[col].clone();
                for (cell, p) in m[r][col..4].iter_mut().zip(&pivot_row[col..4]) {
                    *cell = cell.clone() - factor.clone() * p.clone();
                }
            }
        }
    }
    [0, 1, 2].map(|i| m[i][3].clone() / m[i][i].clone())
}

#[test]
fn dense_f64_solve_matches_exact_rational_solve() {
    let fixtures: [([[i64; 3]; 3], [i64; 3]); 3] = [
        ([[4, 1, 0], [2, 5, 1], [0, 1, 3]], [6, 15, 11]),
        ([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [1, 0, 1]),
        ([[3, 7, -2], [1, -4, 5], [6, 2, 9]], [10, -3, 4]),
    ];
    for (a, b) in fixtures {
        let exact = exact_solve(a, b);
        let dense_a = DMatrix::from_fn(3, 3, |i, j| a[i][j] as f64);
        let dense_b = DVector::from_fn(3, |i| b[i] as f64);
        let approx = dense_a.solve(&dense_b).expect("dense solve");
        for i in 0..3 {
            let exact_f64 = exact[i].to_f64().expect("representable");
            assert!(
                (approx[i] - exact_f64).abs() < 1e-12,
                "x[{i}]: dense {} vs exact {exact_f64} for {a:?}",
                approx[i]
            );
        }
    }
}

#[test]
fn exact_solver_reproduces_known_solution() {
    // Integer system with exact solution (1, 2, 3): the rational solver must
    // return exactly those values, with no rounding.
    let x = exact_solve([[4, 1, 0], [2, 5, 1], [0, 1, 3]], [6, 15, 11]);
    for (i, want) in [1, 2, 3].into_iter().enumerate() {
        assert_eq!(x[i], BigRational::from_integer(BigInt::from(want)));
    }
}
