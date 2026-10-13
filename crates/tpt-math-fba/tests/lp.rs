//! LP-level tests for the sparse bounded simplex.

use tpt_math_fba::{LpSolver, LpStatus, SimplexSolver, SparseLp};
use tpt_math_linalg_sparse::CooMatrix;

const INF: f64 = f64::INFINITY;

/// Build `min cᵀx, Ax = b, lb ≤ x ≤ ub` from dense rows.
fn lp(a: &[Vec<f64>], b: &[f64], c: &[f64], lb: &[f64], ub: &[f64]) -> SparseLp {
    let m = a.len();
    let n = c.len();
    let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for (i, row) in a.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x != 0.0 {
                r.push(i);
                cc.push(j);
                v.push(x);
            }
        }
    }
    SparseLp {
        a: CooMatrix::from_triplets(m, n, r, cc, v).to_csc(),
        b: b.to_vec(),
        c: c.to_vec(),
        lb: lb.to_vec(),
        ub: ub.to_vec(),
    }
}

fn solve(p: &SparseLp) -> tpt_math_fba::LpSolution {
    SimplexSolver::new().solve(p).unwrap()
}

#[test]
fn textbook_production_problem() {
    // max 3x + 5y  s.t. x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18, x,y ≥ 0  →  x=2, y=6, value 36.
    // Written in standard form with slacks s1..s3 ≥ 0.
    let a = vec![
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 1.0, 0.0],
        vec![3.0, 2.0, 0.0, 0.0, 1.0],
    ];
    let p = lp(
        &a,
        &[4.0, 12.0, 18.0],
        &[-3.0, -5.0, 0.0, 0.0, 0.0],
        &[0.0; 5],
        &[INF; 5],
    );
    let s = solve(&p);
    assert_eq!(s.status, LpStatus::Optimal);
    assert!(
        (-s.objective - 36.0).abs() < 1e-7,
        "objective {}",
        s.objective
    );
    assert!((s.x[0] - 2.0).abs() < 1e-7 && (s.x[1] - 6.0).abs() < 1e-7);
}

#[test]
fn bounded_variables_and_bound_flip() {
    // min -x - y  s.t. x + y = 3, 0 ≤ x ≤ 2, 0 ≤ y ≤ 2 → optimum value -3 at any split.
    let p = lp(
        &[vec![1.0, 1.0]],
        &[3.0],
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[2.0, 2.0],
    );
    let s = solve(&p);
    assert_eq!(s.status, LpStatus::Optimal);
    assert!((s.objective + 3.0).abs() < 1e-7);
    assert!((s.x[0] + s.x[1] - 3.0).abs() < 1e-7);
}

#[test]
fn infeasible_is_reported() {
    // x + y = 1 and x + y = 3 cannot both hold.
    let p = lp(
        &[vec![1.0, 1.0], vec![1.0, 1.0]],
        &[1.0, 3.0],
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[INF, INF],
    );
    assert_eq!(solve(&p).status, LpStatus::Infeasible);
}

#[test]
fn unbounded_is_reported() {
    // min -x  s.t. x - y = 0, x,y ≥ 0: x can grow without limit.
    let p = lp(
        &[vec![1.0, -1.0]],
        &[0.0],
        &[-1.0, 0.0],
        &[0.0, 0.0],
        &[INF, INF],
    );
    assert_eq!(solve(&p).status, LpStatus::Unbounded);
}

#[test]
fn free_variable_takes_negative_value() {
    // min s  s.t. x + s = -2, s ≥ 0, x free → s = 0, x = -2 at optimum.
    let p = lp(
        &[vec![1.0, 1.0]],
        &[-2.0],
        &[0.0, 1.0],
        &[-INF, 0.0],
        &[INF, INF],
    );
    let s = solve(&p);
    assert_eq!(s.status, LpStatus::Optimal);
    assert!((s.x[0] + 2.0).abs() < 1e-7, "x = {}", s.x[0]);
}

#[test]
fn degenerate_problem_terminates() {
    // Klee-style degeneracy: several constraints active at the optimum vertex.
    let a = vec![
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let p = lp(
        &a,
        &[1.0, 1.0, 1.0, 1.0],
        &[-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0; 6],
        &[INF; 6],
    );
    let s = solve(&p);
    assert_eq!(s.status, LpStatus::Optimal);
    assert!((s.objective + 1.0).abs() < 1e-7);
}

#[test]
fn matches_dense_simplex_on_random_problems() {
    // Differential test against tpt-math-optimize-general's dense simplex,
    // max cᵀx s.t. Ax ≤ b, x ≥ 0, with a box x ≤ 10 keeping every instance bounded.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / ((1u64 << 31) as f64)
    };
    for case in 0..60 {
        let n = 2 + case % 4;
        let m = 1 + case % 3;
        let mut dense_a: Vec<Vec<f64>> = Vec::new();
        let mut b = Vec::new();
        for _ in 0..m {
            dense_a.push((0..n).map(|_| (next() * 6.0 - 1.0).round()).collect());
            b.push((next() * 20.0).round());
        }
        for j in 0..n {
            let mut row = vec![0.0; n];
            row[j] = 1.0;
            dense_a.push(row);
            b.push(10.0);
        }
        let c: Vec<f64> = (0..n).map(|_| (next() * 8.0 - 3.0).round()).collect();

        let reference =
            tpt_math_optimize_general::linear_program::maximize(&c, &dense_a, &b).unwrap();

        // Slack form: [A | I] x' = b with x' = (x, s) ≥ 0, minimise -cᵀx.
        let rows = dense_a.len();
        let mut sparse_a = vec![vec![0.0; n + rows]; rows];
        for (i, row) in dense_a.iter().enumerate() {
            sparse_a[i][..n].copy_from_slice(row);
            sparse_a[i][n + i] = 1.0;
        }
        let mut cmin = vec![0.0; n + rows];
        for j in 0..n {
            cmin[j] = -c[j];
        }
        let p = lp(
            &sparse_a,
            &b,
            &cmin,
            &vec![0.0; n + rows],
            &vec![INF; n + rows],
        );
        let s = solve(&p);

        match reference.status {
            tpt_math_optimize_general::linear_program::LpStatus::Optimal => {
                assert_eq!(s.status, LpStatus::Optimal, "case {case}");
                assert!(
                    (-s.objective - reference.objective).abs() < 1e-6,
                    "case {case}: sparse {} vs dense {}",
                    -s.objective,
                    reference.objective
                );
            }
            _ => panic!("box-bounded instance {case} unexpectedly not optimal in dense solver"),
        }
    }
}
